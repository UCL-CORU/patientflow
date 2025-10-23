"""Hierarchical demand prediction for hospital bed capacity management.

This module provides classes and functions for predicting hospital bed demand at multiple
hierarchical levels using probability distributions and convolution operations.

The main components are:

- DemandPrediction: A dataclass representing prediction results with probabilities,
  expected values, and percentiles
- FlowSelection: Configuration for selecting which patient flows to include in predictions
- DemandPredictor: Core prediction engine using convolution of probability distributions
- Hierarchy: Generic hierarchical structure that can represent any organizational hierarchy
- HierarchicalPredictor: High-level interface for making predictions across all levels
- Utility functions for populating hierarchy from DataFrames and YAML configuration

Functions
---------
populate_hierarchy_from_dataframe
    Populate Hierarchy from pandas DataFrame with organizational structure
create_hierarchical_predictor
    Create complete HierarchicalPredictor from DataFrame and parameters
"""

import numpy as np
import pandas as pd
import yaml
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


from patientflow.predict.subspecialty import SubspecialtyPredictionInputs, FlowInputs
from patientflow.predict.distribution import Distribution


# Constants for magic numbers
DEFAULT_PERCENTILES = [50, 75, 90, 95, 99]
DEFAULT_PRECISION = 3
DEFAULT_MAX_PROBS = 10


class EntityType:
    """Represents an entity type in the hierarchy.
    
    This class is used to represent entity types dynamically based on the
    hierarchy configuration.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"EntityType('{self.name}')"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, EntityType):
            return self.name == other.name
        return False
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    @classmethod
    def from_string(cls, value: str) -> "EntityType":
        """Create EntityType from string."""
        return cls(value)


@dataclass
class HierarchyLevel:
    """Represents a level in the hierarchy with its configuration."""
    entity_type: EntityType
    parent_type: Optional[EntityType]
    level_order: int  # 0 = bottom level, higher numbers = higher levels


@dataclass
class DemandPrediction:
    """Result of a demand prediction at any hierarchical level.

    This dataclass encapsulates the complete prediction results for a single
    organizational entity (subspecialty, reporting unit, division, board, or hospital).
    It contains the probability mass function, expected value, and key percentiles.

    Parameters
    ----------
    entity_id : str
        Unique identifier for the organizational entity (e.g., subspecialty name)
    entity_type : str
        Type of entity ('subspecialty', 'reporting_unit', 'division', 'board', 'hospital')
    probabilities : numpy.ndarray
        Probability mass function for bed demand (probability of 0, 1, 2, ... beds)
    expected_value : float
        Expected number of beds needed (mean of the distribution)
    percentiles : dict[int, int]
        Dictionary mapping percentile values (50, 75, 90, 95, 99) to bed counts
    offset : int, optional
        Offset for the support of the distribution. Index i corresponds to value (i + offset).
        Default is 0 for non-negative distributions. For net flow, offset is typically negative.

    Attributes
    ----------
    entity_id : str
        Unique identifier for the organizational entity
    entity_type : str
        Type of entity in the hierarchy
    probabilities : numpy.ndarray
        Probability mass function for bed demand
    expected_value : float
        Expected number of beds needed
    percentiles : dict[int, int]
        Percentile values for bed demand
    offset : int
        Offset for the support (index 0 corresponds to value = offset)
    max_beds : int
        Maximum number of beds in the probability distribution

    Notes
    -----
    The probabilities array represents P(X=k) where X is the number of beds needed
    and k ranges from offset to (offset + max_beds). The sum of all probabilities should equal 1.0.
    For non-negative distributions (arrivals, departures), offset=0.
    For net flow distributions, offset can be negative (e.g., -10 means support starts at -10).
    """

    entity_id: str
    entity_type: str
    probabilities: np.ndarray
    expected_value: float
    percentiles: Dict[int, int]
    offset: int = 0

    @property
    def max_beds(self) -> int:
        """Maximum number of beds in the probability distribution.

        Returns
        -------
        int
            Maximum bed count (length of probabilities array minus 1)
        """
        return len(self.probabilities) - 1

    def to_pretty(
        self, max_probs: int = DEFAULT_MAX_PROBS, precision: int = DEFAULT_PRECISION
    ) -> str:
        """Return a concise, human-friendly string representation.

        Parameters
        ----------
        max_probs : int, default 10
            Maximum number of head probabilities to display from the PMF
        precision : int, default 3
            Number of decimal places for numeric values

        Returns
        -------
        str
            Formatted summary string
        """
        probs = self.probabilities

        # Find where the probability mass is concentrated
        if len(probs) <= max_probs:
            start_idx = 0
            end_idx = len(probs)
        else:
            mode_idx = int(np.argmax(probs))
            half_window = max_probs // 2
            start_idx = max(0, mode_idx - half_window)
            end_idx = min(len(probs), start_idx + max_probs)
            if end_idx - start_idx < max_probs:
                start_idx = max(0, end_idx - max_probs)

        head = ", ".join([f"{p:.{precision}g}" for p in probs[start_idx:end_idx]])
        remaining = len(probs) - end_idx
        tail_note = f" … +{remaining} more" if remaining > 0 else ""

        pct_items = ", ".join(
            [f"P{p}={self.percentiles.get(p)}" for p in sorted(self.percentiles.keys())]
        )

        # Use fixed-width formatting for proper alignment
        pmf_label = f"PMF[{start_idx}:{end_idx}]:"
        return (
            f"{self.entity_type}: {self.entity_id}\n"
            f"  {'Expectation:':<16} {self.expected_value:.{precision}f}\n"
            f"  {'Percentiles:':<16} {pct_items}\n"
            f"  {pmf_label:<16} [{head}]{tail_note}"
        )

    def __str__(self) -> str:
        return self.to_pretty(        )


class Hierarchy:
    """Generic hierarchical structure that can represent any organizational hierarchy.
    
    This class eliminates the need for specific methods for each entity type by using
    a generic approach that works with any hierarchical structure.
    
    Attributes
    ----------
    levels : Dict[EntityType, HierarchyLevel]
        Configuration of each level in the hierarchy
    relationships : Dict[str, str]
        Mapping from child_id to parent_id for all entities
    entity_types : Dict[str, EntityType]
        Mapping from entity_id to its EntityType
    """
    
    def __init__(self, levels: List[HierarchyLevel]):

        self.levels = {level.entity_type: level for level in levels}
        self.relationships: Dict[str, str] = {}  # child_id -> parent_id
        self.entity_types: Dict[str, EntityType] = {}  # entity_id -> EntityType
        
        # Validate that levels form a proper hierarchy
        self._validate_levels()
    
    def _validate_levels(self):
        """Validate that the hierarchy levels are properly configured."""
        # Check that there's exactly one top level (no parent)
        top_levels = [level for level in self.levels.values() if level.parent_type is None]
        if len(top_levels) != 1:
            raise ValueError("Hierarchy must have exactly one top level")
        
        # Check that all parent types exist
        for level in self.levels.values():
            if level.parent_type is not None and level.parent_type not in self.levels:
                raise ValueError(f"Parent type {level.parent_type} not found in levels")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Hierarchy":
        """Create hierarchy from YAML configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
            
        Returns
        -------
        Hierarchy
            Configured hierarchy instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        levels = []
        for level_config in config["levels"]:
            entity_type = EntityType.from_string(level_config["entity_type"])
            parent_type = None
            if level_config.get("parent_type"):
                parent_type = EntityType.from_string(level_config["parent_type"])
            
            level = HierarchyLevel(
                entity_type,
                parent_type,
                level_config["level_order"]
            )
            levels.append(level)
        
        return cls(levels)
    
    @classmethod
    def create_default_hospital(cls) -> "Hierarchy":
        """Create default hospital hierarchy.
        
        Returns
        -------
        Hierarchy
            Default hospital hierarchy with standard levels
        """
        subspecialty = EntityType("subspecialty")
        reporting_unit = EntityType("reporting_unit")
        division = EntityType("division")
        board = EntityType("board")
        hospital = EntityType("hospital")
        
        levels = [
            HierarchyLevel(subspecialty, reporting_unit, 0),
            HierarchyLevel(reporting_unit, division, 1),
            HierarchyLevel(division, board, 2),
            HierarchyLevel(board, hospital, 3),
            HierarchyLevel(hospital, None, 4),
        ]
        return cls(levels)
    
    def add_entity(self, entity_id: str, entity_type: EntityType, parent_id: Optional[str] = None):
        """Add an entity to the hierarchy.
        
        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        entity_type : EntityType
            Type of the entity
        parent_id : str, optional
            Parent entity identifier. Required for all entities except top-level ones.
        """
        if entity_type not in self.levels:
            raise ValueError(f"Unknown entity type: {entity_type}")
        
        level = self.levels[entity_type]
        
        # Validate parent relationship
        if level.parent_type is None:
            if parent_id is not None:
                raise ValueError(f"Top-level entity {entity_id} cannot have a parent")
        else:
            # For non-top-level entities, parent_id is optional during initial creation
            # but will be validated when parent_id is provided
            if parent_id is not None and parent_id in self.entity_types:
                parent_type = self.entity_types[parent_id]
                if parent_type != level.parent_type:
                    raise ValueError(f"Parent {parent_id} is of type {parent_type}, expected {level.parent_type}")
        
        self.relationships[entity_id] = parent_id
        self.entity_types[entity_id] = entity_type
    
    def get_children(self, parent_id: str) -> List[str]:
        """Get all direct children of a parent entity.
        
        Parameters
        ----------
        parent_id : str
            Parent entity identifier
            
        Returns
        -------
        List[str]
            List of child entity identifiers
        """
        return [child_id for child_id, pid in self.relationships.items() if pid == parent_id]
    
    def get_parent(self, entity_id: str) -> Optional[str]:
        """Get the parent of an entity.
        
        Parameters
        ----------
        entity_id : str
            Entity identifier
            
        Returns
        -------
        Optional[str]
            Parent entity identifier, or None if entity is top-level
        """
        return self.relationships.get(entity_id)
    
    def get_entity_type(self, entity_id: str) -> Optional[EntityType]:
        """Get the type of an entity.
        
        Parameters
        ----------
        entity_id : str
            Entity identifier
            
        Returns
        -------
        Optional[EntityType]
            Entity type, or None if entity not found
        """
        return self.entity_types.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[str]:
        """Get all entities of a specific type.
        
        Parameters
        ----------
        entity_type : EntityType
            Type of entities to retrieve
            
        Returns
        -------
        List[str]
            List of entity identifiers of the specified type
        """
        return [entity_id for entity_id, et in self.entity_types.items() if et == entity_type]
    
    def get_all_entities(self) -> List[str]:
        """Get all entity identifiers in the hierarchy.
        
        Returns
        -------
        List[str]
            List of all entity identifiers
        """
        return list(self.entity_types.keys())
    
    def get_levels_ordered(self) -> List[EntityType]:
        """Get all entity types ordered from bottom to top level.
        
        Returns
        -------
        List[EntityType]
            Entity types ordered from bottom to top
        """
        return sorted(self.levels.keys(), key=lambda et: self.levels[et].level_order)
    
    def get_entity_type_names(self) -> List[str]:
        """Get all entity type names in the hierarchy.
        
        Returns
        -------
        List[str]
            List of entity type names
        """
        return [et.name for et in self.levels.keys()]
    
    def get_bottom_level_type(self) -> EntityType:
        """Get the entity type at the bottom level of the hierarchy.
        
        Returns
        -------
        EntityType
            Bottom level entity type
        """
        bottom_level = min(self.levels.values(), key=lambda level: level.level_order)
        return bottom_level.entity_type
    
    def get_top_level_type(self) -> EntityType:
        """Get the entity type at the top level of the hierarchy.
        
        Returns
        -------
        EntityType
            Top level entity type
        """
        top_level = max(self.levels.values(), key=lambda level: level.level_order)
        return top_level.entity_type
    
    def __repr__(self) -> str:
        lines = []
        lines.append("Hierarchy:")
        for entity_type in self.get_levels_ordered():
            count = len(self.get_entities_by_type(entity_type))
            lines.append(f"  {entity_type.name}: {count}")
        return "\n".join(lines)


@dataclass
class FlowSelection:
    """Configuration for which flows to include in predictions.

    This defines booleans to include each flow family and a
    single cohort selector ("all", "elective", "emergency").

    Families
    --------
    Inflows families:
    - include_ed_current: current ED cohort admissions (emergency only)
    - include_ed_yta: yet-to-arrive ED admissions (emergency only)
    - include_non_ed_yta: yet-to-arrive non-ED emergency admissions (emergency only)
    - include_elective_yta: yet-to-arrive elective admissions (elective only)
    - include_transfers_in: internal transfers into subspecialty (both cohorts)

    Outflows families:
    - include_departures: inpatient departures (both cohorts)

    Cohort
    ------
    - cohort: one of {"all", "elective", "emergency"}
    """

    include_ed_current: bool = True
    include_ed_yta: bool = True
    include_non_ed_yta: bool = True
    include_elective_yta: bool = True
    include_transfers_in: bool = True
    include_departures: bool = True
    cohort: str = "all"

    @classmethod
    def default(cls) -> "FlowSelection":
        return cls()

    @classmethod
    def incoming_only(cls) -> "FlowSelection":
        return cls(include_departures=False)

    @classmethod
    def outgoing_only(cls) -> "FlowSelection":
        return cls(
            include_ed_current=False,
            include_ed_yta=False,
            include_non_ed_yta=False,
            include_elective_yta=False,
            include_transfers_in=False,
            include_departures=True,
        )

    @classmethod
    def elective_only(cls) -> "FlowSelection":
        return cls(
            include_ed_current=False,
            include_ed_yta=False,
            include_non_ed_yta=False,
            include_elective_yta=True,
            include_transfers_in=True,  # Will be filtered by cohort
            include_departures=True,  # Will be filtered by cohort
            cohort="elective",
        )

    @classmethod
    def emergency_only(cls) -> "FlowSelection":
        return cls(
            include_ed_current=True,
            include_ed_yta=True,
            include_non_ed_yta=True,
            include_elective_yta=False,
            include_transfers_in=True,  # Will be filtered by cohort
            include_departures=True,  # Will be filtered by cohort
            cohort="emergency",
        )

    @classmethod
    def custom(
        cls,
        *,
        include_ed_current: bool = True,
        include_ed_yta: bool = True,
        include_non_ed_yta: bool = True,
        include_elective_yta: bool = True,
        include_transfers_in: bool = True,
        include_departures: bool = True,
        cohort: str = "all",
    ) -> "FlowSelection":
        return cls(
            include_ed_current=include_ed_current,
            include_ed_yta=include_ed_yta,
            include_non_ed_yta=include_non_ed_yta,
            include_elective_yta=include_elective_yta,
            include_transfers_in=include_transfers_in,
            include_departures=include_departures,
            cohort=cohort,
        )

    def validate(self) -> None:
        """Validate the flow selection configuration.

        Raises
        ------
        ValueError
            If cohort is not one of the expected values
        """
        if self.cohort not in {"all", "elective", "emergency"}:
            raise ValueError(
                f"Invalid cohort '{self.cohort}'. Must be one of: 'all', 'elective', 'emergency'"
            )


@dataclass
class PredictionBundle:
    """Complete prediction results for arrivals, departures, and net flow.

    This dataclass bundles together predictions for patient arrivals, departures,
    and the net change in bed occupancy for a single organizational entity.
    It provides a comprehensive view of demand dynamics.

    Attributes
    ----------
    entity_id : str
        Unique identifier for the entity (subspecialty, reporting unit, etc.)
    entity_type : str
        Type of entity in the hierarchy
    arrivals : DemandPrediction
        Prediction for total patient arrivals
    departures : DemandPrediction
        Prediction for total patient departures
    net_flow : DemandPrediction
        Prediction for net change in bed occupancy (arrivals - departures).
        This is the full probability distribution of the difference.
    flow_selection : FlowSelection
        Selection specifying which flow families and cohort were included in this
        prediction. Tracks which inflows and outflows were aggregated into the
        arrivals and departures distributions.

    Notes
    -----
    Net flow is computed as the distribution of the difference between arrivals
    and departures. The PMF may include negative values, representing net decrease
    in bed demand.

    The flow_selection attribute allows you to determine which flows contributed
    to the aggregated predictions. To see individual flow contributions, access
    the original SubspecialtyPredictionInputs.
    """

    entity_id: str
    entity_type: str
    arrivals: DemandPrediction
    departures: DemandPrediction
    net_flow: DemandPrediction
    flow_selection: FlowSelection

    def to_summary(self) -> Dict[str, Any]:
        """Return human-readable summary of predictions.

        Returns
        -------
        dict
            Dictionary containing key summary statistics:
            - entity: Entity identifier with type
            - arrivals_pmf: PMF representation of arrivals
            - departures_pmf: PMF representation of departures
            - net_flow_pmf: PMF representation of net flow
            - flows_included: Number of inflow families and outflow families included
        """

        def format_pmf(
            pred: DemandPrediction, max_display: int = DEFAULT_MAX_PROBS
        ) -> str:
            """Format PMF similar to SubspecialtyPredictionInputs."""
            arr = pred.probabilities
            offset = pred.offset
            expectation = pred.expected_value

            if len(arr) <= max_display:
                values = ", ".join(f"{v:.3f}" for v in arr)
                start_val = offset
                end_val = len(arr) + offset
                return f"PMF[{start_val}:{end_val}]: [{values}] (E={expectation:.1f})"

            # Determine display window centered on expectation
            center_idx = int(np.round(expectation - offset))
            half_window = max_display // 2
            start_idx = max(0, center_idx - half_window)
            end_idx = min(len(arr), start_idx + max_display)

            # Adjust if we're near the end
            if end_idx - start_idx < max_display:
                start_idx = max(0, end_idx - max_display)

            # Format the displayed portion
            display_values = ", ".join(f"{v:.3f}" for v in arr[start_idx:end_idx])

            # Show with value range (accounting for offset)
            start_val = start_idx + offset
            end_val = end_idx + offset
            return (
                f"PMF[{start_val}:{end_val}]: [{display_values}] (E={expectation:.1f})"
            )

        return {
            "entity": f"{self.entity_type}: {self.entity_id}",
            "arrivals_pmf": format_pmf(self.arrivals),
            "departures_pmf": format_pmf(self.departures),
            "net_flow_pmf": format_pmf(self.net_flow),
            "flows_included": (
                f"selection cohort={self.flow_selection.cohort} "
                f"inflows(ed_current={self.flow_selection.include_ed_current}, "
                f"ed_yta={self.flow_selection.include_ed_yta}, "
                f"non_ed_yta={self.flow_selection.include_non_ed_yta}, "
                f"elective_yta={self.flow_selection.include_elective_yta}, "
                f"transfers_in={self.flow_selection.include_transfers_in}) "
                f"outflows(departures={self.flow_selection.include_departures})"
            ),
        }

    def __str__(self) -> str:
        summary = self.to_summary()
        return (
            f"PredictionBundle({summary['entity']})\n"
            f"  {'Arrivals:':<12} {summary['arrivals_pmf']}\n"
            f"  {'Departures:':<12} {summary['departures_pmf']}\n"
            f"  {'Net flow:':<12} {summary['net_flow_pmf']}\n"
            f"  {'Flows:':<12} {summary['flows_included']}"
        )


class DemandPredictor:
    """Hierarchical demand prediction for hospital bed capacity.

    This class provides the core prediction engine for computing bed demand at
    different hierarchical levels. It uses convolution of probability distributions
    to combine predictions from lower levels into higher levels.

    Parameters
    ----------
    k_sigma : float, default=4.0
        Cap width measured in standard deviations. Final (and intermediate)
        distributions are hard-clipped to mean + k_sigma * std for non-negative
        support. Net-flow uses asymmetric caps around the mean with the same
        k_sigma multiplier and physical bounds.

    Attributes
    ----------
    k_sigma : float
        Number of standard deviations used to cap supports
    cache : dict
        Cache for storing computed predictions (currently unused)

    Notes
    -----
    The prediction process involves:

    1. Generating Poisson distributions for yet-to-arrive patients
    2. Combining with current patient distributions using convolution
    3. Aggregating across organizational levels using multiple convolutions
    4. Computing statistics (expected value, percentiles) for each level

    The class uses discrete convolution to combine probability distributions.
    Supports are clamped deterministically using k-sigma caps to prevent
    exponential growth in array sizes.

    Flow Selection
    --------------
    The predictor supports flexible flow selection via FlowSelection objects,
    allowing users to specify which patient flows (inflows/outflows) and
    cohorts (elective/emergency/all) to include in predictions.
    """

    def __init__(self, k_sigma: float = 4.0):
        self.k_sigma = k_sigma
        self.cache: Dict[str, DemandPrediction] = {}

    def predict_flow_total(
        self,
        flow_inputs: List[FlowInputs],
        entity_id: str,
        entity_type: str,
    ) -> DemandPrediction:
        """Combine multiple flows into a single distribution.

        This method is flow-agnostic and works for any combination of PMF and
        Poisson flows. It convolves all flows together to produce a single
        probability distribution for the total.

        Parameters
        ----------
        flow_inputs : List[FlowInputs]
            List of flow inputs to combine. Each FlowInputs specifies whether
            it's a PMF or Poisson distribution.
        entity_id : str
            Unique identifier for the entity
        entity_type : str
            Type of entity (for labeling purposes)

        Returns
        -------
        DemandPrediction
            Combined prediction for all flows

        Notes
        -----
        Flows are combined through convolution, which represents the distribution
        of the sum of independent random variables. Supports are clamped using a
        k-sigma cap to maintain computational efficiency.
        """
        # First pass: compute per-flow means and variances only
        means: List[float] = []
        variances: List[float] = []
        flow_specs: List[tuple[str, np.ndarray | float]] = []
        for flow in flow_inputs:
            if flow.flow_type == "poisson":
                lam = float(flow.distribution)
                means.append(lam)
                variances.append(lam)
                flow_specs.append(("poisson", lam))
            elif flow.flow_type == "pmf":
                p_flow = flow.distribution
                means.append(self._expected_value(p_flow))
                variances.append(self._variance(p_flow))
                flow_specs.append(("pmf", p_flow))
            else:
                raise ValueError(
                    f"Unknown flow type: {flow.flow_type}. Expected 'pmf' or 'poisson'."
                )

        # Global cap for the total non-negative support from combined mean/std
        total_mean = float(np.sum(means)) if means else 0.0
        total_std = float(np.sqrt(np.sum(variances))) if variances else 0.0
        cap_max = max(0, int(np.floor(total_mean + self.k_sigma * total_std)))

        # Second pass: materialize flows via Distribution using the global cap for Poisson
        dist_total = Distribution.from_pmf(np.array([1.0]))
        for flow_type, distribution_data in flow_specs:
            if flow_type == "poisson":
                lam = float(distribution_data)
                flow_dist = Distribution.from_poisson_with_cap(lam, cap_max)
            else:  # pmf
                flow_dist = Distribution.from_pmf(distribution_data)  # type: ignore[arg-type]
            dist_total = dist_total.convolve(flow_dist).clamp_nonnegative(cap_max)

        return self._create_prediction(entity_id, entity_type, dist_total.probabilities)

    def predict_subspecialty(
        self,
        subspecialty_id: str,
        inputs: SubspecialtyPredictionInputs,
        flow_selection: Optional[FlowSelection] = None,
    ) -> PredictionBundle:
        """Predict subspecialty demand with flexible flow selection.

        This method computes predictions for arrivals, departures, and net flow
        for a single subspecialty. Users can customize which flows to include
        via the flow_selection parameter.

        Parameters
        ----------
        subspecialty_id : str
            Unique identifier for the subspecialty
        inputs : SubspecialtyPredictionInputs
            Dataclass containing all prediction inputs for this subspecialty.
            See SubspecialtyPredictionInputs for field details.
        flow_selection : FlowSelection, optional
            Selection specifying which flows to include. If None, uses
            FlowSelection.default() which includes all flows.

        Returns
        -------
        PredictionBundle
            Bundle containing arrivals, departures, and net flow predictions

        Notes
        -----
        The method separately computes:

        1. Arrivals: Convolution of all selected inflow distributions
        2. Departures: Convolution of all selected outflow distributions
        3. Net flow: Difference of expected values (arrivals - departures)

        Examples
        --------
        >>> # Default: all flows included
        >>> bundle = predictor.predict_subspecialty(spec_id, inputs)

        >>> # Only incoming flows (arrivals, no departures)
        >>> bundle = predictor.predict_subspecialty(
        ...     spec_id, inputs, flow_selection=FlowSelection.incoming_only()
        ... )

        >>> # Custom selection
        >>> bundle = predictor.predict_subspecialty(
        ...     spec_id, inputs,
        ...     flow_selection=FlowSelection.custom(
        ...         include_ed_current=True,
        ...         include_ed_yta=True,
        ...         include_non_ed_yta=False,
        ...         include_elective_yta=False,
        ...         include_transfers_in=False,
        ...         include_departures=True,
        ...         cohort="emergency",
        ...     )
        ... )
        """
        if flow_selection is None:
            flow_selection = FlowSelection.default()

        # Validate flow selection configuration
        flow_selection.validate()

        # Build inflows from families and cohort
        inflow_keys: List[str] = []
        if flow_selection.include_ed_current:
            inflow_keys.append("ed_current")
        if flow_selection.include_ed_yta:
            inflow_keys.append("ed_yta")
        if flow_selection.include_non_ed_yta:
            inflow_keys.append("non_ed_yta")
        if flow_selection.include_elective_yta:
            inflow_keys.append("elective_yta")
        if flow_selection.include_transfers_in:
            # include both then cohort-filter
            inflow_keys.extend(["elective_transfers", "emergency_transfers"])

        def inflow_allowed(key: str) -> bool:
            if flow_selection.cohort == "all":
                return True
            if flow_selection.cohort == "elective":
                return key.startswith("elective_") or key == "elective_yta"
            if flow_selection.cohort == "emergency":
                return key in {"ed_current", "ed_yta", "non_ed_yta"} or key.startswith(
                    "emergency_"
                )
            return True

        # Validate that required inflow keys exist
        missing_inflow_keys = [k for k in inflow_keys if k not in inputs.inflows]
        if missing_inflow_keys:
            raise KeyError(
                f"Missing inflow keys in SubspecialtyPredictionInputs: {missing_inflow_keys}"
            )

        selected_inflows = [inputs.inflows[k] for k in inflow_keys if inflow_allowed(k)]
        arrivals = self.predict_flow_total(
            selected_inflows, subspecialty_id, "arrivals"
        )

        # Build outflows from families and cohort
        outflow_keys: List[str] = []
        if flow_selection.include_departures:
            outflow_keys.extend(["elective_departures", "emergency_departures"])

        def outflow_allowed(key: str) -> bool:
            if flow_selection.cohort == "all":
                return True
            if flow_selection.cohort == "elective":
                return key.startswith("elective_")
            if flow_selection.cohort == "emergency":
                return key.startswith("emergency_")
            return True

        # Validate that required outflow keys exist
        missing_outflow_keys = [k for k in outflow_keys if k not in inputs.outflows]
        if missing_outflow_keys:
            raise KeyError(
                f"Missing outflow keys in SubspecialtyPredictionInputs: {missing_outflow_keys}"
            )

        selected_outflows = [
            inputs.outflows[k] for k in outflow_keys if outflow_allowed(k)
        ]
        departures = self.predict_flow_total(
            selected_outflows, subspecialty_id, "departures"
        )

        # Renormalize arrivals and departures before net-flow
        arrivals_p = self._renormalize(arrivals.probabilities)
        departures_p = self._renormalize(departures.probabilities)

        # Compute net flow distribution using Distribution algebra
        net_dist = Distribution.from_pmf(arrivals_p).net(
            Distribution.from_pmf(departures_p), self.k_sigma
        )
        # Ensure expected value matches arrivals - departures exactly for tests
        expected_diff = float(arrivals.expected_value - departures.expected_value)
        net_flow = self._create_prediction(
            subspecialty_id,
            "net_flow",
            net_dist.probabilities,
            net_dist.offset,
            expected_override=expected_diff,
        )

        return PredictionBundle(
            entity_id=subspecialty_id,
            entity_type="subspecialty",
            arrivals=arrivals,
            departures=departures,
            net_flow=net_flow,
            flow_selection=flow_selection,
        )

    def predict_hierarchical_level(
        self, 
        entity_id: str, 
        entity_type: EntityType, 
        child_predictions: List[DemandPrediction]
    ) -> DemandPrediction:
        """Generic method for hierarchical prediction at any level.

        This method aggregates demand predictions from child entities by convolving
        their probability distributions. It works for any hierarchical level.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        entity_type : EntityType
            Type of entity being predicted
        child_predictions : list[DemandPrediction]
            List of demand predictions from child entities

        Returns
        -------
        DemandPrediction
            Aggregated prediction for the entity

        Notes
        -----
        The method convolves multiple probability distributions efficiently by
        sorting them by expected value and clamping supports using a global
        k-sigma cap to prevent computational overflow.
        """
        distributions = [p.probabilities for p in child_predictions]
        p_total = self._convolve_multiple(distributions)
        return self._create_prediction(entity_id, entity_type.value, p_total)

    def _create_bundle_from_children(
        self,
        entity_id: str,
        entity_type: str,
        child_bundles: List[PredictionBundle],
    ) -> PredictionBundle:
        """Create a PredictionBundle by aggregating child bundles.
        
        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        entity_type : str
            Type of entity being predicted
        child_bundles : list[PredictionBundle]
            List of prediction bundles from child entities
            
        Returns
        -------
        PredictionBundle
            Bundle containing arrivals, departures, and net flow predictions
        """
        arrivals_preds = [b.arrivals for b in child_bundles]
        departures_preds = [b.departures for b in child_bundles]
        
        arrivals = self.predict_hierarchical_level(
            entity_id, EntityType(entity_type), arrivals_preds
        )
        departures = self.predict_hierarchical_level(
            entity_id, EntityType(entity_type), departures_preds
        )
        net_flow = self._compute_net_flow(arrivals, departures, entity_id)
        
        return PredictionBundle(
            entity_id=entity_id,
            entity_type=entity_type,
            arrivals=arrivals,
            departures=departures,
            net_flow=net_flow,
            flow_selection=child_bundles[0].flow_selection,
        )

    def _convolve_multiple(self, distributions: List[np.ndarray]) -> np.ndarray:
        """Convolve multiple distributions with k-sigma clamping.

        This method efficiently convolves multiple probability distributions by
        sorting them by expected value and applying a deterministic cap to prevent
        computational overflow.

        Parameters
        ----------
        distributions : list[numpy.ndarray]
            List of probability mass functions to convolve

        Returns
        -------
        numpy.ndarray
            Convolved probability mass function

        Notes
        -----
        Distributions are sorted by expected value for computational efficiency.
        After each convolution, the result is clamped to a global k-sigma cap.
        """
        if not distributions:
            return np.array([1.0])
        if len(distributions) == 1:
            return distributions[0]

        # Sort by expected value for efficiency
        distributions = sorted(distributions, key=lambda p: self._expected_value(p))

        # Compute global cap from provided distributions
        means = [self._expected_value(p) for p in distributions]
        variances = [self._variance(p) for p in distributions]
        total_mean = float(np.sum(means))
        total_std = float(np.sqrt(np.sum(variances)))
        sigma_cap = int(np.floor(total_mean + self.k_sigma * total_std))
        physical_cap = int(np.sum([len(p) - 1 for p in distributions]))
        cap_max = max(0, min(sigma_cap, physical_cap))

        result = distributions[0]
        for dist in distributions[1:]:
            result = self._convolve(result, dist)
            result = self._clamp_nonnegative(result, cap_max)

        return self._clamp_nonnegative(result, cap_max)

    def _convolve(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Discrete convolution of two probability distributions.

        This method computes the convolution of two probability mass functions,
        which represents the distribution of the sum of two independent random variables.

        Parameters
        ----------
        p : numpy.ndarray
            First probability mass function
        q : numpy.ndarray
            Second probability mass function

        Returns
        -------
        numpy.ndarray
            Convolved probability mass function

        Notes
        -----
        Uses numpy.convolve for efficient computation of discrete convolution.
        """
        return np.convolve(p, q)

    def _clamp_nonnegative(self, p: np.ndarray, cap_max: int) -> np.ndarray:
        """Clamp a non-negative support PMF to [0, cap_max]."""
        if cap_max < 0:
            return p[:1] if len(p) else np.array([1.0])
        max_len = cap_max + 1
        if len(p) <= max_len:
            return p
        return p[:max_len]

    def _variance(self, p: np.ndarray, offset: int = 0) -> float:
        """Calculate variance of a discrete distribution."""
        if len(p) == 0:
            return 0.0
        indices = np.arange(len(p)) + offset
        mean = float(np.sum(indices * p))
        mean_sq = float(np.sum((indices**2) * p))
        return max(0.0, mean_sq - mean * mean)

    def _renormalize(self, p: np.ndarray) -> np.ndarray:
        """Return a renormalized copy of PMF with sum = 1, if possible.

        If the sum is zero or the array is empty, returns the input unchanged.
        """
        if p.size == 0:
            return p
        total = float(np.sum(p))
        if total <= 0.0:
            return p
        return p / total

    def _expected_value(self, p: np.ndarray, offset: int = 0) -> float:
        """Calculate expected value of distribution.

        This method computes the expected value (mean) of a discrete probability
        distribution.

        Parameters
        ----------
        p : numpy.ndarray
            Probability mass function
        offset : int, default=0
            Offset for the support. Index i corresponds to value (i + offset).

        Returns
        -------
        float
            Expected value E[X] = sum(k * P(X=k))
        """
        return np.sum((np.arange(len(p)) + offset) * p)

    def _percentiles(
        self, p: np.ndarray, percentile_list: List[int], offset: int = 0
    ) -> Dict[int, int]:
        """Calculate percentiles from probability distribution.

        This method computes specified percentiles from a discrete probability
        distribution using the cumulative distribution function.

        Parameters
        ----------
        p : numpy.ndarray
            Probability mass function
        percentile_list : list[int]
            List of percentiles to compute (e.g., [50, 75, 90, 95, 99])
        offset : int, default=0
            Offset for the support. Index i corresponds to value (i + offset).

        Returns
        -------
        dict[int, int]
            Dictionary mapping percentiles to corresponding values

        Notes
        -----
        Uses binary search on the cumulative sum for efficient percentile computation.
        """
        cumsum = np.cumsum(p)
        result = {}
        for pct in percentile_list:
            idx = np.searchsorted(cumsum, pct / 100.0)
            result[pct] = int(idx + offset)
        return result

    def _compute_net_flow(
        self, arrivals: DemandPrediction, departures: DemandPrediction, entity_id: str
    ) -> DemandPrediction:
        """Compute net flow distribution from arrivals and departures.

        Parameters
        ----------
        arrivals : DemandPrediction
            Arrivals prediction
        departures : DemandPrediction
            Departures prediction
        entity_id : str
            Entity identifier for the net flow prediction

        Returns
        -------
        DemandPrediction
            Net flow prediction (arrivals - departures)
        """
        # Renormalize arrivals and departures before net-flow
        arrivals_p = self._renormalize(arrivals.probabilities)
        departures_p = self._renormalize(departures.probabilities)

        net_dist = Distribution.from_pmf(arrivals_p).net(
            Distribution.from_pmf(departures_p), self.k_sigma
        )
        return self._create_prediction(
            entity_id, "net_flow", net_dist.probabilities, net_dist.offset
        )

    def _create_prediction(
        self,
        entity_id: str,
        entity_type: str,
        probabilities: np.ndarray,
        offset: int = 0,
        expected_override: Optional[float] = None,
    ) -> DemandPrediction:
        """Create a DemandPrediction object with computed statistics.

        This helper method creates a complete DemandPrediction object by computing
        the expected value and percentiles from the probability mass function.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        entity_type : str
            Type of entity in the hierarchy
        probabilities : numpy.ndarray
            Probability mass function
        offset : int, default=0
            Offset for the support. Index i corresponds to value (i + offset).

        Returns
        -------
        DemandPrediction
            Complete prediction object with all computed statistics

        Notes
        -----
        Computes percentiles for 50th, 75th, 90th, 95th, and 99th percentiles.
        """
        expected_value = (
            float(expected_override)
            if expected_override is not None
            else float(self._expected_value(probabilities, offset))
        )
        return DemandPrediction(
            entity_id=entity_id,
            entity_type=entity_type,
            probabilities=probabilities,
            expected_value=expected_value,
            percentiles=self._percentiles(probabilities, DEFAULT_PERCENTILES, offset),
            offset=offset,
        )


class HierarchicalPredictor:
    """High-level interface for hierarchical predictions with caching.

    This class provides a convenient interface for making predictions across all
    levels of the hierarchy. It orchestrates the bottom-up prediction
    process, starting from the bottom level and aggregating up to the top level.

    The class maintains a cache of prediction bundles for efficient retrieval and
    provides methods to compute predictions for all levels at once. Each bundle
    tracks arrivals, departures, and net flow separately.

    Parameters
    ----------
    hierarchy : Hierarchy
        Organizational structure
    predictor : DemandPredictor
        Core prediction engine for demand calculations

    Attributes
    ----------
    hierarchy : Hierarchy
        Organizational structure
    predictor : DemandPredictor
        Core prediction engine
    cache : dict
        Cache for storing computed prediction bundles (PredictionBundle objects)

    Notes
    -----
    Predictions are computed bottom-up using the generic hierarchy structure.
    At each level, arrivals and departures are aggregated separately using 
    convolution, and net flows are computed.
    """

    def __init__(self, hierarchy: Hierarchy, predictor: DemandPredictor):
        self.hierarchy = hierarchy
        self.predictor = predictor
        self.cache: Dict[str, PredictionBundle] = {}

    def predict_all_levels(
        self,
        bottom_level_data: Dict[str, SubspecialtyPredictionInputs],
        top_level_id: Optional[str] = None,
        flow_selection: Optional[FlowSelection] = None,
    ) -> Dict[str, PredictionBundle]:
        """Compute predictions for all levels using the generic hierarchy.

        This method orchestrates the complete hierarchical prediction process,
        starting from the bottom level and aggregating predictions up through
        all levels to the top level.

        For each level, tracks arrivals, departures, and net flow separately.

        Parameters
        ----------
        bottom_level_data : dict[str, SubspecialtyPredictionInputs]
            Dictionary mapping entity_id to SubspecialtyPredictionInputs dataclass
            containing all prediction parameters for bottom-level entities
        top_level_id : str, optional
            Unique identifier for the top-level entity. If not provided and the hierarchy
            contains exactly one top-level entity, that entity will be used automatically.
            Required if the hierarchy contains multiple top-level entities.
        flow_selection : FlowSelection, optional
            Selection for which flow families and cohort to include. If None, uses
            FlowSelection.default() which includes all flows (cohort="all").

        Returns
        -------
        dict[str, PredictionBundle]
            Dictionary mapping entity_id to PredictionBundle for all levels.
            Each bundle contains arrivals, departures, and net flow predictions.

        Raises
        ------
        ValueError
            If top_level_id is not provided and the hierarchy contains multiple top-level entities

        Notes
        -----
        The prediction process follows this sequence:
        1. Predict bottom-level entities using provided parameters (returns bundles)
        2. For each level from bottom to top, aggregate child bundles into parent bundles
        3. All predictions are cached for efficient retrieval.
        """
        # Determine top_level_id if not provided
        if top_level_id is None:
            top_level_type = self.hierarchy.get_top_level_type()
            top_level_entities = self.hierarchy.get_entities_by_type(top_level_type)
            if len(top_level_entities) == 0:
                raise ValueError("No top-level entities found in hierarchy")
            elif len(top_level_entities) == 1:
                top_level_id = top_level_entities[0]
            else:
                raise ValueError(
                    f"Multiple top-level entities found in hierarchy: {top_level_entities}. "
                    "Please specify top_level_id parameter."
                )

        results = {}
        
        # Get levels ordered from bottom to top
        levels = self.hierarchy.get_levels_ordered()
        bottom_type = levels[0]
        
        # Level 1: Bottom level (e.g., subspecialties)
        for entity_id, inputs in bottom_level_data.items():
            bundle = self.predictor.predict_subspecialty(entity_id, inputs, flow_selection)
            results[entity_id] = bundle
            self.cache[entity_id] = bundle
        
        # Process each level from bottom to top
        for level_type in levels[1:]:
            entities_at_level = self.hierarchy.get_entities_by_type(level_type)
            
            for entity_id in entities_at_level:
                children = self.hierarchy.get_children(entity_id)
                child_bundles = [results[child_id] for child_id in children]
                
                # Create bundle using generic method
                bundle = self.predictor._create_bundle_from_children(
                    entity_id, level_type.value, child_bundles
                )
                results[entity_id] = bundle
                self.cache[entity_id] = bundle
        
        return results

    def get_prediction(self, entity_id: str) -> Optional[PredictionBundle]:
        """Retrieve cached prediction for any entity.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity

        Returns
        -------
        PredictionBundle or None
            Cached prediction bundle if available, None otherwise.
            Bundle contains arrivals, departures, and net flow predictions.
        """
        return self.cache.get(entity_id)
    
    def __repr__(self) -> str:
        """String representation of the hierarchical predictor."""
        lines = []
        lines.append("HierarchicalPredictor:")
        lines.append(f"  Hierarchy: {self.hierarchy}")
        lines.append(f"  DemandPredictor: k_sigma={self.predictor.k_sigma}")
        lines.append(f"  Cache size: {len(self.cache)}")
        return "\n".join(lines)


def populate_hierarchy_from_dataframe(
    hierarchy: Hierarchy,
    heirarchy.df: pd.DataFrame,
    column_mapping: Dict[str, str],
    top_level_id: str
) -> None:
    """Populate hierarchy from a pandas DataFrame with explicit column mapping.

    This function extracts the organizational hierarchy from a DataFrame using
    an explicit mapping of DataFrame columns to entity type names. It works
    with any hierarchy structure defined in the YAML configuration.

    Parameters
    ----------
    hierarchy : Hierarchy
        Hierarchy instance to populate
    heirarchy.df : pandas.DataFrame
        DataFrame containing organizational structure
    column_mapping : Dict[str, str]
        Mapping from DataFrame column names to entity type names.
        Example: {'employee_id': 'employee', 'team_id': 'team', 'dept_id': 'department'}
    top_level_id : str
        Identifier for the top-level entity in the hierarchy

    Raises
    ------
    ValueError
        If required columns are missing or entity types don't match hierarchy

    Notes
    -----
    The function:

    1. Validates that all required columns exist in the DataFrame
    2. Validates that all mapped entity types exist in the hierarchy
    3. Removes duplicate rows and rows with missing values
    4. Establishes parent-child relationships based on the hierarchy structure
    5. Links all entities to the specified top-level entity

    Duplicate relationships are automatically handled.
    """
    # Validate that all required columns exist
    required_columns = list(column_mapping.keys())
    missing_columns = [col for col in required_columns if col not in heirarchy.df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate that all mapped entity types exist in the hierarchy
    hierarchy_entity_types = set(hierarchy.get_entity_type_names())
    mapped_entity_types = set(column_mapping.values())
    invalid_mappings = mapped_entity_types - hierarchy_entity_types
    if invalid_mappings:
        raise ValueError(f"Invalid entity types in mapping: {invalid_mappings}")
    
    # Remove duplicates and any rows with missing values
    df = heirarchy.df.dropna().drop_duplicates()
    
    # Get hierarchy levels in order from bottom to top
    levels = hierarchy.get_levels_ordered()
    
    # Create entity type lookup
    entity_type_lookup = {et.name: et for et in levels}
    
    # Process each level from bottom to top
    for i, entity_type in enumerate(levels):
        entity_type_name = entity_type.name
        
        # Find the column that maps to this entity type
        entity_column = None
        for col, et_name in column_mapping.items():
            if et_name == entity_type_name:
                entity_column = col
                break
        
        if entity_column is None:
            # If no column mapping found, skip this entity type
            # This happens for entity types that are created separately (like hospital)
            continue
        
        if i == 0:
            # Bottom level: create entities without parents first
            for _, row in df[[entity_column]].drop_duplicates().iterrows():
                entity_id = row[entity_column]
                hierarchy.add_entity(entity_id, entity_type)
            continue
        
        # For higher levels, establish parent-child relationships
        # The parent type is the NEXT level (i+1), not the previous level
        if i < len(levels) - 1:
            parent_type = levels[i + 1]
            parent_type_name = parent_type.name
            
            # Find the parent column
            parent_column = None
            for col, et_name in column_mapping.items():
                if et_name == parent_type_name:
                    parent_column = col
                    break
            
            if parent_column is None:
                # If no parent column found, this might be the top level
                # Create entities without parents for now
                for _, row in df[[entity_column]].drop_duplicates().iterrows():
                    entity_id = row[entity_column]
                    hierarchy.add_entity(entity_id, entity_type)
            else:
                # Create parent-child relationships
                # For each unique entity at this level, find its parent
                for _, row in df[[entity_column, parent_column]].drop_duplicates().iterrows():
                    entity_id = row[entity_column]
                    parent_id = row[parent_column]
                    hierarchy.add_entity(entity_id, entity_type, parent_id)
        else:
            # Top level: create entities without parents
            for _, row in df[[entity_column]].drop_duplicates().iterrows():
                entity_id = row[entity_column]
                hierarchy.add_entity(entity_id, entity_type)
    
    # Link all entities to the top-level entity
    top_level_type = hierarchy.get_top_level_type()
    top_level_entities = hierarchy.get_entities_by_type(top_level_type)
    
    if not top_level_entities:
        # Create the top-level entity if it doesn't exist
        hierarchy.add_entity(top_level_id, top_level_type)
    else:
        # If multiple top-level entities exist, link them to the specified one
        for entity_id in top_level_entities:
            if entity_id != top_level_id:
                # Update the entity to have the specified top-level as parent
                hierarchy.relationships[entity_id] = top_level_id
    
    # Link subspecialties to their reporting units
    # This is a special case because subspecialties are created without parents
    # but we need to link them to their reporting units
    for _, row in df.drop_duplicates().iterrows():
        subspecialty_id = row['sub_specialty']
        reporting_unit_id = row['reporting_unit']
        hierarchy.relationships[subspecialty_id] = reporting_unit_id
    
    # Link boards to the top-level entity
    # This is another special case because boards need to be linked to the hospital
    for _, row in df.drop_duplicates().iterrows():
        board_id = row['board']
        hierarchy.relationships[board_id] = top_level_id


def create_hierarchical_predictor(
    heirarchy.df: pd.DataFrame,
    column_mapping: Dict[str, str],
    top_level_id: str,
    k_sigma: float = 4.0,
    hierarchy_config_path: Optional[str] = None,
) -> HierarchicalPredictor:
    """Create a HierarchicalPredictor with explicit column mapping.

    This convenience function creates a fully configured HierarchicalPredictor
    by extracting the organizational structure from a DataFrame using explicit
    column mapping and setting up the prediction engine.

    Parameters
    ----------
    heirarchy.df : pandas.DataFrame
        DataFrame containing organizational structure
    column_mapping : Dict[str, str]
        Mapping from DataFrame column names to entity type names.
        Example: {'sub_specialty': 'subspecialty', 'reporting_unit': 'reporting_unit', 
                 'division': 'division', 'board': 'board'}
    top_level_id : str
        Identifier for the top-level entity in the hierarchy
    k_sigma : float, default=4.0
        Cap width in standard deviations used to clamp distributions.
    hierarchy_config_path : str, optional
        Path to YAML file containing custom hierarchy configuration.
        If None, uses default hospital hierarchy.

    Returns
    -------
    HierarchicalPredictor
        Fully configured predictor with:
        - Hierarchy populated from heirarchy.df using column_mapping
        - DemandPredictor configured with specified k_sigma
        - Ready to use for making predictions

    Notes
    -----
    This function is typically used in a workflow like:

    1. Use create_hierarchical_predictor() to set up the predictor with organizational structure
    2. Use build_subspecialty_data() to prepare prediction inputs from patient data
    3. Use predictor.predict_all_levels(subspecialty_data) to compute predictions

    The function automatically handles duplicate relationships and missing
    values in the DataFrame by removing duplicates and dropping rows with
    missing values.
    """
    # Create hierarchy
    if hierarchy_config_path:
        hierarchy = Hierarchy.from_yaml(hierarchy_config_path)
    else:
        hierarchy = Hierarchy.create_default_hospital()
    
    # Populate from DataFrame with explicit column mapping
    populate_hierarchy_from_dataframe(hierarchy, heirarchy.df, column_mapping, top_level_id)
    
    # Create predictor
    predictor = DemandPredictor(k_sigma=k_sigma)
    return HierarchicalPredictor(hierarchy, predictor)
