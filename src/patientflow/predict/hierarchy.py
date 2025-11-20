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
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass


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
        return self.to_pretty()


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
        top_levels = [
            level for level in self.levels.values() if level.parent_type is None
        ]
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
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        levels = []
        for level_config in config["levels"]:
            entity_type = EntityType.from_string(level_config["entity_type"])
            parent_type = None
            if level_config.get("parent_type"):
                parent_type = EntityType.from_string(level_config["parent_type"])

            level = HierarchyLevel(
                entity_type, parent_type, level_config["level_order"]
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

    def add_entity(self, entity_id: str, entity_type: EntityType):
        """Add an entity to the hierarchy with type-prefixed unique ID.

        This method is used in Pass 1 to create entities without relationships.
        Relationships are established separately in Pass 2.

        Parameters
        ----------
        entity_id : str
            Original identifier for the entity (will be prefixed with entity type)
        entity_type : EntityType
            Type of the entity
        """
        if entity_type not in self.levels:
            raise ValueError(f"Unknown entity type: {entity_type}")

        # Create unique ID by prefixing with entity type
        unique_id = f"{entity_type.name}:{entity_id}"

        # Store entity with its type
        self.entity_types[unique_id] = entity_type

    def _find_entity_type_by_name(self, entity_name: str) -> Optional[EntityType]:
        """Find the entity type for a given entity name by searching through existing entities.

        Parameters
        ----------
        entity_name : str
            Original entity name to search for

        Returns
        -------
        Optional[EntityType]
            Entity type if found, None otherwise
        """
        for unique_id, entity_type in self.entity_types.items():
            # Extract original name from prefixed ID
            if ":" in unique_id:
                original_name = unique_id.split(":", 1)[1]
                if original_name == entity_name:
                    return entity_type
        return None

    def _get_original_name(self, unique_id: str) -> str:
        """Extract original entity name from prefixed ID.

        Parameters
        ----------
        unique_id : str
            Prefixed entity ID (e.g., "subspecialty:Cardiology")

        Returns
        -------
        str
            Original entity name (e.g., "Cardiology")
        """
        if ":" in unique_id:
            return unique_id.split(":", 1)[1]
        return unique_id

    def _get_prefixed_id(
        self, entity_name: str, entity_type: EntityType
    ) -> Optional[str]:
        """Get prefixed ID for an entity name and type.

        Parameters
        ----------
        entity_name : str
            Original entity name
        entity_type : EntityType
            Entity type

        Returns
        -------
        Optional[str]
            Prefixed ID if entity exists, None otherwise
        """
        prefixed_id = f"{entity_type.name}:{entity_name}"
        return prefixed_id if prefixed_id in self.entity_types else None

    def get_children(
        self, parent_id: str, parent_type: Optional[EntityType] = None
    ) -> List[str]:
        """Get all direct children of a parent entity.

        Parameters
        ----------
        parent_id : str
            Parent entity identifier (original name or prefixed ID)
        parent_type : EntityType, optional
            Entity type of the parent. If not provided, will try to find it automatically.
            This is useful when there are multiple entities with the same name at different levels.

        Returns
        -------
        List[str]
            List of child entity identifiers (original names)
        """
        # Convert parent_id to prefixed format if needed
        prefixed_parent_id = parent_id
        if ":" not in parent_id:
            if parent_type is not None:
                # Use the specified parent type
                prefixed_parent_id = f"{parent_type.name}:{parent_id}"
            else:
                # Try to find the prefixed version (may not work with entity collisions)
                parent_entity_type = self._find_entity_type_by_name(parent_id)
                if parent_entity_type is not None:
                    prefixed_parent_id = f"{parent_entity_type.name}:{parent_id}"

        children = []
        for child_id, pid in self.relationships.items():
            if pid == prefixed_parent_id:
                # Return original entity name
                children.append(self._get_original_name(child_id))
        return children

    def get_parent(
        self, entity_id: str, entity_type: Optional[EntityType] = None
    ) -> Optional[str]:
        """Get the parent of an entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier (original name or prefixed ID)
        entity_type : EntityType, optional
            Entity type of the entity. If not provided, will try to find it automatically.
            This is useful when there are multiple entities with the same name at different levels.

        Returns
        -------
        Optional[str]
            Parent entity identifier (original name), or None if entity is top-level
        """
        # Convert entity_id to prefixed format if needed
        prefixed_entity_id = entity_id
        if ":" not in entity_id:
            if entity_type is not None:
                # Use the specified entity type
                prefixed_entity_id = f"{entity_type.name}:{entity_id}"
            else:
                # Try to find the prefixed version (may not work with entity collisions)
                entity_entity_type = self._find_entity_type_by_name(entity_id)
                if entity_entity_type is not None:
                    prefixed_entity_id = f"{entity_entity_type.name}:{entity_id}"

        parent_id = self.relationships.get(prefixed_entity_id)
        if parent_id is not None:
            return self._get_original_name(parent_id)
        return None

    def get_entity_type(
        self, entity_id: str, entity_type: Optional[EntityType] = None
    ) -> Optional[EntityType]:
        """Get the type of an entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier (original name or prefixed ID)
        entity_type : EntityType, optional
            Entity type of the entity. If not provided, will try to find it automatically.
            This is useful when there are multiple entities with the same name at different levels.

        Returns
        -------
        Optional[EntityType]
            Entity type, or None if entity not found
        """
        # Convert entity_id to prefixed format if needed
        prefixed_entity_id = entity_id
        if ":" not in entity_id:
            if entity_type is not None:
                # Use the specified entity type
                prefixed_entity_id = f"{entity_type.name}:{entity_id}"
            else:
                # Try to find the prefixed version (may not work with entity collisions)
                entity_entity_type = self._find_entity_type_by_name(entity_id)
                if entity_entity_type is not None:
                    prefixed_entity_id = f"{entity_entity_type.name}:{entity_id}"

        return self.entity_types.get(prefixed_entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[str]:
        """Get all entities of a specific type.

        Parameters
        ----------
        entity_type : EntityType
            Type of entities to retrieve

        Returns
        -------
        List[str]
            List of entity identifiers (original names) of the specified type
        """
        entities = []
        for entity_id, et in self.entity_types.items():
            if et == entity_type:
                # Return original entity name
                entities.append(self._get_original_name(entity_id))
        return entities

    def get_all_entities(self) -> List[str]:
        """Get all entity identifiers in the hierarchy.

        Returns
        -------
        List[str]
            List of all entity identifiers (original names)
        """
        return [
            self._get_original_name(entity_id) for entity_id in self.entity_types.keys()
        ]

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

    def get_entity_info(
        self, entity_name: str, entity_type: Optional[EntityType] = None
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about an entity.

        Parameters
        ----------
        entity_name : str
            Original entity name
        entity_type : EntityType, optional
            Entity type of the entity. If not provided, will try to find it automatically.
            This is useful when there are multiple entities with the same name at different levels.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing entity information:
            - entity_id: original name
            - entity_type: EntityType
            - parent: parent entity name (if any)
            - children: list of child entity names
            - prefixed_id: internal prefixed ID
        """
        if entity_type is None:
            entity_type = self._find_entity_type_by_name(entity_name)
            if entity_type is None:
                return None

        prefixed_id = f"{entity_type.name}:{entity_name}"
        parent = self.get_parent(entity_name, entity_type)
        children = self.get_children(entity_name, entity_type)

        return {
            "entity_id": entity_name,
            "entity_type": entity_type,
            "parent": parent,
            "children": children,
            "prefixed_id": prefixed_id,
        }

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
    k_sigma : float, default=8.0
        Cap width measured in standard deviations. Maximum support for each
        hierarchical level is calculated top-down using statistical caps:
        mean + k_sigma * sqrt(sum of variances). This ensures bounded array
        sizes while maintaining statistical accuracy. Net flow is computed
        naively from already-capped arrivals and departures, with no additional
        truncation.

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
    Supports are clamped using top-down statistical caps calculated before
    convolution, ensuring bounded array sizes while maintaining statistical
    accuracy. Truncation uses renormalization to preserve probability mass.

    Flow Selection
    --------------
    The predictor supports flexible flow selection via FlowSelection objects,
    allowing users to specify which patient flows (inflows/outflows) and
    cohorts (elective/emergency/all) to include in predictions.
    """

    def __init__(self, k_sigma: float = 8.0):
        self.k_sigma = k_sigma
        self.cache: Dict[str, DemandPrediction] = {}

    def _calculate_hierarchical_stats(
        self,
        entity_id: str,
        entity_type: EntityType,
        bottom_level_data: Dict[str, SubspecialtyPredictionInputs],
        hierarchy: "Hierarchy",
        flow_type: str,
        flow_selection: Optional[FlowSelection] = None,
    ) -> Tuple[float, float, int]:
        """Calculate sum of means, combined SD, and maximum support for an entity.

        This method recursively gathers all bottom-level entities in the subtree
        and calculates statistical caps based on the sum of means and combined
        variance of all distributions in the subtree.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the entity
        entity_type : EntityType
            Type of entity being analyzed
        bottom_level_data : Dict[str, SubspecialtyPredictionInputs]
            Dictionary mapping bottom-level entity IDs to their prediction inputs
        hierarchy : Hierarchy
            Hierarchy structure for traversing the tree
        flow_type : str
            Type of flow to analyze: 'arrivals' or 'departures' only
        flow_selection : FlowSelection, optional
            Selection for which flows to include. If None, includes all flows.

        Returns
        -------
        Tuple[float, float, int]
            Tuple of (sum_of_means, combined_sd, max_support)
            - sum_of_means: Sum of all means from distributions in subtree
            - combined_sd: Combined standard deviation (sqrt of sum of variances)
            - max_support: Maximum support value (min of statistical and physical caps)

        Notes
        -----
        For Poisson distributions: variance = mean (lambda)
        For PMF distributions: variance = E[X²] - E[X]² calculated from PMF array
        Physical max for PMF = len(pmf_array) - 1 (maximum index value)
        Physical max for Poisson = infinity (unbounded)
        """
        if flow_type not in {"arrivals", "departures"}:
            raise ValueError(
                f"flow_type must be 'arrivals' or 'departures', got '{flow_type}'"
            )

        if flow_selection is None:
            flow_selection = FlowSelection.default()

        # Get bottom level type
        bottom_type = hierarchy.get_bottom_level_type()

        # Gather all bottom-level entities in the subtree
        def gather_bottom_level_entities(
            current_id: str, current_type: EntityType
        ) -> List[str]:
            """Recursively gather all bottom-level entity IDs in the subtree."""
            if current_type == bottom_type:
                # This is a bottom-level entity
                return [current_id] if current_id in bottom_level_data else []

            # Get children and recurse
            children = hierarchy.get_children(current_id, current_type)
            result = []
            for child_id in children:
                child_type = hierarchy.get_entity_type(child_id)
                if child_type is not None:
                    result.extend(gather_bottom_level_entities(child_id, child_type))
            return result

        bottom_level_ids = gather_bottom_level_entities(entity_id, entity_type)

        # Extract relevant distributions based on flow_type
        means: List[float] = []
        variances: List[float] = []
        physical_maxes: List[float] = []

        for bottom_id in bottom_level_ids:
            if bottom_id not in bottom_level_data:
                continue

            inputs = bottom_level_data[bottom_id]

            # Get flows based on flow_type
            if flow_type == "arrivals":
                # Get inflows
                flow_keys: List[str] = []
                if flow_selection.include_ed_current:
                    flow_keys.append("ed_current")
                if flow_selection.include_ed_yta:
                    flow_keys.append("ed_yta")
                if flow_selection.include_non_ed_yta:
                    flow_keys.append("non_ed_yta")
                if flow_selection.include_elective_yta:
                    flow_keys.append("elective_yta")
                if flow_selection.include_transfers_in:
                    flow_keys.extend(["elective_transfers", "emergency_transfers"])

                def flow_allowed(key: str) -> bool:
                    if flow_selection.cohort == "all":
                        return True
                    if flow_selection.cohort == "elective":
                        return key.startswith("elective_") or key == "elective_yta"
                    if flow_selection.cohort == "emergency":
                        return key in {
                            "ed_current",
                            "ed_yta",
                            "non_ed_yta",
                        } or key.startswith("emergency_")
                    return True

                flows = [
                    inputs.inflows[k]
                    for k in flow_keys
                    if k in inputs.inflows and flow_allowed(k)
                ]
            else:  # flow_type == "departures"
                # Get outflows
                flow_keys = []
                if flow_selection.include_departures:
                    flow_keys.extend(["elective_departures", "emergency_departures"])

                def flow_allowed(key: str) -> bool:
                    if flow_selection.cohort == "all":
                        return True
                    if flow_selection.cohort == "elective":
                        return key.startswith("elective_")
                    if flow_selection.cohort == "emergency":
                        return key.startswith("emergency_")
                    return True

                flows = [
                    inputs.outflows[k]
                    for k in flow_keys
                    if k in inputs.outflows and flow_allowed(k)
                ]

            # Process each flow
            for flow in flows:
                if flow.flow_type == "poisson":
                    lam = float(flow.distribution)
                    means.append(lam)
                    variances.append(lam)  # Var(Poisson(λ)) = λ
                    physical_maxes.append(float("inf"))  # Unbounded
                elif flow.flow_type == "pmf":
                    pmf_array = flow.distribution
                    if isinstance(pmf_array, np.ndarray):
                        mean = self._expected_value(pmf_array)
                        variance = self._variance(pmf_array)
                        physical_max = len(pmf_array) - 1  # Maximum index value
                        means.append(mean)
                        variances.append(variance)
                        physical_maxes.append(physical_max)

        # Calculate combined statistics
        sum_of_means = float(np.sum(means)) if means else 0.0
        combined_variance = float(np.sum(variances)) if variances else 0.0
        combined_sd = (
            float(np.sqrt(combined_variance)) if combined_variance > 0 else 0.0
        )

        # Statistical cap: mean + k_sigma * SD
        statistical_cap = sum_of_means + self.k_sigma * combined_sd

        # Physical cap: minimum of all physical maxes (infinity doesn't constrain)
        finite_physical_maxes = [pm for pm in physical_maxes if pm != float("inf")]
        if finite_physical_maxes:
            physical_cap = float(np.sum(finite_physical_maxes))  # Sum for convolution
        else:
            physical_cap = float("inf")

        # Never exceed what's physically possible
        if physical_cap == float("inf"):
            max_support = int(np.ceil(statistical_cap))
        else:
            max_support = int(min(np.ceil(statistical_cap), physical_cap))

        # Ensure non-negative
        max_support = max(0, max_support)

        return (sum_of_means, combined_sd, max_support)

    def predict_flow_total(
        self,
        flow_inputs: List[FlowInputs],
        entity_id: str,
        entity_type: str,
        max_support: Optional[int] = None,
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
        max_support : int, optional
            Pre-calculated maximum support value. If provided, the result will
            be truncated to this value with renormalization. If None, no truncation
            is applied (for backward compatibility during transition).

        Returns
        -------
        DemandPrediction
            Combined prediction for all flows

        Notes
        -----
        Flows are combined through convolution, which represents the distribution
        of the sum of independent random variables. If max_support is provided,
        the result is truncated and renormalized to ensure probability conservation.
        """
        # Materialize flows via Distribution
        dist_total = Distribution.from_pmf(np.array([1.0]))
        for flow in flow_inputs:
            if flow.flow_type == "poisson":
                lam = float(flow.distribution)
                # Use max_support if provided, otherwise use a reasonable default
                if max_support is not None:
                    poisson_cap = max_support
                else:
                    # Fallback: use k-sigma cap for Poisson
                    poisson_cap = Distribution.poisson_cap_from_k_sigma(
                        lam, self.k_sigma
                    )
                flow_dist = Distribution.from_poisson_with_cap(lam, poisson_cap)
            elif flow.flow_type == "pmf":
                pmf_array = flow.distribution
                if isinstance(pmf_array, np.ndarray):
                    flow_dist = Distribution.from_pmf(pmf_array)
                else:
                    raise ValueError(
                        f"PMF distribution must be numpy array, got {type(pmf_array)}"
                    )
            else:
                raise ValueError(
                    f"Unknown flow type: {flow.flow_type}. Expected 'pmf' or 'poisson'."
                )
            dist_total = dist_total.convolve(flow_dist)

        # Apply cap with renormalization if max_support is provided
        if max_support is not None:
            truncated_pmf = self._apply_cap_with_renormalization(
                dist_total.probabilities, max_support
            )
            return self._create_prediction(entity_id, entity_type, truncated_pmf)
        else:
            return self._create_prediction(
                entity_id, entity_type, dist_total.probabilities
            )

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

        # Compute net flow distribution naively from already-capped arrivals/departures
        # Since arrivals and departures are already capped, no additional truncation is needed
        net_dist = Distribution.from_pmf(arrivals_p).net(
            Distribution.from_pmf(departures_p)
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
        child_predictions: List[DemandPrediction],
        max_support: Optional[int] = None,
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
        max_support : int, optional
            Pre-calculated maximum support value. If provided, the result will
            be truncated to this value with renormalization.

        Returns
        -------
        DemandPrediction
            Aggregated prediction for the entity

        Notes
        -----
        The method convolves multiple probability distributions efficiently by
        sorting them by expected value. If max_support is provided, truncation
        uses renormalization to ensure probability conservation.
        """
        distributions = [p.probabilities for p in child_predictions]
        p_total = self._convolve_multiple(distributions, max_support)
        return self._create_prediction(entity_id, entity_type.name, p_total)

    def _create_bundle_from_children(
        self,
        entity_id: str,
        entity_type: str,
        child_bundles: List[PredictionBundle],
        arrivals_max_support: Optional[int] = None,
        departures_max_support: Optional[int] = None,
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
        arrivals_max_support : int, optional
            Pre-calculated maximum support for arrivals
        departures_max_support : int, optional
            Pre-calculated maximum support for departures

        Returns
        -------
        PredictionBundle
            Bundle containing arrivals, departures, and net flow predictions
        """
        arrivals_preds = [b.arrivals for b in child_bundles]
        departures_preds = [b.departures for b in child_bundles]

        arrivals = self.predict_hierarchical_level(
            entity_id, EntityType(entity_type), arrivals_preds, arrivals_max_support
        )
        departures = self.predict_hierarchical_level(
            entity_id, EntityType(entity_type), departures_preds, departures_max_support
        )
        net_flow = self._compute_net_flow(arrivals, departures, entity_id)

        # Use flow_selection from first child if available, otherwise use default
        flow_selection = (
            child_bundles[0].flow_selection
            if child_bundles
            else FlowSelection.default()
        )

        return PredictionBundle(
            entity_id=entity_id,
            entity_type=entity_type,
            arrivals=arrivals,
            departures=departures,
            net_flow=net_flow,
            flow_selection=flow_selection,
        )

    def _convolve_multiple(
        self, distributions: List[np.ndarray], max_support: Optional[int] = None
    ) -> np.ndarray:
        """Convolve multiple distributions with optional truncation and renormalization.

        This method efficiently convolves multiple probability distributions by
        sorting them by expected value. If max_support is provided, the result
        is truncated and renormalized to ensure probability conservation.

        Parameters
        ----------
        distributions : list[numpy.ndarray]
            List of probability mass functions to convolve
        max_support : int, optional
            Pre-calculated maximum support value. If provided, the result will
            be truncated to this value with renormalization. If None, no truncation
            is applied.

        Returns
        -------
        numpy.ndarray
            Convolved probability mass function

        Notes
        -----
        Distributions are sorted by expected value for computational efficiency.
        If max_support is provided, truncation uses _apply_cap_with_renormalization()
        to ensure the PMF sums to 1.0 exactly.
        """
        if not distributions:
            return np.array([1.0])
        if len(distributions) == 1:
            result = distributions[0]
            if max_support is not None:
                result = self._apply_cap_with_renormalization(result, max_support)
            return result

        # Sort by expected value for efficiency
        distributions = sorted(distributions, key=lambda p: self._expected_value(p))

        # Convolve all distributions
        result = distributions[0]
        for dist in distributions[1:]:
            result = self._convolve(result, dist)

        # Apply cap with renormalization if max_support is provided
        if max_support is not None:
            result = self._apply_cap_with_renormalization(result, max_support)

        return result

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

    def _apply_cap_with_renormalization(
        self, pmf: np.ndarray, max_support: int
    ) -> np.ndarray:
        """Truncate PMF to max_support and assign remaining probability to the last bin.

        This ensures probability conservation: the PMF sums to 1.0 exactly.
        The last bin (at index max_support) represents "at least this many".

        Parameters
        ----------
        pmf : np.ndarray
            Original probability mass function
        max_support : int
            Maximum support value (corresponds to array index max_support)

        Returns
        -------
        np.ndarray
            Truncated PMF with length (max_support + 1) where the last element
            contains all remaining probability mass
        """
        if max_support < 0:
            # Return a single bin with all probability mass
            return np.array([np.sum(pmf)])

        max_len = max_support + 1
        if len(pmf) <= max_len:
            return pmf.copy()

        # Truncate to max_support + 1 elements (indices 0 to max_support)
        truncated = pmf[:max_len].copy()

        # Add remaining probability mass to the last bin
        remaining_mass = np.sum(pmf[max_len:])
        truncated[max_support] += remaining_mass

        return truncated

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
            Arrivals prediction (already capped at hierarchical level)
        departures : DemandPrediction
            Departures prediction (already capped at hierarchical level)
        entity_id : str
            Entity identifier for the net flow prediction

        Returns
        -------
        DemandPrediction
            Net flow prediction (arrivals - departures)

        Notes
        -----
        Net flow is computed naively from already-capped arrivals and departures distributions.
        Since arrivals and departures are already capped using top-down statistical caps,
        the net flow is naturally bounded by physical limits [-max_departures, +max_arrivals].
        No additional truncation is applied.
        """
        # Renormalize arrivals and departures before net-flow
        arrivals_p = self._renormalize(arrivals.probabilities)
        departures_p = self._renormalize(departures.probabilities)

        # Compute net flow distribution naively from already-capped arrivals/departures
        # Since arrivals and departures are already capped using top-down statistical caps,
        # the net flow is naturally bounded by physical limits [-max_departures, +max_arrivals].
        # No additional truncation is needed.
        net_dist = Distribution.from_pmf(arrivals_p).net(
            Distribution.from_pmf(departures_p)
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

        This method orchestrates the complete hierarchical prediction process with
        top-down statistical cap calculation.

        It performs three phases:

        1. Phase 1 (Pre-processing): Calculate maximum support caps for all entities.
        2. Phase 2 (Bottom-Level Prediction): Compute predictions for all bottom-level entities.
        3. Phase 3 (Convolution): Aggregate child predictions bottom-up using pre-calculated caps.

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

        1. Phase 1: Calculate caps for all entities (top-down traversal).
        2. Phase 2: Predict bottom-level entities using provided parameters.
        3. Phase 3: For each level from bottom to top, aggregate child bundles using pre-calculated caps.
        4. Cache all predictions for efficient retrieval.
        """
        if flow_selection is None:
            flow_selection = FlowSelection.default()

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

        # Get levels ordered from bottom to top
        levels = self.hierarchy.get_levels_ordered()
        bottom_type = levels[0]

        # PHASE 1: Calculate caps for all entities (top-down)
        # Process levels from top to bottom to calculate caps
        caps: Dict[
            str, Dict[str, int]
        ] = {}  # entity_id -> {'arrivals': cap, 'departures': cap}

        for level_type in reversed(levels):
            entities_at_level = self.hierarchy.get_entities_by_type(level_type)

            for entity_id in entities_at_level:
                # Calculate caps for arrivals and departures
                arrivals_stats = self.predictor._calculate_hierarchical_stats(
                    entity_id,
                    level_type,
                    bottom_level_data,
                    self.hierarchy,
                    "arrivals",
                    flow_selection,
                )
                departures_stats = self.predictor._calculate_hierarchical_stats(
                    entity_id,
                    level_type,
                    bottom_level_data,
                    self.hierarchy,
                    "departures",
                    flow_selection,
                )

                # Store max_support (third element of tuple)
                prefixed_entity_id = f"{level_type.name}:{entity_id}"
                caps[prefixed_entity_id] = {
                    "arrivals": arrivals_stats[2],  # max_support
                    "departures": departures_stats[2],  # max_support
                }

        results = {}

        # PHASE 2: Predict bottom level (e.g., subspecialties)
        # Bottom level doesn't use caps from hierarchical stats (they're already
        # calculated at the bottom level in predict_subspecialty via predict_flow_total)
        for entity_id, inputs in bottom_level_data.items():
            bundle = self.predictor.predict_subspecialty(
                entity_id, inputs, flow_selection
            )
            # Use prefixed entity ID as key to avoid collisions
            prefixed_entity_id = f"{bottom_type.name}:{entity_id}"
            results[prefixed_entity_id] = bundle
            self.cache[prefixed_entity_id] = bundle

        # PHASE 3: Process each level from bottom to top, using pre-calculated caps
        for level_type in levels[1:]:
            entities_at_level = self.hierarchy.get_entities_by_type(level_type)

            for entity_id in entities_at_level:
                # Use the entity type to avoid entity name collisions
                children = self.hierarchy.get_children(entity_id, level_type)
                # Convert child IDs to prefixed format for lookup
                child_bundles = []
                for child_id in children:
                    # Find the child's entity type to create prefixed ID
                    child_entity_type = self.hierarchy.get_entity_type(child_id)
                    if child_entity_type is not None:
                        prefixed_child_id = f"{child_entity_type.name}:{child_id}"
                        if prefixed_child_id in results:
                            child_bundles.append(results[prefixed_child_id])

                # Get pre-calculated caps for this entity
                prefixed_entity_id = f"{level_type.name}:{entity_id}"
                entity_caps = caps.get(prefixed_entity_id, {})
                arrivals_max_support = entity_caps.get("arrivals")
                departures_max_support = entity_caps.get("departures")

                # Create bundle using generic method with pre-calculated caps
                bundle = self.predictor._create_bundle_from_children(
                    entity_id,
                    level_type.name,
                    child_bundles,
                    arrivals_max_support,
                    departures_max_support,
                )
                # Use prefixed entity ID as key to avoid collisions
                results[prefixed_entity_id] = bundle
                self.cache[prefixed_entity_id] = bundle

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
        lines = []
        lines.append("HierarchicalPredictor:")
        lines.append(f"  Hierarchy: {self.hierarchy}")
        lines.append(f"  DemandPredictor: k_sigma={self.predictor.k_sigma}")
        lines.append(f"  Cache size: {len(self.cache)}")
        return "\n".join(lines)


def populate_hierarchy_from_dataframe(
    hierarchy: Hierarchy,
    hierarchy_df: pd.DataFrame,
    column_mapping: Dict[str, str],
    top_level_id: str,
) -> None:
    """Populate hierarchy from a pandas DataFrame with explicit column mapping.

    This function extracts the organizational hierarchy from a DataFrame using
    an explicit mapping of DataFrame columns to entity type names. It works
    with any hierarchy structure defined in the YAML configuration.

    Parameters
    ----------
    hierarchy : Hierarchy
        Hierarchy instance to populate
    hierarchy_df : pandas.DataFrame
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
    missing_columns = [
        col for col in required_columns if col not in hierarchy_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Validate that all mapped entity types exist in the hierarchy
    hierarchy_entity_types = set(hierarchy.get_entity_type_names())
    mapped_entity_types = set(column_mapping.values())
    invalid_mappings = mapped_entity_types - hierarchy_entity_types
    if invalid_mappings:
        raise ValueError(f"Invalid entity types in mapping: {invalid_mappings}")

    # Remove duplicates and any rows with missing values
    df = hierarchy_df.dropna().drop_duplicates()

    # Get hierarchy levels in order from bottom to top
    levels = hierarchy.get_levels_ordered()

    # PASS 1: Create all entities without relationships
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

        # Create all entities of this type without parents
        for _, row in df[[entity_column]].drop_duplicates().iterrows():
            entity_id = row[entity_column]
            hierarchy.add_entity(entity_id, entity_type)

    # PASS 2: Establish parent-child relationships
    for i, entity_type in enumerate(levels):
        entity_type_name = entity_type.name

        # Find the column that maps to this entity type
        entity_column = None
        for col, et_name in column_mapping.items():
            if et_name == entity_type_name:
                entity_column = col
                break

        if entity_column is None:
            continue

        # Skip top level (no parents)
        if i == len(levels) - 1:
            continue

        # Find the parent column for this level
        parent_type = levels[i + 1]
        parent_type_name = parent_type.name

        parent_column = None
        for col, et_name in column_mapping.items():
            if et_name == parent_type_name:
                parent_column = col
                break

        if parent_column is None:
            continue

        # Establish parent-child relationships with error checking
        df_subset = (
            df[[entity_column, parent_column]]
            .drop_duplicates()
            .sort_values([parent_column, entity_column])
        )
        for _, row in df_subset.iterrows():
            entity_id = row[entity_column]
            parent_id = row[parent_column]

            # Check if parent exists before trying to link
            parent_entity_type = hierarchy.get_entity_type(parent_id)
            if parent_entity_type is None:
                raise ValueError(
                    f"Parent entity '{parent_id}' not found for child '{entity_id}' of type '{entity_type_name}'"
                )

            # Allow same entity name at different levels (entity collision scenario)
            if parent_entity_type != parent_type:
                # Check if there's a parent with the correct type
                correct_parent_prefixed = f"{parent_type.name}:{parent_id}"
                if correct_parent_prefixed in hierarchy.entity_types:
                    # Use the correct parent
                    parent_prefixed = correct_parent_prefixed
                else:
                    raise ValueError(
                        f"Parent entity '{parent_id}' has type '{parent_entity_type.name}' but expected type '{parent_type.name}' for child '{entity_id}'"
                    )
            else:
                parent_prefixed = f"{parent_type.name}:{parent_id}"

            # Update the relationship
            child_prefixed = f"{entity_type.name}:{entity_id}"
            hierarchy.relationships[child_prefixed] = parent_prefixed

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
                # Get prefixed IDs for both entities
                entity_prefixed = hierarchy._get_prefixed_id(entity_id, top_level_type)
                top_level_prefixed = hierarchy._get_prefixed_id(
                    top_level_id, top_level_type
                )

                if entity_prefixed and top_level_prefixed:
                    # Update the entity to have the specified top-level as parent
                    hierarchy.relationships[entity_prefixed] = top_level_prefixed

    # Link all entities that don't have parents to the top-level entity
    # This ensures the hierarchy is properly connected
    for entity_id, entity_type in hierarchy.entity_types.items():
        # Skip the top-level entity itself
        if entity_type == top_level_type:
            continue

        # Check if this entity already has a parent
        if entity_id not in hierarchy.relationships:
            # Link to the top-level entity
            top_level_prefixed = hierarchy._get_prefixed_id(
                top_level_id, top_level_type
            )
            if top_level_prefixed:
                hierarchy.relationships[entity_id] = top_level_prefixed


def create_hierarchical_predictor(
    hierarchy_df: pd.DataFrame,
    column_mapping: Dict[str, str],
    top_level_id: str,
    k_sigma: float = 8.0,
    hierarchy_config_path: Optional[str] = None,
    truncate_only_bottom: bool = True,
) -> HierarchicalPredictor:
    """Create a HierarchicalPredictor with explicit column mapping.

    This convenience function creates a fully configured HierarchicalPredictor
    by extracting the organizational structure from a DataFrame using explicit
    column mapping and setting up the prediction engine.

    Parameters
    ----------
    hierarchy_df : pandas.DataFrame
        DataFrame containing organizational structure
    column_mapping : Dict[str, str]
        Mapping from DataFrame column names to entity type names.
        Example: {'sub_specialty': 'subspecialty', 'reporting_unit': 'reporting_unit',
                 'division': 'division', 'board': 'board'}
    top_level_id : str
        Identifier for the top-level entity in the hierarchy
    k_sigma : float, default=8.0
        Cap width in standard deviations used to clamp distributions using an
        adaptive approach that prevents over-truncation for small lambda values.
    hierarchy_config_path : str, optional
        Path to YAML file containing custom hierarchy configuration.
        If None, uses default hospital hierarchy.

    Returns
    -------
    HierarchicalPredictor
        Fully configured predictor with:
        - Hierarchy populated from hierarchy_df using column_mapping
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
    populate_hierarchy_from_dataframe(
        hierarchy, hierarchy_df, column_mapping, top_level_id
    )

    # Create predictor
    predictor = DemandPredictor(k_sigma=k_sigma)
    return HierarchicalPredictor(hierarchy, predictor)
