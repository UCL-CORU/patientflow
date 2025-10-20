"""Cohort routing for flow distributions.

This module defines a minimal interface for routing flow distributions into
cohorts (e.g., visit_type, age_group), and a neutral flow specification
(``FlowSpec``) used by the router and prediction builders.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional
from dataclasses import dataclass
from typing import Union

import numpy as np


# Simple alias for a hashable cohort key. For now a flat string is sufficient
# (e.g., "overall"), and can later evolve (e.g., "visit_type=elective;age=child").
CohortKey = str


@dataclass(frozen=True)
class FlowSpec:
    """Neutral flow specification used across routing and prediction steps.

    Parameters
    ----------
    flow_id : str
        Identifier for the flow (e.g., "ed_current", "departures").
    flow_type : str
        Either "pmf" or "poisson".
    distribution : numpy.ndarray or float
        For "pmf": array where distribution[k] = P(k). For "poisson": rate (λ).
    display_name : str, optional
        Human-readable name for the flow.
    """

    flow_id: str
    flow_type: str
    distribution: Union[np.ndarray, float]
    display_name: Optional[str] = None


class CohortRouter:
    """Interface for routing flows into cohorts.

    Implementations take a single flow and return one or more cohort-specific
    flows. Identity routing returns the input flow under a single "overall"
    cohort.
    """

    def route_flow(
        self, flow_kind: str, flow: FlowSpec, metadata: Optional[Mapping] = None
    ) -> Dict[CohortKey, FlowSpec]:
        """Route a flow into one or more cohorts.

        Parameters
        ----------
        flow_kind : str
            Identifier for the flow type (e.g., "ed_current", "departures").
        flow : FlowInputs
            The flow to route.
        metadata : Mapping, optional
            Optional context information for routing decisions.

        Returns
        -------
        dict[str, FlowSpec]
            Mapping from cohort key to routed flow specs.
        """

        raise NotImplementedError


class IdentityCohortRouter(CohortRouter):
    """Identity router that passes flows through unchanged under "overall" cohort."""

    def route_flow(
        self, flow_kind: str, flow: FlowSpec, metadata: Optional[Mapping] = None
    ) -> Dict[CohortKey, FlowSpec]:
        return {"overall": flow}


