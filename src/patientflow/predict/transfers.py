"""Subgroup-aware transfer routing for service demand prediction.

This module computes the probability mass function (PMF) of internal-transfer
arrivals at each service, using patient-level routing tables learned by
[TransferProbabilityEstimator][patientflow.predictors.transfer_predictor.TransferProbabilityEstimator].

Production transfer routing resolves each departing inpatient to an age/sex
subgroup and routes them using that subgroup's fitted table
(``transfer_probabilities[cohort]["subgroups"][g]``). This mirrors the ED
admissions pattern in
[patientflow.predict.service][patientflow.predict.service]: heterogeneity is
carried in *per-patient weights* applied through
[pred_proba_to_agg_predicted][patientflow.aggregate.pred_proba_to_agg_predicted],
producing **one** aggregated PMF per flow rather than one PMF per subgroup.

The cohort-pooled ``["services"]`` row remains fitted for diagnostics (e.g.
[get_transition_matrix][patientflow.predictors.transfer_predictor.TransferProbabilityEstimator.get_transition_matrix])
but is **not** used at prediction time. Patients that do not resolve to a
subgroup (typically adults with missing or invalid ``sex``) are excluded from
transfer routing rather than falling back to the pooled row; see
[compute_transfer_arrivals][patientflow.predict.transfers.compute_transfer_arrivals].
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from patientflow.aggregate import pred_proba_to_agg_predicted
from patientflow.predict.distribution import Distribution
from patientflow.predictors.subgroup_definitions import (
    assign_patient_subgroups,
    resolve_patient_subgroup,
)
from patientflow.predictors.transfer_predictor import TransferProbabilityEstimator


def transfer_weight_to_target(
    row: pd.Series,
    source_service: str,
    target_service: str,
    cohort: str,
    transfer_model: TransferProbabilityEstimator,
) -> float:
    """Return ``q_i(T)`` — the routing probability for a single patient.

    This is the probability that, *given the patient departs*, the departure is
    a transfer to ``target_service``. It resolves the patient's age/sex subgroup
    and reads the subgroup-specific transfer table; it does **not** include the
    patient's departure probability ``p_depart_i``.

    Parameters
    ----------
    row : pandas.Series
        A single inpatient row exposing the columns used by the subgroup
        predicates (``age_on_arrival``/``age_group`` and ``sex``).
    source_service : str
        The patient's current subspecialty (where they would depart from).
    target_service : str
        The candidate destination subspecialty.
    cohort : str
        The cohort (admission type) the patient belongs to.
    transfer_model : TransferProbabilityEstimator
        Fitted transfer estimator providing the subgroup tables.

    Returns
    -------
    float
        ``q_transfer * P(destination = target | transfer)`` for the patient's
        subgroup, or ``0.0`` when the patient resolves to no subgroup or to a
        subgroup absent from this cohort's tables.
    """
    assert transfer_model.transfer_probabilities is not None
    g = resolve_patient_subgroup(row, transfer_model.subgroup_functions)
    if g is None or g not in transfer_model.transfer_probabilities[cohort]["subgroups"]:
        return 0.0
    q_transfer = transfer_model.get_transfer_prob(source_service, cohort, subgroup=g)
    dest = transfer_model.get_destination_distribution(
        source_service, cohort, subgroup=g
    )
    return float(q_transfer) * float(dest.get(target_service, 0.0))


@dataclass(frozen=True)
class PerPatientProbabilities:
    """Per-event routing matrix aligned with production subgroup routing.

    Attributes
    ----------
    routing_matrix : numpy.ndarray
        Shape (n_events, n_destinations); row `i` is the departure distribution
        for event `i` and sums to 1.
    n_excluded_unmatched : int
        Events treated as all-discharge because subgroup resolution failed or
        the subgroup is absent from the cohort's fitted tables.
    n_subgroups_used : int
        Distinct resolved subgroups among non-excluded events from this source.
    """

    routing_matrix: np.ndarray
    n_excluded_unmatched: int
    n_subgroups_used: int


def build_per_patient_probabilities(
    events: pd.DataFrame,
    source: str,
    cohort: str | None,
    destinations: Sequence[str],
    transfer_model: TransferProbabilityEstimator,
    *,
    discharge_label: str = "Discharge",
) -> PerPatientProbabilities:
    """Return per-event departure routing probabilities for one source.

    Batch counterpart to `transfer_weight_to_target` for evaluation: each
    departing patient gets a full destination vector `p_i` over destinations
    aligned with `get_transition_matrix(cohort)`, not a single target weight.
    Summing rows gives patient-level expected counts `E_d = sum_i p_i(d)` when
    subgroup mix varies across departures. Probabilities come from subgroup
    tables only; evaluation does not use the pooled `get_transition_matrix` row
    or a homogeneous `N * p` test (that matrix remains diagnostic).

    For a resolved subgroup `g`, each row is built as
    `p_i(T) = q_transfer * q_dest(T)` for transfer destinations `T`, and
    `p_i(Discharge) = 1 - q_transfer`. Uses `assign_patient_subgroups` and the
    same resolved / unmatched rules as `transfer_weight_to_target`. Unmatched
    rows receive `p_i(Discharge)=1`; the pooled ["services"] row is not used as
    a fallback.

    Parameters
    ----------
    events : pandas.DataFrame
        Departure events leaving `source`.
    source : str
        Source subspecialty.
    cohort : str or None
        Cohort key passed to the fitted estimator.
    destinations : sequence of str
        Destination columns (including `discharge_label`).
    transfer_model : TransferProbabilityEstimator
        Fitted transfer model with subgroup tables.
    discharge_label : str, optional
        Label for the discharge column (default "Discharge").

    Returns
    -------
    PerPatientProbabilities
        ``routing_matrix`` has shape (n_events, n_destinations) with rows
        summing to 1.

    Notes
    -----
    `n_excluded_unmatched` flags departures the production path would not route
    (for example adults with missing sex). A high value on a flagged source can
    mean case-mix outside the fitted subgroup tables rather than routing error
    among routed patients. `n_subgroups_used` helps interpret whether a low
    p-value may be driven by a single anomalous subgroup.
    """
    assert transfer_model.transfer_probabilities is not None
    cohort_key = transfer_model._resolve_cohort(cohort)
    cohort_subgroups = set(
        transfer_model.transfer_probabilities[cohort_key]["subgroups"].keys()
    )

    dest_list = list(destinations)
    discharge_idx = dest_list.index(discharge_label)
    n_events = len(events)
    n_dest = len(dest_list)
    p_matrix = np.zeros((n_events, n_dest), dtype=float)

    subgroups = assign_patient_subgroups(events, transfer_model.subgroup_functions)
    n_excluded = 0
    resolved_subgroups: set[str] = set()

    for row_idx, subgroup in enumerate(subgroups):
        if subgroup is None or subgroup not in cohort_subgroups:
            p_matrix[row_idx, discharge_idx] = 1.0
            n_excluded += 1
            continue

        resolved_subgroups.add(subgroup)
        q_transfer = transfer_model.get_transfer_prob(
            source, cohort_key, subgroup=subgroup
        )
        dest_dist = transfer_model.get_destination_distribution(
            source, cohort_key, subgroup=subgroup
        )
        p_matrix[row_idx, discharge_idx] = 1.0 - q_transfer
        for col_idx, dest in enumerate(dest_list):
            if dest == discharge_label:
                continue
            p_matrix[row_idx, col_idx] = q_transfer * float(dest_dist.get(dest, 0.0))

    return PerPatientProbabilities(
        routing_matrix=p_matrix,
        n_excluded_unmatched=n_excluded,
        n_subgroups_used=len(resolved_subgroups),
    )


def _as_pred_proba_frame(
    probabilities: Optional[Union[pd.DataFrame, pd.Series]],
) -> pd.DataFrame:
    """Normalise a departure-probability container to a ``pred_proba`` frame."""
    if probabilities is None:
        return pd.DataFrame(columns=["pred_proba"])
    if isinstance(probabilities, pd.DataFrame):
        if "pred_proba" not in probabilities.columns:
            raise KeyError(
                "Departure probability DataFrame must contain a 'pred_proba' column"
            )
        return probabilities
    return probabilities.to_frame(name="pred_proba")


def _emit_excluded_rows_warning(
    inpatient_snapshots: pd.DataFrame,
    subgroup_of_row: pd.Series,
    transfer_model: TransferProbabilityEstimator,
    admission_types: List[str],
) -> None:
    """Emit a single summary warning for rows excluded from transfer routing.

    A row is excluded when it resolves to no subgroup (typically an adult with
    missing or invalid ``sex``) or to a subgroup absent from its cohort's
    fitted subgroup tables (an empty training slice).
    """
    assert transfer_model.transfer_probabilities is not None
    if "admission_type" not in inpatient_snapshots.columns:
        return

    excluded_index = pd.Index([])
    for cohort in admission_types:
        cohort_subgroups = set(
            transfer_model.transfer_probabilities[cohort]["subgroups"].keys()
        )
        cohort_mask = inpatient_snapshots["admission_type"] == cohort
        cohort_index = inpatient_snapshots.index[cohort_mask]
        if len(cohort_index) == 0:
            continue
        g_cohort = subgroup_of_row.loc[cohort_index]
        excluded = g_cohort.isna() | ~g_cohort.isin(cohort_subgroups)
        excluded_index = excluded_index.union(cohort_index[excluded.to_numpy()])

    n_excluded = len(excluded_index)
    if n_excluded == 0:
        return

    message = (
        f"{n_excluded} inpatient row(s) were excluded from transfer routing "
        "because they did not resolve to a fitted subgroup (e.g. adults with "
        "missing or invalid sex). These rows contribute no transfer arrivals "
        "but still contribute to departure PMFs at their source."
    )
    if "sex" in inpatient_snapshots.columns:
        sex_breakdown = (
            inpatient_snapshots.loc[excluded_index, "sex"]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        message += f" Breakdown by sex: {sex_breakdown}."
    warnings.warn(message, stacklevel=2)


def compute_transfer_arrivals(
    inpatient_snapshots: Optional[pd.DataFrame],
    transfer_model: TransferProbabilityEstimator,
    services: List[str],
    prob_departure_after_elective: Optional[Union[pd.DataFrame, pd.Series]] = None,
    prob_departure_after_emergency: Optional[Union[pd.DataFrame, pd.Series]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute subgroup-aware transfer-arrival PMFs for each service.

    For every ``(target, source, cohort)`` this routes each departing inpatient
    using the transfer table fitted on that patient's age/sex subgroup, then
    aggregates to a single PMF per ``(source, target)`` via
    [pred_proba_to_agg_predicted][patientflow.aggregate.pred_proba_to_agg_predicted]
    with per-patient weights. Contributions across sources are convolved to give
    the arrival distribution at each target.

    Per patient ``i`` at source ``S`` in cohort ``c``:

    - resolve subgroup ``g(i)`` from ``transfer_model.subgroup_functions``;
    - ``q_i(T) = P(transfer | S, c, g) * P(dest = T | transfer, S, c, g)``;
    - effective per-patient Bernoulli probability for a transfer ``S -> T`` is
      ``p_depart_i * q_i(T)``, supplied to
      [pred_proba_to_agg_predicted][patientflow.aggregate.pred_proba_to_agg_predicted]
      as ``predictions_proba = p_depart_i`` and ``weights = q_i(T)``.

    Parameters
    ----------
    inpatient_snapshots : pandas.DataFrame or None
        Current inpatients, indexed consistently with the departure probability
        containers. Must expose ``current_subspecialty``, ``admission_type`` and
        the columns used by the subgroup predicates (``age``/``sex``). When
        ``None`` (or empty), all targets receive a zero-arrivals PMF.
    transfer_model : TransferProbabilityEstimator
        Fitted estimator providing per-subgroup transfer tables.
    services : list of str
        All services in the system (potential sources and targets).
    prob_departure_after_elective : pandas.DataFrame or pandas.Series, optional
        Per-patient departure probability ``p_depart_i`` for elective
        inpatients, indexed by the corresponding rows of *inpatient_snapshots*.
        A DataFrame must contain a ``pred_proba`` column.
    prob_departure_after_emergency : pandas.DataFrame or pandas.Series, optional
        As above for emergency inpatients.

    Returns
    -------
    dict
        ``{"elective": {service: pmf}, "emergency": {service: pmf}}`` where each
        ``pmf`` is a numpy array giving ``P(k transfer arrivals)``.

    Raises
    ------
    ValueError
        If *transfer_model* has not been fitted.

    Notes
    -----
    Subgroup routing is the only prediction path: the cohort-pooled
    ``["services"]`` row is never used here. A patient that resolves to no
    subgroup (or to a subgroup absent from its cohort's tables) contributes
    **no** transfer arrivals — but still contributes to the **departure** PMF at
    its source, since departure is governed by the inpatient classifier rather
    than by transfer routing. One summary warning per call reports the count of
    excluded rows.
    """
    if not getattr(transfer_model, "is_fitted_", False):
        raise ValueError(
            "This TransferProbabilityEstimator instance is not fitted yet. "
            "Call 'fit' before computing transfer arrivals."
        )
    assert transfer_model.transfer_probabilities is not None

    admission_types = ["elective", "emergency"]
    prob_departure_by_type = {
        "elective": _as_pred_proba_frame(prob_departure_after_elective),
        "emergency": _as_pred_proba_frame(prob_departure_after_emergency),
    }

    predicted_arrivals: Dict[str, Dict[str, np.ndarray]] = {
        admission_type: {service: np.array([1.0]) for service in services}
        for admission_type in admission_types
    }

    if inpatient_snapshots is None or inpatient_snapshots.empty:
        return predicted_arrivals

    required_cols = {"current_subspecialty", "admission_type"}
    if not required_cols.issubset(inpatient_snapshots.columns):
        return predicted_arrivals

    # Precompute each row's subgroup once for the whole frame (also validates
    # mutual exclusivity, raising ValueError on overlapping masks).
    subgroup_of_row = assign_patient_subgroups(
        inpatient_snapshots, transfer_model.subgroup_functions
    )

    cohorts_present = [
        c for c in admission_types if c in transfer_model.transfer_probabilities
    ]

    _emit_excluded_rows_warning(
        inpatient_snapshots, subgroup_of_row, transfer_model, cohorts_present
    )

    for admission_type in cohorts_present:
        cohort_subgroups = transfer_model.transfer_probabilities[admission_type][
            "subgroups"
        ]
        if not cohort_subgroups:
            continue

        prob_departure = prob_departure_by_type[admission_type]
        if prob_departure.empty:
            continue

        arrival_dists: Dict[str, Distribution] = {
            service: Distribution.from_pmf(np.array([1.0])) for service in services
        }

        cohort_mask = inpatient_snapshots["admission_type"] == admission_type

        for source_service in services:
            source_mask = cohort_mask & (
                inpatient_snapshots["current_subspecialty"] == source_service
            )
            source_index = inpatient_snapshots.index[source_mask]
            if len(source_index) == 0:
                continue

            # Align departure probabilities to the source rows.
            source_index = source_index.intersection(prob_departure.index)
            if len(source_index) == 0:
                continue

            g_list = subgroup_of_row.loc[source_index]
            pdep_frame = prob_departure.loc[source_index, ["pred_proba"]]

            # Resolve per-subgroup routing tables once for this (source, cohort).
            present_subgroups = {
                g for g in g_list.dropna().unique() if g in cohort_subgroups
            }
            if not present_subgroups:
                continue

            q_transfer_by_g: Dict[str, float] = {}
            dest_by_g: Dict[str, Dict[str, float]] = {}
            candidate_targets: set = set()
            for g in present_subgroups:
                q_transfer_by_g[g] = float(
                    transfer_model.get_transfer_prob(
                        source_service, admission_type, subgroup=g
                    )
                )
                dest = transfer_model.get_destination_distribution(
                    source_service, admission_type, subgroup=g
                )
                dest_by_g[g] = dest
                candidate_targets.update(dest.keys())

            candidate_targets &= set(services)
            candidate_targets.discard(source_service)
            if not candidate_targets:
                continue

            g_values = g_list.to_numpy()
            for target_service in candidate_targets:
                weights = np.array(
                    [
                        (
                            q_transfer_by_g[g] * dest_by_g[g].get(target_service, 0.0)
                            if g in q_transfer_by_g
                            else 0.0
                        )
                        for g in g_values
                    ],
                    dtype=float,
                )
                if not np.any(weights > 0.0):
                    continue

                pmf_frame = pred_proba_to_agg_predicted(pdep_frame, weights=weights)
                pmf = pmf_frame.sort_index()["agg_proba"].to_numpy(dtype=float)
                contribution = Distribution.from_pmf(pmf)
                arrival_dists[target_service] = arrival_dists[target_service].convolve(
                    contribution
                )

        for service in services:
            predicted_arrivals[admission_type][service] = arrival_dists[
                service
            ].probabilities

    return predicted_arrivals
