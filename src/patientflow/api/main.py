"""Simple FastAPI application exposing cached hierarchical predictions."""

from __future__ import annotations

import logging
import os
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from fastapi import FastAPI, HTTPException, Query

from patientflow.predict.hierarchy import (
    DemandPrediction,
    EntityType,
    HierarchicalPredictor,
    PredictionBundle,
)

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="PatientFlow Hierarchical Predictions")

_predictor: Optional[HierarchicalPredictor] = None


class FlowType(str, Enum):
    arrivals = "arrivals"
    departures = "departures"
    net_flow = "net_flow"
    all = "all"


class Cohort(str, Enum):
    all = "all"
    elective = "elective"
    emergency = "emergency"


class DetailLevel(str, Enum):
    summary = "summary"
    full = "full"


def _load_predictor() -> HierarchicalPredictor:
    """Load the pickled ``HierarchicalPredictor`` using environment configuration."""
    path_value = os.getenv("PREDICTOR_PICKLE_PATH")
    if not path_value:
        raise RuntimeError("PREDICTOR_PICKLE_PATH environment variable is not set")

    path = Path(path_value).expanduser()
    if not path.is_file():
        raise RuntimeError(f"PREDICTOR_PICKLE_PATH does not point to a file: {path}")

    try:
        with path.open("rb") as file:
            predictor = pickle.load(file)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to load predictor pickle from %s", path)
        raise RuntimeError("Unable to load predictor pickle") from exc

    if not isinstance(predictor, HierarchicalPredictor):
        raise RuntimeError(
            "Loaded object is not a HierarchicalPredictor; check the pickle content"
        )

    LOGGER.info(
        "Loaded HierarchicalPredictor with %s cached bundles",
        len(predictor.cache),
    )
    return predictor


def _get_predictor() -> HierarchicalPredictor:
    if _predictor is None:
        raise RuntimeError("Predictor not initialised")
    return _predictor


@app.on_event("startup")
def _on_startup() -> None:
    global _predictor
    _predictor = _load_predictor()


def _validate_entity_type(predictor: HierarchicalPredictor, entity_type: str) -> None:
    available_types = predictor.hierarchy.get_entity_type_names()
    if entity_type not in available_types:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown entity type '{entity_type}'",
                "available_types": available_types,
            },
        )


def _prefixed_id(entity_type: str, entity_id: str) -> str:
    return f"{entity_type}:{entity_id}"


def _get_bundle(entity_type: str, entity_id: str) -> PredictionBundle:
    predictor = _get_predictor()
    _validate_entity_type(predictor, entity_type)

    cache_key = _prefixed_id(entity_type, entity_id)
    bundle = predictor.cache.get(cache_key)
    if bundle is None:
        raise HTTPException(
            status_code=404,
            detail={"message": f"No prediction cached for {entity_type}/{entity_id}"},
        )
    return bundle


def _serialize_prediction(
    prediction: DemandPrediction, detail: DetailLevel
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "entity_id": prediction.entity_id,
        "entity_type": prediction.entity_type,
        "expected_value": prediction.expected_value,
        "percentiles": prediction.percentiles,
    }
    if detail is DetailLevel.full:
        payload.update(
            {
                "offset": prediction.offset,
                "probabilities": prediction.probabilities.tolist(),
            }
        )
    return payload


def _bundle_to_response(
    bundle: PredictionBundle, flow_type: FlowType, detail: DetailLevel
) -> Dict[str, Any]:
    flows: Dict[str, Any] = {}
    flow_mapping = {
        FlowType.arrivals: ("arrivals", bundle.arrivals),
        FlowType.departures: ("departures", bundle.departures),
        FlowType.net_flow: ("net_flow", bundle.net_flow),
    }

    selected: Iterable[FlowType]
    if flow_type is FlowType.all:
        selected = (FlowType.arrivals, FlowType.departures, FlowType.net_flow)
    else:
        selected = (flow_type,)

    for key in selected:
        label, result = flow_mapping[key]
        flows[label] = _serialize_prediction(result, detail)

    return {
        "entity_id": bundle.entity_id,
        "entity_type": bundle.entity_type,
        "cohort": bundle.flow_selection.cohort,
        "flows": flows,
    }


@app.get("/api/predictions/{entity_type}/{entity_id}")
def read_prediction(
    entity_type: str,
    entity_id: str,
    flow_type: FlowType = FlowType.all,
    cohort: Cohort = Cohort.all,
    detail_level: DetailLevel = DetailLevel.summary,
) -> Dict[str, Any]:
    bundle = _get_bundle(entity_type, entity_id)

    actual_cohort = bundle.flow_selection.cohort
    if cohort.value != actual_cohort:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Cohort '{cohort.value}' not available for "
                    f"{entity_type}/{entity_id}. Cached cohort is '{actual_cohort}'."
                )
            },
        )

    return _bundle_to_response(bundle, flow_type, detail_level)


@app.get("/api/predictions/{entity_type}/{entity_id}/cohort-comparison")
def read_cohort_comparison(
    entity_type: str,
    entity_id: str,
    flow_type: FlowType = FlowType.net_flow,
    cohorts: str = Query(..., description="Comma-separated cohort list"),
    detail_level: DetailLevel = DetailLevel.summary,
) -> Dict[str, Any]:
    bundle = _get_bundle(entity_type, entity_id)
    actual_cohort = bundle.flow_selection.cohort

    requested_cohorts_raw = [
        item.strip() for item in cohorts.split(",") if item.strip()
    ]
    if not requested_cohorts_raw:
        raise HTTPException(
            status_code=400, detail={"message": "At least one cohort must be provided"}
        )

    parsed_cohorts = []
    for cohort_name in requested_cohorts_raw:
        try:
            parsed_cohorts.append(Cohort(cohort_name))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"message": f"Unsupported cohort '{cohort_name}'"},
            ) from exc

    invalid = [c.value for c in parsed_cohorts if c.value != actual_cohort]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Only cached cohort "
                    f"'{actual_cohort}' is available for {entity_type}/{entity_id}"
                ),
                "unsupported": invalid,
            },
        )

    # Return the same cached cohort data for each requested cohort (since cache is single-cohort)
    flow_payload = _bundle_to_response(bundle, flow_type, detail_level)["flows"]
    return {
        "entity_id": bundle.entity_id,
        "entity_type": bundle.entity_type,
        "cohorts": {actual_cohort: flow_payload},
    }


@app.get("/api/predictions/{entity_type}/{entity_id}/breakdown")
def read_breakdown(
    entity_type: str,
    entity_id: str,
    include_children: bool = False,
    cohort: Cohort = Cohort.all,
    flow_type: FlowType = FlowType.all,
    detail_level: DetailLevel = DetailLevel.summary,
) -> Dict[str, Any]:
    predictor = _get_predictor()
    bundle = _get_bundle(entity_type, entity_id)

    actual_cohort = bundle.flow_selection.cohort
    if cohort.value != actual_cohort:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Cohort '{cohort.value}' not available for "
                    f"{entity_type}/{entity_id}. Cached cohort is '{actual_cohort}'."
                )
            },
        )

    response = {
        "entity": _bundle_to_response(bundle, flow_type, detail_level),
        "children": [],
    }

    if include_children:
        entity_type_obj = EntityType.from_string(entity_type)
        child_ids = predictor.hierarchy.get_children(entity_id, entity_type_obj)
        for child_id in child_ids:
            child_type = predictor.hierarchy.get_entity_type(child_id)
            if child_type is None:
                continue
            child_bundle = _get_bundle(child_type.name, child_id)
            response["children"].append(
                _bundle_to_response(child_bundle, flow_type, detail_level)
            )

    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        predictor = _get_predictor()
    except RuntimeError:
        return {"status": "error", "details": "Predictor not loaded"}

    return {
        "status": "ok",
        "cached_entities": len(predictor.cache),
    }

