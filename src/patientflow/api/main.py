"""Simple FastAPI application exposing cached hierarchical predictions.

Usage
=====
1. Build a pickle file that contains a dictionary mapping cohort names to
   ``HierarchicalPredictor`` instances. Example structure::

       {
           "all": HierarchicalPredictor(...),
           "elective": HierarchicalPredictor(...),
           "emergency": HierarchicalPredictor(...),
       }

   Each predictor should already have its cache populated, typically by calling
   ``predict_all_levels`` after constructing the predictor.

2. Point the API to that pickle via the ``PREDICTOR_PICKLE_PATH`` environment
   variable before starting uvicorn, e.g.::

       export PREDICTOR_PICKLE_PATH=/path/to/hierarchical_predictors.pkl
       uvicorn patientflow.api.main:app

3. Call the HTTP endpoints with the desired cohort. For example::

       GET /api/predictions/subspecialty/Cardiology?cohort=elective

   The API selects the correct predictor dictionary entry and returns the
   cached arrivals/departures/net-flow bundle for that entity.

   cURL example::

       curl "http://localhost:8000/api/predictions/subspecialty/Cardiology?cohort=all&flow_type=arrivals"
"""

from __future__ import annotations

import logging
import os
import pickle
from enum import Enum
from pathlib import Path
import numpy as np
from typing import Any, Dict, Iterable

from fastapi import FastAPI, HTTPException, Query

from patientflow.predict.hierarchy import (
    DEFAULT_MAX_PROBS,
    # DEFAULT_PRECISION,
    DemandPrediction,
    EntityType,
    HierarchicalPredictor,
    PredictionBundle,
)

DEFAULT_PRECISION = 2
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="PatientFlow Hierarchical Predictions")

_predictors: Dict[str, HierarchicalPredictor] = {}


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


def _load_predictors() -> Dict[str, HierarchicalPredictor]:

    path_value = os.getenv("PREDICTOR_PICKLE_PATH")
    if not path_value:
        raise RuntimeError("PREDICTOR_PICKLE_PATH environment variable is not set")

    path = Path(path_value).expanduser()
    if not path.is_file():
        raise RuntimeError(f"PREDICTOR_PICKLE_PATH does not point to a file: {path}")

    try:
        with path.open("rb") as file:
            predictors = pickle.load(file)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to load predictor pickle from %s", path)
        raise RuntimeError("Unable to load predictor pickle") from exc

    LOGGER.info("Loaded %s cohort predictors", len(predictors))
    return predictors


def _get_predictor(cohort: Cohort) -> HierarchicalPredictor:
    if cohort.value not in _predictors:
        raise RuntimeError(f"Predictor not initialised for cohort '{cohort.value}'")
    return _predictors[cohort.value]


@app.on_event("startup")
def _on_startup() -> None:
    global _predictors
    _predictors = _load_predictors()


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


def _get_bundle(
    predictor: HierarchicalPredictor, entity_type: str, entity_id: str
) -> PredictionBundle:
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
    prediction: DemandPrediction, detail: DetailLevel, max_probs: int = DEFAULT_MAX_PROBS
) -> Dict[str, Any]:
    probs = prediction.probabilities
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

    window = probs[start_idx:end_idx].tolist()
    rounded_window = [round(float(value), DEFAULT_PRECISION) for value in window]
    rounded_expectation = round(float(prediction.expected_value), DEFAULT_PRECISION)
    payload: Dict[str, Any] = {
        "expectation": rounded_expectation,
        "pmf_window": {
            "start_index": start_idx + prediction.offset,
            "end_index": end_idx + prediction.offset,
            "values": rounded_window,
        },
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
    predictor = _get_predictor(cohort)
    bundle = _get_bundle(predictor, entity_type, entity_id)
    return _bundle_to_response(bundle, flow_type, detail_level)


@app.get("/api/predictions/{entity_type}/{entity_id}/cohort-comparison")
def read_cohort_comparison(
    entity_type: str,
    entity_id: str,
    flow_type: FlowType = FlowType.net_flow,
    cohorts: str = Query(..., description="Comma-separated cohort list"),
    detail_level: DetailLevel = DetailLevel.summary,
) -> Dict[str, Any]:
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

    cohort_payloads: Dict[str, Dict[str, Any]] = {}
    for cohort in parsed_cohorts:
        predictor = _get_predictor(cohort)
        bundle = _get_bundle(predictor, entity_type, entity_id)
        cohort_payloads[cohort.value] = _bundle_to_response(
            bundle, flow_type, detail_level
        )["flows"]

    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "cohorts": cohort_payloads,
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
    predictor = _get_predictor(cohort)
    bundle = _get_bundle(predictor, entity_type, entity_id)

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
            child_bundle = _get_bundle(predictor, child_type.name, child_id)
            response["children"].append(
                _bundle_to_response(child_bundle, flow_type, detail_level)
            )

    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    if not _predictors:
        return {"status": "error", "details": "Predictor not loaded"}

    cached_entities = {
        cohort: len(predictor.cache) for cohort, predictor in _predictors.items()
    }
    return {"status": "ok", "cached_entities": cached_entities}

