"""Build hierarchical predictors with inputs and save to pickle file.

This script creates hierarchical predictors for different flow cohorts
and saves them along with their associated metadata to a pickle file.
"""

import pickle
from pathlib import Path

from patientflow.predict.hierarchy import (
    FlowSelection,
    create_hierarchical_predictor,
)
from patientflow.predict.subspecialty import build_subspecialty_data


def build_predictor(flow_selection: FlowSelection, return_predictor_only=False):
    """Build a hierarchical predictor with subspecialty data.
    
    Parameters
    ----------
    flow_selection : FlowSelection
        Selection for which flow families and cohort to include
    return_predictor_only : bool, default=False
        If True, return only the predictor. If False, return a dict with
        predictor and associated metadata.
        
    Returns
    -------
    predictor or dict
        If return_predictor_only=True, returns HierarchicalPredictor.
        Otherwise returns dict with keys:
        - 'predictor': HierarchicalPredictor
        - 'subspecialty_data': Dict[str, SubspecialtyPredictionInputs]
        - 'hierarchy_df': DataFrame with hierarchy information
        - 'column_mapping': Column mapping configuration
        - 'top_level_id': Top level entity ID (e.g., "uclh")
    """
    subspecialty_data = build_subspecialty_data(
        models=tuple(
            [
                admissions_models['admissions_0930'],
                admissions_models['discharges_0930'],
                spec_model,
                ed_yta_model,
                non_ed_yta_model,
                elective_yta_model,
                trans_model
            ]
        ),
        prediction_time=(9, 30),
        ed_snapshots=ed_prediction_snapshots,
        inpatient_snapshots=inpat_prediction_snapshots,
        specialties=subspecialties,
        prediction_window=prediction_window,
        x1=4,
        y1=0.8,
        x2=12,
        y2=0.99
    )
    
    predictor = create_hierarchical_predictor(
        hierarchy_df=specs[specs.sub_specialty.isin(subspecialties)],
        column_mapping=column_mapping,
        top_level_id="uclh"
    )
    
    predictor.predict_all_levels(subspecialty_data, flow_selection=flow_selection)
    
    if return_predictor_only:
        return predictor
    else:
        return {
            "predictor": predictor,
            "subspecialty_data": subspecialty_data,
            "hierarchy_df": specs[specs.sub_specialty.isin(subspecialties)],
            "column_mapping": column_mapping,
            "top_level_id": "uclh",
        }


if __name__ == "__main__":
    # Build predictors for different cohorts
    cohort_predictors = {
        "all": build_predictor(FlowSelection.default()),
        "elective": build_predictor(FlowSelection.elective_only()),
        "emergency": build_predictor(FlowSelection.emergency_only()),
    }
    
    # Save to pickle file
    output_path = Path("hierarchical_predictors_with_inputs.pkl")
    with output_path.open("wb") as f:
        pickle.dump(cohort_predictors, f, protocol=pickle.HIGHEST_PROTOCOL)
