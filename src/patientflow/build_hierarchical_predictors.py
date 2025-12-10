"""Build hierarchical predictors with inputs and save to pickle file.

This script creates hierarchical predictors for different flow cohorts
and saves them along with their associated metadata to a pickle file.
"""

import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd

from patientflow.predict.hierarchy import (
    FlowSelection,
    create_hierarchical_predictor,
)
from patientflow.predict.subspecialty import SubspecialtyPredictionInputs


def build_predictor(
    subspecialty_data: Dict[str, SubspecialtyPredictionInputs],
    specs: pd.DataFrame,
    subspecialties: List[str],
    column_mapping: Dict[str, str],
    flow_selection: FlowSelection,
    top_level_id: str = "uclh",
    return_predictor_only: bool = False,
):
    """Build a hierarchical predictor with subspecialty data.
    
    Parameters
    ----------
    subspecialty_data : Dict[str, SubspecialtyPredictionInputs]
        Dictionary mapping subspecialty IDs to their prediction inputs
    specs : pd.DataFrame
        DataFrame containing organizational structure with 'sub_specialty' column
    subspecialties : List[str]
        List of subspecialty names to include in the hierarchy
    column_mapping : Dict[str, str]
        Mapping from DataFrame column names to entity type names
    flow_selection : FlowSelection
        Selection for which flow families and cohort to include
    top_level_id : str, default="uclh"
        Identifier for the top-level entity in the hierarchy
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
    
    predictor = create_hierarchical_predictor(
        hierarchy_df=specs[specs.sub_specialty.isin(subspecialties)],
        column_mapping=column_mapping,
        top_level_id=top_level_id
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
            "top_level_id": top_level_id,
        }


if __name__ == "__main__":
    # TODO: Load or define the following required inputs:
    # - subspecialty_data: Dict[str, SubspecialtyPredictionInputs]
    #   (can be created using build_subspecialty_data from patientflow.predict.subspecialty)
    # - specs: pd.DataFrame with organizational structure (must have 'sub_specialty' column)
    # - subspecialties: List[str] of subspecialty names to include
    # - column_mapping: Dict[str, str] mapping DataFrame columns to entity types
    #   Example: {'sub_specialty': 'subspecialty', 'reporting_unit': 'reporting_unit',
    #             'division': 'division', 'board': 'board'}
    
    # Example calling function:
    def build_predictors_for_cohorts(
        subspecialty_data: Dict[str, SubspecialtyPredictionInputs],
        specs: pd.DataFrame,
        subspecialties: List[str],
        column_mapping: Dict[str, str],
        top_level_id: str = "uclh",
    ):
        """Build hierarchical predictors for different flow cohorts.
        
        Parameters
        ----------
        subspecialty_data : Dict[str, SubspecialtyPredictionInputs]
            Dictionary mapping subspecialty IDs to their prediction inputs
        specs : pd.DataFrame
            DataFrame containing organizational structure with 'sub_specialty' column
        subspecialties : List[str]
            List of subspecialty names to include in the hierarchy
        column_mapping : Dict[str, str]
            Mapping from DataFrame column names to entity type names
        top_level_id : str, default="uclh"
            Identifier for the top-level entity in the hierarchy
            
        Returns
        -------
        Dict[str, dict]
            Dictionary mapping cohort names to predictor results
        """
        cohort_predictors = {
            "all": build_predictor(
                subspecialty_data=subspecialty_data,
                specs=specs,
                subspecialties=subspecialties,
                column_mapping=column_mapping,
                flow_selection=FlowSelection.default(),
                top_level_id=top_level_id,
            ),
            "elective": build_predictor(
                subspecialty_data=subspecialty_data,
                specs=specs,
                subspecialties=subspecialties,
                column_mapping=column_mapping,
                flow_selection=FlowSelection.elective_only(),
                top_level_id=top_level_id,
            ),
            "emergency": build_predictor(
                subspecialty_data=subspecialty_data,
                specs=specs,
                subspecialties=subspecialties,
                column_mapping=column_mapping,
                flow_selection=FlowSelection.emergency_only(),
                top_level_id=top_level_id,
            ),
        }
        return cohort_predictors
    
    # Example usage (uncomment and fill in with actual data):
    # subspecialty_data = ...  # Load or create using build_subspecialty_data
    # specs = ...  # Load DataFrame with organizational structure
    # subspecialties = ...  # List of subspecialty names
    # column_mapping = {
    #     'sub_specialty': 'subspecialty',
    #     'reporting_unit': 'reporting_unit',
    #     'division': 'division',
    #     'board': 'board'
    # }
    # 
    # cohort_predictors = build_predictors_for_cohorts(
    #     subspecialty_data=subspecialty_data,
    #     specs=specs,
    #     subspecialties=subspecialties,
    #     column_mapping=column_mapping,
    #     top_level_id="uclh"
    # )
    # 
    # # Save to pickle file
    # output_path = Path("hierarchical_predictors_with_inputs.pkl")
    # with output_path.open("wb") as f:
    #     pickle.dump(cohort_predictors, f, protocol=pickle.HIGHEST_PROTOCOL)


