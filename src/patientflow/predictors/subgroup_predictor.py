"""
Simple subgroup system for managing multiple patient prediction models.

This module provides an approach to training and using separate
SequenceToOutcomePredictor models for different patient subgroups.

Functions
---------
create_subgroup_system
    Main entry point - creates subgroup functions and multi-model predictor
create_special_category_objects  
    Backward compatibility function (deprecated)
"""

import warnings
from typing import Dict, Any, Union, List, Type, Callable, Optional, Tuple
import pandas as pd
from patientflow.predictors.sequence_to_outcome_predictor import SequenceToOutcomePredictor
from patientflow.predictors.legacy_compatibility import get_age, create_special_category_objects, create_yta_filters


def create_subgroup_functions() -> Dict[str, Callable[[Union[pd.Series, dict]], bool]]:
    """Create the 5 standard subgroup identification functions."""
    
    def is_pediatric(row):
        return get_age(row) < 18
    
    def is_adult_male_young(row):
        age = get_age(row)
        return 18 <= age < 65 and row.get('sex') == 'M'
    
    def is_adult_female_young(row):
        age = get_age(row)
        return 18 <= age < 65 and row.get('sex') == 'F'
    
    def is_adult_male_senior(row):
        age = get_age(row)
        return age >= 65 and row.get('sex') == 'M'
    
    def is_adult_female_senior(row):
        age = get_age(row)
        return age >= 65 and row.get('sex') == 'F'
    
    return {
        'pediatric': is_pediatric,
        'adult_male_young': is_adult_male_young,
        'adult_female_young': is_adult_female_young,
        'adult_male_senior': is_adult_male_senior,
        'adult_female_senior': is_adult_female_senior
    }


class MultiSubgroupPredictor:
    """Manages multiple SequenceToOutcomePredictor models, one per subgroup.
    
    Parameters
    ----------
    subgroup_functions : Dict[str, Callable]
        Dictionary mapping subgroup names to functions that identify patients in each subgroup
    base_predictor_class : Type[SequenceToOutcomePredictor], optional
        The SequenceToOutcomePredictor class to instantiate for each subgroup
    input_var : str, default='consultation_sequence'
        Name of the input sequence column
    grouping_var : str, default='final_sequence'
        Name of the grouping sequence column  
    outcome_var : str, default='observed_specialty'
        Name of the outcome variable column
    min_samples : int, default=50
        Minimum number of samples required to train a model for a subgroup
    """
    
    def __init__(self, 
                 subgroup_functions: Dict[str, Callable[[Union[pd.Series, dict]], bool]], 
                 base_predictor_class: Optional[Type[SequenceToOutcomePredictor]], 
                 input_var: str = 'consultation_sequence',
                 grouping_var: str = 'final_sequence',
                 outcome_var: str = 'observed_specialty',
                 min_samples: int = 50):
        self.subgroup_functions: Dict[str, Callable[[Union[pd.Series, dict]], bool]] = subgroup_functions
        self.base_predictor_class: Optional[Type[SequenceToOutcomePredictor]] = base_predictor_class
        self.input_var: str = input_var
        self.grouping_var: str = grouping_var
        self.outcome_var: str = outcome_var
        self.min_samples: int = min_samples
        self.models: Dict[str, Any] = {}
        self.special_params: Optional[Dict[str, Any]] = None
    
    def fit(self, X: pd.DataFrame) -> "MultiSubgroupPredictor":
        """Train models for each subgroup that has sufficient data."""
        if self.base_predictor_class is None:
            raise ValueError("base_predictor_class must be provided to fit models")
        for name, func in self.subgroup_functions.items():
            # Filter to subgroup
            subgroup_data = X[X.apply(func, axis=1)]
            
            if len(subgroup_data) >= self.min_samples:
                # Train model
                model = self.base_predictor_class(
                    input_var=self.input_var,
                    grouping_var=self.grouping_var, 
                    outcome_var=self.outcome_var,
                    apply_special_category_filtering=False  # We handle subgroups ourselves
                )
                model.fit(subgroup_data)
                self.models[name] = model
            else:
                warnings.warn(f"Skipping {name}: only {len(subgroup_data)} samples")
        
        # Create backward compatibility params
        self.special_params = self._create_legacy_params()
        return self
    
    def predict(self, input_data: Union[Tuple[Any, ...], pd.Series]) -> Dict[str, float]:
        """Predict using appropriate subgroup model."""
        # Handle legacy tuple input (just sequence) - not supported without subgroup context
        if isinstance(input_data, tuple):
            raise RuntimeError(
                "Tuple input is not supported in MultiSubgroupPredictor; provide a full row (pd.Series) so the subgroup can be determined."
            )
        
        # Handle new Series input (full patient row)
        for name, func in self.subgroup_functions.items():
            if func(input_data) and name in self.models:
                sequence = input_data.get(self.input_var, ())
                return self.models[name].predict(sequence)
        
        # No subgroup model matched or model not trained for matched subgroup
        raise RuntimeError(
            "No trained subgroup model is available for this input; ensure fit() has trained models for applicable subgroups."
        )
    
    def _create_legacy_params(self) -> Dict[str, Any]:
        """Create special_params for backward compatibility."""
        pediatric_func = self.subgroup_functions['pediatric']
        
        # Create function map for all subgroups plus legacy keys
        func_map = {
            'paediatric': pediatric_func,
            'default': lambda row: not pediatric_func(row)
        }
        func_map.update(self.subgroup_functions)
        
        return {
            'special_category_func': pediatric_func,
            'special_category_dict': {
                'medical': 0.0, 
                'surgical': 0.0, 
                'haem/onc': 0.0, 
                'paediatric': 1.0
            },
            'special_func_map': func_map
        }


def create_subgroup_system(columns: Union[List[str], pd.Index], 
                          base_predictor_class: Optional[Type[SequenceToOutcomePredictor]] = None,
                          input_var: str = 'consultation_sequence',
                          grouping_var: str = 'final_sequence',
                          outcome_var: str = 'observed_specialty') -> Dict[str, Any]:
    """
    Create subgroup system with 5 standard patient categories.
    
    Parameters
    ----------
    columns : list
        Available columns (for validation)
    base_predictor_class : Type[SequenceToOutcomePredictor], optional
        SequenceToOutcomePredictor class
    input_var : str, default='consultation_sequence'
        Name of the input sequence column
    grouping_var : str, default='final_sequence'
        Name of the grouping sequence column
    outcome_var : str, default='observed_specialty'
        Name of the outcome variable column
        
    Returns
    -------
    dict
        Contains 'predictor' and legacy compatibility fields
    """
    subgroup_functions = create_subgroup_functions()
    
    if base_predictor_class is not None:
        predictor = MultiSubgroupPredictor(
            subgroup_functions, 
            base_predictor_class,
            input_var=input_var,
            grouping_var=grouping_var,
            outcome_var=outcome_var
        )
    else:
        predictor = None
    
    # Create legacy compatibility layer
    pediatric_func = subgroup_functions['pediatric']
    legacy_params = {
        'special_category_func': pediatric_func,
        'special_category_dict': {
            'medical': 0.0, 
            'surgical': 0.0, 
            'haem/onc': 0.0, 
            'paediatric': 1.0
        },
        'special_func_map': {
            'paediatric': pediatric_func,
            'default': lambda row: not pediatric_func(row),
            **subgroup_functions
        }
    }
    
    return {
        'predictor': predictor,
        'subgroup_functions': subgroup_functions,
        # Legacy fields for backward compatibility
        'special_category_func': legacy_params['special_category_func'],
        'special_category_dict': legacy_params['special_category_dict'], 
        'special_func_map': legacy_params['special_func_map']
    }

