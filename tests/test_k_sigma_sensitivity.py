"""Test sensitivity of prediction generation time to k_sigma parameter.

This script benchmarks the time taken to generate hierarchical predictions
across a range of k_sigma values. It loads pickled HierarchicalPredictor objects
and measures performance metrics for each k_sigma value.

To run it: "uv run python tests/test_k_sigma_sensitivity.py <pickle_path> [k_sigma_values...]"
Example: "uv run python tests/test_k_sigma_sensitivity.py hierarchical_predictors.pkl 2 4 6 8 10 12"
"""

import pickle
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from patientflow.predict.hierarchy import (
    HierarchicalPredictor,
    FlowSelection,
    create_hierarchical_predictor,
)
from patientflow.predict.subspecialty import SubspecialtyPredictionInputs


def load_pickled_data(pickle_path: str) -> Dict[str, Dict[str, Any]]:
    """Load pickled data containing predictors and associated metadata.
    
    Parameters
    ----------
    pickle_path : str
        Path to the pickle file
        
    Returns
    -------
    dict[str, dict]
        Dictionary with keys 'all', 'elective', 'emergency', where each value
        contains: 'predictor', 'subspecialty_data', 'hierarchy_df', 
        'column_mapping', 'top_level_id'
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_hierarchy_info(
    predictor: HierarchicalPredictor
) -> tuple[Any, Optional[FlowSelection]]:
    """Extract hierarchy information from a HierarchicalPredictor.
    
    Parameters
    ----------
    predictor : HierarchicalPredictor
        The predictor to extract info from
        
    Returns
    -------
    tuple
        (hierarchy, flow_selection) - hierarchy object and flow selection
    """
    hierarchy = predictor.hierarchy
    # Get flow_selection from first cached bundle
    flow_selection = None
    if predictor.cache:
        first_bundle = next(iter(predictor.cache.values()))
        flow_selection = first_bundle.flow_selection
    
    return hierarchy, flow_selection


def measure_prediction_time(
    predictor: HierarchicalPredictor,
    bottom_level_data: Dict[str, SubspecialtyPredictionInputs],
    flow_selection: FlowSelection,
    n_iterations: int = 10,
) -> tuple[float, Dict[str, Any]]:
    """Measure time to generate predictions across multiple iterations.
    
    Parameters
    ----------
    predictor : HierarchicalPredictor
        Configured hierarchical predictor
    bottom_level_data : dict[str, SubspecialtyPredictionInputs]
        Bottom-level prediction inputs
    flow_selection : FlowSelection
        Flow selection configuration
    n_iterations : int, default=10
        Number of iterations to run for timing
        
    Returns
    -------
    tuple
        (mean_time_seconds, metrics_dict)
        metrics_dict contains timing statistics and PMF size information
    """
    times = []
    pmf_lengths = []
    
    for i in range(n_iterations):
        # Clear cache to ensure fresh computation
        predictor.cache.clear()
        
        start = time.perf_counter()
        results = predictor.predict_all_levels(
            bottom_level_data,
            flow_selection=flow_selection
        )
        end = time.perf_counter()
        
        elapsed = end - start
        times.append(elapsed)
        
        # Collect PMF length metrics on first iteration
        if i == 0:
            for entity_id, bundle in results.items():
                # Calculate maximum value (not just length)
                arrivals_max = len(bundle.arrivals.probabilities) - 1 if len(bundle.arrivals.probabilities) > 0 else 0
                departures_max = len(bundle.departures.probabilities) - 1 if len(bundle.departures.probabilities) > 0 else 0
                pmf_lengths.append({
                    'entity_id': entity_id,
                    'arrivals_pmf_len': len(bundle.arrivals.probabilities),
                    'departures_pmf_len': len(bundle.departures.probabilities),
                    'net_flow_pmf_len': len(bundle.net_flow.probabilities),
                    'arrivals_expected': bundle.arrivals.expected_value,
                    'departures_expected': bundle.departures.expected_value,
                    'arrivals_max': arrivals_max,
                    'departures_max': departures_max,
                })
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    metrics = {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'median_time': median_time,
        'n_iterations': n_iterations,
        'pmf_metrics': pmf_lengths,
        'total_pmf_elements': sum(
            m['arrivals_pmf_len'] + m['departures_pmf_len'] + m['net_flow_pmf_len']
            for m in pmf_lengths
        ),
        'arrivals_pmf_elements': sum(m['arrivals_pmf_len'] for m in pmf_lengths),
        'departures_pmf_elements': sum(m['departures_pmf_len'] for m in pmf_lengths),
        'net_flow_pmf_elements': sum(m['net_flow_pmf_len'] for m in pmf_lengths),
    }
    
    return mean_time, metrics


def test_k_sigma_sensitivity(
    pickle_path: str,
    k_sigma_values: List[float],
    n_iterations: int = 10,
) -> Dict[float, Dict[str, Any]]:
    """Test sensitivity of prediction time to k_sigma parameter.
    
    Parameters
    ----------
    pickle_path : str
        Path to pickle file containing predictor data with structure:
        {
            'all': {
                'predictor': HierarchicalPredictor,
                'subspecialty_data': Dict[str, SubspecialtyPredictionInputs],
                'hierarchy_df': pd.DataFrame,
                'column_mapping': Dict[str, str],
                'top_level_id': str
            },
            ...
        }
    k_sigma_values : list[float]
        List of k_sigma values to test
    n_iterations : int, default=10
        Number of iterations per k_sigma value for timing
        
    Returns
    -------
    dict[float, dict]
        Dictionary mapping k_sigma values to their timing/metrics results
    """
    # Load pickled data
    pickled_data = load_pickled_data(pickle_path)
    
    if 'all' not in pickled_data:
        raise ValueError(
            f"Pickled data must contain 'all' key. "
            f"Available keys: {list(pickled_data.keys())}"
        )
    
    # Extract all required data from 'all' entry
    all_data = pickled_data['all']
    
    required_keys = ['predictor', 'subspecialty_data', 'hierarchy_df', 
                     'column_mapping', 'top_level_id']
    missing_keys = [key for key in required_keys if key not in all_data]
    if missing_keys:
        raise ValueError(
            f"Missing required keys in 'all' data: {missing_keys}. "
            f"Found keys: {list(all_data.keys())}"
        )
    
    predictor = all_data['predictor']
    subspecialty_data = all_data['subspecialty_data']
    hierarchy_df = all_data['hierarchy_df']
    column_mapping = all_data['column_mapping']
    top_level_id = all_data['top_level_id']
    
    # Extract flow_selection from the predictor
    _, flow_selection = extract_hierarchy_info(predictor)
    
    if flow_selection is None:
        flow_selection = FlowSelection.default()
    
    # Test each k_sigma value
    results = {}
    
    # Process all flow types
    flow_types = ['all', 'elective', 'emergency']
    flow_type_data = {}
    for flow_type in flow_types:
        if flow_type in pickled_data:
            flow_type_data[flow_type] = pickled_data[flow_type]
    
    for k_sigma in k_sigma_values:
        # Process each flow type
        flow_type_results = {}
        
        for flow_type in flow_types:
            if flow_type not in flow_type_data:
                continue
                
            flow_data = flow_type_data[flow_type]
            flow_subspecialty_data = flow_data['subspecialty_data']
            flow_hierarchy_df = flow_data['hierarchy_df']
            flow_column_mapping = flow_data['column_mapping']
            flow_top_level_id = flow_data['top_level_id']
            
            # Extract flow_selection from the predictor
            flow_predictor = flow_data['predictor']
            _, flow_flow_selection = extract_hierarchy_info(flow_predictor)
            
            if flow_flow_selection is None:
                flow_flow_selection = FlowSelection.default()
            
            # Create predictor with this k_sigma value
            hierarchical_predictor = create_hierarchical_predictor(
                flow_hierarchy_df,
                flow_column_mapping,
                flow_top_level_id,
                k_sigma=k_sigma,
            )
            
            # Measure prediction time
            mean_time, metrics = measure_prediction_time(
                hierarchical_predictor,
                flow_subspecialty_data,
                flow_flow_selection,
                n_iterations=n_iterations,
            )
            
            flow_type_results[flow_type] = {
                'mean_time': mean_time,
                **metrics
            }
            
            # Group PMF metrics by hierarchy level
            level_sums = {
                'subspecialty': {'arrivals': 0, 'departures': 0, 'net_flow': 0, 'count': 0},
                'reporting_unit': {'arrivals': 0, 'departures': 0, 'net_flow': 0, 'count': 0},
                'division': {'arrivals': 0, 'departures': 0, 'net_flow': 0, 'count': 0},
                'board': {'arrivals': 0, 'departures': 0, 'net_flow': 0, 'count': 0},
                'hospital': {'arrivals': 0, 'departures': 0, 'net_flow': 0, 'count': 0},
            }
            
            for pmf_info in metrics['pmf_metrics']:
                entity_id = pmf_info['entity_id']
                # Extract level from entity_id prefix (e.g., "subspecialty:..." -> "subspecialty")
                if ':' in entity_id:
                    level = entity_id.split(':')[0]
                    if level in level_sums:
                        level_sums[level]['arrivals'] += pmf_info['arrivals_pmf_len']
                        level_sums[level]['departures'] += pmf_info['departures_pmf_len']
                        level_sums[level]['net_flow'] += pmf_info['net_flow_pmf_len']
                        level_sums[level]['count'] += 1
            
            # Display PMF length sums by level
            print(f"\n  k_sigma = {k_sigma}")
            print(f"  Flow type: {flow_type}")
            print(f"    PMF Length Sums by Hierarchy Level:")
            print(f"    {'Level':<20} {'Count':<10} {'Arrivals':<12} {'Departures':<12} {'Net Flow':<12}")
            print(f"    {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
            for level in ['subspecialty', 'reporting_unit', 'division', 'board', 'hospital']:
                if level_sums[level]['count'] > 0:
                    print(f"    {level:<20} {level_sums[level]['count']:<10} "
                          f"{level_sums[level]['arrivals']:<12} {level_sums[level]['departures']:<12} "
                          f"{level_sums[level]['net_flow']:<12}")
        
        # Store results for all flow types
        results[k_sigma] = {
            'flow_type_results': flow_type_results
        }
        
        # For backward compatibility and analysis, use 'all' results as primary
        if 'all' in flow_type_results:
            mean_time = flow_type_results['all']['mean_time']
            metrics = {k: v for k, v in flow_type_results['all'].items() if k != 'mean_time'}
            subspecialty_data = flow_type_data['all']['subspecialty_data']
            analysis_flow_selection = flow_type_data['all'].get('predictor')
            if analysis_flow_selection:
                _, analysis_flow_selection = extract_hierarchy_info(analysis_flow_selection)
        else:
            # Fallback to first available flow type
            first_flow_type = list(flow_type_results.keys())[0]
            mean_time = flow_type_results[first_flow_type]['mean_time']
            metrics = {k: v for k, v in flow_type_results[first_flow_type].items() if k != 'mean_time'}
            subspecialty_data = flow_type_data[first_flow_type]['subspecialty_data']
            analysis_flow_selection = flow_type_data[first_flow_type].get('predictor')
            if analysis_flow_selection:
                _, analysis_flow_selection = extract_hierarchy_info(analysis_flow_selection)
        
        if analysis_flow_selection is None:
            analysis_flow_selection = FlowSelection.default()
        
        # Update results with primary metrics for backward compatibility
        results[k_sigma].update({
            'mean_time': mean_time,
            **metrics
        })
    
    return results


def print_results_summary(results: Dict[float, Dict[str, Any]]) -> None:
    """Print a summary of the sensitivity test results.
    
    Parameters
    ----------
    results : dict[float, dict]
        Results dictionary from test_k_sigma_sensitivity
    """
    print("\n" + "="*80)
    print("K_SIGMA SENSITIVITY TEST RESULTS")
    print("="*80)
    
    print(f"\n{'k_sigma':<10} {'Mean Time (s)':<15} {'Std Time (s)':<15} {'Arrivals PMF':<15} {'Departures PMF':<15} {'Net Flow PMF':<15}")
    print("-" * 100)
    
    for k_sigma in sorted(results.keys()):
        r = results[k_sigma]
        print(
            f"{k_sigma:<10.2f} "
            f"{r['mean_time']:<15.4f} "
            f"{r['std_time']:<15.4f} "
            f"{r['arrivals_pmf_elements']:<15} "
            f"{r['departures_pmf_elements']:<15} "
            f"{r['net_flow_pmf_elements']:<15}"
        )
    
    # Calculate relative performance (slowdown/speedup)
    if len(results) > 1:
        print("\n" + "-" * 80)
        print("Performance relative to smallest k_sigma:")
        k_sigmas = sorted(results.keys())
        baseline_k = k_sigmas[0]
        baseline_time = results[baseline_k]['mean_time']
        
        for k_sigma in k_sigmas[1:]:
            current_time = results[k_sigma]['mean_time']
            slowdown = current_time / baseline_time
            print(f"  k_sigma={k_sigma:.2f} is {slowdown:.2f}x slower than k_sigma={baseline_k:.2f}")


if __name__ == "__main__":
    """Run k_sigma sensitivity test from command line or by modifying this script."""
    import sys
    
    if len(sys.argv) < 2:
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    
    # Default k_sigma values if not provided
    if len(sys.argv) > 2:
        k_sigma_values = [float(x) for x in sys.argv[2:]]
    else:
        k_sigma_values = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    
    try:
        results = test_k_sigma_sensitivity(
            pickle_path=pickle_path,
            k_sigma_values=k_sigma_values,
            n_iterations=10,
        )
        
        print_results_summary(results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

