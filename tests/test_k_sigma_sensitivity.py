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
    print(f"Loading pickled data from {pickle_path}...")
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
        print("Using default FlowSelection (could not extract from predictor)")
    else:
        print(f"Using FlowSelection: {flow_selection.cohort}")
    
    print(f"Found {len(subspecialty_data)} subspecialties")
    print(f"Top level ID: {top_level_id}")
    
    # Test each k_sigma value
    results = {}
    
    for k_sigma in k_sigma_values:
        print(f"\nTesting k_sigma = {k_sigma}...")
        
        # Create predictor with this k_sigma value
        hierarchical_predictor = create_hierarchical_predictor(
            hierarchy_df,
            column_mapping,
            top_level_id,
            k_sigma=k_sigma,
        )
        
        # Measure prediction time
        mean_time, metrics = measure_prediction_time(
            hierarchical_predictor,
            subspecialty_data,
            flow_selection,
            n_iterations=n_iterations,
        )
        
        # Analyze individual flows from input data to see how physical_maxes accumulate
        if k_sigma == k_sigma_values[0]:  # Only do this detailed analysis once
            print(f"\n  Analyzing individual departure flows from input data:")
            elective_totals = []
            emergency_totals = []
            
            for spec_id, inputs in subspecialty_data.items():
                # Get individual flows
                elective_flow = inputs.outflows.get("elective_departures")
                emergency_flow = inputs.outflows.get("emergency_departures")
                
                if elective_flow and elective_flow.flow_type == "pmf":
                    pmf_array = elective_flow.distribution
                    if isinstance(pmf_array, np.ndarray):
                        physical_max = len(pmf_array) - 1
                        elective_totals.append(physical_max)
                
                if emergency_flow and emergency_flow.flow_type == "pmf":
                    pmf_array = emergency_flow.distribution
                    if isinstance(pmf_array, np.ndarray):
                        physical_max = len(pmf_array) - 1
                        emergency_totals.append(physical_max)
            
            total_elective_max = sum(elective_totals)
            total_emergency_max = sum(emergency_totals)
            total_individual_max = total_elective_max + total_emergency_max
            
            print(f"    Sum of elective_departures physical_maxes: {total_elective_max}")
            print(f"    Sum of emergency_departures physical_maxes: {total_emergency_max}")
            print(f"    Total (elective + emergency): {total_individual_max}")
            print(f"    Expected (total inpatients): 305")
            print(f"    Difference: {total_individual_max - 305}")
            
            # Show distribution
            elective_dist = {}
            for val in elective_totals:
                elective_dist[val] = elective_dist.get(val, 0) + 1
            
            emergency_dist = {}
            for val in emergency_totals:
                emergency_dist[val] = emergency_dist.get(val, 0) + 1
            
            print(f"\n    Distribution of elective_departures physical_maxes:")
            for max_val in sorted(elective_dist.keys())[:15]:
                count = elective_dist[max_val]
                contribution = max_val * count
                print(f"      {max_val} patients: {count} subspecialties (contributes {contribution})")
            if len(elective_dist) > 15:
                print(f"      ... and {len(elective_dist) - 15} more values")
            
            print(f"\n    Distribution of emergency_departures physical_maxes:")
            for max_val in sorted(emergency_dist.keys())[:15]:
                count = emergency_dist[max_val]
                contribution = max_val * count
                print(f"      {max_val} patients: {count} subspecialties (contributes {contribution})")
            if len(emergency_dist) > 15:
                print(f"      ... and {len(emergency_dist) - 15} more values")
            
            # Show examples of subspecialties with physical_max = 1 to see if they actually have patients
            print(f"\n    Examples of subspecialties with physical_max = 1:")
            examples_shown = 0
            for spec_id, inputs in subspecialty_data.items():
                if examples_shown >= 5:
                    break
                elective_flow = inputs.outflows.get("elective_departures")
                emergency_flow = inputs.outflows.get("emergency_departures")
                
                if elective_flow and elective_flow.flow_type == "pmf":
                    pmf_array = elective_flow.distribution
                    if isinstance(pmf_array, np.ndarray) and len(pmf_array) - 1 == 1:
                        # Check if this is actually [1.0, 0.0] (1 patient, p=0) or [1.0] (0 patients)
                        if len(pmf_array) == 2 and pmf_array[1] == 0.0:
                            print(f"      {spec_id} - elective: PMF={pmf_array} (1 patient, p=0)")
                        elif len(pmf_array) == 1:
                            print(f"      {spec_id} - elective: PMF={pmf_array} (0 patients)")
                        examples_shown += 1
                
                if emergency_flow and emergency_flow.flow_type == "pmf" and examples_shown < 10:
                    pmf_array = emergency_flow.distribution
                    if isinstance(pmf_array, np.ndarray) and len(pmf_array) - 1 == 1:
                        if len(pmf_array) == 2 and pmf_array[1] == 0.0:
                            print(f"      {spec_id} - emergency: PMF={pmf_array} (1 patient, p=0)")
                        elif len(pmf_array) == 1:
                            print(f"      {spec_id} - emergency: PMF={pmf_array} (0 patients)")
                        examples_shown += 1
        
        results[k_sigma] = {
            'mean_time': mean_time,
            **metrics
        }
        
        print(f"  Mean time: {mean_time:.4f}s")
        print(f"  Arrivals PMF elements: {metrics['arrivals_pmf_elements']}")
        print(f"  Departures PMF elements: {metrics['departures_pmf_elements']}")
        print(f"  Net flow PMF elements: {metrics['net_flow_pmf_elements']}")
        
        # Diagnostic: Show maximum departures at hospital level
        hospital_departures_max = None
        for pmf_info in metrics['pmf_metrics']:
            if pmf_info['entity_id'].startswith('hospital:'):
                hospital_departures_max = pmf_info['departures_max']
                break
        
        if hospital_departures_max is not None:
            print(f"  Hospital level max departures: {hospital_departures_max}")
        
        # Show detailed breakdown of departures at subspecialty level
        subspecialty_data = []
        for pmf_info in metrics['pmf_metrics']:
            if pmf_info['entity_id'].startswith('subspecialty:'):
                subspecialty_data.append({
                    'entity_id': pmf_info['entity_id'],
                    'departures_max': pmf_info['departures_max'],
                    'departures_pmf_len': pmf_info['departures_pmf_len'],
                })
        
        if subspecialty_data:
            total_max = sum(d['departures_max'] for d in subspecialty_data)
            print(f"\n  Detailed subspecialty departures breakdown:")
            print(f"  Total subspecialties: {len(subspecialty_data)}")
            print(f"  Sum of max departures (from aggregated PMF): {total_max}")
            print(f"  Expected (total inpatients): 305")
            print(f"  Difference: {total_max - 305}")
            
            # Now check the individual flows from bottom_level_data
            # We need to access the predictor's bottom_level_data
            # Actually, we can get it from the test function - let me add it to metrics
            print(f"\n  Analyzing individual flows (elective + emergency) from input data:")
            
            # Group by departures_max value to see distribution
            max_distribution = {}
            for d in subspecialty_data:
                max_val = d['departures_max']
                max_distribution[max_val] = max_distribution.get(max_val, 0) + 1
            
            print(f"\n  Distribution of aggregated departures_max values:")
            for max_val in sorted(max_distribution.keys())[:20]:  # Show first 20
                count = max_distribution[max_val]
                contribution = max_val * count
                print(f"    {max_val} patients: {count} subspecialties (contributes {contribution})")
            if len(max_distribution) > 20:
                print(f"    ... and {len(max_distribution) - 20} more values")
            
            # Show examples of subspecialties with high departures_max
            sorted_by_max = sorted(subspecialty_data, key=lambda x: x['departures_max'], reverse=True)
            print(f"\n  Top 10 subspecialties by departures_max:")
            for i, d in enumerate(sorted_by_max[:10], 1):
                print(f"    {i}. {d['entity_id']}: max={d['departures_max']}, PMF_len={d['departures_pmf_len']}")
            
            # Check for cases where PMF_len != max + 1 (should always be true)
            mismatches = [d for d in subspecialty_data if d['departures_pmf_len'] != d['departures_max'] + 1]
            if mismatches:
                print(f"\n  ⚠️  Found {len(mismatches)} subspecialties where PMF_len != max + 1:")
                for d in mismatches[:5]:
                    print(f"    {d['entity_id']}: max={d['departures_max']}, PMF_len={d['departures_pmf_len']}")
            
            print(f"\n  Note: 4394 is the sum of PMF array LENGTHS across all entities,")
            print(f"        not the maximum departures. Each entity has a PMF array,")
            print(f"        and we're summing their lengths, which is expected.")
    
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
        print("Usage: python test_k_sigma_sensitivity.py <pickle_path> [k_sigma_values...]")
        print("Example: python test_k_sigma_sensitivity.py hierarchical_predictors.pkl 2 4 6 8 10 12")
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
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

