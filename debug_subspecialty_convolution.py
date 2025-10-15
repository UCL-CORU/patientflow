#!/usr/bin/env python3
"""Debug script for checking subspecialty convolution logic step by step.

This script loads saved SubspecialtyPredictionInputs for a specific subspecialty
and walks through the convolution logic, showing intermediate results at each step
up to calculating the net flow.

Usage:
    python debug_subspecialty_convolution.py <subspecialty_data_path> <check_spec>

Arguments:
    subspecialty_data_path: Path to the saved subspecialty_data dictionary (joblib/pickle)
    check_spec: Name of the subspecialty to debug (e.g., 'Cardiology')

Example:
    python debug_subspecialty_convolution.py subspecialty_data.joblib "Cardiology"
"""

import sys
import numpy as np
import joblib
from typing import Dict, List
from dataclasses import dataclass

# Import the classes from your patientflow package
from patientflow.predict.subspecialty import SubspecialtyPredictionInputs, FlowInputs
from patientflow.predict.hierarchy import (
    DemandPredictor,
    DemandPrediction,
    FlowSelection,
    PredictionBundle,
)


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """Print a formatted separator line."""
    if title:
        title_str = f" {title} "
        padding = (width - len(title_str)) // 2
        print(f"\n{char * padding}{title_str}{char * padding}")
    else:
        print(f"\n{char * width}")


def print_pmf_summary(pmf: np.ndarray, name: str, offset: int = 0, max_display: int = 15):
    """Print a summary of a PMF with key statistics."""
    expectation = np.sum((np.arange(len(pmf)) + offset) * pmf)
    
    # Find mode
    mode_idx = np.argmax(pmf)
    mode_value = mode_idx + offset
    
    # Calculate percentiles
    cumsum = np.cumsum(pmf)
    p50_idx = np.searchsorted(cumsum, 0.50)
    p75_idx = np.searchsorted(cumsum, 0.75)
    p95_idx = np.searchsorted(cumsum, 0.95)
    
    print(f"\n{name}:")
    print(f"  Length: {len(pmf)} values (support: {offset} to {len(pmf) - 1 + offset})")
    print(f"  Expectation (mean): {expectation:.3f}")
    print(f"  Mode: {mode_value} (probability: {pmf[mode_idx]:.4f})")
    print(f"  Percentiles: P50={p50_idx + offset}, P75={p75_idx + offset}, P95={p95_idx + offset}")
    print(f"  Sum of probabilities: {np.sum(pmf):.6f}")
    
    # Display the PMF values (centered around expectation)
    if len(pmf) <= max_display:
        display_values = ", ".join(f"{v:.4f}" for v in pmf)
        print(f"  PMF: [{display_values}]")
    else:
        center_idx = int(np.round(expectation - offset))
        half_window = max_display // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(pmf), start_idx + max_display)
        
        if end_idx - start_idx < max_display:
            start_idx = max(0, end_idx - max_display)
        
        display_values = ", ".join(f"{v:.4f}" for v in pmf[start_idx:end_idx])
        start_val = start_idx + offset
        end_val = end_idx - 1 + offset
        print(f"  PMF[{start_val}:{end_val}]: [{display_values}]")
        
        # Show where the mass is concentrated
        top_5_indices = np.argsort(pmf)[-5:][::-1]
        print(f"  Top 5 probabilities:")
        for idx in top_5_indices:
            val = idx + offset
            print(f"    P(X={val}) = {pmf[idx]:.4f}")


def print_poisson_summary(lambda_param: float, name: str):
    """Print a summary of a Poisson distribution."""
    print(f"\n{name}:")
    print(f"  Type: Poisson distribution")
    print(f"  Lambda (rate parameter): {lambda_param:.3f}")
    print(f"  Expected value: {lambda_param:.3f}")
    print(f"  Standard deviation: {np.sqrt(lambda_param):.3f}")
    
    # Show a few probabilities
    from scipy.stats import poisson
    print(f"  Probabilities:")
    for k in range(min(10, int(lambda_param + 10))):
        prob = poisson.pmf(k, lambda_param)
        if prob > 0.001:  # Only show non-negligible probabilities
            print(f"    P(X={k}) = {prob:.4f}")


def debug_flow_inputs(inputs: SubspecialtyPredictionInputs, check_spec: str):
    """Print detailed information about the flow inputs."""
    print_separator(f"SUBSPECIALTY: {check_spec}", "=")
    
    print(f"\nPrediction window: {inputs.prediction_window}")
    
    # INFLOWS
    print_separator("INFLOWS (Arrivals)", "-")
    print(f"\nNumber of inflow sources: {len(inputs.inflows)}")
    
    total_expected_arrivals = 0.0
    
    for flow_id, flow in inputs.inflows.items():
        print_separator(f"Inflow: {flow.get_display_name()}", "·")
        print(f"  Flow ID: {flow_id}")
        print(f"  Flow type: {flow.flow_type}")
        
        if flow.flow_type == "pmf":
            print_pmf_summary(flow.distribution, "Distribution", offset=0)
            expectation = np.sum(np.arange(len(flow.distribution)) * flow.distribution)
            total_expected_arrivals += expectation
        elif flow.flow_type == "poisson":
            print_poisson_summary(flow.distribution, "Distribution")
            total_expected_arrivals += flow.distribution
    
    print(f"\n>>> Total expected arrivals (sum of all inflows): {total_expected_arrivals:.3f}")
    
    # OUTFLOWS
    print_separator("OUTFLOWS (Departures)", "-")
    print(f"\nNumber of outflow sources: {len(inputs.outflows)}")
    
    total_expected_departures = 0.0
    
    for flow_id, flow in inputs.outflows.items():
        print_separator(f"Outflow: {flow.get_display_name()}", "·")
        print(f"  Flow ID: {flow_id}")
        print(f"  Flow type: {flow.flow_type}")
        
        if flow.flow_type == "pmf":
            print_pmf_summary(flow.distribution, "Distribution", offset=0)
            expectation = np.sum(np.arange(len(flow.distribution)) * flow.distribution)
            total_expected_departures += expectation
        elif flow.flow_type == "poisson":
            print_poisson_summary(flow.distribution, "Distribution")
            total_expected_departures += flow.distribution
    
    print(f"\n>>> Total expected departures (sum of all outflows): {total_expected_departures:.3f}")
    
    print_separator("EXPECTED NET FLOW", "-")
    expected_net_flow = total_expected_arrivals - total_expected_departures
    print(f"\nExpected arrivals: {total_expected_arrivals:.3f}")
    print(f"Expected departures: {total_expected_departures:.3f}")
    print(f"Expected net flow: {expected_net_flow:.3f}")


def manual_convolve(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Manual convolution implementation for comparison."""
    result = np.convolve(p, q)
    return result


def manual_truncate(p: np.ndarray, epsilon: float) -> np.ndarray:
    """Manual truncation implementation for comparison."""
    cumsum = np.cumsum(p)
    cutoff_idx = np.searchsorted(cumsum, 1 - epsilon) + 1
    return p[:cutoff_idx]


def manual_poisson_pmf(lambda_param: float, max_k: int) -> np.ndarray:
    """Manual Poisson PMF generation for comparison."""
    from scipy.stats import poisson
    if lambda_param == 0:
        return np.array([1.0])
    k = np.arange(max_k + 1)
    return poisson.pmf(k, lambda_param)


def manual_poisson_max(lambda_param: float, epsilon: float) -> int:
    """Manual calculation of Poisson max_k for comparison."""
    from scipy.stats import poisson
    if lambda_param == 0:
        return 0
    return poisson.ppf(1 - epsilon, lambda_param).astype(int)


def manual_expected_value(p: np.ndarray, offset: int = 0) -> float:
    """Manual expected value calculation for comparison."""
    return np.sum((np.arange(len(p)) + offset) * p)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str, tolerance: float = 1e-10):
    """Compare two arrays and report differences."""
    if len(arr1) != len(arr2):
        print(f"  ⚠️  {name}: DIFFERENT LENGTHS ({len(arr1)} vs {len(arr2)})")
        return False
    
    if np.allclose(arr1, arr2, atol=tolerance):
        print(f"  ✓ {name}: Arrays match (within tolerance {tolerance})")
        return True
    else:
        max_diff = np.max(np.abs(arr1 - arr2))
        print(f"  ✗ {name}: Arrays differ (max difference: {max_diff:.2e})")
        # Show where they differ
        diff_indices = np.where(np.abs(arr1 - arr2) > tolerance)[0]
        print(f"    Differences at indices: {diff_indices[:10]}...")  # Show first 10
        return False


def manual_compute_net_flow_pmf(
    p_arrivals: np.ndarray, p_departures: np.ndarray, epsilon: float
) -> tuple:
    """Manual implementation of net flow PMF computation for comparison.
    
    Computes PMF for net flow (arrivals - departures) using nested loops.
    """
    max_arrivals = len(p_arrivals) - 1
    max_departures = len(p_departures) - 1
    
    # Net flow ranges from -max_departures to +max_arrivals
    net_flow_size = max_arrivals + max_departures + 1
    p_net = np.zeros(net_flow_size)
    
    # Compute probability for each possible net flow value
    for a in range(len(p_arrivals)):
        for d in range(len(p_departures)):
            net = a - d  # Net flow value
            idx = net + max_departures  # Array index (shifted to handle negatives)
            p_net[idx] += p_arrivals[a] * p_departures[d]
    
    # Truncate from both ends
    initial_offset = -max_departures
    
    # Find where cumulative probability exceeds epsilon (from left)
    cumsum = np.cumsum(p_net)
    total_prob = cumsum[-1]
    
    if total_prob == 0:
        return np.array([1.0]), 0
    
    left_cutoff = np.searchsorted(cumsum, epsilon)
    
    # Find where remaining probability is less than epsilon (from right)
    cumsum_reversed = np.cumsum(p_net[::-1])
    right_cutoff = len(p_net) - np.searchsorted(cumsum_reversed, epsilon)
    
    # Ensure we keep at least one element
    left_cutoff = max(0, left_cutoff)
    right_cutoff = min(len(p_net), max(right_cutoff, left_cutoff + 1))
    
    p_truncated = p_net[left_cutoff:right_cutoff]
    final_offset = initial_offset + left_cutoff
    
    return p_truncated, final_offset


def debug_convolution_steps(inputs: SubspecialtyPredictionInputs, check_spec: str, epsilon: float = 1e-7):
    """Walk through the convolution steps and show intermediate results."""
    
    print_separator("STEP-BY-STEP CONVOLUTION WITH VERIFICATION", "=")
    
    print(f"\nThis section compares the DemandPredictor methods with manual calculations")
    print(f"to verify the convolution logic is implemented correctly.\n")
    
    predictor = DemandPredictor(epsilon=epsilon)
    
    # ARRIVALS CONVOLUTION
    print_separator("COMPUTING ARRIVALS DISTRIBUTION", "-")
    
    print("\nStarting with degenerate distribution at 0: [1.0]")
    p_arrivals = np.array([1.0])
    p_arrivals_manual = np.array([1.0])
    
    flow_count = 0
    for flow_id, flow in inputs.inflows.items():
        flow_count += 1
        print_separator(f"Step {flow_count}: Adding {flow.get_display_name()}", "·")
        
        if flow.flow_type == "poisson":
            # Generate Poisson PMF using both predictor and manual methods
            print(f"\n  POISSON GENERATION (λ={flow.distribution:.3f}):")
            
            max_k_predictor = predictor._calculate_poisson_max(flow.distribution)
            max_k_manual = manual_poisson_max(flow.distribution, epsilon)
            print(f"    Predictor max_k: {max_k_predictor}")
            print(f"    Manual max_k: {max_k_manual}")
            if max_k_predictor == max_k_manual:
                print(f"    ✓ max_k values match")
            else:
                print(f"    ✗ max_k values differ!")
            
            p_flow = predictor._poisson_pmf(flow.distribution, max_k_predictor)
            p_flow_manual = manual_poisson_pmf(flow.distribution, max_k_manual)
            
            print(f"\n  Comparing Poisson PMF generation:")
            compare_arrays(p_flow, p_flow_manual, "Poisson PMF")
        else:
            p_flow = flow.distribution
            p_flow_manual = flow.distribution.copy()
            print(f"  Using provided PMF (no generation needed)")
        
        print_pmf_summary(p_flow, f"  Flow PMF ({flow_id})")
        
        # Before convolution
        print(f"\n  Before convolution:")
        print(f"    Running total length: {len(p_arrivals)}")
        exp_before = predictor._expected_value(p_arrivals)
        exp_before_manual = manual_expected_value(p_arrivals_manual)
        print(f"    Predictor expectation: {exp_before:.6f}")
        print(f"    Manual expectation: {exp_before_manual:.6f}")
        
        # Convolve using both methods
        print(f"\n  CONVOLUTION:")
        p_arrivals_before = p_arrivals.copy()
        p_arrivals = predictor._convolve(p_arrivals, p_flow)
        p_arrivals_manual = manual_convolve(p_arrivals_manual, p_flow_manual)
        
        print(f"    Predictor result length: {len(p_arrivals)}")
        print(f"    Manual result length: {len(p_arrivals_manual)}")
        compare_arrays(p_arrivals, p_arrivals_manual, "Convolution result")
        
        exp_after = predictor._expected_value(p_arrivals)
        exp_after_manual = manual_expected_value(p_arrivals_manual)
        print(f"    Predictor expectation: {exp_after:.6f}")
        print(f"    Manual expectation: {exp_after_manual:.6f}")
        
        # Verify expectation addition property
        exp_flow = predictor._expected_value(p_flow)
        expected_sum = exp_before + exp_flow
        print(f"    Expected sum (E[before] + E[flow]): {expected_sum:.6f}")
        if abs(exp_after - expected_sum) < 1e-6:
            print(f"    ✓ Convolution preserves expectation correctly")
        else:
            print(f"    ✗ Expectation mismatch: {abs(exp_after - expected_sum):.2e}")
        
        # Truncate using both methods
        print(f"\n  TRUNCATION (epsilon={epsilon}):")
        before_truncate_len = len(p_arrivals)
        p_arrivals = predictor._truncate(p_arrivals)
        p_arrivals_manual = manual_truncate(p_arrivals_manual, epsilon)
        
        print(f"    Predictor: truncated from {before_truncate_len} to {len(p_arrivals)} values")
        print(f"    Manual: truncated to {len(p_arrivals_manual)} values")
        compare_arrays(p_arrivals, p_arrivals_manual, "Truncated result")
        
        exp_after_trunc = predictor._expected_value(p_arrivals)
        exp_after_trunc_manual = manual_expected_value(p_arrivals_manual)
        print(f"    Predictor expectation after truncation: {exp_after_trunc:.6f}")
        print(f"    Manual expectation after truncation: {exp_after_trunc_manual:.6f}")
    
    print_separator("FINAL ARRIVALS DISTRIBUTION", "·")
    print_pmf_summary(p_arrivals, "Arrivals PMF (via Predictor)")
    print(f"\nVerifying final arrivals:")
    compare_arrays(p_arrivals, p_arrivals_manual, "Final arrivals PMF")
    arrivals_prediction = predictor._create_prediction(check_spec, "arrivals", p_arrivals)
    
    # DEPARTURES CONVOLUTION
    print_separator("COMPUTING DEPARTURES DISTRIBUTION", "-")
    
    print("\nStarting with degenerate distribution at 0: [1.0]")
    p_departures = np.array([1.0])
    p_departures_manual = np.array([1.0])
    
    flow_count = 0
    for flow_id, flow in inputs.outflows.items():
        flow_count += 1
        print_separator(f"Step {flow_count}: Adding {flow.get_display_name()}", "·")
        
        if flow.flow_type == "poisson":
            # Generate Poisson PMF using both methods
            print(f"\n  POISSON GENERATION (λ={flow.distribution:.3f}):")
            
            max_k_predictor = predictor._calculate_poisson_max(flow.distribution)
            max_k_manual = manual_poisson_max(flow.distribution, epsilon)
            print(f"    Predictor max_k: {max_k_predictor}")
            print(f"    Manual max_k: {max_k_manual}")
            if max_k_predictor == max_k_manual:
                print(f"    ✓ max_k values match")
            else:
                print(f"    ✗ max_k values differ!")
            
            p_flow = predictor._poisson_pmf(flow.distribution, max_k_predictor)
            p_flow_manual = manual_poisson_pmf(flow.distribution, max_k_manual)
            
            print(f"\n  Comparing Poisson PMF generation:")
            compare_arrays(p_flow, p_flow_manual, "Poisson PMF")
        else:
            p_flow = flow.distribution
            p_flow_manual = flow.distribution.copy()
            print(f"  Using provided PMF (no generation needed)")
        
        print_pmf_summary(p_flow, f"  Flow PMF ({flow_id})")
        
        # Before convolution
        print(f"\n  Before convolution:")
        print(f"    Running total length: {len(p_departures)}")
        exp_before = predictor._expected_value(p_departures)
        exp_before_manual = manual_expected_value(p_departures_manual)
        print(f"    Predictor expectation: {exp_before:.6f}")
        print(f"    Manual expectation: {exp_before_manual:.6f}")
        
        # Convolve using both methods
        print(f"\n  CONVOLUTION:")
        p_departures = predictor._convolve(p_departures, p_flow)
        p_departures_manual = manual_convolve(p_departures_manual, p_flow_manual)
        
        print(f"    Predictor result length: {len(p_departures)}")
        print(f"    Manual result length: {len(p_departures_manual)}")
        compare_arrays(p_departures, p_departures_manual, "Convolution result")
        
        exp_after = predictor._expected_value(p_departures)
        exp_after_manual = manual_expected_value(p_departures_manual)
        print(f"    Predictor expectation: {exp_after:.6f}")
        print(f"    Manual expectation: {exp_after_manual:.6f}")
        
        # Verify expectation addition property
        exp_flow = predictor._expected_value(p_flow)
        expected_sum = exp_before + exp_flow
        print(f"    Expected sum (E[before] + E[flow]): {expected_sum:.6f}")
        if abs(exp_after - expected_sum) < 1e-6:
            print(f"    ✓ Convolution preserves expectation correctly")
        else:
            print(f"    ✗ Expectation mismatch: {abs(exp_after - expected_sum):.2e}")
        
        # Truncate using both methods
        print(f"\n  TRUNCATION (epsilon={epsilon}):")
        before_truncate_len = len(p_departures)
        p_departures = predictor._truncate(p_departures)
        p_departures_manual = manual_truncate(p_departures_manual, epsilon)
        
        print(f"    Predictor: truncated from {before_truncate_len} to {len(p_departures)} values")
        print(f"    Manual: truncated to {len(p_departures_manual)} values")
        compare_arrays(p_departures, p_departures_manual, "Truncated result")
        
        exp_after_trunc = predictor._expected_value(p_departures)
        exp_after_trunc_manual = manual_expected_value(p_departures_manual)
        print(f"    Predictor expectation after truncation: {exp_after_trunc:.6f}")
        print(f"    Manual expectation after truncation: {exp_after_trunc_manual:.6f}")
    
    print_separator("FINAL DEPARTURES DISTRIBUTION", "·")
    print_pmf_summary(p_departures, "Departures PMF (via Predictor)")
    print(f"\nVerifying final departures:")
    compare_arrays(p_departures, p_departures_manual, "Final departures PMF")
    departures_prediction = predictor._create_prediction(check_spec, "departures", p_departures)
    
    # NET FLOW CALCULATION
    print_separator("COMPUTING NET FLOW DISTRIBUTION", "-")
    
    print(f"\nComputing net flow as difference: Arrivals - Departures")
    print(f"  Arrivals support: 0 to {len(p_arrivals) - 1}")
    print(f"  Departures support: 0 to {len(p_departures) - 1}")
    
    max_arrivals = len(p_arrivals) - 1
    max_departures = len(p_departures) - 1
    
    print(f"\nNet flow will range from -{max_departures} to +{max_arrivals}")
    net_flow_size = max_arrivals + max_departures + 1
    print(f"Initial net flow PMF size (before truncation): {net_flow_size}")
    
    # Compute net flow PMF using both methods
    print(f"\n  COMPUTING NET FLOW PMF:")
    p_net, net_offset = predictor._compute_net_flow_pmf(p_arrivals, p_departures)
    p_net_manual, net_offset_manual = manual_compute_net_flow_pmf(
        p_arrivals_manual, p_departures_manual, epsilon
    )
    
    print(f"\n    Predictor result:")
    print(f"      PMF size: {len(p_net)}")
    print(f"      Offset: {net_offset} (index 0 → value {net_offset})")
    print(f"      Support: {net_offset} to {len(p_net) - 1 + net_offset}")
    
    print(f"\n    Manual result:")
    print(f"      PMF size: {len(p_net_manual)}")
    print(f"      Offset: {net_offset_manual} (index 0 → value {net_offset_manual})")
    print(f"      Support: {net_offset_manual} to {len(p_net_manual) - 1 + net_offset_manual}")
    
    print(f"\n  COMPARING NET FLOW CALCULATIONS:")
    if net_offset == net_offset_manual:
        print(f"    ✓ Offsets match: {net_offset}")
    else:
        print(f"    ✗ Offsets differ: predictor={net_offset}, manual={net_offset_manual}")
    
    compare_arrays(p_net, p_net_manual, "Net flow PMF")
    
    # Compare expectations
    exp_net = predictor._expected_value(p_net, net_offset)
    exp_net_manual = manual_expected_value(p_net_manual, net_offset_manual)
    print(f"\n    Predictor expectation: {exp_net:.6f}")
    print(f"    Manual expectation: {exp_net_manual:.6f}")
    if abs(exp_net - exp_net_manual) < 1e-6:
        print(f"    ✓ Expectations match")
    else:
        print(f"    ✗ Expectations differ by {abs(exp_net - exp_net_manual):.2e}")
    
    print_separator("FINAL NET FLOW DISTRIBUTION", "·")
    print_pmf_summary(p_net, "Net Flow PMF (via Predictor)", offset=net_offset)
    
    net_flow_prediction = predictor._create_prediction(check_spec, "net_flow", p_net, net_offset)
    
    # SUMMARY
    print_separator("FINAL SUMMARY", "=")
    
    print(f"\n{check_spec} Prediction Summary:")
    print(f"\nARRIVALS:")
    print(f"  Expected value: {arrivals_prediction.expected_value:.3f}")
    print(f"  Percentiles: {arrivals_prediction.percentiles}")
    
    print(f"\nDEPARTURES:")
    print(f"  Expected value: {departures_prediction.expected_value:.3f}")
    print(f"  Percentiles: {departures_prediction.percentiles}")
    
    print(f"\nNET FLOW:")
    print(f"  Expected value: {net_flow_prediction.expected_value:.3f}")
    print(f"  Percentiles: {net_flow_prediction.percentiles}")
    
    # Verify expectation calculation
    print(f"\nVERIFICATION:")
    expected_net = arrivals_prediction.expected_value - departures_prediction.expected_value
    print(f"  E[Arrivals] - E[Departures] = {expected_net:.3f}")
    print(f"  E[Net Flow] = {net_flow_prediction.expected_value:.3f}")
    diff = abs(expected_net - net_flow_prediction.expected_value)
    print(f"  Difference: {diff:.6f} {'✓' if diff < 0.001 else '✗'}")


def main():
    """Main entry point for the debug script."""
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nERROR: Missing required arguments")
        sys.exit(1)
    
    data_path = sys.argv[1]
    check_spec = sys.argv[2]
    
    # Optional: epsilon parameter
    epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-7
    
    print_separator("LOADING DATA", "=")
    print(f"\nLoading subspecialty data from: {data_path}")
    
    try:
        subspecialty_data = joblib.load(data_path)
    except Exception as e:
        print(f"\nERROR: Failed to load data: {e}")
        print("\nTrying pickle format...")
        try:
            import pickle
            with open(data_path, 'rb') as f:
                subspecialty_data = pickle.load(f)
        except Exception as e2:
            print(f"ERROR: Also failed with pickle: {e2}")
            sys.exit(1)
    
    print(f"Loaded data with {len(subspecialty_data)} subspecialties")
    print(f"Available subspecialties: {list(subspecialty_data.keys())}")
    
    if check_spec not in subspecialty_data:
        print(f"\nERROR: Subspecialty '{check_spec}' not found in data")
        print(f"Available options: {', '.join(sorted(subspecialty_data.keys()))}")
        sys.exit(1)
    
    inputs = subspecialty_data[check_spec]
    
    if not isinstance(inputs, SubspecialtyPredictionInputs):
        print(f"\nERROR: Expected SubspecialtyPredictionInputs, got {type(inputs)}")
        sys.exit(1)
    
    # Part 1: Show the flow inputs
    debug_flow_inputs(inputs, check_spec)
    
    # Part 2: Walk through the convolution steps
    debug_convolution_steps(inputs, check_spec, epsilon=epsilon)
    
    print_separator("DEBUG COMPLETE", "=")
    print()


if __name__ == "__main__":
    main()

