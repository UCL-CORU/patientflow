# Subspecialty Convolution Debug Guide

This guide explains how to debug the convolution logic for subspecialty predictions.

## Quick Start

### Method 1: Using the standalone script (for remote environments)

```bash
# Run the debug script on saved data
python debug_subspecialty_convolution.py <path_to_saved_data> "<subspecialty_name>"

# Example:
python debug_subspecialty_convolution.py subspecialty_data.joblib "Cardiology"

# With custom epsilon:
python debug_subspecialty_convolution.py subspecialty_data.joblib "Cardiology" 1e-6
```

### Method 2: Using the inline debug function (for interactive sessions)

```python
from debug_subspecialty_inline import debug_subspecialty_convolution

# Assuming you have subspecialty_data dict
debug_subspecialty_convolution(subspecialty_data, "Cardiology")
```

## What the Debug Script Shows

The debug script walks through the complete convolution pipeline for a subspecialty and **compares the DemandPredictor's internal methods with manual calculations** to verify correctness:

### 1. **Flow Inputs Summary**
   - Lists all inflow sources (ED current, ED YTA, non-ED YTA, elective, transfers)
   - Lists all outflow sources (departures, etc.)
   - Shows distribution type (PMF or Poisson) and parameters for each
   - Displays expected values

### 2. **Arrivals Convolution (Step-by-Step with Verification)**
   For each inflow:
   - **Poisson Generation** (if applicable):
     - Compares `predictor._calculate_poisson_max()` with manual calculation
     - Compares `predictor._poisson_pmf()` with manual Poisson PMF
     - Shows ✓ or ✗ for agreement
   - **Convolution**:
     - Compares `predictor._convolve()` with manual `np.convolve()`
     - Verifies expectation addition: E[A+B] = E[A] + E[B]
     - Shows array comparison results
   - **Truncation**:
     - Compares `predictor._truncate()` with manual truncation
     - Shows how many values were removed
     - Verifies expectation is preserved
   - **Running verification**: Each step confirms predictor matches manual calculation

### 3. **Departures Convolution (Step-by-Step with Verification)**
   For each outflow:
   - Same detailed verification as arrivals
   - Independent manual calculation tracks alongside predictor

### 4. **Net Flow Calculation (with Verification)**
   - Compares `predictor._compute_net_flow_pmf()` with manual nested-loop computation
   - Verifies offset calculation for negative support
   - Compares truncation from both ends
   - Shows ✓ or ✗ for each comparison
   - Verifies that E[Net Flow] = E[Arrivals] - E[Departures]

### 5. **Final Summary**
   - Expected values for arrivals, departures, and net flow
   - Percentiles (P50, P75, P90, P95, P99) for each distribution
   - Final verification that all expectations match

## Understanding the Output

### Verification Symbols

The script uses these symbols to indicate agreement between methods:
- **✓** Green checkmark: Predictor and manual calculations match (within tolerance)
- **✗** Red X: Calculations differ (investigation needed)
- **⚠️** Warning: Different array lengths or other structural differences

### Comparison Output Format

When comparing arrays:
```
  ✓ Convolution result: Arrays match (within tolerance 1e-10)
```
or
```
  ✗ Net flow PMF: Arrays differ (max difference: 3.45e-08)
    Differences at indices: [0, 1, 2, 5, 8, 9, 10]...
```

### PMF Display Format

When the script shows a PMF, you'll see:
```
  Length: 15 values (support: 0 to 14)
  Expectation (mean): 3.456
  Mode: 3 (probability: 0.2543)
  Percentiles: P50=3, P75=5, P95=8
  Sum of probabilities: 1.000000
  PMF[0:9]: [0.0045, 0.0234, 0.0567, 0.0893, 0.1123, 0.1234, 0.1098, 0.0876, 0.0654]
```

- **Length**: How many values in the PMF array
- **Support**: Range of values the random variable can take
- **Expectation**: The mean (expected value)
- **Mode**: Most likely value
- **Percentiles**: Values at key percentiles
- **PMF**: The probability mass function values (truncated for display)

### Convolution Step Format

For each convolution step:
```
Step 2: Adding ED yet-to-arrive admissions
  Generated Poisson PMF with λ=2.500, max_k=12
  
  Flow PMF (ed_yta):
    [shows PMF details]
  
  Before convolution:
    Running total length: 10
    Running total expectation: 1.234
  
  After convolution:
    New length: 22
    New expectation: 3.734
  
  After truncation (epsilon=1e-07):
    Truncated from 22 to 18 values
    Expectation: 3.734
```

This shows:
1. The flow being added
2. Its distribution
3. The state before adding it
4. The state after convolution (sum of independent RVs)
5. The state after truncation (removing negligible tail probabilities)

### Net Flow Offset

Net flow can be negative, so the PMF uses an offset:
```
  Offset: -5 (index 0 corresponds to value -5)
  Support: -5 to 12
```

This means:
- `p_net[0]` = P(Net Flow = -5)
- `p_net[1]` = P(Net Flow = -4)
- ...
- `p_net[5]` = P(Net Flow = 0)
- ...
- `p_net[17]` = P(Net Flow = 12)

## Saving Data for the Script

To use the standalone script, save your subspecialty_data:

```python
import joblib

# After running build_subspecialty_data
subspecialty_data = build_subspecialty_data(
    models=models,
    prediction_time=prediction_time,
    ed_snapshots=ed_snapshots,
    inpatient_snapshots=inpatient_snapshots,
    specialties=specialties,
    prediction_window=prediction_window,
    x1=x1, y1=y1, x2=x2, y2=y2,
)

# Save it
joblib.dump(subspecialty_data, 'subspecialty_data.joblib')
```

## Interpreting Verification Results

### All checks pass (✓)
If all comparisons show ✓, the predictor is working correctly:
- Poisson generation matches scipy.stats
- Convolution matches np.convolve
- Truncation preserves probability mass
- Net flow calculation is mathematically correct
- Expectations are additive as expected

This confirms the implementation is sound.

### Some checks fail (✗)
If you see ✗ symbols, investigate:

1. **Small numerical differences (< 1e-8)**: Usually harmless floating-point rounding
2. **Larger differences**: Could indicate:
   - Bug in predictor implementation
   - Bug in manual calculation
   - Numerical instability (try smaller epsilon)
   - Different handling of edge cases

The script shows where differences occur and their magnitude to help pinpoint issues.

### Expected property violations
The script checks mathematical properties:
- **E[A+B] = E[A] + E[B]**: Convolution should preserve expectation addition
- **E[Net] = E[Arr] - E[Dep]**: Net flow expectation should equal difference
- **Sum(PMF) ≈ 1.0**: All PMFs should sum to 1

Violations suggest numerical issues or logic errors.

## Common Debugging Scenarios

### 1. Checking if flows are correctly computed

Look at the "INFLOWS" and "OUTFLOWS" sections to verify:
- ED current PMF matches your snapshot aggregation
- YTA Poisson rates are reasonable
- Transfer PMF has expected shape
- All flows are present

### 2. Understanding why net flow doesn't match expectations

Compare:
- Individual flow expectations (shown in flow summaries)
- Sum of inflow expectations
- Sum of outflow expectations
- Final net flow expectation

The convolution process is exact, so any discrepancy suggests an issue with the input flows.

### 3. Investigating truncation effects

Check the "After truncation" steps:
- How much was truncated?
- Did expectation change significantly?
- Is epsilon too large?

If truncation is removing too much mass, consider using a smaller epsilon (e.g., 1e-9).

### 4. Verifying Poisson generation

For Poisson flows, check:
- The lambda parameter
- The max_k chosen (should capture ~99.99999% of mass)
- The generated PMF expectation (should equal lambda)

## Example Output Snippet

Here's what a typical verification section looks like:

```
Step 2: Adding ED yet-to-arrive admissions
··············································································

  POISSON GENERATION (λ=2.345):
    Predictor max_k: 12
    Manual max_k: 12
    ✓ max_k values match

  Comparing Poisson PMF generation:
  ✓ Poisson PMF: Arrays match (within tolerance 1e-10)

  Flow PMF (ed_yta):
    Length: 13 values (support: 0 to 12)
    Expectation (mean): 2.345
    ...

  Before convolution:
    Running total length: 8
    Predictor expectation: 1.234000
    Manual expectation: 1.234000

  CONVOLUTION:
    Predictor result length: 20
    Manual result length: 20
  ✓ Convolution result: Arrays match (within tolerance 1e-10)
    Predictor expectation: 3.579000
    Manual expectation: 3.579000
    Expected sum (E[before] + E[flow]): 3.579000
    ✓ Convolution preserves expectation correctly

  TRUNCATION (epsilon=1e-07):
    Predictor: truncated from 20 to 18 values
    Manual: truncated to 18 values
  ✓ Truncated result: Arrays match (within tolerance 1e-10)
    Predictor expectation after truncation: 3.579000
    Manual expectation after truncation: 3.579000
```

This shows:
- Both methods generate identical Poisson PMFs
- Convolution results match exactly
- Expectations add correctly (1.234 + 2.345 = 3.579)
- Truncation is consistent
- All verification checks pass (✓)

## Files

- `debug_subspecialty_convolution.py` - Standalone script for saved data with verification
- `DEBUG_CONVOLUTION_GUIDE.md` - This guide

## Tips

1. **Start with one subspecialty**: Debug one at a time to understand the pattern
2. **Check expectations first**: They should match your manual calculations
3. **Look for zero flows**: Missing flows will appear as λ=0 or PMF=[1.0, 0.0]
4. **Compare to simple calculations**: For small numbers, hand-calculate to verify
5. **Use smaller epsilon for precision**: If you need more accuracy, use 1e-9 or 1e-10

## Example Output Interpretation

If you see:
```
Expected arrivals: 5.234
Expected departures: 3.123
Expected net flow: 2.111
```

But your beds are decreasing, check:
- Are departures being counted correctly (not reversed)?
- Are transfers being double-counted?
- Is the sign convention correct?

The script shows the full computation, so you can trace exactly where the numbers come from.

