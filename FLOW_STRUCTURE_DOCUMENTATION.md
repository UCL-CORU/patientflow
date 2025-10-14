# Patient Flow Structure Documentation

## Overview

This document describes the refactored flow-based architecture for subspecialty demand prediction in the patientflow package. The new structure provides a flexible, extensible way to model patient arrivals and departures.

## Key Components

### 1. `FlowInputs` Dataclass

Represents a single source of patient flow (either arriving or departing).

**Attributes:**
- `flow_id` (str): Unique identifier (e.g., "ed_current", "transfers_in", "departures")
- `flow_type` (str): Either "pmf" (probability mass function) or "poisson" (Poisson distribution)
- `distribution` (np.ndarray | float): PMF array or Poisson lambda parameter
- `display_name` (str, optional): Human-readable name for display

**Standard Flow IDs:**

**Inflows (arrivals):**
- `"ed_current"`: Current ED patients who will be admitted (PMF)
- `"ed_yta"`: Yet-to-arrive ED patients (Poisson)
- `"non_ed_yta"`: Non-ED emergency admissions (Poisson)
- `"elective_yta"`: Elective admissions (Poisson)
- `"transfers_in"`: Patients transferring from other subspecialties (PMF)

**Outflows (departures):**
- `"departures"`: Current inpatients who will depart (PMF)
- Future: `"transfers_out"`, `"deaths"`, etc.

### 2. `SubspecialtyPredictionInputs` Dataclass

Organizes all flows for a single subspecialty.

**Attributes:**
- `subspecialty_id` (str): Unique identifier
- `prediction_window` (timedelta): Time window for predictions
- `inflows` (Dict[str, FlowInputs]): Dictionary of arrival flows
- `outflows` (Dict[str, FlowInputs]): Dictionary of departure flows

### 3. `FlowSelection` Configuration

Controls which flows to include in predictions.

**Methods:**
- `FlowSelection.default()`: All flows included
- `FlowSelection.admissions_only()`: Excludes transfers
- `FlowSelection.custom(inflow_keys, outflow_keys)`: Custom selection

### 4. `PredictionBundle` Results

Contains complete prediction results.

**Attributes:**
- `entity_id` (str): Subspecialty identifier
- `entity_type` (str): Type ("subspecialty", "reporting_unit", etc.)
- `arrivals` (DemandPrediction): Total arrivals distribution
- `departures` (DemandPrediction): Total departures distribution
- `net_flow_expected` (float): Expected net change (arrivals - departures)

## Usage Examples

### Example 1: Basic Usage with All Flows

```python
from patientflow.predict.subspecialty import build_subspecialty_data
from patientflow.predict.hierarchy import DemandPredictor, FlowSelection

# Build subspecialty inputs (automatically includes all flows)
subspecialty_data = build_subspecialty_data(
    models=models,
    prediction_time=(7, 0),
    ed_snapshots=ed_snapshots,
    inpatient_snapshots=inpatient_snapshots,
    specialties=["cardiology", "surgery", "medicine"],
    prediction_window=timedelta(hours=8),
    x1=0.0, y1=0.0, x2=8.0, y2=0.9
)

# Make prediction with all flows included
predictor = DemandPredictor()
bundle = predictor.predict_subspecialty(
    subspecialty_id="cardiology",
    inputs=subspecialty_data["cardiology"],
    flow_selection=FlowSelection.default()  # Optional, this is the default
)

# Access results
print(f"Expected arrivals: {bundle.arrivals.expected_value:.1f}")
print(f"Expected departures: {bundle.departures.expected_value:.1f}")
print(f"Expected net flow: {bundle.net_flow_expected:.1f}")
print(f"95th percentile arrivals: {bundle.arrivals.percentiles[95]}")
```

### Example 2: Exclude Transfers (Admissions Only)

```python
# Predict admissions only (no transfers)
bundle = predictor.predict_subspecialty(
    subspecialty_id="cardiology",
    inputs=subspecialty_data["cardiology"],
    flow_selection=FlowSelection.admissions_only()
)

# Now arrivals only include:
# - ed_current, ed_yta, non_ed_yta, elective_yta
# (transfers_in is excluded)
```

### Example 3: Custom Flow Selection

```python
# Only include current ED and yet-to-arrive ED
bundle = predictor.predict_subspecialty(
    subspecialty_id="cardiology",
    inputs=subspecialty_data["cardiology"],
    flow_selection=FlowSelection.custom(
        include_inflows=["ed_current", "ed_yta"],
        include_outflows=["departures"]
    )
)
```

### Example 4: Inspect Flow Structure

```python
# Examine the flow structure
inputs = subspecialty_data["cardiology"]

print(inputs)  # Pretty-printed with __repr__

# Output:
# SubspecialtyPredictionInputs(subspecialty='cardiology')
#   INFLOWS:
#     Admissions from current ED                PMF[0:10]: [0.234, 0.456, ...] (E=2.3 of 15)
#     ED yet-to-arrive admissions               λ = 0.500
#     Non-ED emergency admissions               λ = 0.300
#     Elective admissions                       λ = 0.200
#     Transfers from other subspecialties       PMF[0:10]: [0.890, 0.089, ...] (E=0.2)
#   OUTFLOWS:
#     Inpatient departures                      PMF[0:10]: [0.123, 0.345, ...] (E=5.6 of 12)

# Access individual flows
ed_current_flow = inputs.inflows["ed_current"]
print(f"Flow type: {ed_current_flow.flow_type}")  # "pmf"
print(f"Distribution: {ed_current_flow.distribution}")  # numpy array

ed_yta_flow = inputs.inflows["ed_yta"]
print(f"Flow type: {ed_yta_flow.flow_type}")  # "poisson"
print(f"Lambda: {ed_yta_flow.distribution}")  # float
```

### Example 5: Create Custom Flows (for testing or custom scenarios)

```python
from patientflow.predict.subspecialty import FlowInputs
import numpy as np

# Create custom test data
custom_inputs = SubspecialtyPredictionInputs(
    subspecialty_id="test_subspecialty",
    prediction_window=timedelta(hours=8),
    inflows={
        "ed_current": FlowInputs(
            flow_id="ed_current",
            flow_type="pmf",
            distribution=np.array([0.5, 0.3, 0.2]),
            display_name="Current ED patients"
        ),
        "ed_yta": FlowInputs(
            flow_id="ed_yta",
            flow_type="poisson",
            distribution=2.5,
            display_name="Yet-to-arrive ED"
        ),
    },
    outflows={
        "departures": FlowInputs(
            flow_id="departures",
            flow_type="pmf",
            distribution=np.array([0.2, 0.5, 0.3]),
            display_name="Discharges"
        ),
    }
)
```

## Display Names

All flows have human-readable display names that appear in the `__repr__` output:

| Flow ID | Display Name (default) |
|---------|------------------------|
| `ed_current` | Admissions from current ED |
| `ed_yta` | ED yet-to-arrive admissions |
| `non_ed_yta` | Non-ED emergency admissions |
| `elective_yta` | Elective admissions |
| `transfers_in` | Transfers from other subspecialties |
| `departures` | Inpatient departures |

## Net Flow Calculation

Net flow is computed as the **difference in expected values** (mean arrivals - mean departures).

For more detailed net flow analysis:
```python
# Access full distributions
arrivals_pmf = bundle.arrivals.probabilities
departures_pmf = bundle.departures.probabilities

# Compute percentile-based net flows
p50_net = bundle.arrivals.percentiles[50] - bundle.departures.percentiles[50]
p95_net = bundle.arrivals.percentiles[95] - bundle.departures.percentiles[95]
```

## Future Extensions

The structure is designed to easily accommodate new flow types:

```python
# Future: Add transfer-out flows
outflows={
    "departures": FlowInputs(...),
    "transfers_out": FlowInputs(
        flow_id="transfers_out",
        flow_type="pmf",
        distribution=transfer_out_pmf,
        display_name="Transfers to other subspecialties"
    ),
    "deaths": FlowInputs(
        flow_id="deaths",
        flow_type="poisson",
        distribution=death_rate,
        display_name="In-hospital deaths"
    ),
}
```

## Migration from Old Structure

**Old structure (deprecated):**
```python
inputs.pmf_ed_current_within_window
inputs.lambda_ed_yta_within_window
inputs.pmf_inpatient_departures_within_window
```

**New structure:**
```python
inputs.inflows["ed_current"].distribution
inputs.inflows["ed_yta"].distribution
inputs.outflows["departures"].distribution
```

## Benefits

1. **Extensible**: Easy to add new flow types without changing function signatures
2. **Self-documenting**: Flow types and names are explicit
3. **Flexible**: Users can select which flows to include
4. **Type-safe**: Each flow declares whether it's PMF or Poisson
5. **Queryable**: Can iterate over flows, check existence, etc.

