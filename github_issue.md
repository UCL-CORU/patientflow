# Improve Documentation and Add Notebook for Hierarchical Prediction Module

## Summary

This issue tracks improvements to the `predict.hierarchy` module documentation and proposes a new demonstration notebook. The module has solid technical documentation but lacks workflow guidance and practical examples for new users.

## Completed Tasks ✅

### 1. Remove Docstrings from Internal Methods

**Status:** ✅ Completed

Removed docstrings from the following internal methods to prevent them from appearing in user API documentation:

- `Hierarchy._validate_levels()`
- `Hierarchy._find_entity_type_by_name()`
- `Hierarchy._get_original_name()`
- `Hierarchy._get_prefixed_id()`
- `HierarchicalPredictor._predict_subtree()`

**Rationale:** Internal methods (prefixed with `_`) should not have docstrings as they appear in user-facing API documentation. These methods are implementation details and should remain undocumented for users.

## Documentation Improvements Needed

### 2. Add Module-Level Usage Guide

**Priority:** High

**Current State:** The `__init__.py` module docstring is minimal and doesn't explain the workflow.

**Proposed Addition:** Expand the module docstring in `src/patientflow/predict/hierarchy/__init__.py` to include:

1. **Usage Section:** Step-by-step workflow showing:
   - Creating/loading hierarchy structure
   - Populating hierarchy with organizational data
   - Preparing prediction inputs using `build_subspecialty_data()`
   - Creating predictor and running predictions
   - Accessing and interpreting results

2. **Components Section:** Explain the relationship between:
   - `Hierarchy`: Defines organizational structure
   - `DemandPredictor`: Core calculation engine
   - `HierarchicalPredictor`: Orchestrates predictions across levels
   - `FlowSelection`: Configures which flows to include

**Example Structure:**
```python
"""Hierarchical demand prediction module.

This module provides tools for predicting hospital bed demand across different
organisational levels (subspecialty, division, hospital, etc.) using a
hierarchical structure.

Usage
-----
The typical workflow is:

1. Create or load a hierarchy structure:
   >>> hierarchy = Hierarchy.create_default_hospital()
   >>> # or
   >>> hierarchy = Hierarchy.from_yaml("config.yaml")

2. Populate the hierarchy with organizational data:
   >>> populate_hierarchy_from_dataframe(
   ...     hierarchy, df, column_mapping={"subspecialty": "subspecialty", ...}, 
   ...     top_level_id="Hospital"
   ... )

3. Prepare prediction inputs (using functions from predict.subspecialty):
   >>> from patientflow.predict.subspecialty import build_subspecialty_data
   >>> bottom_level_data = build_subspecialty_data(...)

4. Create predictor and run predictions:
   >>> predictor = create_hierarchical_predictor(...)
   >>> results = predictor.predict_all_levels(bottom_level_data)

5. Access results:
   >>> bundle = results["Cardiology"]
   >>> print(bundle.arrivals.expected_value)

Components
----------
- **Hierarchy**: Defines the organizational structure (levels, relationships)
- **DemandPredictor**: Core calculation engine for probability distributions
- **HierarchicalPredictor**: Orchestrates predictions across all hierarchy levels
- **FlowSelection**: Configures which patient flows to include in predictions
"""
```

### 3. Add Examples to Key Functions

**Priority:** High

**Functions Needing Examples:**

1. **`create_hierarchical_predictor()`** (`orchestrator.py`)
   - Example DataFrame structure
   - Example column mapping format
   - Example YAML config format
   - Show different initialization paths

2. **`populate_hierarchy_from_dataframe()`** (`structure.py`)
   - Example DataFrame format
   - Explanation of the two-pass process (entities first, then relationships)
   - What happens with missing relationships

3. **`Hierarchy.from_yaml()`** (`structure.py`)
   - Example YAML structure
   - Required fields and format

4. **`predict_all_levels()`** (`orchestrator.py`)
   - Complete end-to-end example
   - Data format requirements
   - Relationship between subspecialty IDs and hierarchy entities
   - What the returned dictionary contains

### 4. Expand `predict_all_levels()` Documentation

**Priority:** Medium

**Current Issues:**
- Unclear data format requirements
- Relationship between subspecialty IDs and hierarchy entities not explained
- What happens if a subspecialty in data isn't in hierarchy

**Proposed Additions:**
- Clear specification of `bottom_level_data` format
- Explanation that subspecialty IDs must match hierarchy entity names
- Error handling documentation
- Example of returned dictionary structure

### 5. Add Architecture/Components Section

**Priority:** Medium

**Proposed Addition:** Add a section explaining:
- When to use each component
- The relationship between components
- The prediction flow (3-phase algorithm)
- When to use `create_hierarchical_predictor()` vs manual construction
- When to use `DemandPredictor` directly vs `HierarchicalPredictor`

### 6. Document Prefixed ID System

**Priority:** Low

**Current Issue:** The prefixed ID system (`"subspecialty:Cardiology"`) is an implementation detail that leaks into the API.

**Proposed Addition:** Add a note in `Hierarchy` class docstring explaining:
- When users see prefixed IDs vs original names
- Why some methods return original names while internal storage uses prefixed IDs
- How to handle entity name collisions

### 7. Expand `populate_hierarchy_from_dataframe()` Documentation

**Priority:** Low

**Current Issue:** Mentions "Pass 1" and "Pass 2" but doesn't explain the overall process clearly.

**Proposed Additions:**
- Clearer explanation of the two-pass process
- What happens if relationships are missing
- How to handle top-level entities

## New Notebook Proposal

### 8. Create `5a_Predict_demand_with_hierarchical_bundles.ipynb`

**Priority:** High

**Purpose:** Demonstrate hierarchical demand prediction from start to finish using only ED inflows (matching available public data).

**Why This Notebook is Needed:**
- No existing notebook demonstrates the complete hierarchical workflow
- `explore_prediction_bundles.ipynb` only explores pre-created predictions (and will be deleted)
- Notebook 4b uses the older `create_predictions()` function, not hierarchical prediction
- Bridges the gap between `build_subspecialty_data()` and hierarchical prediction

**Proposed Structure:**

1. **Introduction**
   - Relationship to notebook 4b
   - What hierarchical prediction adds
   - What flows are available (ED only)

2. **Set Up (Reuse from 4b)**
   - Load data
   - Train models (same as 4b)
   - Select prediction moment

3. **Create Hierarchy Structure**
   - Simple 2-level hierarchy: subspecialty → hospital
   - Four specialties: `'medical'`, `'surgical'`, `'haem/onc'`, `'paediatric'`
   - Single hospital entity
   - Show manual creation approach

4. **Prepare Subspecialty Data**
   - Manually create `SubspecialtyPredictionInputs` with only ED flows:
     - `ed_current` (PMF from current ED patients)
     - `ed_yta` (Poisson from yet-to-arrive model)
     - Empty departures (no inpatient data)
     - Empty transfers (not available)
   - Show how to set unavailable flows to zero/empty distributions

5. **Create Hierarchical Predictor**
   - Use `create_hierarchical_predictor()` with simple hierarchy
   - Configure `FlowSelection` for ED-only flows:
     ```python
     flow_selection = FlowSelection.custom(
         include_ed_current=True,
         include_ed_yta=True,
         include_non_ed_yta=False,
         include_elective_yta=False,
         include_transfers_in=False,
         include_departures=False,
         cohort="emergency"
     )
     ```

6. **Run Predictions**
   - `predict_all_levels()` to get bundles
   - Show results at subspecialty level
   - Show aggregated results at hospital level

7. **Explore Prediction Bundles**
   - Access arrivals predictions
   - Understand PMF structure
   - Compare subspecialty vs. hospital-level predictions
   - Show percentiles and expected values

8. **Visualization**
   - Plot PMFs at different levels
   - Compare aggregated vs. individual predictions

**Hierarchy Configuration Details:**

```python
# Simple 2-level hierarchy
Level 0 (bottom): subspecialty
  - 'medical'
  - 'surgical'
  - 'haem/onc'
  - 'paediatric'

Level 1 (top): hospital
  - 'hospital' (single entity)
```

**Implementation Approach:**

```python
from patientflow.predict.hierarchy import Hierarchy, EntityType, HierarchyLevel

# Create entity types
subspecialty_type = EntityType("subspecialty")
hospital_type = EntityType("hospital")

# Create hierarchy levels
levels = [
    HierarchyLevel(subspecialty_type, hospital_type, level_order=0),  # Bottom level
    HierarchyLevel(hospital_type, None, level_order=1),  # Top level
]

hierarchy = Hierarchy(levels)

# Add entities
specialties = ['medical', 'surgical', 'haem/onc', 'paediatric']
for spec in specialties:
    hierarchy.add_entity(spec, subspecialty_type)

hierarchy.add_entity("hospital", hospital_type)

# Establish relationships
for spec in specialties:
    spec_prefixed = f"subspecialty:{spec}"
    hospital_prefixed = "hospital:hospital"
    hierarchy.relationships[spec_prefixed] = hospital_prefixed
```

**Data Preparation:**

Since `build_subspecialty_data()` requires all 7 models (including inpatient and transfer models), manually create `SubspecialtyPredictionInputs`:

```python
from patientflow.predict.subspecialty import SubspecialtyPredictionInputs, FlowInputs
import numpy as np

subspecialty_data = {}
for spec in specialties:
    inflows = {
        'ed_current': FlowInputs(flow_id='ed_current', flow_type='pmf', 
                                distribution=ed_current_pmf, ...),
        'ed_yta': FlowInputs(flow_id='ed_yta', flow_type='poisson', 
                            distribution=ed_yta_lambda, ...),
        # Set unavailable flows to zero/empty
        'non_ed_yta': FlowInputs(..., distribution=0.0),
        'elective_yta': FlowInputs(..., distribution=0.0),
        'elective_transfers': FlowInputs(..., distribution=np.array([1.0])),
        'emergency_transfers': FlowInputs(..., distribution=np.array([1.0])),
    }
    outflows = {
        'elective_departures': FlowInputs(..., distribution=np.array([1.0])),
        'emergency_departures': FlowInputs(..., distribution=np.array([1.0])),
    }
    subspecialty_data[spec] = SubspecialtyPredictionInputs(
        subspecialty_id=spec,
        prediction_window=prediction_window,
        inflows=inflows,
        outflows=outflows
    )
```

**Placement:** After `4d_Predict_emergency_demand_with_special_categories.ipynb` in the notebook sequence.

## Documentation Assessment Summary

**Current State:**
- ✅ Strong technical documentation at method/class level
- ✅ Good examples in some methods (e.g., `predict_subspecialty()`)
- ❌ Missing high-level workflow guidance
- ❌ Missing examples in key functions
- ❌ Unclear component relationships
- ❌ Missing data format specifications

**For New Users:**
- Can understand individual components ✅
- Will struggle to piece together complete workflow ❌
- Needs more examples and clearer entry points ❌

**Overall Documentation Quality:** 7/10
- Strong technical details
- Good method-level docs
- Missing workflow guidance and examples

## Implementation Notes

### For Documentation Improvements:
- All changes should be made to existing docstrings
- Maintain consistency with existing docstring format (NumPy style)
- Add examples using doctest format where appropriate
- Ensure all code examples are tested/valid

### For Notebook:
- Use data from `data-public` folder
- Reuse models and setup from notebook 4b where possible
- Focus on clarity and educational value
- Include both code and explanatory markdown
- Show common pitfalls and how to avoid them

## Related Files

**Documentation Files:**
- `src/patientflow/predict/hierarchy/__init__.py`
- `src/patientflow/predict/hierarchy/orchestrator.py`
- `src/patientflow/predict/hierarchy/structure.py`
- `src/patientflow/predict/hierarchy/calculation.py`
- `src/patientflow/predict/hierarchy/types.py`

**Notebook Files:**
- `notebooks/4b_Predict_emergency_demand.ipynb` (reference)
- `notebooks/explore_prediction_bundles.ipynb` (temporary, will be deleted)
- `notebooks/5a_Predict_demand_with_hierarchical_bundles.ipynb` (to be created)

**Data Files:**
- `data-public/ed_visits.csv`
- `data-public/inpatient_arrivals.csv`
- `data-dictionaries/ed_visits_data_dictionary.md`

## Acceptance Criteria

- [ ] Module-level usage guide added to `__init__.py`
- [ ] Examples added to `create_hierarchical_predictor()`
- [ ] Examples added to `populate_hierarchy_from_dataframe()`
- [ ] Examples added to `Hierarchy.from_yaml()`
- [ ] `predict_all_levels()` documentation expanded with data format details
- [ ] Architecture/components section added
- [ ] Prefixed ID system documented
- [ ] `populate_hierarchy_from_dataframe()` documentation expanded
- [ ] New notebook `5a_Predict_demand_with_hierarchical_bundles.ipynb` created
- [ ] Notebook demonstrates complete workflow from hierarchy setup to results
- [ ] Notebook works with public data (ED flows only)
- [ ] Notebook includes clear explanations and examples

## Notes

- The `explore_prediction_bundles.ipynb` notebook is temporary and will be deleted
- The new notebook should replace its functionality while showing how to create predictions from scratch
- All documentation improvements should maintain backward compatibility
- Code examples in documentation should be tested to ensure they work


