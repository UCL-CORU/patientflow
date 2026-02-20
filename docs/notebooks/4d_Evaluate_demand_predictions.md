# 4d. Evaluate demand predictions

In the 3x_ notebooks, I showed how to evaluate individual model components in isolation: group snapshot predictions (3b), bed demand by hospital service (3d), and demand from patients yet to arrive (3f). Each of those notebooks contained bespoke evaluation code for one component at a time.

In this notebook, I show how the production data structures introduced in notebooks 4a–4c make it straightforward to evaluate all components of a configured pipeline in a uniform, systematic way — and to slice the evaluation by flow type, cohort, or service. No new evaluation methods are introduced; the same EPUDD plots, delta plots, MAE/MPE, and arrival rate comparisons are used, but they are driven by iterating over the production data structures rather than by writing ad hoc code for each component.

## Approach

### 1. Build `ServicePredictionInputs` across the test set

For each snapshot date and prediction time in the test set, I construct the production-format inputs (as in notebook 4c), collecting the predicted distributions and corresponding observed values. Each `ServicePredictionInputs` object bundles all the flow components for a single service: the `inflows` dictionary contains entries like `ed_current`, `ed_yta`, `non_ed_yta`, `elective_yta`, and so on; the `outflows` dictionary contains departure flows.

### 2. Iterate over flows systematically

It is now possible to loop through `inputs.inflows.items()` for each service. Each `FlowInputs` object has a `flow_type` attribute (“pmf” or “poisson”) that determines which evaluation approach to use:

* For PMF flows (like `ed_current`): evaluate using the EPUDD and delta plot methods from notebooks 3b and 3d
* For Poisson flows (like `ed_yta`): evaluate using the arrival rate comparison methods from notebook 3f

The evaluation method can be selected by `flow_type`, giving a uniform loop over all components.

Note, however, that if the emergency flows are aspirational (expressing demand as if ED 4-hour targets are met) they cannot be directly evaluated against observed numbers of admissions without careful thought, as we've discussed in notebook 3f. 

### 3. Use `FlowSelection` to evaluate subsets

The `FlowSelection` class from notebook 4a would allow the same evaluation loop to produce different views of performance depending on which flows are selected. For example:

* `FlowSelection.emergency_only()` — evaluate only emergency flows
* `FlowSelection.incoming_only()` — evaluate only inflows, excluding departures
* `FlowSelection.default()` — evaluate everything

This demonstrates how the production configuration can drive what gets evaluated, without changing the evaluation code itself. (But again, noting the point about aspirational flows.)

### 4. Evaluate at service level

Finally, I show evaluation of the combined `PredictionBundle` (arrivals, departures, net flow) for each service. This is something the 3x_ notebooks never do, because they don't have the concept of combining flows. It gives an overall picture of how the assembled pipeline performs for each service.

Code will follow to illustrate some of these points.


