# Support Multiple Prediction Windows in Admission Predictors

## Summary
Modify `IncomingAdmissionPredictor` and its subclasses to support training and prediction with multiple prediction windows while maintaining backward compatibility.

## Changes Required

### 1. Modify `fit()` method signature
- Accept `prediction_window: Union[timedelta, List[timedelta]]` instead of single `timedelta`
- Normalize single `timedelta` to `[timedelta]` internally
- Set `self._multi_window_mode = True` when list length > 1
- Store `self.prediction_windows = [timedelta, ...]` (always a list)
- For backward compatibility: `self.prediction_window = prediction_windows[0]` if single window

### 2. Update weights data structure
- Single window mode (backward compatible): `weights[filter_key][prediction_time] = {"arrival_rates": [...]}`
- Multi-window mode: `weights[filter_key][prediction_window][prediction_time] = {"arrival_rates": [...]}`
- Loop through all windows in `fit()` to calculate and store parameters for each

### 3. Add `_window_metadata` dictionary
- Store per-window metadata: `{prediction_window: {"NTimes": int, "prediction_window_hours": float, "yta_time_interval_hours": float}}`
- For single-window mode, maintain existing attributes (`self.NTimes`, etc.) for backward compatibility

### 4. Modify `_get_window_and_interval_hours()`
- Add optional parameter: `prediction_window: Optional[timedelta] = None`
- Lookup metadata from `_window_metadata` dictionary
- Default to first window if `prediction_window` is None

### 5. Modify `_iter_prediction_inputs()`
- Add optional parameter: `prediction_window: Optional[timedelta] = None`
- Conditionally access weights based on `_multi_window_mode` flag
- Yield `prediction_window` as additional value: `(filter_key, prediction_time, arrival_rates, prediction_window)`

### 6. Modify `predict()` methods
- Add optional parameter: `prediction_window: Optional[timedelta] = None` to all three predictor classes
- Validate `prediction_window` exists in `self.prediction_windows` if provided
- Default to first window in multi-window mode, use stored window in single-window mode
- Pass `prediction_window` to helper methods

### 7. Update `predict_mean()`
- Add optional `prediction_window` parameter
- Pass to `_iter_prediction_inputs()` and use in calculations

### 8. Validation
- Validate all windows in list are positive and > `yta_time_interval`
- Validate `prediction_window` parameter in `predict()` exists in trained windows
- Raise appropriate errors for invalid inputs

## Backward Compatibility
- Single `timedelta` input to `fit()` works as before
- `predict()` without `prediction_window` parameter works as before
- Existing attributes (`self.prediction_window`, `self.NTimes`, etc.) preserved for single-window mode
- `get_weights()` returns same structure for single-window mode




