# Emergency Demand Predictors

This module contains custom predictors used in predicting bed demand. They are built to be compatible with scikit-learn pipelines and follow the estimator interface pattern.

## Predictors

### WeightedPoissonPredictor

Estimates the number of hospital admissions within a specified prediction window using historical admission data. It applies Poisson and binomial distributions to forecast future admissions, excluding already arrived patients.

**Key Features:**

- Time-varying arrival rate modeling
- Prediction based on time of day
- Support for different hospital contexts through configurable filters
- Aspirational approach to probability of admission within a prediction window

### SequenceToOutcomePredictor

Models and predicts the probability distribution of sequences in categorical data. This predictor builds a model that maps input sequences to specific outcome categories, making it useful for predicting patient pathways.

**Key Features:**

- Sequence-based prediction for categorical outcomes
- Support for grouping variables to capture complex patterns
- Special category filtering capabilities
- Probability distribution outputs for all possible outcomes

## Integration

Both predictors implement scikit-learn's `BaseEstimator` and `TransformerMixin` interfaces, allowing them to be:

- Incorporated into scikit-learn pipelines
- Easily combined with other preprocessing steps
- Configured with standard parameter optimization techniques

## Dependencies

- pandas: For data manipulation and analysis
- numpy: For numerical operations
- scikit-learn: For model building and pipeline construction
- scipy: For statistical distributions
- datetime: For time-based operations

## Notes on Implementation

- Both predictors include error handling and logging capabilities
- Time-based functionality accounts for different prediction times throughout the day
- Configurable parameters allow for customisation to different hospital environments
- Missing value handling is included

## Performance Considerations

For large datasets or time-sensitive predictions:

- The WeightedPoissonPredictor computes time-varying arrival rates which can be computationally intensive
- Consider precomputing time-varying arrival rates for common scenarios
- The SequenceToOutcomePredictor's performance depends on the number of unique sequences in the training data
