"""
Custom exception classes for model loading and validation.

This module defines specialized exceptions used during model loading

Classes
-------
ModelLoadError
    Raised when a model fails to load due to an unspecified error.

MissingKeysError
    Raised when expected keys are missing from a dictionary of special parameters.
"""


class ModelLoadError(Exception):
    """
    Exception raised when a model fails to load.

    This generic exception can be used to signal a failure during the model
    loading process due to unexpected issues such as file corruption,
    invalid formats, or unsupported configurations.
    """

    pass


class MissingKeysError(ValueError):
    """
    Exception raised when required keys are missing from special_params.

    Parameters
    ----------
    missing_keys : list or set
        The keys that are required but missing from the input dictionary.

    Attributes
    ----------
    missing_keys : list or set
        Stores the missing keys that caused the exception.
    """

    def __init__(self, missing_keys):
        super().__init__(f"special_params is missing required keys: {missing_keys}")
        self.missing_keys = missing_keys
