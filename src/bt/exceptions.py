"""Custom exception hierarchy for bt-flow.

All public exceptions are re-exported from the top-level ``bt`` namespace so
callers can catch them without reaching into internal sub-modules:

    from bt import ModelLoadError
"""


class BTFlowError(Exception):
    """Base exception for all bt-flow errors.

    Catch this if you want a single handler for any library error.
    """


class ModelLoadError(BTFlowError):
    """Raised when a model artifact cannot be loaded from disk.

    Args:
        path: The filesystem path that was attempted.
        reason: A human-readable explanation of why loading failed.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load model from '{path}': {reason}")


class UnsupportedModelError(BTFlowError):
    """Raised when the provided object is not a supported model type.

    bt-flow v0.1.0 exclusively supports fitted scikit-learn estimators.
    """


class SchemaInferenceError(BTFlowError):
    """Raised when the input schema cannot be inferred from the model.

    This typically means the model was not fitted or does not expose
    ``n_features_in_``.
    """


class PredictionError(BTFlowError):
    """Raised when the model's ``predict`` call raises an unexpected error."""
