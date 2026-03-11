"""bt-flow — One-line scikit-learn model deployment.

The ``bt`` package is the top-level import namespace for the ``bt-flow``
PyPI distribution. Everything a user needs for day-to-day usage is exported
directly from this namespace::

    from bt import APIGenerator
    from bt import ModelLoadError, UnsupportedModelError

Version follows `Semantic Versioning <https://semver.org/>`_.
"""

from bt.core import APIGenerator
from bt.exceptions import (
    BTFlowError,
    ModelLoadError,
    PredictionError,
    SchemaInferenceError,
    UnsupportedModelError,
)
from bt.schemas import HealthResponse, PredictionResponse

__version__ = "0.1.0"
__all__ = [
    # Core
    "APIGenerator",
    # Exceptions
    "BTFlowError",
    # Response schemas (useful for type-annotating clients)
    "HealthResponse",
    "ModelLoadError",
    "PredictionError",
    "PredictionResponse",
    "SchemaInferenceError",
    "UnsupportedModelError",
]
