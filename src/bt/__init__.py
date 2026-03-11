"""bt-flow — One-line scikit-learn model deployment.

The ``bt`` package is the top-level import namespace for the ``bt-flow``
PyPI distribution. Everything a user needs for day-to-day usage is exported
directly from this namespace::

    from bt import APIGenerator
    from bt import ModelLoadError, UnsupportedModelError

Version follows `Semantic Versioning <https://semver.org/>`_.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from bt.core import APIGenerator
from bt.exceptions import (
    BTFlowError,
    ModelLoadError,
    PredictionError,
    SchemaInferenceError,
    UnsupportedModelError,
)
from bt.schemas import HealthResponse, PredictionResponse

try:
    __version__: str = _pkg_version("bt-flow")
except PackageNotFoundError:  # pragma: no cover — only outside an installed env
    __version__ = "0.0.0"
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
