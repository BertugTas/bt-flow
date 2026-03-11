"""Core APIGenerator class for bt-flow.

This module is the heart of the library. It exposes a single, ergonomic class
that handles model loading, schema inference, FastAPI app construction, and
server lifecycle management.

Typical usage::

    from bt import APIGenerator

    gen = APIGenerator("model.pkl")
    gen.run()                          # blocking — starts uvicorn
    app = gen.app                      # or grab the ASGI app for Gunicorn / etc.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from bt.exceptions import (
    ModelLoadError,
    PredictionError,
    SchemaInferenceError,
    UnsupportedModelError,
)
from bt.schemas import (
    HealthResponse,
    PredictionResponse,
    build_named_input_schema,
    build_positional_input_schema,
    numpy_scalar_to_python,
)

if TYPE_CHECKING:
    pass  # keep TYPE_CHECKING block for future annotations


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_SUFFIXES = frozenset({".pkl", ".joblib", ".pickle"})
_SKLEARN_MODULE_PREFIX = "sklearn"


# ---------------------------------------------------------------------------
# APIGenerator
# ---------------------------------------------------------------------------


class APIGenerator:
    """Wrap a scikit-learn estimator and serve it as a production FastAPI app.

    ``APIGenerator`` performs three jobs automatically:

    1. **Model loading** — accepts a filesystem path (``.pkl`` / ``.joblib``)
       or a pre-loaded sklearn estimator object.
    2. **Schema inference** — inspects ``model.feature_names_in_`` (if present)
       or ``model.n_features_in_`` to generate a validated Pydantic request body.
    3. **App construction** — wires up ``/health`` and ``/predict`` routes on a
       ``FastAPI`` instance that you can hand to any ASGI server.

    Args:
        model_source: Either a path-like object / string pointing to a
            serialised sklearn model (``.pkl`` or ``.joblib``), or a
            *fitted* sklearn estimator instance.
        title: Title shown in the auto-generated OpenAPI docs.
        description: Description shown in the OpenAPI docs.
        version: API version string injected into OpenAPI metadata.
        feature_names: Optional explicit list of feature names. Overrides
            names inferred from ``model.feature_names_in_``. Must match
            the length of ``model.n_features_in_``.
        docs_url: Path for Swagger UI. Set to ``None`` to disable.
        redoc_url: Path for ReDoc UI. Set to ``None`` to disable.

    Raises:
        ModelLoadError: If a path is provided but the file cannot be read.
        UnsupportedModelError: If the object is not a fitted sklearn estimator.
        SchemaInferenceError: If feature count cannot be determined.
        ValueError: If ``feature_names`` length mismatches ``n_features_in_``.

    Example::

        # From a saved model file
        api = APIGenerator("iris_classifier.pkl", title="Iris API")
        api.run(port=8080)

        # From a live object
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression().fit(X_train, y_train)
        api = APIGenerator(clf)
        app = api.app  # plug into Gunicorn / any ASGI runner
    """

    def __init__(
        self,
        model_source: str | Path | Any,
        *,
        title: str = "bt-flow Model API",
        description: str = (
            "Auto-generated FastAPI REST API powered by **bt-flow**. "
            "Send POST requests to `/predict` to run model inference."
        ),
        version: str = "0.1.0",
        feature_names: Sequence[str] | None = None,
        docs_url: str | None = "/docs",
        redoc_url: str | None = "/redoc",
    ) -> None:
        self._start_time: float = time.time()
        self._model_source_repr: str = (
            str(model_source)
            if isinstance(model_source, (str, Path))
            else type(model_source).__name__
        )

        # --- Load & validate ---
        self._model: Any = self._load_model(model_source)
        self._validate_model(self._model)

        # --- Feature metadata ---
        self._feature_names: list[str] | None = None
        self._n_features: int = 0
        self._input_schema: type[BaseModel] = self._infer_schema(feature_names)

        # --- FastAPI app ---
        self._app: FastAPI = FastAPI(
            title=title,
            description=description,
            version=version,
            docs_url=docs_url,
            redoc_url=redoc_url,
        )
        self._register_routes(self._app)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def app(self) -> FastAPI:
        """The underlying ``FastAPI`` application instance.

        Use this to integrate with an external ASGI runner:

        .. code-block:: bash

            gunicorn -k uvicorn.workers.UvicornWorker mymodule:api.app
        """
        return self._app

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "info",
        **uvicorn_kwargs: Any,
    ) -> None:
        """Start a ``uvicorn`` server and block until interrupted.

        This is a convenience wrapper. For production deployments prefer
        passing ``self.app`` to Gunicorn or another process manager.

        Args:
            host: Network interface to bind to. Defaults to ``"0.0.0.0"``.
            port: TCP port. Defaults to ``8000``.
            log_level: Uvicorn log level (``"debug"``, ``"info"``, etc.).
            **uvicorn_kwargs: Any additional keyword arguments forwarded
                verbatim to ``uvicorn.run()``.
        """
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "uvicorn is required to call APIGenerator.run(). "
                "Install it with: pip install 'bt-flow[uvicorn]'"
            ) from exc

        uvicorn.run(
            self._app,
            host=host,
            port=port,
            log_level=log_level,
            **uvicorn_kwargs,
        )

    # ASGI callable — lets you use the generator directly as an ASGI app.
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Delegate ASGI calls to the internal FastAPI application.

        This lets you pass an ``APIGenerator`` instance directly to any
        ASGI server without extracting ``.app`` first::

            import uvicorn
            uvicorn.run(api_generator_instance, port=8000)
        """
        await self._app(scope, receive, send)

    def __repr__(self) -> str:
        return (
            f"APIGenerator("
            f"model={type(self._model).__name__}, "
            f"n_features={self._n_features}, "
            f"named={'yes' if self._feature_names else 'no'}"
            f")"
        )

    # ------------------------------------------------------------------
    # Public — model metadata
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Number of input features the model expects."""
        return self._n_features

    @property
    def feature_names(self) -> list[str] | None:
        """Ordered feature names, or ``None`` if the model was trained on a numpy array."""
        return self._feature_names

    @property
    def model_type(self) -> str:
        """Class name of the underlying sklearn estimator (e.g. ``"LogisticRegression"``)."""
        return type(self._model).__name__

    # ------------------------------------------------------------------
    # Private — model loading
    # ------------------------------------------------------------------

    def _load_model(self, source: str | Path | Any) -> Any:
        """Load a model from a path or return the object as-is.

        Args:
            source: Filesystem path (``.pkl`` / ``.joblib``) or a live object.

        Returns:
            The deserialised (or passed-through) sklearn estimator.

        Raises:
            ModelLoadError: If the file is missing, unreadable, or has an
                unsupported extension.
        """
        if not isinstance(source, (str, Path)):
            return source  # caller already has a model object

        path = Path(source)

        if not path.exists():
            raise ModelLoadError(str(path), "file does not exist")

        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ModelLoadError(
                str(path),
                f"unsupported file extension '{path.suffix}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED_SUFFIXES))}",
            )

        try:
            return joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(str(path), str(exc)) from exc

    # ------------------------------------------------------------------
    # Private — model validation
    # ------------------------------------------------------------------

    def _validate_model(self, model: Any) -> None:
        """Assert that *model* is a fitted sklearn estimator.

        Checks (in order):
        - Has a ``predict`` callable.
        - Exposes ``n_features_in_`` (set by sklearn after ``fit``).
        - Originates from the ``sklearn`` package.

        Args:
            model: The candidate model object.

        Raises:
            UnsupportedModelError: On any check failure.
        """
        if not callable(getattr(model, "predict", None)):
            raise UnsupportedModelError(
                f"Object of type '{type(model).__name__}' has no callable ``predict`` "
                "method. bt-flow requires a fitted scikit-learn estimator."
            )

        if not hasattr(model, "n_features_in_"):
            raise UnsupportedModelError(
                f"'{type(model).__name__}' does not expose ``n_features_in_``. "
                "Ensure the model has been fitted (i.e. ``.fit()`` was called) before "
                "passing it to APIGenerator."
            )

        model_module: str = getattr(type(model), "__module__", "") or ""
        if not model_module.startswith(_SKLEARN_MODULE_PREFIX):
            raise UnsupportedModelError(
                f"bt-flow v0.1.0 supports scikit-learn estimators only. "
                f"The provided model's module is '{model_module}'. "
                "Support for additional frameworks is planned for v0.2.0."
            )

    # ------------------------------------------------------------------
    # Private — schema inference
    # ------------------------------------------------------------------

    def _infer_schema(
        self,
        user_feature_names: Sequence[str] | None,
    ) -> type[BaseModel]:
        """Derive the Pydantic input schema from model metadata.

        Resolution order for feature names:

        1. ``feature_names`` argument passed to ``__init__``.
        2. ``model.feature_names_in_`` (set when trained on a DataFrame).
        3. Fall back to a positional ``List[float]`` schema using
           ``model.n_features_in_``.

        Args:
            user_feature_names: Explicit feature names supplied by the caller.

        Returns:
            A dynamically created Pydantic ``BaseModel`` subclass.

        Raises:
            SchemaInferenceError: If ``n_features_in_`` is not a positive int.
            ValueError: If ``user_feature_names`` length mismatches the model.
        """
        n_features: int = int(self._model.n_features_in_)
        if n_features <= 0:
            raise SchemaInferenceError(
                f"model.n_features_in_ reported {n_features}, which is invalid. "
                "Ensure the model is properly fitted."
            )
        self._n_features = n_features

        # Resolve feature names
        raw_names: Any | None = (
            list(user_feature_names)
            if user_feature_names is not None
            else getattr(self._model, "feature_names_in_", None)
        )

        if raw_names is not None:
            feature_names: list[str] = [str(n) for n in raw_names]
            if len(feature_names) != n_features:
                raise ValueError(
                    f"Provided {len(feature_names)} feature name(s) but the model "
                    f"expects {n_features}."
                )
            self._feature_names = feature_names
            return build_named_input_schema(feature_names)

        return build_positional_input_schema(n_features)

    # ------------------------------------------------------------------
    # Private — route registration
    # ------------------------------------------------------------------

    def _register_routes(self, app: FastAPI) -> None:
        """Wire up ``/health`` and ``/predict`` on *app*.

        Uses closures to correctly bind the dynamically generated input schema
        to the route handler so FastAPI's dependency injection reads the right
        Pydantic model for request parsing and OpenAPI schema generation.

        Args:
            app: The ``FastAPI`` instance to register routes on.
        """
        # Capture in closure — avoids `self` references inside async handlers
        _input_schema = self._input_schema
        _model = self._model
        _feature_names = self._feature_names
        _n_features = self._n_features
        _has_proba = hasattr(_model, "predict_proba")
        _is_classifier = hasattr(_model, "classes_")
        _model_type_name = type(_model).__name__
        _start_time = self._start_time

        # ---- PredictionError → HTTP 500 ----------------------------

        @app.exception_handler(PredictionError)
        async def _prediction_error_handler(request: Request, exc: PredictionError) -> JSONResponse:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(exc)},
            )

        # ---- /health ------------------------------------------------

        @app.get(
            "/health",
            response_model=HealthResponse,
            summary="Health Check",
            description=(
                "Returns the health status and metadata of the deployed model. "
                "Suitable for use as a Kubernetes liveness/readiness probe."
            ),
            tags=["Operations"],
        )
        async def health() -> HealthResponse:
            return HealthResponse(
                status="ok",
                model_type=_model_type_name,
                n_features=_n_features,
                feature_names=_feature_names,
                uptime_seconds=round(time.time() - _start_time, 3),
            )

        # ---- /predict -----------------------------------------------

        async def predict_handler(request: _input_schema) -> PredictionResponse:  # type: ignore[valid-type]
            """Run inference against the loaded model.

            Accepts a JSON payload whose structure is derived from the model's
            training-time feature metadata. Returns the prediction and, for
            classifiers that implement ``predict_proba``, per-class probabilities.
            """
            try:
                payload: dict[str, Any] = request.model_dump()  # type: ignore[attr-defined]

                if _feature_names is not None:
                    feature_values: list[float] = [float(payload[name]) for name in _feature_names]
                else:
                    feature_values = [float(v) for v in payload["features"]]

                X: np.ndarray = np.array([feature_values], dtype=np.float64)
                raw_prediction: np.ndarray = _model.predict(X)
                prediction = numpy_scalar_to_python(raw_prediction[0])

                probabilities: dict[str, float] | None = None
                if _has_proba and _is_classifier:
                    raw_proba: np.ndarray = _model.predict_proba(X)[0]
                    class_labels: list[str] = [str(c) for c in _model.classes_]
                    probabilities = {
                        label: round(float(prob), 6) for label, prob in zip(class_labels, raw_proba)
                    }

                return PredictionResponse(
                    prediction=prediction,
                    probabilities=probabilities,
                    model_type=_model_type_name,
                )

            except (KeyError, IndexError, ValueError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Input processing error: {exc}",
                ) from exc
            except Exception as exc:
                raise PredictionError(f"Model inference failed: {exc}") from exc

        # Bind the dynamic schema into __annotations__ so FastAPI's
        # introspection resolves the correct request body model.
        predict_handler.__annotations__["request"] = _input_schema
        predict_handler.__annotations__["return"] = PredictionResponse

        app.add_api_route(
            "/predict",
            predict_handler,
            methods=["POST"],
            response_model=PredictionResponse,
            summary="Run Model Inference",
            description=(
                "Accepts a JSON payload matching the model's input schema and "
                "returns a prediction. For classifiers that implement "
                "``predict_proba``, per-class probabilities are also returned."
            ),
            tags=["Inference"],
            status_code=status.HTTP_200_OK,
        )
