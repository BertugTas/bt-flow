"""Pydantic schema factories and response models for bt-flow.

This module is responsible for:
- Dynamically generating request body schemas from model metadata.
- Defining static, strongly-typed response models shared across routes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, create_model

# ---------------------------------------------------------------------------
# Static response models
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Schema for the ``/predict`` endpoint response.

    Attributes:
        prediction: The model output. Scalar for single-row inference.
            May be an int (classifier), float (regressor), or str (label).
        probabilities: Class-probability mapping returned by classifiers that
            implement ``predict_proba``. ``None`` for regressors.
        model_type: The ``__name__`` of the underlying estimator class,
            e.g. ``"LogisticRegression"``.
    """

    prediction: int | float | str
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Per-class probabilities (classifiers with predict_proba only).",
    )
    model_type: str = Field(description="Class name of the underlying sklearn estimator.")


class HealthResponse(BaseModel):
    """Schema for the ``/health`` endpoint response.

    Attributes:
        status: Always ``"ok"`` when the service is running.
        model_type: Estimator class name.
        n_features: Number of input features expected by the model.
        feature_names: Ordered feature names, or ``None`` if not available.
        uptime_seconds: Seconds elapsed since the API was created.
    """

    status: str = Field(default="ok")
    model_type: str
    n_features: int
    feature_names: list[str] | None = None
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Dynamic input schema factories
# ---------------------------------------------------------------------------


def build_named_input_schema(feature_names: list[str]) -> type[BaseModel]:
    """Build a Pydantic model with one typed field per named feature.

    The generated schema looks like::

        class PredictionInput(BaseModel):
            sepal_length: float = Field(..., description="Feature: sepal_length")
            sepal_width: float  = Field(..., description="Feature: sepal_width")
            ...

    Args:
        feature_names: Ordered list of feature names sourced from
            ``model.feature_names_in_`` or provided by the caller.

    Returns:
        A dynamically created Pydantic ``BaseModel`` subclass.
    """
    fields: dict[str, Any] = {
        name: (
            float,
            Field(..., description=f"Feature: {name}"),
        )
        for name in feature_names
    }
    return create_model("PredictionInput", **fields)


def build_positional_input_schema(n_features: int) -> type[BaseModel]:
    """Build a Pydantic model that accepts an ordered feature vector.

    The generated schema looks like::

        class PredictionInput(BaseModel):
            features: List[float] = Field(..., min_length=4, max_length=4)

    Args:
        n_features: The exact number of features the model expects
            (``model.n_features_in_``).

    Returns:
        A dynamically created Pydantic ``BaseModel`` subclass.
    """
    return create_model(
        "PredictionInput",
        features=(
            list[float],
            Field(
                ...,
                min_length=n_features,
                max_length=n_features,
                description=(
                    f"Ordered list of exactly {n_features} numeric feature value(s). "
                    "Order must match the feature order used during training."
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def numpy_scalar_to_python(value: Any) -> int | float | str:
    """Coerce a numpy scalar to a native Python type safe for JSON serialisation.

    Args:
        value: The raw prediction value from ``model.predict()``.

    Returns:
        A native Python ``int``, ``float``, or ``str``.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return int(value)
    return str(value)
