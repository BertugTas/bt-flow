# Python API

## `APIGenerator`

The main class — wraps a scikit-learn estimator in a FastAPI app.

```python
from bt import APIGenerator

api = APIGenerator(
    "model.pkl",          # or a live fitted estimator
    title="My Model API",
    description="Auto-generated API.",
    version="1.0.0",
    feature_names=["a", "b", "c"],  # optional override
    docs_url="/docs",     # set None to disable
    redoc_url="/redoc",   # set None to disable
)
```

### Constructor Parameters

| Parameter       | Type                      | Default              | Description                                              |
|-----------------|---------------------------|----------------------|----------------------------------------------------------|
| `model_source`  | `str \| Path \| estimator` | *required*          | Path to `.pkl`/`.joblib` or a fitted sklearn estimator   |
| `title`         | `str`                     | `"bt-flow Model API"` | OpenAPI / Swagger UI title                              |
| `description`   | `str`                     | Auto text            | Markdown description shown in docs                       |
| `version`       | `str`                     | `"0.1.0"`            | API version string in OpenAPI schema                     |
| `feature_names` | `list[str] \| None`       | `None`               | Explicit feature names; overrides `model.feature_names_in_` |
| `docs_url`      | `str \| None`             | `"/docs"`            | Swagger UI path; `None` to disable                       |
| `redoc_url`     | `str \| None`             | `"/redoc"`           | ReDoc path; `None` to disable                            |

### Key Methods & Properties

| Member         | Type              | Description                                                  |
|----------------|-------------------|--------------------------------------------------------------|
| `.app`         | `FastAPI`         | The underlying FastAPI instance (use with Gunicorn/uvicorn)  |
| `.run(...)`    | `None` (blocking) | Start uvicorn server                                         |
| `.__call__(...)`| ASGI callable    | Pass the generator directly to any ASGI server               |

### Exceptions raised

| Exception              | When                                                        |
|------------------------|-------------------------------------------------------------|
| `ModelLoadError`       | File does not exist or has unsupported extension            |
| `UnsupportedModelError`| Object is not a fitted sklearn estimator                    |
| `SchemaInferenceError` | `n_features_in_` is missing or non-positive                 |
| `ValueError`           | `feature_names` length mismatches `n_features_in_`         |

## Programmatic API reference

::: bt.core.APIGenerator
