# bt-flow

> **Bridge the gap between model training and deployment.**  
> Serve scikit-learn models as production-ready FastAPI REST APIs — in a single line of code.

## Overview

**bt-flow** takes a trained scikit-learn model and instantly wraps it in a production-grade [FastAPI](https://fastapi.tiangolo.com/) REST API with:

- 📋 **Auto-generated schemas** — inferred from `model.feature_names_in_` or `n_features_in_`
- 🩺 **`/health`** — Kubernetes-ready liveness/readiness probe
- 🔮 **`/predict`** — inference with optional per-class probabilities
- 📖 **Swagger UI** at `/docs`, ReDoc at `/redoc`
- ⚡ **CLI** — zero Python required

```python
from bt import APIGenerator

api = APIGenerator("my_model.pkl")
api.run()  # ← that's it
```

Continue to [Installation](getting-started/installation.md) to get started.
