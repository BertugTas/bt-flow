# bt-flow 🚀

> **Bridge the gap between model training and deployment.**  
> Serve scikit-learn models as production-ready FastAPI REST APIs — in a single line of code.

[![CI](https://github.com/BertugTas/bt-flow/actions/workflows/ci.yml/badge.svg)](https://github.com/BertugTas/bt-flow/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BertugTas/bt-flow/branch/main/graph/badge.svg)](https://codecov.io/gh/BertugTas/bt-flow)
[![PyPI - Version](https://img.shields.io/pypi/v/bt-flow)](https://pypi.org/project/bt-flow/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bt-flow)](https://pypi.org/project/bt-flow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ What is bt-flow?

**bt-flow** takes a trained scikit-learn model and instantly wraps it in a production-grade [FastAPI](https://fastapi.tiangolo.com/) REST API — complete with:

- 📋 **Auto-generated Pydantic schemas** inferred from your model's training-time feature metadata  
- 🩺 **Health endpoint** (`/health`) ready for Kubernetes liveness/readiness probes  
- 🔮 **Predict endpoint** (`/predict`) returning predictions and per-class probabilities  
- 📖 **Interactive Swagger UI** at `/docs` and ReDoc at `/redoc`  
- ⚡ **CLI** — serve any `.pkl` or `.joblib` file without writing a single line of Python  

---

## 📦 Installation

```bash
pip install bt-flow
```

Requires Python ≥ 3.9.

---

## 🚀 Quick Start

### Option 1 — Python API

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from bt import APIGenerator

# Train a model
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)

# One line to serve it
api = APIGenerator(clf, title="Iris Classifier")
api.run(port=8000)  # blocks — visit http://localhost:8000/docs
```

### Option 2 — CLI

```bash
# Save your model first, then serve it directly
bt-flow serve iris_classifier.joblib

# With options
bt-flow serve model.pkl --port 9000 --host 127.0.0.1 --title "My API"

# Override or supply feature names
bt-flow serve model.pkl --feature-names 'sepal_len,sepal_wid,petal_len,petal_wid'
```

---

## 🌐 Auto-Generated Endpoints

Once running, your API exposes:

| Method | Path          | Description                                        |
|--------|---------------|----------------------------------------------------|
| `GET`  | `/health`     | Liveness probe — returns model metadata + uptime   |
| `POST` | `/predict`    | Run inference — returns prediction + probabilities |
| `GET`  | `/docs`       | Interactive Swagger UI                             |
| `GET`  | `/redoc`      | ReDoc documentation                               |
| `GET`  | `/openapi.json` | Raw OpenAPI schema                              |

### Example: Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_type": "LogisticRegression",
  "n_features": 4,
  "feature_names": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
  "uptime_seconds": 12.345
}
```

### Example: Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}'
```

```json
{
  "prediction": 0,
  "probabilities": {"0": 0.978, "1": 0.015, "2": 0.007},
  "model_type": "LogisticRegression"
}
```

---

## 🔧 Advanced Usage

### Use with Gunicorn (production)

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 mymodule:api.app
```

Where `mymodule.py` contains:

```python
from bt import APIGenerator
api = APIGenerator("model.pkl")
```

### ASGI-compatible — plug into any runner

`APIGenerator` implements the ASGI interface directly:

```python
import uvicorn
from bt import APIGenerator

api = APIGenerator("model.pkl")
uvicorn.run(api, host="0.0.0.0", port=8000)  # pass the generator directly
```

### Programmatic testing (no server needed)

```python
from fastapi.testclient import TestClient
from bt import APIGenerator

api = APIGenerator(clf)
client = TestClient(api.app)

resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
assert resp.status_code == 200
```

### Disable docs UI

```python
api = APIGenerator(clf, docs_url=None, redoc_url=None)
```

---

## 📐 Schema Inference

bt-flow automatically detects the best input schema:

| Training data              | Inferred schema               | Example payload                    |
|----------------------------|-------------------------------|------------------------------------|
| `pd.DataFrame` with columns | Named fields (one per feature) | `{"sepal_len": 5.1, ...}`         |
| Raw `numpy` array          | Positional `features` list    | `{"features": [5.1, 3.5, 1.4, 0.2]}` |
| `--feature-names` override | Named fields (user-provided)  | `{"a": 5.1, "b": 3.5, ...}`       |

---

## 🧪 Development

```bash
git clone https://github.com/BertugTas/bt-flow.git
cd bt-flow
pip install hatch

# Run tests
hatch run test

# Lint + format
hatch run lint
hatch run fmt

# Type-check
hatch run typecheck

# Run all checks
hatch run all
```

---

## 🗺️ Roadmap

- [ ] XGBoost and LightGBM support
- [ ] Batch prediction endpoint (`/predict/batch`)
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] Docker image generator (`bt-flow dockerize`)
- [ ] Async prediction support

---

## 📄 License

[MIT](LICENSE) © Bertug Tas
