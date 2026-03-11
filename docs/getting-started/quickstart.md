# Quick Start

## Train a model and serve it

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from bt import APIGenerator

# 1. Train
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)

# 2. Serve — one line
api = APIGenerator(clf, title="Iris Classifier")
api.run(port=8000)
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

## From the command line

```bash
# Save your model
import joblib; joblib.dump(clf, "model.joblib")

# Serve it
bt-flow serve model.joblib
```

## Test without a server

```python
from fastapi.testclient import TestClient

client = TestClient(api.app)
resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
print(resp.json())
# {"prediction": 0, "probabilities": {...}, "model_type": "LogisticRegression"}
```
