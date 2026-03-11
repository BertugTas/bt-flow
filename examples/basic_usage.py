"""basic_usage.py — End-to-end bt-flow demonstration.

This script shows the three canonical ways to use bt-flow:

    1. Train → save → serve from a file path.
    2. Train in-process → serve directly from the live object.
    3. Access the FastAPI app for use with an external ASGI runner.

Run it with:

    cd bt-flow
    pip install -e ".[dev]"
    python examples/basic_usage.py
"""

from __future__ import annotations

import pathlib
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ── Import bt-flow ────────────────────────────────────────────────────────────
from bt import APIGenerator

# =============================================================================
# Step 1 — Train a simple classifier
# =============================================================================

print("=" * 60)
print("bt-flow  Basic Usage Demo")
print("=" * 60)

iris = load_iris()

# Use a DataFrame so the model records feature names automatically.
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=200, random_state=42)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"\n[1] Trained LogisticRegression  —  test accuracy: {accuracy:.2%}")

# =============================================================================
# Step 2 — Save the model and serve it from disk
# =============================================================================

with tempfile.TemporaryDirectory() as tmpdir:
    model_path = pathlib.Path(tmpdir) / "iris_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"\n[2] Model saved to: {model_path}")

    # One line to create the API ↓
    api = APIGenerator(
        model_path,
        title="Iris Species Classifier",
        description=(
            "Classifies Iris flowers into Setosa, Versicolor, or Virginica "
            "based on sepal and petal measurements."
        ),
        version="1.0.0",
    )

    print(f"    {api}")
    print(f"    Feature names : {api._feature_names}")
    print(f"    N features    : {api._n_features}")

# =============================================================================
# Step 3 — Serve directly from the live object (no file needed)
# =============================================================================

print("\n[3] Serving directly from the fitted estimator (no file)…")

api_live = APIGenerator(
    clf,
    title="Iris Classifier (live object)",
)
print(f"    {api_live}")

# =============================================================================
# Step 4 — Programmatic test against the FastAPI app (no server needed)
# =============================================================================

print("\n[4] Running programmatic test via FastAPI TestClient…")

from fastapi.testclient import TestClient  # noqa: E402 (imported after demo output)

client = TestClient(api_live.app)

# Health check
health = client.get("/health").json()
print(f"    /health  →  status={health['status']}  model={health['model_type']}")

# Named-feature prediction (a known Setosa sample)
setosa_sample = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}

pred_resp = client.post("/predict", json=setosa_sample).json()
predicted_class = int(pred_resp["prediction"])
class_name = iris.target_names[predicted_class]
print(f"    /predict →  prediction={predicted_class} ({class_name})")
print(f"              probabilities={pred_resp['probabilities']}")

# =============================================================================
# Step 5 — How to start a real server (commented out)
# =============================================================================

print("\n[5] To start a production server, uncomment the line below:")
print("""
    api_live.run(host="0.0.0.0", port=8000)

    # Or hand the ASGI app to Gunicorn:
    # gunicorn -k uvicorn.workers.UvicornWorker -w 4 examples.basic_usage:api_live
""")

# =============================================================================
# Step 6 — Regressor example (no probabilities)
# =============================================================================

print("[6] Regressor example (LinearRegression)…")

from sklearn.datasets import make_regression  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

Xr, yr = make_regression(n_samples=200, n_features=3, noise=5.0, random_state=42)
reg = LinearRegression().fit(Xr, yr)

api_reg = APIGenerator(
    reg,
    title="Regression Model",
    feature_names=["x1", "x2", "x3"],
)
client_reg = TestClient(api_reg.app)
reg_resp = client_reg.post("/predict", json={"x1": 1.0, "x2": -0.5, "x3": 2.1}).json()
print(f"    prediction     = {reg_resp['prediction']:.4f}")
print(f"    probabilities  = {reg_resp['probabilities']}  (None for regressors)")

print("\nDemo complete.")
