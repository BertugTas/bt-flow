# bt-flow

> Deploy scikit-learn models as production-ready FastAPI REST APIs in one line.

```python
from bt import APIGenerator

api = APIGenerator("my_model.pkl")
api.run()  # ← that's it
```

## Install

```bash
pip install bt-flow
```

## Quick Start

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from bt import APIGenerator

X, y = load_iris(return_X_y=True)
clf = LogisticRegression().fit(X, y)

api = APIGenerator(clf, title="Iris Classifier")
api.run(port=8000)
```

Auto-generated endpoints:

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Liveness probe + model metadata      |
| POST   | `/predict` | Run inference, returns probabilities |
| GET    | `/docs`    | Interactive Swagger UI               |

## License

MIT
