# Schema Inference

bt-flow automatically determines the best request body schema for your model.

## Resolution order

```
1. feature_names argument passed to APIGenerator()
        ↓
2. model.feature_names_in_  (set when trained on a DataFrame)
        ↓
3. Positional List[float] schema using model.n_features_in_
```

## Named schema (DataFrame-trained model)

When your model was trained on a `pd.DataFrame`, sklearn records column names in `model.feature_names_in_`. bt-flow generates one typed field per feature:

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from bt import APIGenerator

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = LogisticRegression().fit(X, iris.target)

api = APIGenerator(clf)
# Request body: {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, ...}
```

## Positional schema (numpy-trained model)

When trained on a raw numpy array, bt-flow generates a list-based schema:

```python
X, y = load_iris(return_X_y=True)
clf = LogisticRegression().fit(X, y)

api = APIGenerator(clf)
# Request body: {"features": [5.1, 3.5, 1.4, 0.2]}
```

## Manual override

Provide `feature_names` to force named fields regardless of training data:

```python
api = APIGenerator(clf, feature_names=["sepal_len", "sepal_wid", "petal_len", "petal_wid"])
# Request body: {"sepal_len": 5.1, "sepal_wid": 3.5, ...}
```

The list length must exactly match `model.n_features_in_`, or a `ValueError` is raised.
