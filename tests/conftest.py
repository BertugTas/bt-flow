"""Shared pytest fixtures for bt-flow tests.

Fixtures are scoped at the *session* level wherever safe to do so, since
training and serialising sklearn models is relatively expensive.
"""

from __future__ import annotations

import pathlib
import pickle

import joblib
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def iris_data():  # type: ignore[return]
    """Return the Iris dataset as (X_array, y)."""
    return load_iris(return_X_y=True)


@pytest.fixture(scope="session")
def iris_data_named():  # type: ignore[return]
    """Return the Iris dataset with a pandas DataFrame for X (named features)."""
    iris = load_iris()
    X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return X_df, iris.target


@pytest.fixture(scope="session")
def regression_data():  # type: ignore[return]
    """Return a synthetic regression dataset."""
    return make_regression(n_samples=100, n_features=5, random_state=42)


# ---------------------------------------------------------------------------
# Fitted estimators (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_classifier(iris_data):  # type: ignore[return]
    """A fitted LogisticRegression — no feature names."""
    X, y = iris_data
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="session")
def fitted_classifier_named(iris_data_named):  # type: ignore[return]
    """A fitted LogisticRegression trained on a DataFrame — has feature_names_in_."""
    X_df, y = iris_data_named
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_df, y)
    return clf


@pytest.fixture(scope="session")
def fitted_regressor(regression_data):  # type: ignore[return]
    """A fitted LinearRegression (no predict_proba)."""
    X, y = regression_data
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


# ---------------------------------------------------------------------------
# Serialised model paths (session-scoped, tmp_path_factory)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pkl_model_path(tmp_path_factory, fitted_classifier) -> pathlib.Path:  # type: ignore[return]
    """Write a fitted classifier to a ``.pkl`` file and return its path."""
    path = tmp_path_factory.mktemp("models") / "classifier.pkl"
    with open(path, "wb") as fh:
        pickle.dump(fitted_classifier, fh)
    return path


@pytest.fixture(scope="session")
def joblib_model_path(tmp_path_factory, fitted_classifier) -> pathlib.Path:  # type: ignore[return]
    """Write a fitted classifier to a ``.joblib`` file and return its path."""
    path = tmp_path_factory.mktemp("models") / "classifier.joblib"
    joblib.dump(fitted_classifier, path)
    return path
