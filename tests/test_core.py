"""Test suite for bt.core.APIGenerator.

Covers:
- Model loading from .pkl and .joblib paths.
- Model loading from a live object.
- Schema inference (named vs. positional features).
- /health endpoint contract.
- /predict endpoint contract (classifier, classifier+proba, regressor).
- Error handling: missing file, bad extension, unfitted model, bad payload.
- ASGI interface.
- numpy scalar coercion.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from bt import APIGenerator, ModelLoadError, UnsupportedModelError
from bt.schemas import numpy_scalar_to_python

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client(generator: APIGenerator) -> TestClient:
    return TestClient(generator.app)


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------


class TestModelLoading:
    def test_load_from_pkl_path(self, pkl_model_path: pathlib.Path) -> None:
        gen = APIGenerator(pkl_model_path)
        assert gen._model is not None

    def test_load_from_joblib_path(self, joblib_model_path: pathlib.Path) -> None:
        gen = APIGenerator(joblib_model_path)
        assert gen._model is not None

    def test_load_from_string_path(self, pkl_model_path: pathlib.Path) -> None:
        gen = APIGenerator(str(pkl_model_path))
        assert gen._model is not None

    def test_load_from_live_object(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier)
        assert gen._model is fitted_classifier

    def test_missing_file_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ModelLoadError, match="does not exist"):
            APIGenerator(tmp_path / "nonexistent.pkl")

    def test_unsupported_extension_raises(self, tmp_path: pathlib.Path) -> None:
        bad_path = tmp_path / "model.h5"
        bad_path.touch()
        with pytest.raises(ModelLoadError, match="unsupported file extension"):
            APIGenerator(bad_path)

    def test_unfitted_model_raises(self) -> None:
        unfitted = LogisticRegression()
        with pytest.raises(UnsupportedModelError, match="n_features_in_"):
            APIGenerator(unfitted)

    def test_non_sklearn_object_raises(self) -> None:
        class FakeModel:
            n_features_in_ = 4

            def predict(self, X):  # type: ignore[no-untyped-def]
                return X

        with pytest.raises(UnsupportedModelError, match="scikit-learn"):
            APIGenerator(FakeModel())

    def test_object_without_predict_raises(self) -> None:
        with pytest.raises(UnsupportedModelError, match="predict"):
            APIGenerator({"not": "a model"})

    def test_repr_contains_model_name(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier)
        assert "LogisticRegression" in repr(gen)


# ---------------------------------------------------------------------------
# Schema inference tests
# ---------------------------------------------------------------------------


class TestSchemaInference:
    def test_positional_schema_no_feature_names(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier)
        assert gen._feature_names is None
        assert gen._n_features == 4  # Iris has 4 features

    def test_named_schema_from_dataframe_training(self, fitted_classifier_named) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier_named)
        assert gen._feature_names is not None
        assert len(gen._feature_names) == 4

    def test_named_schema_from_user_override(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        names = ["a", "b", "c", "d"]
        gen = APIGenerator(fitted_classifier, feature_names=names)
        assert gen._feature_names == names

    def test_user_feature_names_wrong_length_raises(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="4"):
            APIGenerator(fitted_classifier, feature_names=["only_one"])


# ---------------------------------------------------------------------------
# /health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        client = make_client(APIGenerator(fitted_classifier))
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        client = make_client(APIGenerator(fitted_classifier))
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_model_type_in_response(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        client = make_client(APIGenerator(fitted_classifier))
        data = client.get("/health").json()
        assert data["model_type"] == "LogisticRegression"

    def test_n_features_in_response(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        client = make_client(APIGenerator(fitted_classifier))
        data = client.get("/health").json()
        assert data["n_features"] == 4

    def test_feature_names_present_when_named(self, fitted_classifier_named) -> None:  # type: ignore[no-untyped-def]
        client = make_client(APIGenerator(fitted_classifier_named))
        data = client.get("/health").json()
        assert isinstance(data["feature_names"], list)
        assert len(data["feature_names"]) == 4

    def test_feature_names_null_when_positional(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        client = make_client(APIGenerator(fitted_classifier))
        data = client.get("/health").json()
        assert data["feature_names"] is None

    def test_uptime_is_non_negative(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        client = make_client(APIGenerator(fitted_classifier))
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# /predict endpoint tests — positional schema
# ---------------------------------------------------------------------------


class TestPredictPositional:
    """Classifier trained on raw numpy array — positional feature schema."""

    @pytest.fixture(autouse=True)
    def setup(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        self.client = make_client(APIGenerator(fitted_classifier))

    def test_returns_200(self) -> None:
        resp = self.client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        assert resp.status_code == 200

    def test_prediction_present(self) -> None:
        resp = self.client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        data = resp.json()
        assert "prediction" in data

    def test_probabilities_present_for_classifier(self) -> None:
        resp = self.client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        data = resp.json()
        assert data["probabilities"] is not None
        assert len(data["probabilities"]) == 3  # 3 Iris classes

    def test_probabilities_sum_to_one(self) -> None:
        resp = self.client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        proba = resp.json()["probabilities"]
        assert abs(sum(proba.values()) - 1.0) < 1e-5

    def test_wrong_feature_count_returns_422(self) -> None:
        resp = self.client.post("/predict", json={"features": [1.0, 2.0]})  # too few
        assert resp.status_code == 422

    def test_missing_body_returns_422(self) -> None:
        resp = self.client.post("/predict", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /predict endpoint tests — named schema
# ---------------------------------------------------------------------------


class TestPredictNamed:
    """Classifier trained on DataFrame — named feature schema."""

    @pytest.fixture(autouse=True)
    def setup(self, fitted_classifier_named) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier_named)
        self.client = make_client(gen)
        self.feature_names = gen._feature_names

    def _valid_payload(self):  # type: ignore[no-untyped-def]
        assert self.feature_names is not None
        return dict.fromkeys(self.feature_names, 1.0)

    def test_returns_200_with_named_payload(self) -> None:
        resp = self.client.post("/predict", json=self._valid_payload())
        assert resp.status_code == 200

    def test_missing_feature_returns_422(self) -> None:
        payload = self._valid_payload()
        del payload[self.feature_names[0]]  # type: ignore[index]
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_model_type_in_response(self) -> None:
        resp = self.client.post("/predict", json=self._valid_payload())
        assert resp.json()["model_type"] == "LogisticRegression"


# ---------------------------------------------------------------------------
# /predict endpoint tests — regressor (no probabilities)
# ---------------------------------------------------------------------------


class TestPredictRegressor:
    @pytest.fixture(autouse=True)
    def setup(self, fitted_regressor) -> None:  # type: ignore[no-untyped-def]
        self.client = make_client(APIGenerator(fitted_regressor))
        self.n_features = fitted_regressor.n_features_in_

    def test_returns_200(self) -> None:
        payload = {"features": [1.0] * self.n_features}
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_probabilities_are_null_for_regressor(self) -> None:
        payload = {"features": [1.0] * self.n_features}
        resp = self.client.post("/predict", json=payload)
        assert resp.json()["probabilities"] is None

    def test_prediction_is_numeric(self) -> None:
        payload = {"features": [1.0] * self.n_features}
        resp = self.client.post("/predict", json=payload)
        assert isinstance(resp.json()["prediction"], float)


# ---------------------------------------------------------------------------
# Public properties
# ---------------------------------------------------------------------------


class TestPublicProperties:
    def test_n_features(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier)
        assert gen.n_features == 4

    def test_feature_names_none_for_numpy_training(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier)
        assert gen.feature_names is None

    def test_feature_names_set_for_dataframe_training(self, fitted_classifier_named) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier_named)
        assert gen.feature_names is not None
        assert len(gen.feature_names) == 4

    def test_model_type(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier)
        assert gen.model_type == "LogisticRegression"


# ---------------------------------------------------------------------------
# PredictionError — model inference failure
# ---------------------------------------------------------------------------


class TestPredictionError:
    def test_model_crash_returns_500(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        from unittest.mock import patch

        gen = APIGenerator(fitted_classifier)
        client = make_client(gen)
        with patch.object(fitted_classifier, "predict", side_effect=RuntimeError("GPU exploded")):
            resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        assert resp.status_code == 500
        assert "Model inference failed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# OpenAPI docs availability
# ---------------------------------------------------------------------------


class TestOpenAPIDocs:
    def test_openapi_json_available(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        client = make_client(APIGenerator(fitted_classifier))
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_docs_ui_disabled(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        gen = APIGenerator(fitted_classifier, docs_url=None, redoc_url=None)
        client = make_client(gen)
        assert client.get("/docs").status_code == 404


# ---------------------------------------------------------------------------
# ASGI interface — APIGenerator as a first-class ASGI callable
# ---------------------------------------------------------------------------


class TestASGIInterface:
    async def test_asgi_call_delegates_to_app(self, fitted_classifier) -> None:  # type: ignore[no-untyped-def]
        """APIGenerator.__call__ must forward ASGI traffic to the FastAPI app."""
        import httpx

        gen = APIGenerator(fitted_classifier)
        # ASGITransport calls gen.__call__ directly, not gen.app
        transport = httpx.ASGITransport(app=gen)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Model loading — corrupt file
# ---------------------------------------------------------------------------


class TestCorruptModelFile:
    def test_corrupt_joblib_raises_model_load_error(self, tmp_path: pathlib.Path) -> None:
        corrupt = tmp_path / "bad.joblib"
        corrupt.write_bytes(b"this is not a valid joblib file")
        with pytest.raises(ModelLoadError, match=r"bad\.joblib"):
            APIGenerator(corrupt)


# ---------------------------------------------------------------------------
# numpy_scalar_to_python — edge cases
# ---------------------------------------------------------------------------


class TestNumpyScalarToPython:
    def test_numpy_integer(self) -> None:
        assert numpy_scalar_to_python(np.int64(42)) == 42
        assert isinstance(numpy_scalar_to_python(np.int64(42)), int)

    def test_numpy_float(self) -> None:
        assert numpy_scalar_to_python(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(numpy_scalar_to_python(np.float64(3.14)), float)

    def test_numpy_bool(self) -> None:
        assert numpy_scalar_to_python(np.bool_(True)) == 1
        assert numpy_scalar_to_python(np.bool_(False)) == 0

    def test_string_fallback(self) -> None:
        assert numpy_scalar_to_python("setosa") == "setosa"
        assert isinstance(numpy_scalar_to_python("setosa"), str)
