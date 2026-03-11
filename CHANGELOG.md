# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- XGBoost and LightGBM support
- Batch prediction endpoint (`/predict/batch`)
- Prometheus metrics endpoint (`/metrics`)
- Docker image generator (`bt-flow dockerize`)
- Async prediction support

---

## [0.1.0] — 2026-03-11

### Added
- **`APIGenerator`** class — wraps any fitted scikit-learn estimator as a FastAPI ASGI app
- **Automatic schema inference** — detects named features from `model.feature_names_in_` (DataFrame-trained models) or falls back to a positional `features: List[float]` schema
- **`/health`** endpoint — returns model metadata, feature info, and uptime; suitable as a Kubernetes liveness/readiness probe
- **`/predict`** endpoint — runs inference and returns prediction + per-class probabilities for classifiers
- **CLI** (`bt-flow serve`) — serve any `.pkl` or `.joblib` file with zero Python required
- **Rich startup panel** — coloured, information-dense terminal output on server start
- **ASGI compatibility** — `APIGenerator` implements the ASGI callable interface directly
- **`py.typed` marker** — full PEP 561 inline type annotation support
- Comprehensive test suite (50+ tests) covering model loading, schema inference, endpoints, and CLI
- CI workflow (GitHub Actions) with test matrix across Python 3.9 – 3.12

[Unreleased]: https://github.com/BertugTas/bt-flow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BertugTas/bt-flow/releases/tag/v0.1.0
