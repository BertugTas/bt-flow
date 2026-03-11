# Contributing to bt-flow

Thank you for considering a contribution! This document explains how to set up your environment, run tests, and submit a pull request.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)

---

## Code of Conduct

Be kind, inclusive, and constructive. We follow the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## Getting Started

### Prerequisites

- Python ≥ 3.9
- [Hatch](https://hatch.pypa.io/) — the project's build and environment manager

```bash
pip install hatch
```

### Clone and install

```bash
git clone https://github.com/BertugTas/bt-flow.git
cd bt-flow
pip install -e ".[dev]"
```

### Optional: set up pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

---

## Development Workflow

| Command              | Description                          |
|----------------------|--------------------------------------|
| `hatch run test`     | Run the full test suite              |
| `hatch run lint`     | Ruff linting                         |
| `hatch run fmt`      | Ruff auto-formatting                 |
| `hatch run typecheck`| Mypy strict type-checking            |
| `hatch run all`      | Lint + typecheck + test in sequence  |

---

## Running Tests

```bash
hatch run test
```

Tests live in `tests/`. All tests use `pytest`. Async tests are handled automatically via `asyncio_mode = "auto"`.

To run a single test file:

```bash
hatch run pytest tests/test_core.py -v
```

---

## Code Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) — `hatch run fmt`
- **Linter**: Ruff — `hatch run lint`
- **Type checker**: Mypy strict — `hatch run typecheck`
- **Line length**: 100 characters
- **Docstrings**: Google style

All CI checks must pass before a PR is merged.

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
2. **Make your changes** — include tests for new behaviour.
3. **Ensure all checks pass**:
   ```bash
   hatch run all
   ```
4. **Update `CHANGELOG.md`** under `[Unreleased]`.
5. **Open a PR** against `main` with a descriptive title and summary.

---

## Reporting Bugs

Open an issue at [GitHub Issues](https://github.com/BertugTas/bt-flow/issues) and include:

- bt-flow version (`bt-flow --version`)
- Python version
- scikit-learn version
- Minimal reproducible example
- Full traceback
