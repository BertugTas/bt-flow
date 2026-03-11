"""Test suite for bt.cli (the ``bt-flow`` console script).

Strategy
--------
* All tests use Typer's ``CliRunner`` which redirects ``sys.stdout`` /
  ``sys.stderr`` before invoking the command.  Because ``cli.py`` creates its
  ``Console`` instances *inside* the command function (not at module level),
  Rich output is written to the runner's captured streams.
* ``APIGenerator.run`` is patched to a no-op in every test that reaches the
  "start server" code path, preventing uvicorn from actually binding a port.
* ``mix_stderr=False`` keeps stdout and stderr separate so we can assert that
  errors go to the right stream.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from bt.cli import app

# One shared runner for the whole module — stateless, safe to reuse.
# Typer's CliRunner always captures stderr separately (result.stderr);
# mix_stderr is not a constructor param in Typer >= 0.12.
runner = CliRunner()

# Patch target: the method on the class so every instance is covered.
_RUN_TARGET = "bt.core.APIGenerator.run"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def invoke(*args: str, **kwargs: Any) -> Any:
    """Thin wrapper — invoke the CLI and return the result object."""
    return runner.invoke(app, list(args), **kwargs)


def invoke_serve(*args: str, **kwargs: Any) -> Any:
    """Invoke the ``serve`` sub-command."""
    return runner.invoke(app, ["serve", *args], **kwargs)


# ---------------------------------------------------------------------------
# Top-level help
# ---------------------------------------------------------------------------


class TestHelp:
    def test_root_help_exits_zero(self) -> None:
        result = invoke("--help")
        assert result.exit_code == 0

    def test_root_help_mentions_serve(self) -> None:
        result = invoke("--help")
        assert "serve" in result.output

    def test_version_flag_exits_zero(self) -> None:
        result = invoke("--version")
        assert result.exit_code == 0

    def test_version_flag_prints_version(self) -> None:
        from bt import __version__
        result = invoke("--version")
        assert __version__ in result.output

    def test_serve_help_exits_zero(self) -> None:
        result = invoke_serve("--help")
        assert result.exit_code == 0

    def test_serve_help_shows_all_options(self) -> None:
        result = invoke_serve("--help")
        for flag in ("--port", "--host", "--title", "--log-level", "--feature-names"):
            assert flag in result.output, f"Missing option: {flag}"

    def test_serve_help_shows_examples(self) -> None:
        result = invoke_serve("--help")
        assert ".pkl" in result.output or ".joblib" in result.output


# ---------------------------------------------------------------------------
# Error cases — file / model problems
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_missing_file_exits_one(self, tmp_path: pathlib.Path) -> None:
        result = invoke_serve(str(tmp_path / "ghost.pkl"))
        assert result.exit_code == 1

    def test_missing_file_error_on_stderr(self, tmp_path: pathlib.Path) -> None:
        result = invoke_serve(str(tmp_path / "ghost.pkl"))
        # Error panel goes to stderr; stdout should be empty / minimal.
        assert "ghost.pkl" in (result.stderr or "")

    def test_unsupported_extension_exits_one(self, tmp_path: pathlib.Path) -> None:
        bad = tmp_path / "model.h5"
        bad.touch()
        result = invoke_serve(str(bad))
        assert result.exit_code == 1

    def test_unsupported_extension_mentions_format(self, tmp_path: pathlib.Path) -> None:
        bad = tmp_path / "model.h5"
        bad.touch()
        result = invoke_serve(str(bad))
        assert ".h5" in (result.stderr or "") or "unsupported" in (result.stderr or "").lower()

    def test_unfitted_model_exits_one(self, tmp_path: pathlib.Path) -> None:
        from sklearn.linear_model import LogisticRegression

        path = tmp_path / "unfitted.pkl"
        with open(path, "wb") as fh:
            pickle.dump(LogisticRegression(), fh)
        result = invoke_serve(str(path))
        assert result.exit_code == 1

    def test_unfitted_model_error_on_stderr(self, tmp_path: pathlib.Path) -> None:
        from sklearn.linear_model import LogisticRegression

        path = tmp_path / "unfitted.pkl"
        with open(path, "wb") as fh:
            pickle.dump(LogisticRegression(), fh)
        result = invoke_serve(str(path))
        stderr = result.stderr or ""
        assert "n_features_in_" in stderr or "fitted" in stderr.lower()

    def test_empty_feature_names_exits_one(self, pkl_model_path: pathlib.Path) -> None:
        result = invoke_serve(str(pkl_model_path), "--feature-names", "  ,  ,")
        assert result.exit_code == 1

    def test_wrong_feature_name_count_exits_one(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        # Iris has 4 features; supply only 2.
        result = invoke_serve(
            str(pkl_model_path), "--feature-names", "only_one,only_two"
        )
        assert result.exit_code == 1

    def test_wrong_feature_name_count_error_on_stderr(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        result = invoke_serve(
            str(pkl_model_path), "--feature-names", "only_one,only_two"
        )
        stderr = result.stderr or ""
        assert "4" in stderr or "feature" in stderr.lower()


# ---------------------------------------------------------------------------
# Happy-path — valid model, mocked server
# ---------------------------------------------------------------------------


class TestHappyPath:
    """All tests in this class patch ``APIGenerator.run`` to a no-op."""

    def test_exits_zero_with_valid_pkl(self, pkl_model_path: pathlib.Path) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path))
        assert result.exit_code == 0

    def test_exits_zero_with_valid_joblib(
        self, joblib_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(joblib_model_path))
        assert result.exit_code == 0

    def test_startup_panel_shows_model_name(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path))
        assert "LogisticRegression" in result.output

    def test_startup_panel_shows_default_port(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path))
        assert "8000" in result.output

    def test_startup_panel_shows_swagger_path(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path))
        assert "/docs" in result.output

    def test_startup_panel_shows_predict_path(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path))
        assert "/predict" in result.output

    def test_startup_panel_shows_health_path(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path))
        assert "/health" in result.output

    def test_startup_panel_shows_localhost_not_0000(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        """0.0.0.0 should be humanised to 'localhost' in the display URL."""
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path))
        assert "localhost" in result.output
        assert "0.0.0.0" not in result.output


# ---------------------------------------------------------------------------
# Argument forwarding — verify options reach APIGenerator.run correctly
# ---------------------------------------------------------------------------


class TestArgumentForwarding:
    def test_custom_port_forwarded_to_run(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        mock = MagicMock()
        with patch(_RUN_TARGET, mock):
            result = invoke_serve(str(pkl_model_path), "--port", "9123")
        assert result.exit_code == 0
        _assert_kwarg(mock, "port", 9123)

    def test_custom_host_forwarded_to_run(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        mock = MagicMock()
        with patch(_RUN_TARGET, mock):
            result = invoke_serve(str(pkl_model_path), "--host", "127.0.0.1")
        assert result.exit_code == 0
        _assert_kwarg(mock, "host", "127.0.0.1")

    def test_custom_log_level_forwarded_to_run(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        mock = MagicMock()
        with patch(_RUN_TARGET, mock):
            result = invoke_serve(str(pkl_model_path), "--log-level", "warning")
        assert result.exit_code == 0
        _assert_kwarg(mock, "log_level", "warning")

    def test_custom_port_appears_in_startup_panel(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path), "--port", "9999")
        assert "9999" in result.output

    def test_custom_host_appears_in_startup_panel(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(str(pkl_model_path), "--host", "192.168.1.100")
        assert "192.168.1.100" in result.output

    def test_run_called_exactly_once(self, pkl_model_path: pathlib.Path) -> None:
        mock = MagicMock()
        with patch(_RUN_TARGET, mock):
            invoke_serve(str(pkl_model_path))
        mock.assert_called_once()


# ---------------------------------------------------------------------------
# Feature names override
# ---------------------------------------------------------------------------


class TestFeatureNamesOption:
    def test_valid_feature_names_accepted(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(
                str(pkl_model_path), "--feature-names", "a,b,c,d"
            )
        assert result.exit_code == 0

    def test_feature_names_with_spaces_stripped(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        """Spaces around commas should be tolerated."""
        with patch(_RUN_TARGET):
            result = invoke_serve(
                str(pkl_model_path), "--feature-names", " a , b , c , d "
            )
        assert result.exit_code == 0

    def test_named_features_appear_in_panel(
        self, pkl_model_path: pathlib.Path
    ) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(
                str(pkl_model_path), "--feature-names", "alpha,beta,gamma,delta"
            )
        assert "alpha" in result.output

    def test_schema_label_is_named(self, pkl_model_path: pathlib.Path) -> None:
        with patch(_RUN_TARGET):
            result = invoke_serve(
                str(pkl_model_path), "--feature-names", "w,x,y,z"
            )
        assert "named" in result.output


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _assert_kwarg(mock: MagicMock, key: str, expected: Any) -> None:
    """Assert that a mock was called with a specific keyword argument value.

    Handles both positional (``call_args.args``) and keyword
    (``call_args.kwargs``) call styles.
    """
    call = mock.call_args
    assert call is not None, "Mock was never called"
    actual = call.kwargs.get(key)
    assert actual == expected, (
        f"Expected {key}={expected!r}, got {actual!r}. "
        f"Full call: {call}"
    )
