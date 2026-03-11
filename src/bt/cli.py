"""Command-line interface for bt-flow.

Provides the ``bt-flow`` console script, which serves a serialised
scikit-learn model as a FastAPI REST API with a single shell command::

    bt-flow serve iris_classifier.pkl
    bt-flow serve model.joblib --port 9000 --host 127.0.0.1
    bt-flow serve model.pkl --feature-names 'age,income,score'

The startup experience is powered by ``rich`` for a professional,
information-dense terminal output.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from bt import __version__
from bt.core import APIGenerator
from bt.exceptions import BTFlowError

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="bt-flow",
    help=(
        "Serve scikit-learn models as production-ready FastAPI REST APIs.\n\n"
        "Run [bold cyan]bt-flow serve --help[/] for full usage."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Eager callback for --version / -V.

    Typer evaluates option callbacks before the command body, so raising
    ``typer.Exit`` here cleanly terminates without requiring a subcommand.
    """
    if value:
        typer.echo(f"bt-flow v{__version__}")
        raise typer.Exit()


@app.callback()
def _root_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print the bt-flow version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """bt-flow — scikit-learn → FastAPI. Zero boilerplate.

    This callback serves two purposes:

    1. Exposes ``--version`` / ``-V`` at the root level.
    2. Forces Typer into group-mode (multi-command routing) so that
       ``bt-flow serve <path>`` always works — without the callback,
       Typer 0.12+ hoists a single command to the app level, swallowing
       the subcommand name as a positional argument.
    """


# ---------------------------------------------------------------------------
# Private renderables — pure functions that build Rich objects without I/O
# ---------------------------------------------------------------------------


def _startup_panel(
    api: APIGenerator,
    host: str,
    port: int,
    model_path: Path,
) -> Panel:
    """Build the Rich startup information panel.

    This is a pure function; it builds and returns a renderable without
    printing anything. Separation from I/O makes unit testing straightforward.

    Args:
        api: The fully initialised ``APIGenerator`` instance.
        host: The network host the server will bind to.
        port: The TCP port the server will listen on.
        model_path: Filesystem path of the loaded model artifact.

    Returns:
        A Rich ``Panel`` ready to be passed to ``console.print()``.
    """
    display_host = "localhost" if host in ("0.0.0.0", "::") else host
    base_url = f"http://{display_host}:{port}"

    # ── Feature summary line ────────────────────────────────────────────────
    n = api._n_features
    names = api._feature_names
    if names:
        preview = ", ".join(names[:3])
        suffix = ", …" if len(names) > 3 else ""
        feature_cell = f"{n}  ({preview}{suffix})"
        schema_kind = "named"
    else:
        feature_cell = str(n)
        schema_kind = "positional"

    # ── Grid table (two-column, no borders) ─────────────────────────────────
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="bold dim", no_wrap=True, min_width=14)
    grid.add_column()

    # Model metadata section
    grid.add_row("Model", Text(type(api._model).__name__, style="bold cyan"))
    grid.add_row("Source", Text(str(model_path), style="dim"))
    grid.add_row("Features", feature_cell)
    grid.add_row("Schema", Text(schema_kind, style="green"))

    # Separator
    grid.add_row("", "")
    grid.add_row("", Rule(style="dim"))
    grid.add_row("", "")

    # URLs section
    docs_url = f"{base_url}/docs"
    grid.add_row(
        "Swagger UI",
        Text(docs_url, style=f"underline bright_cyan link {docs_url}"),
    )
    grid.add_row(
        "ReDoc",
        Text(f"{base_url}/redoc", style="dim"),
    )
    grid.add_row(
        "Predict",
        Text(f"POST  {base_url}/predict", style="bold green"),
    )
    grid.add_row(
        "Health",
        Text(f"GET   {base_url}/health", style="bold green"),
    )

    return Panel(
        Padding(grid, pad=(1, 1)),
        title=Text.assemble(
            ("✦ ", "yellow"),
            ("bt-flow", "bold white"),
            (f"  v{__version__}", "dim"),
        ),
        subtitle=Text(" Press Ctrl+C to quit ", style="dim"),
        border_style="bright_blue",
        expand=False,
    )


def _error_panel(message: str, hint: str | None = None) -> Panel:
    """Build a Rich error panel for display on stderr.

    Args:
        message: The primary error description.
        hint: An optional actionable suggestion shown below the message.

    Returns:
        A Rich ``Panel`` styled for errors.
    """
    body = Text(message, style="bold")
    if hint:
        body.append(f"\n\n  Hint: {hint}", style="yellow dim")

    return Panel(
        Padding(body, pad=(1, 1)),
        title=Text(" Error ", style="bold red"),
        border_style="red",
        expand=False,
    )


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def serve(
    model_path: Path = typer.Argument(
        ...,
        help="Path to the serialised scikit-learn model (.pkl or .joblib).",
        show_default=False,
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-H",
        help="Network interface to bind to.",
        show_default=True,
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        min=1,
        max=65535,
        help="TCP port to listen on.",
        show_default=True,
    ),
    title: str = typer.Option(
        "bt-flow Model API",
        "--title",
        "-t",
        help="Title displayed in the Swagger UI.",
        show_default=True,
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        "-l",
        help="Uvicorn log level: debug | info | warning | error | critical.",
        show_default=True,
    ),
    feature_names: str | None = typer.Option(
        None,
        "--feature-names",
        "-f",
        help=(
            "Comma-separated feature names. Overrides names inferred from the model. "
            "Example: --feature-names 'age,income,score'"
        ),
    ),
) -> None:
    """Serve a scikit-learn model as a FastAPI REST API.

    \b
    Examples:
        bt-flow serve model.pkl
        bt-flow serve model.joblib --port 9000 --host 127.0.0.1
        bt-flow serve model.pkl --feature-names 'petal_len,petal_wid'
        bt-flow serve model.pkl --title "Iris API" --log-level debug
    """
    # Create console instances inside the command so that Typer's CliRunner
    # (which redirects sys.stdout before invoking commands) captures all output.
    console = Console(highlight=False)
    err_console = Console(stderr=True, highlight=False)

    # ── Parse --feature-names ────────────────────────────────────────────────
    parsed_names: list[str] | None = None
    if feature_names is not None:
        parsed_names = [n.strip() for n in feature_names.split(",") if n.strip()]
        if not parsed_names:
            err_console.print(
                _error_panel(
                    "The --feature-names option produced an empty list.",
                    hint="Use a comma-separated list, e.g. --feature-names 'age,income,score'",
                )
            )
            raise typer.Exit(code=1)

    # ── Load model & build API ───────────────────────────────────────────────
    try:
        api = APIGenerator(
            model_path,
            title=title,
            feature_names=parsed_names,
        )
    except BTFlowError as exc:
        err_console.print(_error_panel(str(exc)))
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover — safety net for unexpected errors
        err_console.print(
            _error_panel(
                f"Unexpected error while loading model: {exc}",
                hint="Run with --log-level debug for more detail.",
            )
        )
        raise typer.Exit(code=1) from exc

    # ── Print startup panel ──────────────────────────────────────────────────
    console.print()
    console.print(_startup_panel(api, host=host, port=port, model_path=model_path))
    console.print()

    # ── Start server (blocking) ──────────────────────────────────────────────
    try:
        api.run(host=host, port=port, log_level=log_level)
    except KeyboardInterrupt:
        console.print("[dim]\nServer stopped.[/dim]")
        raise typer.Exit(code=0) from None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point registered as the ``bt-flow`` console script.

    Declared in ``pyproject.toml`` as::

        [project.scripts]
        bt-flow = "bt.cli:main"
    """
    app()


if __name__ == "__main__":
    main()
