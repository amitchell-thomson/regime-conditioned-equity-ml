"""
CLI entry point for regime-conditioned equity ML pipeline.

Usage:
    regime-ml --help
    regime-ml data [--log-level LEVEL]
    regime-ml features [--log-level LEVEL]
    regime-ml regime [--log-level LEVEL] [--log-dir DIR]
    regime-ml all [--log-level LEVEL]        # run data → features → regime in sequence
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Callable

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

console = Console()

_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

_PIPELINE_REGISTRY: dict[str, tuple[str, Callable]] = {}


def _register_pipelines() -> None:
    """Lazily import and register pipeline callables to avoid slow imports at parse time."""
    from regime_ml.data.macro.pipeline import run_macro_data_pipeline
    from regime_ml.features.macro.pipeline import run_macro_feature_pipeline
    from regime_ml.regimes.pipeline import run_regime_pipeline

    _PIPELINE_REGISTRY["data"] = ("Macro data pipeline", run_macro_data_pipeline)
    _PIPELINE_REGISTRY["features"] = (
        "Macro feature pipeline",
        run_macro_feature_pipeline,
    )
    _PIPELINE_REGISTRY["regime"] = ("Regime detection pipeline", run_regime_pipeline)


def _setup_logging(level: str) -> None:
    level_int = getattr(logging, level.upper())
    logging.basicConfig(
        level=level_int,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console, rich_tracebacks=True, show_path=level == "DEBUG"
            )
        ],
        force=True,
    )
    # Silence noisy third-party loggers unless the user asked for DEBUG
    if level_int > logging.DEBUG:
        for noisy in ("hmmlearn", "matplotlib", "numexpr", "PIL"):
            logging.getLogger(noisy).setLevel(logging.ERROR)


def _run_pipeline(name: str, fn: Callable, log_level: str, **kwargs) -> int:
    """Run a single pipeline function, printing a header and timing info. Returns exit code."""
    console.print(Panel(Text(name, style="bold cyan"), expand=False))
    start = time.perf_counter()
    try:
        result = fn(**kwargs)
        elapsed = time.perf_counter() - start
        console.print(
            f"[green]✓[/green] {name} completed in [bold]{elapsed:.1f}s[/bold]"
        )
        if result is not None and hasattr(result, "shape"):
            console.print(f"  Output shape: [dim]{result.shape}[/dim]")
        return 0
    except Exception as exc:
        elapsed = time.perf_counter() - start
        console.print(f"[red]✗[/red] {name} failed after [bold]{elapsed:.1f}s[/bold]")
        logging.getLogger(__name__).exception("Pipeline error: %s", exc)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regime-ml",
        description="Run regime-conditioned equity ML pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    commands:
    data      Load, clean, align and validate raw macro data
    features  Build transform-chain features from processed macro data
    regime    Fit HMM grid, select best model, label regimes, validate episodes
    all       Run data → features → regime in sequence

    examples:
    regime-ml data
    regime-ml features --log-level DEBUG
    regime-ml regime
    regime-ml all --log-level WARNING
    """,
    )
    parser.add_argument(
        "command",
        choices=["data", "features", "regime", "all"],
        help="Pipeline to run",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=_LOG_LEVELS,
        metavar="LEVEL",
        help=f"Logging verbosity: {', '.join(_LOG_LEVELS)} (default: INFO)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        metavar="DIR",
        help="Write a timestamped pipeline log file to DIR (regime command only).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)
    _register_pipelines()

    if args.command == "all":
        sequence = ["data", "features", "regime"]
    else:
        sequence = [args.command]

    exit_code = 0
    for step in sequence:
        name, fn = _PIPELINE_REGISTRY[step]
        kwargs = {}
        if step == "regime" and args.log_dir is not None:
            from pathlib import Path

            kwargs["log_dir"] = Path(args.log_dir)
        code = _run_pipeline(name, fn, args.log_level, **kwargs)
        if code != 0:
            exit_code = code
            break  # stop on first failure

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
