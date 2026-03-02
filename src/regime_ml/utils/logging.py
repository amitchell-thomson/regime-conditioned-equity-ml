"""Pipeline logging utility.

Call configure_pipeline_logging() once at the start of a pipeline run to attach
a timestamped file handler to the regime_ml logger. Each call creates a new
log file — multiple calls in the same process add multiple handlers.

Do NOT call this in tests. Use pytest's caplog or logging.getLogger() directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def configure_pipeline_logging(
    log_dir: Path,
    level: int = logging.DEBUG,
) -> None:
    """Attach a timestamped file handler to the regime_ml package logger.

    Creates log_dir if it does not exist. The log file is named
    pipeline_YYYYMMDD_HHMMSS.log. The handler is attached to
    logging.getLogger("regime_ml") only — the root logger is not affected,
    so test isolation is preserved.

    Args:
        log_dir: Directory in which to create the log file.
        level:   Minimum log level for the file handler. Defaults to DEBUG
                 so all pipeline output is captured regardless of the
                 console handler level set by the caller.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_{ts}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    pkg_logger = logging.getLogger("regime_ml")
    pkg_logger.addHandler(handler)
    pkg_logger.info("configure_pipeline_logging: writing to %s", log_path)
