"""Shared fixtures for regime_ml test suite."""

import logging
import pytest

# Suppress verbose third-party output that clutters test logs.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


@pytest.fixture(autouse=True)
def configure_logging(caplog):
    with caplog.at_level(logging.WARNING, logger="regime_ml"):
        yield caplog
