"""Shared fixtures for regime_ml test suite."""

import logging
import pytest


@pytest.fixture(autouse=True)
def configure_logging(caplog):
    with caplog.at_level(logging.WARNING, logger="regime_ml"):
        yield caplog
