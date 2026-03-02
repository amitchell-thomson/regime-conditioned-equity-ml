"""Tests for configure_pipeline_logging() utility (Phase 6.4)."""

import logging
import re


from regime_ml.utils.logging import configure_pipeline_logging


class TestConfigurePipelineLogging:
    def test_creates_log_file_in_directory(self, tmp_path):
        """Calling configure_pipeline_logging creates a .log file in log_dir."""
        configure_pipeline_logging(tmp_path)
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1, f"Expected 1 log file, found {log_files}"

    def test_log_filename_matches_timestamp_pattern(self, tmp_path):
        """Log filename must match pipeline_YYYYMMDD_HHMMSS.log."""
        configure_pipeline_logging(tmp_path)
        log_files = list(tmp_path.glob("*.log"))
        name = log_files[0].name
        assert re.match(r"pipeline_\d{8}_\d{6}\.log", name), f"Log filename '{name}' does not match expected pattern"

    def test_log_file_contains_emitted_messages(self, tmp_path):
        """Messages logged to regime_ml are written to the file."""
        configure_pipeline_logging(tmp_path)
        test_logger = logging.getLogger("regime_ml.test_pipeline_logging_marker")
        test_logger.setLevel(logging.DEBUG)
        test_logger.info("sentinel message for test_pipeline_logging")

        log_file = list(tmp_path.glob("*.log"))[0]
        content = log_file.read_text(encoding="utf-8")
        assert "sentinel message for test_pipeline_logging" in content

    def test_creates_log_dir_if_missing(self, tmp_path):
        """configure_pipeline_logging creates the log_dir if it does not exist."""
        new_dir = tmp_path / "nested" / "logs"
        assert not new_dir.exists()
        configure_pipeline_logging(new_dir)
        assert new_dir.exists()

    def test_does_not_add_handler_to_root_logger(self, tmp_path):
        """Root logger handler count must not increase after configure_pipeline_logging."""
        root_handler_count_before = len(logging.getLogger().handlers)
        configure_pipeline_logging(tmp_path)
        root_handler_count_after = len(logging.getLogger().handlers)
        assert (
            root_handler_count_after == root_handler_count_before
        ), "configure_pipeline_logging must not add handlers to the root logger"

    def teardown_method(self, method):
        """Remove any FileHandlers added to the regime_ml logger during a test."""
        pkg_logger = logging.getLogger("regime_ml")
        pkg_logger.handlers = [h for h in pkg_logger.handlers if not isinstance(h, logging.FileHandler)]
