"""Tests for logging configuration."""

import json
import logging
from unittest.mock import patch, MagicMock

import pytest

from app.logging_config import JSONFormatter, setup_logging


def test_json_formatter_normal_record():
    """JSONFormatter produces valid JSON with level, logger, message."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="app.main",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    result = formatter.format(record)
    data = json.loads(result)
    assert data["level"] == "INFO"
    assert data["logger"] == "app.main"
    assert data["message"] == "Test message"
    assert "timestamp" in data


def test_json_formatter_with_exc_info():
    """JSONFormatter includes exception when exc_info is set."""
    import sys

    formatter = JSONFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Failed",
            args=(),
            exc_info=sys.exc_info(),
        )
        result = formatter.format(record)
        data = json.loads(result)
        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "test error" in data["exception"]


def test_json_formatter_with_extra_fields():
    """JSONFormatter includes query, chunk_ids, scores, latency_ms, token_usage."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="app.main",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Query processed",
        args=(),
        exc_info=None,
    )
    record.query = "What is DGESV?"
    record.chunk_ids = ["c1", "c2"]
    record.scores = [0.9, 0.8]
    record.latency_ms = 150.5
    record.token_usage = {"prompt_tokens": 100}

    result = formatter.format(record)
    data = json.loads(result)
    assert data["query"] == "What is DGESV?"
    assert data["chunk_ids"] == ["c1", "c2"]
    assert data["scores"] == [0.9, 0.8]
    assert data["latency_ms"] == 150.5
    assert data["token_usage"] == {"prompt_tokens": 100}


def test_setup_logging_configures_root_logger():
    """setup_logging configures root logger with console handler."""
    setup_logging()
    root = logging.getLogger()
    assert root.level == logging.INFO
    handlers = root.handlers
    assert len(handlers) >= 1
    # At least one StreamHandler for console
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)


@patch("app.logging_config.Path")
def test_setup_logging_oserror_disables_file_handler(mock_path):
    """When logs/ cannot be created, file handler is skipped."""
    log_dir_mock = MagicMock()
    log_dir_mock.mkdir.side_effect = OSError("Permission denied")
    mock_path.return_value = log_dir_mock

    setup_logging()
    root = logging.getLogger()
    assert len(root.handlers) >= 1
