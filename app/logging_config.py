"""Structured JSON logging with rotating file handler."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path


_STANDARD_LOG_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
)


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and key not in log_entry:
                log_entry[key] = value
        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO"):
    """Configure root logger with console + rotating JSON file handler."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # Console — human-readable
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # File — structured JSON, rotating 5 MB x 3 backups (skip if not writable)
    log_dir = Path("logs")
    try:
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "legacylens.jsonl",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not create logs/ directory — file logging disabled")
