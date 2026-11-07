"""
Multi-target Rotating Logging System.
Configures dedicated log files with structured formatting and secret sanitization.
"""

import os
import re
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import settings

# Regex patterns to sanitize sensitive tokens/passwords if ever accidentally printed
SENSITIVE_PATTERNS = [
    re.compile(r'(api_key|token|secret|password)=([^\s&]+)', re.IGNORECASE),
    re.compile(r'Bearer\s+([A-Za-z0-9\-\._~\+\/]+=*)', re.IGNORECASE)
]

class SanitizedFormatter(logging.Formatter):
    """Custom formatter that removes sensitive patterns from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        orig = super().format(record)
        sanitized = orig
        for pattern in SENSITIVE_PATTERNS:
            sanitized = pattern.sub(r'\1=***REDACTED***', sanitized)
        return sanitized

def _build_handler(file_path: Path, level: int = logging.INFO) -> RotatingFileHandler:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024, # 10 MB per log file
        backupCount=5,
        encoding="utf-8"
    )
    handler.setLevel(level)
    formatter = SanitizedFormatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    return handler

# Root application logger setup
_INITIALIZED = False

def setup_loggers():
    global _INITIALIZED
    if _INITIALIZED:
        return
    
    log_dir = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        SanitizedFormatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(console_handler)

    # Global application log handler
    root.addHandler(_build_handler(log_dir / "application.log", logging.INFO))

    # Dedicated Errors log handler
    error_handler = _build_handler(log_dir / "errors.log", logging.ERROR)
    root.addHandler(error_handler)

    _INITIALIZED = True

setup_loggers()

def get_logger(name: str, specific_log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a named logger. If specific_log_file is provided, attaches an additional dedicated
    rotating file handler (e.g. training.log, inference.log, hardware.log, etc.).
    """
    logger = logging.getLogger(name)
    if specific_log_file:
        file_path = settings.log_dir / specific_log_file
        # Check if already added to avoid duplicate handlers
        for h in logger.handlers:
            if isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == file_path.resolve():
                return logger
        handler = _build_handler(file_path, logging.INFO)
        logger.addHandler(handler)
    return logger

# Predefined loggers for primary system components
app_logger = get_logger("app", "application.log")
train_logger = get_logger("training", "training.log")
training_error_logger = get_logger("training_errors", "training_errors.log")
inference_logger = get_logger("inference", "inference.log")
hardware_logger = get_logger("hardware", "hardware.log")
diagnostics_logger = get_logger("diagnostics", "agent_diagnostics.log")
experiment_logger = get_logger("experiments", "experiments.log")

