"""Logging configuration for the METIS framework."""

import logging
import sys
import warnings
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
    suppress_warnings: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        format_string: Custom format string for log messages
        suppress_warnings: Whether to suppress common noisy warnings
    """
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Clear existing handlers to ensure file handler is added
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure root logger
    root_logger.setLevel(numeric_level)

    # Create and add handlers
    formatter = logging.Formatter(format_string)
    for handler in _create_handlers(log_file):
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # set specific logger levels to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("multiprocessing").setLevel(logging.WARNING)

    # Suppress common noisy warnings
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", message=".*could represent a regression problem.*")
        warnings.filterwarnings("ignore", message=".*Series.view.*")
        warnings.filterwarnings("ignore", message=".*Large number of metrics.*")


def _create_handlers(log_file: str | None) -> list:
    """Create logging handlers for console and optional file output."""
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    return handlers


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Configure default logging on import
setup_logging()
