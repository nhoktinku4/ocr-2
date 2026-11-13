"""Application logging with daily rotation and request context."""

import datetime
import logging
import os
import re
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


request_id_context = ContextVar("ocr_request_id", default="")


class MicrosecondFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            "%(asctime)s [%(levelname)s] [%(filename_only)s:%(lineno)d] "
            "[idRequest: %(id)s] - Message: %(message)s"
        )

    def formatTime(self, record, datefmt=None):
        return datetime.datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )

    def format(self, record):
        record.filename_only = Path(record.filename).stem
        if not hasattr(record, "id"):
            record.id = request_id_context.get()
        # Standard formatting also preserves exception tracebacks.
        return super().format(record)


def create_logger(logfile="ocr.log"):
    """Configure the shared OCR logger; environment settings override defaults."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    if level_name not in levels:
        raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
    backup_count = int(os.environ.get("LOG_BACKUP_COUNT", "5"))
    if backup_count < 1:
        raise ValueError("LOG_BACKUP_COUNT must be at least 1")

    log_dir = Path(os.environ.get(
        "LOG_DIR", str(Path(__file__).resolve().parent / "logs")
    ))
    log_path = log_dir / logfile
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = MicrosecondFormatter()
    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y%m%d"
    # Retention must match the custom suffix instead of the default YYYY-MM-DD.
    file_handler.extMatch = re.compile(r"^\d{8}(\.\w+)?$", re.ASCII)
    console_handler = logging.StreamHandler()

    app_logger = logging.getLogger("ocr_app")
    app_logger.setLevel(levels[level_name])
    app_logger.propagate = False
    # Reconfiguration/reload must not duplicate console or file output.
    for handler in list(app_logger.handlers):
        if getattr(handler, "_ocr_handler", False):
            app_logger.removeHandler(handler)
            handler.close()
    for handler in (console_handler, file_handler):
        handler.setLevel(levels[level_name])
        handler.setFormatter(formatter)
        handler._ocr_handler = True
        app_logger.addHandler(handler)
    return app_logger


logger = create_logger()
 