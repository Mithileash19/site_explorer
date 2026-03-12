"""
core/logging.py — Unified structured stdout logger.
"""
import logging
import sys


def _configure() -> None:
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Silence noisy uvicorn access log; keep error/warning
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


_configure()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
