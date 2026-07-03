"""
Timing helpers for performance logging.

Usage:
    from src.utils.timing import timed

    with timed("onnx_inference") as t:
        result = session.run(...)
    logger.info("inference took %.1f ms", t.ms)
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    name: str
    ms: float = 0.0


@contextmanager
def timed(name: str):
    """Context manager that measures wall-clock time in milliseconds."""
    t = Timer(name=name)
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.ms = (time.perf_counter() - start) * 1000
