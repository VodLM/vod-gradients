import math


def linear_warmup(step: float, period: float, start: float, end: float) -> float:
    """Linearly warm up from `start` to `end` over `period` steps."""
    t = max(0.0, min(step / period, 1.0))
    return start + t * (end - start)

def cosine_warmup(step: float, period: float, start: float, end: float) -> float:
    """Cosine warm up from `start` to `end` over `period` steps."""
    u = max(0.0, min(step / period, 1.0))
    t = 0.5 * (1 - math.cos(math.pi * u))
    return start + t * (end - start)
