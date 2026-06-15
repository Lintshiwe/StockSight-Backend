from __future__ import annotations

from collections import deque
from time import perf_counter


class FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self.times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self.times.append(perf_counter())
        if len(self.times) < 2:
            return 0.0
        elapsed = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0

