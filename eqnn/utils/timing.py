# eqnn/utils/timing.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class RuntimeProfile:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, seconds: float) -> None:
        self.totals[key] = self.totals.get(key, 0.0) + float(seconds)
        self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self) -> dict[str, dict[str, float | int]]:
        keys = sorted(self.totals)
        return {
            key: {
                "total_seconds": self.totals[key],
                "count": self.counts.get(key, 0),
                "mean_seconds": self.totals[key] / max(self.counts.get(key, 1), 1),
            }
            for key in keys
        }


@contextmanager
def timed(profile: RuntimeProfile | None, key: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        if profile is not None:
            profile.add(key, perf_counter() - start)