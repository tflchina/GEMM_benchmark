from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkResult:
    backend: str
    m: int
    n: int
    k: int
    dtype: str
    device: str
    avg_ms: float
    min_ms: float
    max_ms: float
    tflops: float


class GemmBackend(ABC):
    """Interface for GEMM backends."""

    name: str
    supported_dtypes: tuple[str, ...]

    @abstractmethod
    def benchmark(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str,
        warmup: int,
        iters: int,
        seed: Optional[int] = None,
    ) -> BenchmarkResult:
        raise NotImplementedError
