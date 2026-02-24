from __future__ import annotations

import time
from typing import Optional

import cupy as cp

from .base import BenchmarkResult, GemmBackend


class CuBLASBackend(GemmBackend):
    name = "cublas"
    supported_dtypes = (
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int8",
        "fp8_e4m3fn",
        "fp8_e5m2",
        "fp4_e2m1",
    )

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
        if seed is not None:
            cp.random.seed(seed)

        a, b = _create_inputs(m, n, k, dtype)

        for _ in range(warmup):
            _ = cp.matmul(a, b)
        cp.cuda.Stream.null.synchronize()

        times_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)

        avg_ms = sum(times_ms) / len(times_ms)
        tflops = (2.0 * m * n * k) / (avg_ms / 1000.0) / 1e12
        return BenchmarkResult(
            backend=self.name,
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            device=f"cuda:{cp.cuda.Device().id}",
            avg_ms=avg_ms,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            tflops=tflops,
        )


def _create_inputs(m: int, n: int, k: int, dtype: str):
    if dtype == "int8":
        a = cp.random.randint(-128, 127, size=(m, k), dtype=cp.int8)
        b = cp.random.randint(-128, 127, size=(k, n), dtype=cp.int8)
        return a, b

    if dtype == "fp4_e2m1":
        # Emulated FP4 values in float16 tensors.
        a = _quantize_fp4(cp.random.standard_normal((m, k), dtype=cp.float16))
        b = _quantize_fp4(cp.random.standard_normal((k, n), dtype=cp.float16))
        return a, b

    cp_dtype = _to_cupy_dtype(dtype)
    a = cp.random.standard_normal((m, k), dtype=cp_dtype)
    b = cp.random.standard_normal((k, n), dtype=cp_dtype)
    return a, b


def _quantize_fp4(x):
    abs_x = cp.abs(x)
    eps = cp.array(1e-6, dtype=x.dtype)
    exp = cp.floor(cp.log2(cp.maximum(abs_x, eps)))
    exp = cp.clip(exp, -2, 1)
    base = cp.power(cp.array(2.0, dtype=x.dtype), exp)
    mant = cp.where(abs_x / base >= 1.5, cp.array(1.5, dtype=x.dtype), cp.array(1.0, dtype=x.dtype))
    return cp.sign(x) * base * mant


def _to_cupy_dtype(dtype: str):
    mapping = {
        "float16": cp.float16,
        "float32": cp.float32,
        "float64": cp.float64,
        "int8": cp.int8,
    }

    special_names = {
        "bfloat16": "bfloat16",
        "fp8_e4m3fn": "float8_e4m3fn",
        "fp8_e5m2": "float8_e5m2",
    }
    if dtype in special_names:
        try:
            return cp.dtype(special_names[dtype])
        except TypeError as exc:
            raise ValueError(f"This CuPy build does not support {dtype}.") from exc

    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for cuBLAS backend: {dtype}") from exc
