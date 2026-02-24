from __future__ import annotations

import time
from typing import Optional

import torch

from .base import BenchmarkResult, GemmBackend


class PyTorchBackend(GemmBackend):
    name = "pytorch"
    supported_dtypes = (
        "float16",
        "float32",
        "float64",
        "bfloat16",
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
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch backend requires a CUDA-enabled GPU.")

        device = torch.device("cuda")
        if seed is not None:
            torch.manual_seed(seed)

        a, b = _create_inputs(m, n, k, dtype, device)

        for _ in range(warmup):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)

        return _build_result(self.name, m, n, k, dtype, "cuda", times_ms)


def _create_inputs(m: int, n: int, k: int, dtype: str, device: torch.device):
    if dtype == "fp4_e2m1":
        # PyTorch has no native FP4 tensor type yet; emulate by quantizing to FP4 levels
        # and storing in FP16 tensors for kernel execution.
        a = _quantize_fp4(torch.randn((m, k), device=device, dtype=torch.float16))
        b = _quantize_fp4(torch.randn((k, n), device=device, dtype=torch.float16))
        return a, b

    torch_dtype = _to_torch_dtype(dtype)
    a = torch.randn((m, k), device=device, dtype=torch_dtype)
    b = torch.randn((k, n), device=device, dtype=torch_dtype)
    return a, b


def _quantize_fp4(x: torch.Tensor) -> torch.Tensor:
    # Very small FP4-like emulation: sign + 2-bit exponent + 1-bit mantissa.
    # This is approximate and intended for benchmarking workflow compatibility.
    abs_x = x.abs()
    sign = torch.sign(x)
    eps = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    exp = torch.floor(torch.log2(torch.maximum(abs_x, eps)))
    exp = torch.clamp(exp, -2, 1)
    base = torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), exp)
    mant = torch.where(abs_x / base >= 1.5, torch.tensor(1.5, device=x.device, dtype=x.dtype), 1.0)
    q = sign * base * mant
    return q.to(torch.float16)


def _to_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp8_e4m3fn":
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
        raise ValueError("This PyTorch build does not support torch.float8_e4m3fn.")
    if dtype == "fp8_e5m2":
        if hasattr(torch, "float8_e5m2"):
            return torch.float8_e5m2
        raise ValueError("This PyTorch build does not support torch.float8_e5m2.")

    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for PyTorch backend: {dtype}") from exc


def _build_result(
    backend: str,
    m: int,
    n: int,
    k: int,
    dtype: str,
    device: str,
    times_ms: list[float],
) -> BenchmarkResult:
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    tflops = (2.0 * m * n * k) / (avg_ms / 1000.0) / 1e12
    return BenchmarkResult(
        backend=backend,
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        device=device,
        avg_ms=avg_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        tflops=tflops,
    )
