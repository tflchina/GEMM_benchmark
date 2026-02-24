from __future__ import annotations

import time
from typing import Optional

import torch
import triton
import triton.language as tl

from .base import BenchmarkResult, GemmBackend


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        a_mask = (offs_m[:, None] < m) & (offs_k[None, :] + k_start < k)
        b_mask = (offs_k[:, None] + k_start < k) & (offs_n[None, :] < n)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)


class TritonBackend(GemmBackend):
    name = "triton"
    supported_dtypes = ("float16", "float32", "bfloat16")

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
            raise RuntimeError("Triton backend requires a CUDA-enabled GPU.")
        torch_dtype = _to_torch_dtype(dtype)

        if seed is not None:
            torch.manual_seed(seed)

        a = torch.randn((m, k), device="cuda", dtype=torch_dtype)
        b = torch.randn((k, n), device="cuda", dtype=torch_dtype)
        c = torch.empty((m, n), device="cuda", dtype=torch.float32)

        grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))

        def _launch() -> None:
            _matmul_kernel[grid](
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_M=64,
                BLOCK_N=64,
                BLOCK_K=32,
            )

        for _ in range(warmup):
            _launch()
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            _launch()
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)

        avg_ms = sum(times_ms) / len(times_ms)
        tflops = (2.0 * m * n * k) / (avg_ms / 1000.0) / 1e12
        return BenchmarkResult(
            backend=self.name,
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            device="cuda",
            avg_ms=avg_ms,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            tflops=tflops,
        )


def _to_torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for Triton backend: {dtype}") from exc
