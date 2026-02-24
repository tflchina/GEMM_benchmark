from __future__ import annotations

import time
from typing import Optional

import cupy as cp

from .base import BenchmarkResult, GemmBackend


_KERNEL = r"""
#include <cuda_fp16.h>

extern "C" __global__
void matmul_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            acc += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = acc;
    }
}

extern "C" __global__
void matmul_f16(const half* A, const half* B, half* C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            acc += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
        }
        C[row * N + col] = __float2half(acc);
    }
}
"""


class ManualCUDAKernelBackend(GemmBackend):
    name = "cuda-kernel"
    supported_dtypes = ("float16", "float32")

    def __init__(self) -> None:
        self._kernels = {
            "float16": cp.RawKernel(_KERNEL, "matmul_f16"),
            "float32": cp.RawKernel(_KERNEL, "matmul_f32"),
        }

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
        if dtype not in self.supported_dtypes:
            raise ValueError(f"Manual CUDA kernel backend supports only {self.supported_dtypes}.")

        if seed is not None:
            cp.random.seed(seed)

        cp_dtype = cp.float16 if dtype == "float16" else cp.float32
        a = cp.random.standard_normal((m, k), dtype=cp_dtype)
        b = cp.random.standard_normal((k, n), dtype=cp_dtype)
        c = cp.zeros((m, n), dtype=cp_dtype)

        kernel = self._kernels[dtype]
        block = (16, 16)
        grid = ((n + block[0] - 1) // block[0], (m + block[1] - 1) // block[1])

        for _ in range(warmup):
            kernel(grid, block, (a, b, c, m, n, k))
        cp.cuda.Stream.null.synchronize()

        times_ms = []
        for _ in range(iters):
            start = time.perf_counter()
            kernel(grid, block, (a, b, c, m, n, k))
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
