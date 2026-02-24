from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backends.base import BenchmarkResult, GemmBackend

DTYPE_CHOICES = ["float16", "float32", "float64", "bfloat16", "int8", "fp8_e4m3fn", "fp8_e5m2", "fp4_e2m1"]


@dataclass
class GemmShape:
    m: int
    n: int
    k: int
    dtype: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GEMM with multiple backends.")
    parser.add_argument("--m", type=int, help="Rows of A / C.")
    parser.add_argument("--n", type=int, help="Cols of B / C.")
    parser.add_argument("--k", type=int, help="Cols of A / rows of B.")
    parser.add_argument(
        "--shape-file",
        type=Path,
        help=(
            "Path to JSON file containing shapes, e.g. "
            "{'shapes': [{'m': 1024, 'n': 1024, 'k': 1024, 'dtype': 'float16'}]}"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "cublas", "triton", "cuda-kernel", "all"],
        default="all",
        help="GEMM backend to benchmark.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=DTYPE_CHOICES,
        help="Default dtype when not provided per-shape in the JSON file.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    return parser.parse_args()


def _load_shapes(args: argparse.Namespace) -> list[GemmShape]:
    if args.shape_file:
        payload = json.loads(args.shape_file.read_text())
        raw_shapes = payload.get("shapes", [])
        if not raw_shapes:
            raise ValueError("Shape file is missing 'shapes' entries.")
        return [GemmShape(**entry) for entry in raw_shapes]

    if args.m is None or args.n is None or args.k is None:
        raise ValueError("Provide either --shape-file or all of --m --n --k.")
    return [GemmShape(m=args.m, n=args.n, k=args.k)]


def _load_backends(requested: str) -> list[GemmBackend]:
    backend_names = [requested] if requested != "all" else ["pytorch", "cublas", "triton", "cuda-kernel"]
    resolved: list[GemmBackend] = []

    for name in backend_names:
        try:
            if name == "pytorch":
                from backends.pytorch_backend import PyTorchBackend

                resolved.append(PyTorchBackend())
            elif name == "cublas":
                from backends.cublas_backend import CuBLASBackend

                resolved.append(CuBLASBackend())
            elif name == "triton":
                from backends.triton_backend import TritonBackend

                resolved.append(TritonBackend())
            elif name == "cuda-kernel":
                from backends.cuda_kernel_backend import ManualCUDAKernelBackend

                resolved.append(ManualCUDAKernelBackend())
            else:
                raise ValueError(f"Unknown backend: {name}")
        except Exception as exc:
            print(f"[WARN] backend={name} unavailable: {exc}")

    return resolved


def _print_results(results: Iterable[BenchmarkResult]) -> None:
    print(
        "backend\tm\tn\tk\tdtype\tdevice\tavg_ms\tmin_ms\tmax_ms\ttflops",
    )
    for res in results:
        print(
            f"{res.backend}\t{res.m}\t{res.n}\t{res.k}\t{res.dtype}\t{res.device}\t"
            f"{res.avg_ms:.4f}\t{res.min_ms:.4f}\t{res.max_ms:.4f}\t{res.tflops:.4f}"
        )


def main() -> None:
    args = _parse_args()
    shapes = _load_shapes(args)
    backends = _load_backends(args.backend)
    if not backends:
        raise RuntimeError("No backends are available to run.")

    results: list[BenchmarkResult] = []
    for shape in shapes:
        shape_dtype = shape.dtype or args.dtype
        for backend in backends:
            if shape_dtype not in backend.supported_dtypes:
                print(
                    f"[WARN] backend={backend.name} shape=({shape.m},{shape.n},{shape.k}) "
                    f"skipped: dtype={shape_dtype} not in {backend.supported_dtypes}"
                )
                continue
            try:
                result = backend.benchmark(
                    m=shape.m,
                    n=shape.n,
                    k=shape.k,
                    dtype=shape_dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    seed=args.seed,
                )
                results.append(result)
            except Exception as exc:
                print(
                    f"[WARN] backend={backend.name} shape=({shape.m},{shape.n},{shape.k}) "
                    f"dtype={shape_dtype} skipped: {exc}"
                )

    _print_results(results)


if __name__ == "__main__":
    main()
