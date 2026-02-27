#!/usr/bin/env python3
"""
Benchmark GEMM MFU (Model FLOP Utilization) on NVIDIA GPU.

- Reads a JSON config containing m,n,k (and optional fields).
- Runs torch matmul (GEMM) in FP8 where supported, else falls back to BF16/FP16.
- Measures time with CUDA events and/or torch.profiler.
- Reports achieved TFLOP/s and MFU vs a user-supplied peak TFLOP/s.

Usage:
  python bench_gemm_mfu.py --config gemm.json --peak-tflops 3958

Example gemm.json:
{
  "m": 9450,
  "n": 5120,
  "k": 5120,
  "dtype": "fp8",                // "fp8" | "bf16" | "fp16"
  "fp8_format": "e4m3fn",        // "e4m3fn" | "e5m2"
  "layout": "nn",                // "nn" | "nt" | "tn" | "tt"
  "warmup": 30,
  "iters": 200,
  "torch_compile": false,
  "tf32": false
}

Notes:
- FP8 in PyTorch requires recent PyTorch + CUDA and Hopper/Blackwell. If unavailable, we fall back.
- Peak TFLOP/s is NOT auto-detected; pass --peak-tflops for your GPU + dtype.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


DTypeStr = Literal["fp8", "bf16", "fp16"]


@dataclass
class GemmConfig:
    m: int
    n: int
    k: int
    dtype: DTypeStr = "fp8"
    fp8_format: str = "e4m3fn"
    layout: str = "nn"  # "nn", "nt", "tn", "tt"
    warmup: int = 20
    iters: int = 100
    torch_compile: bool = False
    tf32: bool = False
    seed: int = 0


def _load_config(path: str) -> GemmConfig:
    with open(path, "r") as f:
        d = json.load(f)

    def get(name, default):
        return d[name] if name in d else default

    cfg = GemmConfig(
        m=int(d["m"]),
        n=int(d["n"]),
        k=int(d["k"]),
        dtype=str(get("dtype", "fp8")).lower(),  # type: ignore
        fp8_format=str(get("fp8_format", "e4m3fn")).lower(),
        layout=str(get("layout", "nn")).lower(),
        warmup=int(get("warmup", 20)),
        iters=int(get("iters", 100)),
        torch_compile=bool(get("torch_compile", False)),
        tf32=bool(get("tf32", False)),
        seed=int(get("seed", 0)),
    )
    if cfg.layout not in ("nn", "nt", "tn", "tt"):
        raise ValueError(f"Unsupported layout: {cfg.layout}")
    if cfg.dtype not in ("fp8", "bf16", "fp16"):
        raise ValueError(f"Unsupported dtype: {cfg.dtype}")
    if cfg.fp8_format not in ("e4m3fn", "e5m2"):
        raise ValueError(f"Unsupported fp8_format: {cfg.fp8_format}")
    return cfg


def _pick_fp8_dtype(fmt: str):
    # PyTorch uses torch.float8_e4m3fn and torch.float8_e5m2 on supported builds.
    if not hasattr(torch, "float8_e4m3fn"):
        return None
    if fmt == "e4m3fn":
        return torch.float8_e4m3fn
    if fmt == "e5m2":
        return torch.float8_e5m2
    return None


def _make_matrices(cfg: GemmConfig, device: torch.device):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Shapes depend on layout:
    # A: (m,k) or (k,m) if transposed in compute? We'll keep physical A always (m,k)
    # and then use transpose views for NT/TN/TT.
    A_shape = (cfg.m, cfg.k)
    B_shape = (cfg.k, cfg.n)

    if cfg.dtype == "fp8":
        fp8dtype = _pick_fp8_dtype(cfg.fp8_format)
        if fp8dtype is None:
            raise RuntimeError("FP8 dtype not available in this PyTorch build.")
        # Create in FP16 then cast down to FP8 for more realistic range.
        A = (torch.randn(*A_shape, device=device, dtype=torch.float16) * 0.5).to(fp8dtype)
        B = (torch.randn(*B_shape, device=device, dtype=torch.float16) * 0.5).to(fp8dtype)
        # Output/accum: use FP16 by default (matmul will typically accumulate in FP16/FP32 internally)
        out_dtype = torch.float16
    elif cfg.dtype == "bf16":
        A = torch.randn(*A_shape, device=device, dtype=torch.bfloat16)
        B = torch.randn(*B_shape, device=device, dtype=torch.bfloat16)
        out_dtype = torch.bfloat16
    else:  # fp16
        A = torch.randn(*A_shape, device=device, dtype=torch.float16)
        B = torch.randn(*B_shape, device=device, dtype=torch.float16)
        out_dtype = torch.float16

    # Apply layout as transpose *views*.
    A_mat = A.t() if cfg.layout[0] == "t" else A
    B_mat = B.t() if cfg.layout[1] == "t" else B

    # Ensure contiguous if requested; often better to keep contiguous, but for benchmarking
    # you may want to model your real layout. We'll make them contiguous to reduce noise.
    A_mat = A_mat.contiguous()
    B_mat = B_mat.contiguous()

    # Create output to avoid allocation in loop.
    C = torch.empty((cfg.m, cfg.n), device=device, dtype=out_dtype)
    return A_mat, B_mat, C


def _gemm(A, B, C):
    # Use out= to avoid allocating each iteration.
    # torch.matmul supports out for 2D inputs via torch.mm / torch.matmul? torch.matmul does not accept out
    # reliably; use torch.mm for 2D.
    return torch.mm(A, B, out=C)


def _benchmark(cfg: GemmConfig, peak_tflops: float) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Need an NVIDIA GPU + CUDA runtime.")

    device = torch.device("cuda")

    # TF32 controls (matters for fp32, not fp16/bf16/fp8, but keep configurable)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)

    # Optional: enable best matmul algo
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    A, B, C = _make_matrices(cfg, device)

    fn = _gemm
    if cfg.torch_compile:
        # Requires PyTorch 2.x
        fn = torch.compile(fn, mode="max-autotune")

    # Warmup (also triggers autotune/caching)
    torch.cuda.synchronize()
    for _ in range(cfg.warmup):
        fn(A, B, C)
    torch.cuda.synchronize()

    # Timed loop with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(cfg.iters):
        fn(A, B, C)
    end.record()
    torch.cuda.synchronize()

    ms_total = start.elapsed_time(end)  # milliseconds
    ms_per = ms_total / cfg.iters

    # GEMM FLOPs: 2*M*N*K
    flops = 2.0 * cfg.m * cfg.n * cfg.k
    tflops_achieved = (flops / (ms_per / 1e3)) / 1e12  # FLOP/s -> TFLOP/s
    mfu = tflops_achieved / peak_tflops if peak_tflops > 0 else float("nan")

    # Simple sanity: return a scalar to prevent dead-code elimination (unlikely in eager)
    checksum = float(C[0, 0].float().item())

    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "config": cfg.__dict__,
        "iters": cfg.iters,
        "warmup": cfg.warmup,
        "ms_per_iter": ms_per,
        "tflops_achieved": tflops_achieved,
        "peak_tflops": peak_tflops,
        "mfu": mfu,
        "checksum": checksum,
    }


def main():
    props = torch.cuda.get_device_properties(0)

    print("Name:", props.name)
    print("SM count:", props.multi_processor_count)
    #print("Max clock (MHz):", props.clock_rate / 1000)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to json with m,n,k and optional fields.")
    ap.add_argument(
        "--peak-tflops",
        type=float,
        required=True,
        help="Peak tensor TFLOP/s for your GPU at this dtype (used for MFU).",
    )
    ap.add_argument("--json-out", default="", help="Optional output json path.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    res = _benchmark(cfg, args.peak_tflops)

    # Pretty print
    print("\n=== GEMM MFU Benchmark ===")
    print(f"GPU: {res['gpu']}")
    print(f"Shape (m,n,k): {cfg.m}, {cfg.n}, {cfg.k}")
    print(f"DType: {cfg.dtype} (fp8_format={cfg.fp8_format if cfg.dtype=='fp8' else 'n/a'})")
    print(f"Layout: {cfg.layout}")
    print(f"Time: {res['ms_per_iter']:.6f} ms / iter (iters={cfg.iters}, warmup={cfg.warmup})")
    print(f"Achieved: {res['tflops_achieved']:.2f} TFLOP/s")
    print(f"Peak: {res['peak_tflops']:.2f} TFLOP/s")
    print(f"MFU: {res['mfu']*100:.2f}%")
    print(f"Checksum: {res['checksum']}\n")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote: {args.json_out}")


if __name__ == "__main__":
    main()
