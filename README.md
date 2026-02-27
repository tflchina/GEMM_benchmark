# GEMM Benchmark

Benchmark different GEMM shapes across multiple backends:

- `pytorch`
- `cublas` (via CuPy)
- `triton`
- `cuda-kernel` (manual CUDA kernel via CuPy RawKernel)

## Usage

Single shape from CLI:

```bash
python benchmark.py --backend all --m 1024 --n 1024 --k 1024 --dtype fp8_e4m3fn
```

Multiple shapes from JSON (with optional per-shape dtype):

```bash
python benchmark.py --backend all --shape-file shapes_example.json --dtype float32
```

## JSON format

```json
{
  "shapes": [
    {"m": 512, "n": 512, "k": 512, "dtype": "float16"},
    {"m": 1024, "n": 1024, "k": 1024, "dtype": "fp8_e4m3fn"},
    {"m": 4096, "n": 4096, "k": 4096, "dtype": "fp4_e2m1"}
  ]
}
```

If `dtype` is missing in a shape entry, the CLI `--dtype` value is used.

## Dtype support

- CLI dtypes: `float16`, `float32`, `float64`, `bfloat16`, `int8`, `fp8_e4m3fn`, `fp8_e5m2`, `fp4_e2m1`
- Backend support:
  - `pytorch`: `float16`, `float32`, `float64`, `bfloat16`, `fp8_e4m3fn`, `fp8_e5m2`, `fp4_e2m1` (FP4 is emulated)
  - `cublas`: `float16`, `float32`, `float64`, `bfloat16`, `int8`, `fp8_e4m3fn`, `fp8_e5m2`, `fp4_e2m1` (depends on CuPy build; FP4 is emulated)
  - `triton`: `float16`, `float32`, `bfloat16`
  - `cuda-kernel`: `float16`, `float32`

Notes:
- FP8 dtypes are only available if your framework build exposes them.
- `fp4_e2m1` is an emulated format in this benchmark (stored/computed using `float16` after quantization).
- Unsupported dtype/backend combinations are skipped with a warning.

## GPU spec dataset

A curated AI GPU specification dataset is provided at `gpu_specs_ai.json`, including major NVIDIA and AMD accelerators with peak throughput by datatype (where published), memory bandwidth/capacity, and power envelope.

