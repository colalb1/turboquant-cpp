# turboquant-cpp

A C++23 port of the **TurboQuant** quantization algorithm (ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), targeting Apple Silicon (ARM64) and exposed to inference pipelines as **ONNX Runtime custom operators**.

TurboQuant is a learned-codebook quantizer that compresses float vectors (e.g. LLM key/value tensors) into 2–4 bits per coordinate with provable error and inner-product bounds. This repository reimplements the reference PyTorch pipeline in C++ with NEON SIMD, Apple Accelerate for BLAS/LAPACK, and a shared-library entry point (`libturboquant_onnx.dylib`) that plugs directly into any ONNX Runtime session.

## What you get

| Artifact | What it is |
|---|---|
| `turboquant_core` | Static C++ library. The algorithm, codec kernels, and KV-cache machinery. |
| `libturboquant_onnx.dylib` | ORT custom-op shared library. Registers seven ops under the `com.turboquant` domain. |
| `turboquant_tests` | Catch2 test suite — 82 tests covering parity with Python, NEON vs scalar, Theorems 1–3. |
| `turboquant_bench` | Google Benchmark microbenchmarks for every hot path. |
| `turboquant_onnx_smoke` | Minimal test that loads the dylib and registers the op domain. |

## Scope

- **Core quantizers**: `TurboQuantMSE<Bits>` (Algorithm 1) and `TurboQuantProd<Bits>` (Algorithm 2, inner-product preserving) for `Bits ∈ {1, 2, 3, 4}`.
- **KV cache system**: `RingBuffer` (recent tokens), `CompressedKVStore` (chunked, compressed history), `KVCaptureEngine` (orchestrator), `compute_hybrid_attention` (three-path attention).
- **Codebooks**: the nine bundled JSON codebooks from the Python reference are embedded into the binary via a build-time code generator, plus a C++ Lloyd–Max implementation for computing new `(dim, bits)` pairs.
- **NEON kernels**: hand-written SIMD for L2 norm, scaling, pack/unpack, searchsorted, QJL projection, and group-wise value quantization. Scalar twins exist for verification.

Everything is FP32 end-to-end — this is the reference port, not an fp16/bf16 fork.

## Requirements

- **macOS on Apple Silicon** (M1 or later). CMake configure fails (`FATAL_ERROR`) on any other host — the code depends unconditionally on Accelerate (cblas_sgemm, vDSP, vvexpf) and NEON intrinsics.
- **Apple Clang 16+** (ships with Xcode 16) or LLVM Clang $\geq$ 17. C++23 required — the project uses `std::expected` from `<expected>` with no fallback.
- **CMake 3.24+**, **Ninja** (recommended), and **Python 3** (build-time only, for `tools/codebooks_to_cpp.py` which generates the embedded codebook translation unit).
- Everything else — nlohmann/json, Catch2 v3, Google Benchmark, ONNX Runtime 1.17 — is fetched by CMake's `FetchContent`. No system packages to install.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The first configure downloads ONNX Runtime (~35 MB) and builds Catch2 + Google Benchmark from source. Subsequent builds are incremental.

Useful CMake options:

| Option | Default | Effect |
|---|---|---|
| `TURBOQUANT_BUILD_TESTS` | `ON` | Build `turboquant_tests`. |
| `TURBOQUANT_BUILD_BENCH` | `ON` | Build `turboquant_bench`. |
| `TURBOQUANT_BUILD_ONNX` | `ON` | Build `libturboquant_onnx.dylib` + smoke test. |
| `TURBOQUANT_ENABLE_LTO` | `ON` | Thin LTO in Release. |
| `TURBOQUANT_CODEBOOK_SRC_DIR` | `./codebooks` | Where the bundled JSON codebooks live. |

## Test

```bash
ctest --test-dir build --output-on-failure -j
```

82 tests currently pass on macOS 14/Apple M-series. They cover: rotation round-trips, pack/unpack round-trips, codebook loading, Lloyd–Max vs bundled JSONs, quantizer MSE + Prod parity, NEON-vs-scalar bit-exactness, value quant round-trips, KV capture + flush, three-path hybrid attention, and ORT domain registration.

## Benchmark

```bash
./build/bench/turboquant_bench --benchmark_filter='MSE_Quantize/b=3/128/.*'
```

Baseline numbers from Apple M4 Pro are recorded in [`bench/BASELINES.md`](bench/BASELINES.md). The bench suite is the data source for tuning `kBlasThreshold` (rotation BLAS cutover) and the NEON pack fusion work.

## Using the ORT custom-op dylib

```cpp
#include <onnxruntime_cxx_api.h>
#include <turboquant/onnx/custom_ops.hpp>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_app");
Ort::SessionOptions opts;

// Register com.turboquant with this session.
OrtStatus* st = RegisterCustomOps(opts, OrtGetApiBase());
if (st) { /* handle error, release status */ }

Ort::Session session(env, "model.onnx", opts);
```

Then author an `.onnx` model that calls any of these ops in domain `com.turboquant`:

| Op | Inputs | Outputs | Attributes |
|---|---|---|---|
| `TurboQuantMSE_Quantize` | `x: float[..., D]` | `indices: uint8`, `norms: float` | `dim`, `bits`, `seed=42` |
| `TurboQuantMSE_Dequantize` | `indices`, `norms` | `x: float` | same |
| `TurboQuantProd_Quantize` | `x` | `mse_indices`, `qjl_signs`, `residual_norms`, `norms` | same |
| `TurboQuantProd_Dequantize` | four tensors above | `x` | same |
| `TurboQuantProd_AttentionScore` | `query`, four key tensors | `scores: float` | same |
| `TurboQuantValue_Quantize` | `v: float[..., D]` | `data`, `scales`, `zeros` | `bits ∈ {2,4,8}`, `group_size`, `head_dim` |
| `TurboQuantValue_Dequantize` | three above | `v: float` | same |

Quantizer state (`Pi`, `S`, centroids) is **not** a graph input — it's derived deterministically from `(dim, bits, seed)` and cached process-wide inside the dylib. This keeps model files small at the cost of a one-time construction when an unseen key is first seen in a session.

## Layout

```
turboquant-cpp/
├── include/turboquant/      # public headers (templates, NEON kernels, policies)
├── src/                     # non-template implementations + explicit instantiations
├── onnx/                    # ORT custom-op dylib (entry point + 7 op kernels)
├── bench/                   # Google Benchmark microbenchmarks
├── tests/                   # Catch2 unit/parity/theorem tests
├── codebooks/               # 9 bundled JSON codebooks (embedded at build time)
├── cmake/                   # FetchContent modules
├── tools/                   # codebooks_to_cpp.py code generator
├── CPP_IMPLEMENTATION_RULES.md   # project-wide C++ performance rules
├── IMPLEMENTATION_PLAN.md        # design doc / milestone plan
└── .clang-format, .clang-tidy    # lint + formatting config
```

## Reference of truth

Numerical behavior is validated against the PyTorch reference implementation that accompanies the paper. The port reproduces its outputs exactly up to the documented ULP budgets in `tests/unit/`. When in doubt, the Python source (quantizer, rotation, codebook, kv_cache, store, capture, score) is authoritative — C++ comments cite specific line numbers.

## License

See [LICENSE](LICENSE).
