# turboquant_bench baselines

First-pass numbers for the M9 microbench suite. Reproduce with:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bench/turboquant_bench
```

Filter a subset with `--benchmark_filter=<regex>`, e.g.
`--benchmark_filter='MSE_Quantize/b=3/.*'`.

## Host

- Apple M4 Pro (12 cores), 24 GiB RAM
- macOS 26.3.1 (Darwin 25.3.0, arm64)
- Apple clang 17.0.0 (clang-1700.6.3.2)
- Build: Release, `-O3 -march=armv8.5-a -mtune=apple-m1 -flto=thin`

All runs are single-threaded, no thread affinity (macOS does not expose it).
Bencher's "CPU frequency" warning is cosmetic — wall-clock numbers are fine.

## Headline numbers

### MSE / Prod quantize (batch=1024, bits=3)

| op        | dim | MB/s   |
|-----------|----:|-------:|
| MSE_Quantize  |  64 | 432 |
| MSE_Quantize  | 128 | 600 |
| MSE_Quantize  | 256 | 622 |
| Prod_Quantize |  64 | 262 |
| Prod_Quantize | 128 | 347 |
| Prod_Quantize | 256 | 349 |

### MSE / Prod dequantize (batch=1024, bits=3)

| op              | dim | MB/s |
|-----------------|----:|-----:|
| MSE_Dequantize  |  64 |  787 |
| MSE_Dequantize  | 128 |  886 |
| MSE_Dequantize  | 256 |  840 |
| Prod_Dequantize |  64 |  687 |
| Prod_Dequantize | 128 |  798 |
| Prod_Dequantize | 256 |  755 |

### Prod attention_score (dim=128, n_q=1, bits=3)

| n_k  | µs / iter |
|-----:|----------:|
|  128 |     87.6  |
|  512 |    299.6  |
| 2048 |   1135.8  |

### Rotation forward (dim=128) — BLAS threshold sweep

| batch | GFLOP/s |
|------:|--------:|
|     1 |    136  |
|     4 |     58  |  (crossover point — worst case)
|     8 |    117  |
|    64 |    668  |
|  1024 |   1305  |

The `kBlasThreshold = 4` constant cuts over to `cblas_sgemm` right where the
BLAS call's fixed overhead dominates; at batch=4–7 we briefly do worse than
the NEON small-batch path would. Candidate retune: raise the threshold to 8
(or ~16) and re-sweep. Not done in this pass — recording the shape here so
the follow-up has a before/after.

### KV cycle (dim=128, kv_heads=8, KeyBits=4, ValBits=4)

| scenario                                   | time / iter |
|--------------------------------------------|------------:|
| CaptureIngest prefill=256,  decode=64      | 3.48 ms     |
| CaptureIngest prefill=1024, decode=128     | 12.96 ms    |
| HybridAttention history=512, recent=0      | 2.73 ms     |
| HybridAttention history=512, recent=64     | 2.73 ms     |

## Observations

- **Quantize is ~2× slower than dequantize at matching (d, b, batch).** The
  quantize hot path is still scalar-per-coord for searchsorted+pack (the
  NEON searchsorted_pack kernel is planned but not yet wired into the
  templated code). Dequantize is already NEON-lit via unpack_gather. This
  is the biggest headroom item.
- **Prod is ~60% of MSE throughput.** Expected — Prod does an extra QJL
  project + sign-pack on the residual after the MSE stage.
- **M9 plan target of ≥ 3 GB/s at (d=128, b=3, batch=1024) is not met.**
  Measured 600 MB/s. The gap is the missing NEON searchsorted_pack fusion;
  instrumenting quantize under Time Profiler and adding that kernel are
  the natural M9 follow-ups (deferred — user opted to skip M9 originally
  and we came back to it only for the bench infra itself).
- **Rotation at batch=4 is a clear anti-sweet-spot.** Raise
  `kBlasThreshold` or add a 4×4-tile NEON path for the [4, 8) range.

## What's NOT covered yet

- Value codec (`ValueCodec<Bits>`) throughput — captured implicitly in
  KV_CaptureIngest but not isolated. Add a `bench_value_quant.cpp` when
  tuning the NEON group_quant kernel.
- Lloyd-Max codebook compute cost — cold path; ignore for now.
- Rotation::backward — symmetric to forward, same BLAS crossover.
- Instruments profiling pass (Time Profiler + Counters) — separate task.
