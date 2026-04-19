// Quantize throughput benchmarks.
//
// Metric: input bytes/sec = batch * dim * sizeof(float) per iteration.
// The M9 gate in IMPLEMENTATION_PLAN.md targets ≥ 3 GB/s at (d=128, b=3,
// batch=1024). Run with:
//   ./build/bench/turboquant_bench --benchmark_filter=Quantize
//
// Explicit instantiations for (Bits) are already in turboquant_core, so
// each bench registration just dispatches to a templated fn.

#include "bench_util.hpp"

#include "turboquant/quantizer_mse.hpp"
#include "turboquant/quantizer_prod.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

template <int Bits>
void bench_mse_quantize(benchmark::State& state) {
    const std::size_t dim   = static_cast<std::size_t>(state.range(0));
    const std::size_t batch = static_cast<std::size_t>(state.range(1));

    auto q = tq::bench::must(tq::TurboQuantMSE<Bits>::make(dim, /*seed=*/42));

    std::vector<float>        x(batch * dim);
    std::vector<std::uint8_t> indices(batch * tq::TurboQuantMSE<Bits>::packed_bytes(dim));
    std::vector<float>        norms(batch);
    tq::bench::fill_gaussian(x, 0xC0DE0001);

    for (auto _ : state) {
        tq::Error e = q.quantize(x, batch, indices, norms);
        benchmark::DoNotOptimize(e);
        benchmark::DoNotOptimize(indices.data());
        benchmark::DoNotOptimize(norms.data());
        benchmark::ClobberMemory();
    }

    const std::int64_t bytes_in = static_cast<std::int64_t>(batch * dim * sizeof(float));
    state.SetBytesProcessed(bytes_in * state.iterations());
    state.SetItemsProcessed(static_cast<std::int64_t>(batch) * state.iterations());
}

template <int Bits>
void bench_prod_quantize(benchmark::State& state) {
    const std::size_t dim   = static_cast<std::size_t>(state.range(0));
    const std::size_t batch = static_cast<std::size_t>(state.range(1));

    auto q = tq::bench::must(tq::TurboQuantProd<Bits>::make(dim, /*seed=*/42));

    const std::size_t mse_pb = tq::TurboQuantProd<Bits>::mse_packed_bytes(dim);
    const std::size_t qjl_pb = tq::TurboQuantProd<Bits>::qjl_packed_bytes(dim);

    std::vector<float>        x(batch * dim);
    std::vector<std::uint8_t> mse_indices(batch * mse_pb);
    std::vector<std::uint8_t> qjl_signs(batch * qjl_pb);
    std::vector<float>        residual_norms(batch);
    std::vector<float>        norms(batch);
    tq::bench::fill_gaussian(x, 0xC0DE0002);

    for (auto _ : state) {
        tq::Error e = q.quantize(x, batch, mse_indices, qjl_signs,
                                 residual_norms, norms);
        benchmark::DoNotOptimize(e);
        benchmark::DoNotOptimize(mse_indices.data());
        benchmark::DoNotOptimize(qjl_signs.data());
        benchmark::ClobberMemory();
    }

    const std::int64_t bytes_in = static_cast<std::int64_t>(batch * dim * sizeof(float));
    state.SetBytesProcessed(bytes_in * state.iterations());
    state.SetItemsProcessed(static_cast<std::int64_t>(batch) * state.iterations());
}

// (dim, batch) grid. Keep compact — Benchmark CLI can filter further.
void quant_args(benchmark::internal::Benchmark* b) {
    for (int d : {64, 128, 256}) {
        for (int n : {64, 256, 1024, 4096}) {
            b->Args({d, n});
        }
    }
}

} // namespace

BENCHMARK(bench_mse_quantize<1>)->Apply(quant_args)->Name("MSE_Quantize/b=1");
BENCHMARK(bench_mse_quantize<2>)->Apply(quant_args)->Name("MSE_Quantize/b=2");
BENCHMARK(bench_mse_quantize<3>)->Apply(quant_args)->Name("MSE_Quantize/b=3");
BENCHMARK(bench_mse_quantize<4>)->Apply(quant_args)->Name("MSE_Quantize/b=4");

BENCHMARK(bench_prod_quantize<2>)->Apply(quant_args)->Name("Prod_Quantize/b=2");
BENCHMARK(bench_prod_quantize<3>)->Apply(quant_args)->Name("Prod_Quantize/b=3");
BENCHMARK(bench_prod_quantize<4>)->Apply(quant_args)->Name("Prod_Quantize/b=4");
