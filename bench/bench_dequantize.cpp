// Dequantize throughput benchmarks.
//
// Quantize once outside the loop, then measure dequantize in isolation.
// Metric: output bytes/sec = batch * dim * sizeof(float).

#include "bench_util.hpp"

#include "turboquant/quantizer_mse.hpp"
#include "turboquant/quantizer_prod.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

template <int Bits>
void bench_mse_dequantize(benchmark::State& state) {
    const std::size_t dim   = static_cast<std::size_t>(state.range(0));
    const std::size_t batch = static_cast<std::size_t>(state.range(1));

    auto q = tq::bench::must(tq::TurboQuantMSE<Bits>::make(dim, /*seed=*/42));

    std::vector<float>        x(batch * dim);
    std::vector<std::uint8_t> indices(batch * tq::TurboQuantMSE<Bits>::packed_bytes(dim));
    std::vector<float>        norms(batch);
    std::vector<float>        x_out(batch * dim);
    tq::bench::fill_gaussian(x, 0xDEC0DE01);

    if (q.quantize(x, batch, indices, norms) != tq::Error::Ok) std::abort();

    for (auto _ : state) {
        tq::Error e = q.dequantize(indices, norms, batch, x_out);
        benchmark::DoNotOptimize(e);
        benchmark::DoNotOptimize(x_out.data());
        benchmark::ClobberMemory();
    }

    const std::int64_t bytes_out = static_cast<std::int64_t>(batch * dim * sizeof(float));
    state.SetBytesProcessed(bytes_out * state.iterations());
    state.SetItemsProcessed(static_cast<std::int64_t>(batch) * state.iterations());
}

template <int Bits>
void bench_prod_dequantize(benchmark::State& state) {
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
    std::vector<float>        x_out(batch * dim);
    tq::bench::fill_gaussian(x, 0xDEC0DE02);

    if (q.quantize(x, batch, mse_indices, qjl_signs, residual_norms, norms) != tq::Error::Ok)
        std::abort();

    for (auto _ : state) {
        tq::Error e = q.dequantize(mse_indices, qjl_signs, residual_norms,
                                   norms, batch, x_out);
        benchmark::DoNotOptimize(e);
        benchmark::DoNotOptimize(x_out.data());
        benchmark::ClobberMemory();
    }

    const std::int64_t bytes_out = static_cast<std::int64_t>(batch * dim * sizeof(float));
    state.SetBytesProcessed(bytes_out * state.iterations());
    state.SetItemsProcessed(static_cast<std::int64_t>(batch) * state.iterations());
}

void dequant_args(benchmark::internal::Benchmark* b) {
    for (int d : {64, 128, 256}) {
        for (int n : {64, 256, 1024, 4096}) {
            b->Args({d, n});
        }
    }
}

} // namespace

BENCHMARK(bench_mse_dequantize<1>)->Apply(dequant_args)->Name("MSE_Dequantize/b=1");
BENCHMARK(bench_mse_dequantize<2>)->Apply(dequant_args)->Name("MSE_Dequantize/b=2");
BENCHMARK(bench_mse_dequantize<3>)->Apply(dequant_args)->Name("MSE_Dequantize/b=3");
BENCHMARK(bench_mse_dequantize<4>)->Apply(dequant_args)->Name("MSE_Dequantize/b=4");

BENCHMARK(bench_prod_dequantize<2>)->Apply(dequant_args)->Name("Prod_Dequantize/b=2");
BENCHMARK(bench_prod_dequantize<3>)->Apply(dequant_args)->Name("Prod_Dequantize/b=3");
BENCHMARK(bench_prod_dequantize<4>)->Apply(dequant_args)->Name("Prod_Dequantize/b=4");
