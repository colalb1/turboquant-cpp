// Prod attention_score throughput.
//
// Prepares a key batch of size n_k quantized with TurboQuantProd once,
// then times `attention_score(query[n_q], keys, scores)` in the hot loop.
// Grid is (dim, n_q, n_k, bits). Metric: key pairs scored/sec and
// output-scores bytes/sec.

#include "bench_util.hpp"

#include "turboquant/quantizer_prod.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

template <int Bits>
void bench_prod_score(benchmark::State& state) {
    const std::size_t dim = static_cast<std::size_t>(state.range(0));
    const std::size_t n_q = static_cast<std::size_t>(state.range(1));
    const std::size_t n_k = static_cast<std::size_t>(state.range(2));

    auto q = tq::bench::must(tq::TurboQuantProd<Bits>::make(dim, /*seed=*/42));

    const std::size_t mse_pb = tq::TurboQuantProd<Bits>::mse_packed_bytes(dim);
    const std::size_t qjl_pb = tq::TurboQuantProd<Bits>::qjl_packed_bytes(dim);

    std::vector<float> keys_x(n_k * dim);
    std::vector<float> query(n_q * dim);
    tq::bench::fill_gaussian(keys_x, 0x5C0DE001u);
    tq::bench::fill_gaussian(query, 0x5C0DE002u);

    std::vector<std::uint8_t> mse_indices(n_k * mse_pb);
    std::vector<std::uint8_t> qjl_signs(n_k * qjl_pb);
    std::vector<float>        residual_norms(n_k);
    std::vector<float>        norms(n_k);
    std::vector<float>        scores(n_q * n_k);

    if (q.quantize(keys_x, n_k, mse_indices, qjl_signs, residual_norms, norms) != tq::Error::Ok)
        std::abort();

    for (auto _ : state) {
        tq::Error e = q.attention_score(query, n_q, mse_indices, qjl_signs, residual_norms, norms,
                                        n_k, scores);
        benchmark::DoNotOptimize(e);
        benchmark::DoNotOptimize(scores.data());
        benchmark::ClobberMemory();
    }

    // Report both "pairs/sec" (q×k dot products) and bytes of score output.
    state.SetItemsProcessed(static_cast<std::int64_t>(n_q * n_k) * state.iterations());
    state.SetBytesProcessed(static_cast<std::int64_t>(n_q * n_k * sizeof(float)) *
                            state.iterations());
}

void score_args(benchmark::internal::Benchmark* b) {
    for (int d : {64, 128}) {
        for (int q : {1, 16}) {
            for (int k : {128, 512, 2048}) {
                b->Args({d, q, k});
            }
        }
    }
}

}  // namespace

BENCHMARK(bench_prod_score<2>)->Apply(score_args)->Name("Prod_Score/b=2");
BENCHMARK(bench_prod_score<3>)->Apply(score_args)->Name("Prod_Score/b=3");
BENCHMARK(bench_prod_score<4>)->Apply(score_args)->Name("Prod_Score/b=4");
