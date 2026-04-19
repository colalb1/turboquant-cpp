// Rotation::forward throughput across batches.
//
// Purpose: characterize the threshold where Accelerate cblas_sgemm
// overtakes the small-batch NEON path. IMPLEMENTATION_PLAN M9 calls out
// tuning kBlasThreshold; this sweep is the data feeding that decision.

#include "bench_util.hpp"

#include "turboquant/rotation.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

void bench_rotation_forward(benchmark::State& state) {
    const std::size_t dim   = static_cast<std::size_t>(state.range(0));
    const std::size_t batch = static_cast<std::size_t>(state.range(1));

    auto rot = tq::bench::must(tq::Rotation::make(dim, /*seed=*/42));

    std::vector<float> x(batch * dim);
    std::vector<float> y(batch * dim);
    tq::bench::fill_gaussian(x, 0xA0A71017u);

    for (auto _ : state) {
        tq::Error e = rot.forward(x, y, batch);
        benchmark::DoNotOptimize(e);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }

    const std::int64_t flops_per_iter =
        2 * static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(dim) * dim;
    state.counters["GFLOP/s"] = benchmark::Counter(static_cast<double>(flops_per_iter),
                                                   benchmark::Counter::kIsIterationInvariantRate,
                                                   benchmark::Counter::kIs1000);
    state.SetBytesProcessed(static_cast<std::int64_t>(batch * dim * sizeof(float)) *
                            state.iterations());
    state.SetItemsProcessed(static_cast<std::int64_t>(batch) * state.iterations());
}

void rotation_args(benchmark::internal::Benchmark* b) {
    for (int d : {64, 128, 256}) {
        // Sweep around kBlasThreshold (4) to see the crossover.
        for (int n : {1, 2, 4, 8, 16, 64, 256, 1024}) {
            b->Args({d, n});
        }
    }
}

}  // namespace

BENCHMARK(bench_rotation_forward)->Apply(rotation_args)->Name("Rotation_Forward");
