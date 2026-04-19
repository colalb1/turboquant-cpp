// End-to-end KV cycle: prefill ingest → decode ingest loop →
// compute_hybrid_attention. Mirrors the typical per-layer cost during
// LLM decode and flushes out regressions that would be invisible to the
// standalone quantize/dequantize benches (e.g. chunk bookkeeping, the
// lazy flat-view materialization, GQA permutes in score.cpp).

#include "bench_util.hpp"

#include "turboquant/kv_capture.hpp"
#include "turboquant/score.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

namespace {

template <int KeyBits, int ValBits>
void bench_capture_ingest_decode(benchmark::State& state) {
    const std::size_t     dim           = static_cast<std::size_t>(state.range(0));
    const std::size_t     num_kv_heads  = static_cast<std::size_t>(state.range(1));
    const std::size_t     prefill_n     = static_cast<std::size_t>(state.range(2));
    const std::size_t     decode_steps  = static_cast<std::size_t>(state.range(3));
    constexpr std::size_t group_size    = 32;
    constexpr std::size_t ring_capacity = 128;

    std::vector<float> k_pref(prefill_n * num_kv_heads * dim);
    std::vector<float> v_pref(prefill_n * num_kv_heads * dim);
    std::vector<float> k_dec(num_kv_heads * dim);
    std::vector<float> v_dec(num_kv_heads * dim);
    tq::bench::fill_gaussian(k_pref, 0xAABBCC01u);
    tq::bench::fill_gaussian(v_pref, 0xAABBCC02u);
    tq::bench::fill_gaussian(k_dec, 0xAABBCC03u);
    tq::bench::fill_gaussian(v_dec, 0xAABBCC04u);

    for (auto _ : state) {
        auto      eng = tq::bench::must(tq::KVCaptureEngine<KeyBits, ValBits>::make(
            dim, num_kv_heads, group_size, ring_capacity, /*seed=*/42));
        tq::Error e   = eng.ingest_prefill(k_pref, v_pref, prefill_n);
        benchmark::DoNotOptimize(e);
        for (std::size_t t = 0; t < decode_steps; ++t) {
            e = eng.ingest_decode(k_dec, v_dec, 1);
            benchmark::DoNotOptimize(e);
        }
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(static_cast<std::int64_t>(prefill_n + decode_steps) *
                            state.iterations());
}

template <int KeyBits, int ValBits>
void bench_hybrid_attention(benchmark::State& state) {
    const std::size_t dim          = static_cast<std::size_t>(state.range(0));
    const std::size_t num_kv_heads = static_cast<std::size_t>(state.range(1));
    const std::size_t num_q_heads  = num_kv_heads;  // GQA group = 1 for baseline
    const std::size_t history_n    = static_cast<std::size_t>(state.range(2));
    const std::size_t recent_n     = static_cast<std::size_t>(state.range(3));
    const std::size_t n_q_tokens   = 1;  // typical decode step

    constexpr std::size_t group_size    = 32;
    constexpr std::size_t ring_capacity = 128;

    auto eng = tq::bench::must(tq::KVCaptureEngine<KeyBits, ValBits>::make(
        dim, num_kv_heads, group_size, ring_capacity, /*seed=*/42));

    // Fill the store with `history_n` tokens via ingest_prefill. Anything
    // over ring_capacity lands in the compressed store; the trailing
    // ring contents we drain explicitly so the history-vs-recent split is
    // deterministic for timing.
    std::vector<float> k_hist(history_n * num_kv_heads * dim);
    std::vector<float> v_hist(history_n * num_kv_heads * dim);
    tq::bench::fill_gaussian(k_hist, 0x5C0DEA01u);
    tq::bench::fill_gaussian(v_hist, 0x5C0DEA02u);
    if (eng.ingest_prefill(k_hist, v_hist, history_n) != tq::Error::Ok) std::abort();
    if (eng.flush() != tq::Error::Ok) std::abort();

    std::vector<float> recent_k(recent_n * num_kv_heads * dim);
    std::vector<float> recent_v(recent_n * num_kv_heads * dim);
    tq::bench::fill_gaussian(recent_k, 0x5C0DEA03u);
    tq::bench::fill_gaussian(recent_v, 0x5C0DEA04u);

    std::vector<float> query(n_q_tokens * num_q_heads * dim);
    std::vector<float> out(n_q_tokens * num_q_heads * dim);
    tq::bench::fill_gaussian(query, 0x5C0DEA05u);

    for (auto _ : state) {
        tq::Error e = tq::compute_hybrid_attention<KeyBits, ValBits>(
            query, n_q_tokens, num_q_heads, eng.store(), recent_k, recent_v, recent_n,
            /*scale=*/0.0f, out);
        benchmark::DoNotOptimize(e);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    // Keys scored per iteration = history + recent.
    const std::int64_t k_total = static_cast<std::int64_t>(history_n + recent_n);
    state.SetItemsProcessed(k_total * static_cast<std::int64_t>(num_q_heads) * state.iterations());
}

void capture_args(benchmark::internal::Benchmark* b) {
    // (dim, kv_heads, prefill_n, decode_steps)
    for (int d : {64, 128}) {
        for (int h : {4, 8}) {
            b->Args({d, h, 256, 64});
            b->Args({d, h, 1024, 128});
        }
    }
}

void score_args(benchmark::internal::Benchmark* b) {
    // (dim, kv_heads, history_n, recent_n)
    for (int d : {64, 128}) {
        for (int h : {4, 8}) {
            b->Args({d, h, 512, 0});    // compressed-only
            b->Args({d, h, 0, 128});    // recent-only (history below MIN)
            b->Args({d, h, 512, 64});   // hybrid
            b->Args({d, h, 2048, 64});  // hybrid, bigger history
        }
    }
}

}  // namespace

BENCHMARK(bench_capture_ingest_decode<2, 4>)
    ->Apply(capture_args)
    ->Name("KV_CaptureIngest/kb=2/vb=4");
BENCHMARK(bench_capture_ingest_decode<4, 4>)
    ->Apply(capture_args)
    ->Name("KV_CaptureIngest/kb=4/vb=4");

BENCHMARK(bench_hybrid_attention<2, 4>)->Apply(score_args)->Name("KV_HybridAttention/kb=2/vb=4");
BENCHMARK(bench_hybrid_attention<4, 4>)->Apply(score_args)->Name("KV_HybridAttention/kb=4/vb=4");
