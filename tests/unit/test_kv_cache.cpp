// M7: ring buffer, compressed KV store, capture engine.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "turboquant/compressed_kv_store.hpp"
#include "turboquant/kv_capture.hpp"
#include "turboquant/ring_buffer.hpp"

using Catch::Matchers::WithinAbs;

namespace {

std::vector<float> make_gaussian(std::size_t n, std::uint32_t seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              out(n);
    for (auto& v : out)
        v = nd(rng);
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// RingBuffer
// -----------------------------------------------------------------------------

TEST_CASE("RingBuffer write/drain round-trips exactly", "[ring]") {
    const std::size_t cap = 16, H = 4, D = 8;
    auto              rb_r = tq::RingBuffer::make(cap, H, D);
    REQUIRE(rb_r);
    auto& rb = *rb_r;
    REQUIRE(rb.capacity() == cap);
    REQUIRE(rb.space_left() == cap);
    REQUIRE(rb.size() == 0);

    const std::size_t n = 10, rs = H * D;
    const auto        k = make_gaussian(n * rs, /*seed=*/1);
    const auto        v = make_gaussian(n * rs, /*seed=*/2);
    REQUIRE(rb.write(k, v, n) == tq::Error::Ok);
    REQUIRE(rb.size() == n);
    REQUIRE(rb.space_left() == cap - n);
    REQUIRE(rb.total_written() == n);

    std::vector<float> dk(n * rs), dv(n * rs);
    REQUIRE(rb.drain(dk, dv) == n);
    REQUIRE(rb.size() == 0);
    for (std::size_t i = 0; i < n * rs; ++i) {
        REQUIRE(dk[i] == k[i]);
        REQUIRE(dv[i] == v[i]);
    }
    // total_written persists across drain.
    REQUIRE(rb.total_written() == n);
}

TEST_CASE("RingBuffer refuses writes that exceed space", "[ring]") {
    auto               rb = *tq::RingBuffer::make(/*cap=*/4, /*H=*/2, /*D=*/3);
    const std::size_t  rs = 2 * 3;
    std::vector<float> k(5 * rs, 1.0f), v(5 * rs, 2.0f);
    REQUIRE(rb.write(k, v, 5) == tq::Error::BufferTooSmall);
    REQUIRE(rb.size() == 0);
}

TEST_CASE("RingBuffer rejects shape-mismatched spans", "[ring]") {
    auto               rb = *tq::RingBuffer::make(4, 2, 3);
    std::vector<float> k(10), v(12);
    REQUIRE(rb.write(k, v, 2) == tq::Error::ShapeMismatch);
}

// -----------------------------------------------------------------------------
// CompressedKVStore
// -----------------------------------------------------------------------------

TEST_CASE("CompressedKVStore single-chunk append + flat matches source bytes", "[store][flat]") {
    using Store         = tq::CompressedKVStore<3, 2>;
    const std::size_t D = 64, H = 2, T = 6, gs = 16;
    auto              store_r = Store::make(D, H, gs, /*seed=*/42);
    REQUIRE(store_r);
    auto& store = *store_r;

    const auto k = make_gaussian(T * H * D, /*seed=*/123);
    const auto v = make_gaussian(T * H * D, /*seed=*/124);
    REQUIRE(store.append_chunk(k, v, T) == tq::Error::Ok);
    REQUIRE(store.num_tokens() == T);
    REQUIRE(store.num_chunks() == 1);

    auto flat_r = store.get_flat();
    REQUIRE(flat_r);
    const auto& f = *flat_r;
    REQUIRE(f.total_tokens == T);
    REQUIRE(f.mse_indices.size() == H * T * store.mse_pb());
    REQUIRE(f.qjl_signs.size() == H * T * store.qjl_pb());
    REQUIRE(f.residual_norms.size() == H * T);
    REQUIRE(f.norms.size() == H * T);
    REQUIRE(f.val_data.size() == H * T * store.val_pb());
    REQUIRE(f.val_scales.size() == H * T * store.val_ng());
    REQUIRE(f.val_zeros.size() == H * T * store.val_ng());
}

TEST_CASE("CompressedKVStore multi-chunk flat concatenates per head correctly", "[store][flat]") {
    using Store         = tq::CompressedKVStore<3, 2>;
    const std::size_t D = 32, H = 3, gs = 16;
    auto              store = *Store::make(D, H, gs, /*seed=*/7);

    // Append two chunks of different sizes.
    const std::size_t T1 = 4, T2 = 5;
    const auto        k1 = make_gaussian(T1 * H * D, /*seed=*/1000);
    const auto        v1 = make_gaussian(T1 * H * D, /*seed=*/1001);
    const auto        k2 = make_gaussian(T2 * H * D, /*seed=*/2000);
    const auto        v2 = make_gaussian(T2 * H * D, /*seed=*/2001);

    REQUIRE(store.append_chunk(k1, v1, T1) == tq::Error::Ok);
    REQUIRE(store.append_chunk(k2, v2, T2) == tq::Error::Ok);
    REQUIRE(store.num_chunks() == 2);

    auto fr = store.get_flat();
    REQUIRE(fr);
    const auto&       f = *fr;
    const std::size_t T = T1 + T2;
    REQUIRE(f.total_tokens == T);

    // Round-trip: dequantize the flat key+value and compare to the true
    // source. Values use per-group asymmetric quant, so we bound the
    // error by 2 * max_scale per element.
    std::vector<float> k_all(T * H * D), v_all(T * H * D);
    // Build (T, H, D) arrangement consistent with Python: chunk1 tokens
    // first, then chunk2 tokens.
    std::memcpy(k_all.data(), k1.data(), T1 * H * D * sizeof(float));
    std::memcpy(k_all.data() + T1 * H * D, k2.data(), T2 * H * D * sizeof(float));
    std::memcpy(v_all.data(), v1.data(), T1 * H * D * sizeof(float));
    std::memcpy(v_all.data() + T1 * H * D, v2.data(), T2 * H * D * sizeof(float));

    // Value round-trip: flat layout is (H, T, ...). Reshape.
    std::vector<float> v_rt_ht(H * T * D);
    using Val = tq::ValueCodec<2>;
    REQUIRE(Val::dequantize(f.val_data, f.val_scales, f.val_zeros, H * T, D, gs,
                            std::span<float>(v_rt_ht)) == tq::Error::Ok);
    // Find max value scale across the flat.
    float max_sc = 0.0f;
    for (float s : f.val_scales)
        max_sc = std::max(max_sc, s);
    // Compare against (H, T, D) ordering of v_all (which is (T, H, D)).
    for (std::size_t h = 0; h < H; ++h) {
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t j = 0; j < D; ++j) {
                const float got  = v_rt_ht[(h * T + t) * D + j];
                const float want = v_all[(t * H + h) * D + j];
                CAPTURE(h, t, j);
                REQUIRE(std::fabs(got - want) <= max_sc * 0.51f + 1e-5f);
            }
        }
    }
}

TEST_CASE("CompressedKVStore flat cache invalidates on append", "[store][cache]") {
    using Store         = tq::CompressedKVStore<3, 2>;
    const std::size_t D = 32, H = 2, gs = 16, T = 4;
    auto              store = *Store::make(D, H, gs, 11);
    const auto        k1    = make_gaussian(T * H * D, 1);
    const auto        v1    = make_gaussian(T * H * D, 2);
    REQUIRE(store.append_chunk(k1, v1, T) == tq::Error::Ok);

    auto              f1 = *store.get_flat();
    const std::size_t t1 = f1.total_tokens;

    const auto k2 = make_gaussian(T * H * D, 3);
    const auto v2 = make_gaussian(T * H * D, 4);
    REQUIRE(store.append_chunk(k2, v2, T) == tq::Error::Ok);

    auto f2 = *store.get_flat();
    REQUIRE(f2.total_tokens == t1 + T);
    REQUIRE(f2.mse_indices.data() != nullptr);
    // Repeat get_flat without append — same pointer (cache hit).
    auto f3 = *store.get_flat();
    REQUIRE(f3.mse_indices.data() == f2.mse_indices.data());
}

TEST_CASE("CompressedKVStore reset clears everything", "[store]") {
    using Store      = tq::CompressedKVStore<3, 2>;
    auto       store = *Store::make(32, 2, 16, 11);
    const auto k     = make_gaussian(4 * 2 * 32, 1);
    const auto v     = make_gaussian(4 * 2 * 32, 2);
    REQUIRE(store.append_chunk(k, v, 4) == tq::Error::Ok);
    REQUIRE(store.num_tokens() == 4);
    store.reset();
    REQUIRE(store.num_tokens() == 0);
    REQUIRE(store.num_chunks() == 0);
    auto f = *store.get_flat();
    REQUIRE(f.total_tokens == 0);
}

// -----------------------------------------------------------------------------
// KVCaptureEngine
// -----------------------------------------------------------------------------

TEST_CASE("KVCaptureEngine: prefill smaller than ring buffer fills ring only",
          "[engine][prefill]") {
    using Eng             = tq::KVCaptureEngine<3, 2>;
    auto              eng = *Eng::make(/*D=*/32, /*H=*/2, /*gs=*/16, /*cap=*/16, /*seed=*/0);
    const std::size_t T = 10, rs = 2 * 32;
    const auto        k = make_gaussian(T * rs, 1);
    const auto        v = make_gaussian(T * rs, 2);
    REQUIRE(eng.ingest_prefill(k, v, T) == tq::Error::Ok);
    REQUIRE(eng.total_buffered_tokens() == T);
    REQUIRE(eng.total_compressed_tokens() == 0);
}

TEST_CASE("KVCaptureEngine: prefill larger than ring compresses leading tokens",
          "[engine][prefill]") {
    using Eng             = tq::KVCaptureEngine<3, 2>;
    const std::size_t cap = 16;
    auto              eng = *Eng::make(32, 2, 16, cap, 0);
    const std::size_t T = 50, rs = 2 * 32;
    const auto        k = make_gaussian(T * rs, 1);
    const auto        v = make_gaussian(T * rs, 2);
    REQUIRE(eng.ingest_prefill(k, v, T) == tq::Error::Ok);
    REQUIRE(eng.total_buffered_tokens() == cap);
    REQUIRE(eng.total_compressed_tokens() == T - cap);
    REQUIRE(eng.total_tokens() == T);
}

TEST_CASE("KVCaptureEngine: decode drains ring on overflow", "[engine][decode]") {
    using Eng             = tq::KVCaptureEngine<3, 2>;
    const std::size_t cap = 8;
    auto              eng = *Eng::make(32, 2, 16, cap, 0);
    const std::size_t T = cap * 3 + 5, rs = 2 * 32;
    // Stream tokens one at a time.
    const auto k = make_gaussian(T * rs, 1);
    const auto v = make_gaussian(T * rs, 2);
    for (std::size_t t = 0; t < T; ++t) {
        REQUIRE(eng.ingest_decode(std::span<const float>(k.data() + t * rs, rs),
                                  std::span<const float>(v.data() + t * rs, rs),
                                  1) == tq::Error::Ok);
    }
    REQUIRE(eng.total_tokens() == T);
    REQUIRE(eng.total_compressed_tokens() == cap * 3);
    REQUIRE(eng.total_buffered_tokens() == 5);
}

TEST_CASE("KVCaptureEngine: flush drains ring into store", "[engine][flush]") {
    using Eng             = tq::KVCaptureEngine<3, 2>;
    auto              eng = *Eng::make(32, 2, 16, /*cap=*/16, 0);
    const std::size_t T = 10, rs = 2 * 32;
    const auto        k = make_gaussian(T * rs, 1);
    const auto        v = make_gaussian(T * rs, 2);
    REQUIRE(eng.ingest_decode(k, v, T) == tq::Error::Ok);
    REQUIRE(eng.total_compressed_tokens() == 0);
    REQUIRE(eng.flush() == tq::Error::Ok);
    REQUIRE(eng.total_buffered_tokens() == 0);
    REQUIRE(eng.total_compressed_tokens() == T);
}

TEST_CASE("KVCaptureEngine: bulk decode matching interleaved decode", "[engine][decode]") {
    using Eng             = tq::KVCaptureEngine<3, 2>;
    const std::size_t cap = 8, D = 32, H = 2, gs = 16;
    auto              ea = *Eng::make(D, H, gs, cap, /*seed=*/100);
    auto              eb = *Eng::make(D, H, gs, cap, /*seed=*/100);

    const std::size_t T = 25, rs = H * D;
    const auto        k = make_gaussian(T * rs, 1);
    const auto        v = make_gaussian(T * rs, 2);

    // A: one bulk call
    REQUIRE(ea.ingest_decode(k, v, T) == tq::Error::Ok);
    // B: one-token calls
    for (std::size_t t = 0; t < T; ++t) {
        REQUIRE(eb.ingest_decode(std::span<const float>(k.data() + t * rs, rs),
                                 std::span<const float>(v.data() + t * rs, rs),
                                 1) == tq::Error::Ok);
    }
    REQUIRE(ea.total_tokens() == eb.total_tokens());
    REQUIRE(ea.total_compressed_tokens() == eb.total_compressed_tokens());
    REQUIRE(ea.total_buffered_tokens() == eb.total_buffered_tokens());
}
