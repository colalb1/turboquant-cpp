// M8: compute_hybrid_attention — three paths.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "turboquant/compressed_kv_store.hpp"
#include "turboquant/score.hpp"
#include "turboquant/value_quant.hpp"

using Catch::Matchers::WithinAbs;

namespace {

std::vector<float> make_gaussian(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> out(n);
    for (auto& v : out) v = nd(rng);
    return out;
}

// Reference: plain GQA attention over (H, N, D) K/V and (n_q, Q, D) query.
void reference_attention(const std::vector<float>& query,
                         std::size_t               n_q,
                         std::size_t               Q,
                         const std::vector<float>& k_all,
                         const std::vector<float>& v_all,
                         std::size_t               H,
                         std::size_t               N,
                         std::size_t               D,
                         float                     scale,
                         std::vector<float>&       out)
{
    out.assign(n_q * Q * D, 0.0f);
    if (N == 0) return;
    const std::size_t G = Q / H;

    std::vector<float> scores(N);
    for (std::size_t h = 0; h < H; ++h) {
        for (std::size_t g = 0; g < G; ++g) {
            for (std::size_t t = 0; t < n_q; ++t) {
                const float* q = query.data() + t * Q * D + (h * G + g) * D;
                for (std::size_t n = 0; n < N; ++n) {
                    const float* k = k_all.data() + h * N * D + n * D;
                    float s = 0.0f;
                    for (std::size_t d = 0; d < D; ++d) s += q[d] * k[d];
                    scores[n] = s * scale;
                }
                float mx = scores[0];
                for (std::size_t n = 1; n < N; ++n)
                    if (scores[n] > mx) mx = scores[n];
                float sum = 0.0f;
                for (std::size_t n = 0; n < N; ++n) {
                    scores[n] = std::exp(scores[n] - mx);
                    sum      += scores[n];
                }
                const float inv = 1.0f / sum;
                for (std::size_t n = 0; n < N; ++n) scores[n] *= inv;

                float* o = out.data() + t * Q * D + (h * G + g) * D;
                for (std::size_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (std::size_t n = 0; n < N; ++n) {
                        acc += scores[n] * v_all[h * N * D + n * D + d];
                    }
                    o[d] = acc;
                }
            }
        }
    }
}

// Transpose (T, H, D) → (H, T, D) — used to convert recent tensors to
// per-head layout for the reference attention.
std::vector<float> thd_to_htd(const std::vector<float>& src,
                              std::size_t T, std::size_t H, std::size_t D)
{
    std::vector<float> out(T * H * D);
    for (std::size_t h = 0; h < H; ++h)
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t d = 0; d < D; ++d)
                out[(h * T + t) * D + d] = src[(t * H + h) * D + d];
    return out;
}

template <int KeyBits, int ValBits>
void dequant_flat(tq::CompressedKVStore<KeyBits, ValBits>& store,
                  std::vector<float>&                      k_out,
                  std::vector<float>&                      v_out)
{
    auto flat_r = store.get_flat();
    REQUIRE(flat_r);
    auto flat = *flat_r;
    const std::size_t D = store.head_dim();
    const std::size_t H = store.num_kv_heads();
    const std::size_t T = flat.total_tokens;
    k_out.assign(H * T * D, 0.0f);
    v_out.assign(H * T * D, 0.0f);
    if (T == 0) return;

    REQUIRE(store.quantizer().dequantize(
        flat.mse_indices, flat.qjl_signs, flat.residual_norms, flat.norms,
        H * T, std::span<float>(k_out.data(), H * T * D)) == tq::Error::Ok);
    REQUIRE(tq::ValueCodec<ValBits>::dequantize(
        flat.val_data, flat.val_scales, flat.val_zeros,
        H * T, D, store.value_group_size(),
        std::span<float>(v_out.data(), H * T * D)) == tq::Error::Ok);
}

} // namespace

// -----------------------------------------------------------------------------
// Zero-output path: no history (< MIN_HISTORY) and no recent.
// -----------------------------------------------------------------------------

TEST_CASE("compute_hybrid_attention with no history and no recent is zero",
          "[score]") {
    const std::size_t D = 32, H = 2, Q = 4, n_q = 1, gs = 16;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/1);

    auto query = make_gaussian(n_q * Q * D, 9);
    std::vector<float> actual(n_q * Q * D, /*sentinel=*/42.0f);

    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        query, n_q, Q, store, {}, {}, 0, 0.0f, actual) == tq::Error::Ok);

    for (float v : actual) REQUIRE(v == 0.0f);
}

TEST_CASE("compute_hybrid_attention with < MIN_HISTORY and no recent is zero",
          "[score]") {
    const std::size_t D = 32, H = 2, Q = 4, n_q = 1, gs = 16;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/2);

    const std::size_t T = tq::MIN_HISTORY_FOR_TQ - 1;  // 15 < 16
    const auto k = make_gaussian(T * H * D, 3);
    const auto v = make_gaussian(T * H * D, 4);
    REQUIRE(store.append_chunk(k, v, T) == tq::Error::Ok);

    auto query = make_gaussian(n_q * Q * D, 5);
    std::vector<float> actual(n_q * Q * D, 42.0f);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        query, n_q, Q, store, {}, {}, 0, 0.0f, actual) == tq::Error::Ok);
    for (float x : actual) REQUIRE(x == 0.0f);
}

// -----------------------------------------------------------------------------
// Compressed-only path
// -----------------------------------------------------------------------------

TEST_CASE("compute_hybrid_attention compressed-only matches reference",
          "[score]") {
    const std::size_t D = 32, H = 2, Q = 4, n_q = 2, gs = 16;
    const std::size_t T = 20;  // >= 16
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/7);

    const auto keys   = make_gaussian(T * H * D, 11);
    const auto values = make_gaussian(T * H * D, 12);
    REQUIRE(store.append_chunk(keys, values, T) == tq::Error::Ok);

    std::vector<float> k_ref, v_ref;
    dequant_flat(store, k_ref, v_ref);

    const auto query = make_gaussian(n_q * Q * D, 99);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> expected;
    reference_attention(query, n_q, Q, k_ref, v_ref, H, T, D, scale, expected);

    std::vector<float> actual(n_q * Q * D);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        query, n_q, Q, store, {}, {}, 0, 0.0f, actual) == tq::Error::Ok);

    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE_THAT(actual[i], WithinAbs(expected[i], 1e-4f));
    }
}

TEST_CASE("compute_hybrid_attention compressed-only across multiple chunks",
          "[score]") {
    const std::size_t D = 16, H = 2, Q = 4, n_q = 1, gs = 8;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/13);

    const auto k1 = make_gaussian(10 * H * D, 21);
    const auto v1 = make_gaussian(10 * H * D, 22);
    const auto k2 = make_gaussian(12 * H * D, 23);
    const auto v2 = make_gaussian(12 * H * D, 24);
    REQUIRE(store.append_chunk(k1, v1, 10) == tq::Error::Ok);
    REQUIRE(store.append_chunk(k2, v2, 12) == tq::Error::Ok);

    std::vector<float> k_ref, v_ref;
    dequant_flat(store, k_ref, v_ref);

    const auto query = make_gaussian(n_q * Q * D, 77);
    const float scale = 0.25f;  // custom scale

    std::vector<float> expected;
    reference_attention(query, n_q, Q, k_ref, v_ref, H, 22, D, scale, expected);

    std::vector<float> actual(n_q * Q * D);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        query, n_q, Q, store, {}, {}, 0, scale, actual) == tq::Error::Ok);

    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE_THAT(actual[i], WithinAbs(expected[i], 1e-4f));
    }
}

// -----------------------------------------------------------------------------
// Recent-only path
// -----------------------------------------------------------------------------

TEST_CASE("compute_hybrid_attention recent-only matches fp32 reference",
          "[score]") {
    const std::size_t D = 32, H = 2, Q = 4, n_q = 1, gs = 16;
    const std::size_t N_rec = 8;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/31);

    const auto recent_k = make_gaussian(N_rec * H * D, 41);
    const auto recent_v = make_gaussian(N_rec * H * D, 42);
    const auto k_ref    = thd_to_htd(recent_k, N_rec, H, D);
    const auto v_ref    = thd_to_htd(recent_v, N_rec, H, D);

    const auto query = make_gaussian(n_q * Q * D, 61);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> expected;
    reference_attention(query, n_q, Q, k_ref, v_ref, H, N_rec, D, scale, expected);

    std::vector<float> actual(n_q * Q * D);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        query, n_q, Q, store, recent_k, recent_v, N_rec, 0.0f, actual)
            == tq::Error::Ok);

    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE_THAT(actual[i], WithinAbs(expected[i], 1e-5f));
    }
}

TEST_CASE("compute_hybrid_attention uses recent-only when history < MIN_HISTORY",
          "[score]") {
    const std::size_t D = 16, H = 2, Q = 4, n_q = 1, gs = 8;
    const std::size_t T_small = 5, N_rec = 6;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/51);

    // Small compressed history — must be ignored.
    const auto k_small = make_gaussian(T_small * H * D, 52);
    const auto v_small = make_gaussian(T_small * H * D, 53);
    REQUIRE(store.append_chunk(k_small, v_small, T_small) == tq::Error::Ok);

    const auto recent_k = make_gaussian(N_rec * H * D, 54);
    const auto recent_v = make_gaussian(N_rec * H * D, 55);
    const auto k_ref    = thd_to_htd(recent_k, N_rec, H, D);
    const auto v_ref    = thd_to_htd(recent_v, N_rec, H, D);

    const auto query = make_gaussian(n_q * Q * D, 56);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> expected;
    reference_attention(query, n_q, Q, k_ref, v_ref, H, N_rec, D, scale, expected);

    std::vector<float> actual(n_q * Q * D);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        query, n_q, Q, store, recent_k, recent_v, N_rec, 0.0f, actual)
            == tq::Error::Ok);

    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE_THAT(actual[i], WithinAbs(expected[i], 1e-5f));
    }
}

// -----------------------------------------------------------------------------
// Hybrid path
// -----------------------------------------------------------------------------

TEST_CASE("compute_hybrid_attention hybrid concats compressed + recent",
          "[score]") {
    const std::size_t D = 32, H = 2, Q = 4, n_q = 1, gs = 16;
    const std::size_t T_hist = 20, N_rec = 6;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/71);

    const auto k_hist = make_gaussian(T_hist * H * D, 81);
    const auto v_hist = make_gaussian(T_hist * H * D, 82);
    REQUIRE(store.append_chunk(k_hist, v_hist, T_hist) == tq::Error::Ok);

    std::vector<float> k_dq, v_dq;
    dequant_flat(store, k_dq, v_dq);

    const auto recent_k = make_gaussian(N_rec * H * D, 83);
    const auto recent_v = make_gaussian(N_rec * H * D, 84);
    const auto k_rec_htd = thd_to_htd(recent_k, N_rec, H, D);
    const auto v_rec_htd = thd_to_htd(recent_v, N_rec, H, D);

    // Build concatenated (H, T_hist + N_rec, D) reference.
    const std::size_t N_all = T_hist + N_rec;
    std::vector<float> k_ref(H * N_all * D), v_ref(H * N_all * D);
    for (std::size_t h = 0; h < H; ++h) {
        std::copy_n(k_dq.data() + h * T_hist * D, T_hist * D,
                    k_ref.data() + h * N_all * D);
        std::copy_n(v_dq.data() + h * T_hist * D, T_hist * D,
                    v_ref.data() + h * N_all * D);
        std::copy_n(k_rec_htd.data() + h * N_rec * D, N_rec * D,
                    k_ref.data() + h * N_all * D + T_hist * D);
        std::copy_n(v_rec_htd.data() + h * N_rec * D, N_rec * D,
                    v_ref.data() + h * N_all * D + T_hist * D);
    }

    const auto query = make_gaussian(n_q * Q * D, 91);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> expected;
    reference_attention(query, n_q, Q, k_ref, v_ref, H, N_all, D, scale, expected);

    std::vector<float> actual(n_q * Q * D);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        query, n_q, Q, store, recent_k, recent_v, N_rec, 0.0f, actual)
            == tq::Error::Ok);

    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE_THAT(actual[i], WithinAbs(expected[i], 1e-4f));
    }
}

// -----------------------------------------------------------------------------
// GQA expansion
// -----------------------------------------------------------------------------

TEST_CASE("compute_hybrid_attention handles gqa_ratio > 1", "[score]") {
    const std::size_t D = 16, H = 2, gqa = 4, gs = 8;
    const std::size_t Q = H * gqa;   // 8 query heads, 2 kv heads
    const std::size_t n_q = 1;
    const std::size_t T_hist = 18, N_rec = 4;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/101);

    const auto k_hist = make_gaussian(T_hist * H * D, 111);
    const auto v_hist = make_gaussian(T_hist * H * D, 112);
    REQUIRE(store.append_chunk(k_hist, v_hist, T_hist) == tq::Error::Ok);

    std::vector<float> k_dq, v_dq;
    dequant_flat(store, k_dq, v_dq);

    const auto recent_k = make_gaussian(N_rec * H * D, 113);
    const auto recent_v = make_gaussian(N_rec * H * D, 114);
    const auto k_rec_htd = thd_to_htd(recent_k, N_rec, H, D);
    const auto v_rec_htd = thd_to_htd(recent_v, N_rec, H, D);

    const std::size_t N_all = T_hist + N_rec;
    std::vector<float> k_ref(H * N_all * D), v_ref(H * N_all * D);
    for (std::size_t h = 0; h < H; ++h) {
        std::copy_n(k_dq.data() + h * T_hist * D, T_hist * D,
                    k_ref.data() + h * N_all * D);
        std::copy_n(v_dq.data() + h * T_hist * D, T_hist * D,
                    v_ref.data() + h * N_all * D);
        std::copy_n(k_rec_htd.data() + h * N_rec * D, N_rec * D,
                    k_ref.data() + h * N_all * D + T_hist * D);
        std::copy_n(v_rec_htd.data() + h * N_rec * D, N_rec * D,
                    v_ref.data() + h * N_all * D + T_hist * D);
    }

    const auto query = make_gaussian(n_q * Q * D, 121);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> expected;
    reference_attention(query, n_q, Q, k_ref, v_ref, H, N_all, D, scale, expected);

    std::vector<float> actual(n_q * Q * D);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        query, n_q, Q, store, recent_k, recent_v, N_rec, 0.0f, actual)
            == tq::Error::Ok);

    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE_THAT(actual[i], WithinAbs(expected[i], 1e-4f));
    }
}

TEST_CASE("compute_hybrid_attention rejects mismatched GQA shapes", "[score]") {
    const std::size_t D = 16, H = 2, gs = 8;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/131);
    std::vector<float> q(1 * 3 * D), o(1 * 3 * D);  // 3 query heads, H=2 → 3%2!=0
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        q, 1, 3, store, {}, {}, 0, 0.0f, o) == tq::Error::InvalidArgument);
}

TEST_CASE("compute_hybrid_attention rejects shape mismatches", "[score]") {
    const std::size_t D = 16, H = 2, Q = 4, gs = 8;
    using Store = tq::CompressedKVStore<4, 4>;
    auto store = *Store::make(D, H, gs, /*seed=*/141);

    std::vector<float> q_bad(3);            // wrong size
    std::vector<float> o(1 * Q * D);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        q_bad, 1, Q, store, {}, {}, 0, 0.0f, o) == tq::Error::ShapeMismatch);

    std::vector<float> q(1 * Q * D);
    std::vector<float> rec_k_bad(5);        // wrong size for N_rec=4
    std::vector<float> rec_v(4 * H * D);
    REQUIRE(tq::compute_hybrid_attention<4, 4>(
        q, 1, Q, store, rec_k_bad, rec_v, 4, 0.0f, o)
            == tq::Error::ShapeMismatch);
}
