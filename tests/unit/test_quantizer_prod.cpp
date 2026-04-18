// TurboQuantProd scalar tests: round-trip, shape checks, determinism,
// attention_score vs brute-force <query, dequantized key>.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "turboquant/quantizer_prod.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

std::vector<float> make_signal(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

} // namespace

TEST_CASE("TurboQuantProd::make rejects invalid dims", "[quantizer][prod]") {
    REQUIRE_FALSE(tq::TurboQuantProd<3>::make(0, 42).has_value());
    REQUIRE_FALSE(tq::TurboQuantProd<3>::make(tq::kMaxDim + 1, 42).has_value());
}

TEST_CASE("TurboQuantProd: byte sizes match pack policies", "[quantizer][prod]") {
    // Bits=3 → MSE uses 2-bit, packed 4/byte; QJL is 8/byte.
    REQUIRE(tq::TurboQuantProd<3>::mse_packed_bytes(128) == 128 / 4);
    REQUIRE(tq::TurboQuantProd<3>::qjl_packed_bytes(128) == 128 / 8);

    // Bits=4 → MSE 3-bit (aliased to 4-bit = 2/byte).
    REQUIRE(tq::TurboQuantProd<4>::mse_packed_bytes(128) == 128 / 2);
    REQUIRE(tq::TurboQuantProd<4>::qjl_packed_bytes(128) == 128 / 8);
}

TEST_CASE("TurboQuantProd: qjl_scale = sqrt(pi/2)/dim", "[quantizer][prod]") {
    const std::size_t d = 128;
    auto q = tq::TurboQuantProd<3>::make(d, 42);
    REQUIRE(q.has_value());
    const float expected = static_cast<float>(std::sqrt(M_PI / 2.0) / d);
    REQUIRE_THAT(q->qjl_scale(), WithinRel(expected, 1e-6f));
}

TEST_CASE("TurboQuantProd: quantize is deterministic for fixed seed",
          "[quantizer][prod]") {
    const std::size_t d = 128, batch = 3;
    auto a = tq::TurboQuantProd<3>::make(d, 42);
    auto b = tq::TurboQuantProd<3>::make(d, 42);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());

    const auto x = make_signal(batch * d, 1234);

    const std::size_t mb = tq::TurboQuantProd<3>::mse_packed_bytes(d);
    const std::size_t qb = tq::TurboQuantProd<3>::qjl_packed_bytes(d);
    std::vector<std::uint8_t> mA(batch * mb), mB(batch * mb);
    std::vector<std::uint8_t> sA(batch * qb), sB(batch * qb);
    std::vector<float>        rA(batch), rB(batch);
    std::vector<float>        nA(batch), nB(batch);

    REQUIRE(a->quantize(x, batch, mA, sA, rA, nA) == tq::Error::Ok);
    REQUIRE(b->quantize(x, batch, mB, sB, rB, nB) == tq::Error::Ok);

    REQUIRE(mA == mB);
    REQUIRE(sA == sB);
    for (std::size_t i = 0; i < batch; ++i) {
        REQUIRE(rA[i] == rB[i]);
        REQUIRE(nA[i] == nB[i]);
    }
}

TEST_CASE("TurboQuantProd: attention_score is unbiased when averaged over S seeds",
          "[quantizer][prod][.slow]") {
    // Algorithm 2 claim: E[prod_score] = <q, k>   (unbiased asymmetric estimator).
    // MSE-only has a deterministic residual bias <q, r>. Run the same (q, k)
    // pair through many independent QJL seeds — Prod's mean should approach
    // truth while the sample stddev scales as 1/sqrt(n_seeds).
    const std::size_t d   = 128;
    const std::size_t n_q = 4, n_k = 4;
    const std::size_t n_seeds = 64;

    const auto query = make_signal(n_q * d, 11);
    const auto keys  = make_signal(n_k * d, 22);

    // Truth via brute-force.
    std::vector<double> truth(n_q * n_k);
    for (std::size_t q = 0; q < n_q; ++q) {
        for (std::size_t k = 0; k < n_k; ++k) {
            double acc = 0.0;
            for (std::size_t i = 0; i < d; ++i) {
                acc += static_cast<double>(query[q * d + i]) *
                       static_cast<double>(keys[k * d + i]);
            }
            truth[q * n_k + k] = acc;
        }
    }

    const std::size_t mb = tq::TurboQuantProd<3>::mse_packed_bytes(d);
    const std::size_t qb = tq::TurboQuantProd<3>::qjl_packed_bytes(d);

    std::vector<double> sum_prod(n_q * n_k, 0.0);
    for (std::uint32_t s = 0; s < n_seeds; ++s) {
        auto prod = tq::TurboQuantProd<3>::make(d, /*seed=*/42 + s);
        REQUIRE(prod.has_value());
        std::vector<std::uint8_t> mi(n_k * mb), si(n_k * qb);
        std::vector<float>        res(n_k), norms(n_k);
        REQUIRE(prod->quantize(keys, n_k, mi, si, res, norms) == tq::Error::Ok);
        std::vector<float> sc(n_q * n_k);
        REQUIRE(prod->attention_score(query, n_q, mi, si, res, norms, n_k, sc)
                == tq::Error::Ok);
        for (std::size_t i = 0; i < sc.size(); ++i) sum_prod[i] += sc[i];
    }

    // Mean bias << single-realization stddev.
    double mean_abs_bias = 0.0;
    for (std::size_t i = 0; i < sum_prod.size(); ++i) {
        const double mean_prod = sum_prod[i] / static_cast<double>(n_seeds);
        mean_abs_bias += std::fabs(mean_prod - truth[i]);
    }
    mean_abs_bias /= static_cast<double>(sum_prod.size());
    CAPTURE(mean_abs_bias);
    // Truth magnitudes are ~sqrt(d) ≈ 11 for random Gaussian q, k.
    // With 64 seeds the residual bias should be well below 2.
    REQUIRE(mean_abs_bias < 2.0);
}

TEST_CASE("TurboQuantProd::attention_score matches brute-force on unbiased estimator",
          "[quantizer][prod]") {
    // Build random keys, a random query, and verify
    //   attention_score(q, keys)  ==  q · dequantize(keys)   (byte-identical).
    // Both paths use the same k_mse and the same signs → GEMM of the same
    // matrices in different order, so floating-point order differs slightly;
    // we assert a small absolute tolerance.
    const std::size_t d   = 128;
    const std::size_t n_k = 6;
    const std::size_t n_q = 3;

    auto prod = tq::TurboQuantProd<3>::make(d, 42);
    REQUIRE(prod.has_value());

    const auto keys  = make_signal(n_k * d, 11);
    const auto query = make_signal(n_q * d, 22);

    const std::size_t mb = tq::TurboQuantProd<3>::mse_packed_bytes(d);
    const std::size_t qb = tq::TurboQuantProd<3>::qjl_packed_bytes(d);
    std::vector<std::uint8_t> mse_idx(n_k * mb), signs(n_k * qb);
    std::vector<float>        res(n_k), norms(n_k);
    REQUIRE(prod->quantize(keys, n_k, mse_idx, signs, res, norms) == tq::Error::Ok);

    // Path A: attention_score
    std::vector<float> scores(n_q * n_k);
    REQUIRE(prod->attention_score(query, n_q, mse_idx, signs, res, norms, n_k, scores)
            == tq::Error::Ok);

    // Path B: dequantize keys, then q @ k_hat.T
    std::vector<float> k_hat(n_k * d);
    REQUIRE(prod->dequantize(mse_idx, signs, res, norms, n_k, k_hat) == tq::Error::Ok);

    std::vector<float> scores_ref(n_q * n_k);
    for (std::size_t q = 0; q < n_q; ++q) {
        for (std::size_t k = 0; k < n_k; ++k) {
            double acc = 0.0;
            for (std::size_t i = 0; i < d; ++i) {
                acc += static_cast<double>(query[q * d + i]) *
                       static_cast<double>(k_hat[k * d + i]);
            }
            scores_ref[q * n_k + k] = static_cast<float>(acc);
        }
    }

    for (std::size_t q = 0; q < n_q; ++q) {
        for (std::size_t k = 0; k < n_k; ++k) {
            CAPTURE(q, k);
            REQUIRE_THAT(scores[q * n_k + k],
                         WithinAbs(scores_ref[q * n_k + k], 1e-3f));
        }
    }
}

TEST_CASE("TurboQuantProd: shape mismatch surfaces as ShapeMismatch",
          "[quantizer][prod]") {
    const std::size_t d = 128, batch = 2;
    auto q = tq::TurboQuantProd<3>::make(d, 42);
    REQUIRE(q.has_value());
    std::vector<float>        x(batch * d);
    std::vector<std::uint8_t> mse_idx(batch * q->mse_packed_bytes(d));
    std::vector<std::uint8_t> signs(batch * q->qjl_packed_bytes(d) + 1);  // wrong
    std::vector<float>        res(batch), nr(batch);
    REQUIRE(q->quantize(x, batch, mse_idx, signs, res, nr) == tq::Error::ShapeMismatch);
}

TEST_CASE("TurboQuantProd::from_matrices accepts external Pi and S",
          "[quantizer][prod][fixture]") {
    const std::size_t d = 64;
    auto seed_prod = tq::TurboQuantProd<3>::make(d, 5);
    REQUIRE(seed_prod.has_value());

    std::vector<float> pi(seed_prod->mse().rotation().matrix().begin(),
                          seed_prod->mse().rotation().matrix().end());
    std::vector<float> s (seed_prod->s_matrix().begin(),
                          seed_prod->s_matrix().end());

    auto loaded = tq::TurboQuantProd<3>::from_matrices(pi, s, d);
    REQUIRE(loaded.has_value());

    const auto x = make_signal(1 * d, 99);
    const std::size_t mb = tq::TurboQuantProd<3>::mse_packed_bytes(d);
    const std::size_t qb = tq::TurboQuantProd<3>::qjl_packed_bytes(d);

    std::vector<std::uint8_t> mA(mb), mB(mb), sA(qb), sB(qb);
    std::vector<float>        rA(1), rB(1), nA(1), nB(1);
    REQUIRE(seed_prod->quantize(x, 1, mA, sA, rA, nA) == tq::Error::Ok);
    REQUIRE(loaded   ->quantize(x, 1, mB, sB, rB, nB) == tq::Error::Ok);
    REQUIRE(mA == mB);
    REQUIRE(sA == sB);
    REQUIRE(rA[0] == rB[0]);
    REQUIRE(nA[0] == nB[0]);
}
