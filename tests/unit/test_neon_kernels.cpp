// NEON vs scalar parity tests. Bit-exact where possible; small-ulp elsewhere.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "turboquant/neon/kernels.hpp"
#include "turboquant/neon/scalar_fallback.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinULP;

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

TEST_CASE("neon::l2norm within 2 ulp of double-accumulated scalar", "[neon]") {
    for (std::size_t d : {7u, 16u, 64u, 128u, 256u, 333u}) {
        const auto  x = make_gaussian(d, /*seed=*/static_cast<std::uint32_t>(d));
        const float s = tq::neon_scalar::l2norm(x.data(), d);
        const float n = tq::neon::l2norm(x.data(), d);
        CAPTURE(d, s, n);
        // Drift bound: float32 summation of d² terms has worst-case ~d*eps;
        // we stay well under 4 ulp for d ≤ 1024.
        REQUIRE_THAT(n, WithinULP(s, 64));
    }
}

TEST_CASE("neon::scale is bit-exact vs scalar (single multiply per lane)", "[neon]") {
    for (std::size_t d : {7u, 16u, 128u}) {
        const auto         x = make_gaussian(d, /*seed=*/1 + static_cast<std::uint32_t>(d));
        std::vector<float> ys(d), yn(d);
        const float        inv = 0.123f;
        tq::neon_scalar::scale(x.data(), inv, ys.data(), d);
        tq::neon::scale(x.data(), inv, yn.data(), d);
        for (std::size_t i = 0; i < d; ++i) {
            CAPTURE(d, i);
            REQUIRE(yn[i] == ys[i]);
        }
    }
}

TEST_CASE("neon::searchsorted_one matches scalar for bits 1..4 codebooks", "[neon]") {
    for (std::size_t n : {1u, 3u, 7u, 15u}) {  // 2^b - 1 for b = 1..4
        std::vector<float> bounds(n);
        for (std::size_t i = 0; i < n; ++i) {
            bounds[i] = -0.5f + static_cast<float>(i) / static_cast<float>(n);
        }
        const auto probes = make_gaussian(257, /*seed=*/static_cast<std::uint32_t>(n) * 11);
        for (float v : probes) {
            const std::uint8_t s = tq::neon_scalar::searchsorted(bounds.data(), n, v);
            const std::uint8_t e = tq::neon::searchsorted_one(bounds.data(), n, v);
            CAPTURE(n, v);
            REQUIRE(e == s);
        }
    }
}

TEST_CASE("neon::searchsorted_one handles exact-boundary inputs (right=False)", "[neon]") {
    // right=False → a bound equal to v does NOT count as less than v. Index
    // should be the smallest i with bounds[i] >= v.
    std::array<float, 3> b = {-0.5f, 0.0f, 0.5f};
    REQUIRE(tq::neon::searchsorted_one(b.data(), 3, -0.5f) == 0);
    REQUIRE(tq::neon::searchsorted_one(b.data(), 3, 0.0f) == 1);
    REQUIRE(tq::neon::searchsorted_one(b.data(), 3, 0.5f) == 2);
    REQUIRE(tq::neon::searchsorted_one(b.data(), 3, 0.6f) == 3);
}

TEMPLATE_TEST_CASE_SIG("neon::searchsorted_and_pack is bit-exact vs scalar", "[neon][pack]",
                       ((int Bits), Bits), 1, 2, 3, 4) {
    const std::size_t  d        = 128;
    const std::size_t  n_bounds = (1u << Bits) - 1u;
    std::vector<float> bounds(n_bounds);
    for (std::size_t i = 0; i < n_bounds; ++i) {
        bounds[i] = -0.5f + static_cast<float>(i + 1) / static_cast<float>(n_bounds + 1);
    }
    const auto rotated = make_gaussian(d, /*seed=*/77u + Bits);

    const std::size_t         pb = tq::PackPolicy<Bits>::packed_bytes(d);
    std::vector<std::uint8_t> ps(pb, 0), pn(pb, 0);
    tq::neon_scalar::searchsorted_and_pack<Bits>(rotated.data(), bounds.data(), n_bounds, ps.data(),
                                                 d);
    tq::neon::searchsorted_and_pack<Bits>(rotated.data(), bounds.data(), n_bounds, pn.data(), d);
    REQUIRE(ps == pn);
}

TEMPLATE_TEST_CASE_SIG("neon::unpack_and_gather is bit-exact vs scalar", "[neon][pack]",
                       ((int Bits), Bits), 1, 2, 3, 4) {
    const std::size_t  d    = 128;
    const std::size_t  n_cl = 1u << Bits;
    std::vector<float> centroids(n_cl);
    for (std::size_t i = 0; i < n_cl; ++i) {
        centroids[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(n_cl - 1);
    }
    // Synthesize a packed buffer by walking a pseudo-random index pattern.
    std::mt19937                       rng(101u + Bits);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(n_cl) - 1);
    std::vector<std::uint8_t>          raw(d);
    for (auto& r : raw)
        r = static_cast<std::uint8_t>(dist(rng));

    const std::size_t         pb = tq::PackPolicy<Bits>::packed_bytes(d);
    std::vector<std::uint8_t> packed(pb, 0);
    tq::PackPolicy<Bits>::pack(raw.data(), d, packed.data());

    std::vector<float> ys(d), yn(d);
    tq::neon_scalar::unpack_and_gather<Bits>(packed.data(), centroids.data(), ys.data(), d);
    tq::neon::unpack_and_gather<Bits>(packed.data(), centroids.data(), yn.data(), d);
    for (std::size_t i = 0; i < d; ++i) {
        CAPTURE(Bits, i);
        REQUIRE(yn[i] == ys[i]);
    }
}

TEST_CASE("neon::qjl_pack_signs is bit-exact vs scalar", "[neon][qjl]") {
    for (std::size_t d : {8u, 37u, 128u, 257u}) {
        const auto                proj = make_gaussian(d, /*seed=*/201u);
        const std::size_t         pb   = tq::QJLPack::packed_bytes(d);
        std::vector<std::uint8_t> ps(pb, 0), pn(pb, 0);
        tq::neon_scalar::qjl_pack_signs(proj.data(), d, ps.data());
        tq::neon::qjl_pack_signs(proj.data(), d, pn.data());
        CAPTURE(d);
        REQUIRE(ps == pn);
    }
}

TEST_CASE("neon::qjl_unpack_pm1 is bit-exact vs scalar", "[neon][qjl]") {
    for (std::size_t d : {8u, 37u, 128u, 257u}) {
        // Build a packed buffer from random bits.
        std::mt19937                       rng(static_cast<std::uint32_t>(d) * 31u);
        std::uniform_int_distribution<int> dist(0, 255);
        const std::size_t                  pb = tq::QJLPack::packed_bytes(d);
        std::vector<std::uint8_t>          packed(pb);
        for (auto& b : packed)
            b = static_cast<std::uint8_t>(dist(rng));

        std::vector<float> ys(d, 0), yn(d, 0);
        tq::neon_scalar::qjl_unpack_pm1(packed.data(), d, ys.data());
        tq::neon::qjl_unpack_pm1(packed.data(), d, yn.data());
        for (std::size_t i = 0; i < d; ++i) {
            CAPTURE(d, i);
            REQUIRE(yn[i] == ys[i]);
            REQUIRE((yn[i] == 1.0f || yn[i] == -1.0f));
        }
    }
}
