// Per-group value quantization: round-trip error, shape handling,
// NEON vs scalar parity.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "turboquant/neon/kernels.hpp"
#include "turboquant/neon/scalar_fallback.hpp"
#include "turboquant/value_quant.hpp"

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

TEMPLATE_TEST_CASE_SIG("ValueCodec round-trip is within one level of error", "[value][roundtrip]",
                       ((int Bits), Bits), 2, 4, 8) {
    using Codec             = tq::ValueCodec<Bits>;
    const std::size_t dim   = 128;
    const std::size_t gs    = 32;
    const std::size_t batch = 8;
    const auto        v     = make_gaussian(batch * dim, /*seed=*/1000u + Bits);

    const std::size_t         pb = Codec::packed_bytes(dim);
    const std::size_t         ng = dim / gs;
    std::vector<std::uint8_t> data(batch * pb);
    std::vector<float>        scales(batch * ng), zeros(batch * ng);
    REQUIRE(Codec::quantize(v, batch, dim, gs, data, scales, zeros) == tq::Error::Ok);

    std::vector<float> v_rt(batch * dim);
    REQUIRE(Codec::dequantize(data, scales, zeros, batch, dim, gs, v_rt) == tq::Error::Ok);

    // Per-element reconstruction error must be ≤ scale_g (one quant level).
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t g = 0; g < ng; ++g) {
            const float sc = scales[b * ng + g];
            for (std::size_t i = 0; i < gs; ++i) {
                const std::size_t idx = b * dim + g * gs + i;
                const float       err = std::fabs(v_rt[idx] - v[idx]);
                CAPTURE(Bits, b, g, i, sc);
                // Asymmetric rounding: up to sc/2 ideally, but with the
                // 1e-10 scale floor and boundary rounding we allow sc + 1e-6.
                REQUIRE(err <= sc * 0.5001f + 1e-6f);
            }
        }
    }
}

TEST_CASE("ValueCodec<2> matches explicit Python packing layout", "[value][pack]") {
    // Hand-authored: a single row of 16 floats arranged so each 2-bit
    // index slot lands at a known value. bits=2, gs=16 → 1 group.
    const std::size_t dim = 16, gs = 16;
    // min=0, max=3 per group → scale=1.0, zero=0.0. Values 0,1,2,3 cycle.
    std::vector<float> v(dim);
    for (std::size_t i = 0; i < dim; ++i)
        v[i] = static_cast<float>(i % 4);

    std::vector<std::uint8_t> data(tq::ValueCodec<2>::packed_bytes(dim));
    std::vector<float>        scales(1), zeros(1);
    REQUIRE(tq::ValueCodec<2>::quantize(v, 1, dim, gs, data, scales, zeros) == tq::Error::Ok);

    REQUIRE_THAT(scales[0], WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(zeros[0], WithinAbs(0.0f, 1e-6f));
    // LSB-first pack of [0,1,2,3,0,1,2,3,...] with 4 vals per byte:
    // byte = 0 | (1<<2) | (2<<4) | (3<<6) = 0b11100100 = 0xE4
    for (std::uint8_t byte : data)
        REQUIRE(byte == 0xE4);
}

TEST_CASE("ValueCodec rejects dim not divisible by group_size", "[value][shape]") {
    std::vector<float>        v(30);
    std::vector<std::uint8_t> data(30);
    std::vector<float>        scales(1), zeros(1);
    REQUIRE(tq::ValueCodec<2>::quantize(v, 1, /*dim=*/30, /*gs=*/32, data, scales, zeros) ==
            tq::Error::InvalidDim);
}

TEST_CASE("neon::group_quant_row matches scalar on random input", "[neon][value]") {
    for (int n_levels : {3, 15, 255}) {
        for (std::size_t gs : {8u, 16u, 32u}) {
            const std::size_t         dim = 4 * gs;
            const auto                v   = make_gaussian(dim,
                                                          /*seed=*/401u + static_cast<std::uint32_t>(n_levels) +
                                                              static_cast<std::uint32_t>(gs));
            std::vector<std::uint8_t> is(dim), in(dim);
            std::vector<float>        ss(dim / gs), sn(dim / gs);
            std::vector<float>        zs(dim / gs), zn(dim / gs);
            tq::neon_scalar::group_quant_row(v.data(), dim, gs, n_levels, is.data(), ss.data(),
                                             zs.data());
            tq::neon::group_quant_row(v.data(), dim, gs, n_levels, in.data(), sn.data(), zn.data());
            for (std::size_t g = 0; g < dim / gs; ++g) {
                CAPTURE(n_levels, gs, g);
                REQUIRE(ss[g] == sn[g]);
                REQUIRE(zs[g] == zn[g]);
            }
            for (std::size_t i = 0; i < dim; ++i) {
                CAPTURE(n_levels, gs, i);
                REQUIRE(is[i] == in[i]);
            }
        }
    }
}

TEST_CASE("neon::group_dequant_row matches scalar on random input", "[neon][value]") {
    std::mt19937                       rng(1234);
    std::uniform_int_distribution<int> di(0, 255);
    std::normal_distribution<float>    nd(0.0f, 1.0f);
    for (std::size_t gs : {8u, 16u, 32u}) {
        const std::size_t         dim = 4 * gs;
        std::vector<std::uint8_t> idx(dim);
        std::vector<float>        scales(dim / gs), zeros(dim / gs);
        for (auto& v : idx)
            v = static_cast<std::uint8_t>(di(rng));
        for (auto& v : scales)
            v = std::fabs(nd(rng)) + 1e-3f;
        for (auto& v : zeros)
            v = nd(rng);

        std::vector<float> xs(dim), xn(dim);
        tq::neon_scalar::group_dequant_row(idx.data(), dim, gs, scales.data(), zeros.data(),
                                           xs.data());
        tq::neon::group_dequant_row(idx.data(), dim, gs, scales.data(), zeros.data(), xn.data());
        for (std::size_t i = 0; i < dim; ++i) {
            CAPTURE(gs, i);
            REQUIRE(xs[i] == xn[i]);
        }
    }
}
