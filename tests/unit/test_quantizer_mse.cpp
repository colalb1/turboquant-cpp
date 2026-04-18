// TurboQuantMSE scalar-path tests: factory, round-trip, norm preservation,
// determinism, per-bit sweep. Parity vs Python lives in tests/parity/ (M3 end).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include "turboquant/quantizer_mse.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

template <int Bits>
void roundtrip_close(std::size_t d, std::size_t batch, double tol) {
    auto q = tq::TurboQuantMSE<Bits>::make(d, /*seed=*/42);
    REQUIRE(q.has_value());

    // Build a batch of unit-ish vectors with known norms.
    std::vector<float> x(batch * d);
    for (std::size_t b = 0; b < batch; ++b) {
        double s = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            const float v = std::sin(0.13f * static_cast<float>(i) + 0.7f * static_cast<float>(b));
            x[b * d + i] = v;
            s += static_cast<double>(v) * v;
        }
        const float norm = static_cast<float>(std::sqrt(s));
        // Rescale so each row has a distinct norm in [1, 3].
        const float target = 1.0f + static_cast<float>(b) * 2.0f / static_cast<float>(batch);
        const float factor = target / norm;
        for (std::size_t i = 0; i < d; ++i) x[b * d + i] *= factor;
    }

    const std::size_t pb = tq::TurboQuantMSE<Bits>::packed_bytes(d);
    std::vector<std::uint8_t> idx(batch * pb);
    std::vector<float>        norms(batch);
    std::vector<float>        x_hat(batch * d);

    REQUIRE(q->quantize(x, batch, idx, norms) == tq::Error::Ok);
    REQUIRE(q->dequantize(idx, norms, batch, x_hat) == tq::Error::Ok);

    // Norms preserved exactly (they're stored verbatim).
    for (std::size_t b = 0; b < batch; ++b) {
        double s = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            const double v = static_cast<double>(x[b * d + i]);
            s += v * v;
        }
        const float expected = static_cast<float>(std::sqrt(s));
        CAPTURE(b);
        REQUIRE_THAT(norms[b], WithinRel(expected, 1e-5f));
    }

    // Reconstruction MSE per-coordinate should be bounded by something
    // generous but monotone in Bits.
    double sse = 0.0;
    double ref = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d_ = static_cast<double>(x[i]) - static_cast<double>(x_hat[i]);
        sse += d_ * d_;
        ref += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    const double rel_mse = sse / ref;
    CAPTURE(Bits, rel_mse);
    REQUIRE(rel_mse < tol);
}

} // namespace

TEST_CASE("TurboQuantMSE::make rejects invalid dims", "[quantizer][mse]") {
    REQUIRE_FALSE(tq::TurboQuantMSE<3>::make(0, 42).has_value());
    REQUIRE_FALSE(tq::TurboQuantMSE<3>::make(tq::kMaxDim + 1, 42).has_value());
}

TEST_CASE("TurboQuantMSE: packed_bytes matches PackPolicy", "[quantizer][mse]") {
    REQUIRE(tq::TurboQuantMSE<1>::packed_bytes(128) == 128 / 8);
    REQUIRE(tq::TurboQuantMSE<2>::packed_bytes(128) == 128 / 4);
    REQUIRE(tq::TurboQuantMSE<3>::packed_bytes(128) == 128 / 2);  // 3-bit → 4-bit storage
    REQUIRE(tq::TurboQuantMSE<4>::packed_bytes(128) == 128 / 2);
}

TEST_CASE("TurboQuantMSE: quantize is deterministic for fixed seed",
          "[quantizer][mse]") {
    const std::size_t d = 128, batch = 3;
    auto a = tq::TurboQuantMSE<3>::make(d, 42);
    auto b = tq::TurboQuantMSE<3>::make(d, 42);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());

    std::vector<float> x(batch * d);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::cos(0.09f * static_cast<float>(i));

    const std::size_t pb = tq::TurboQuantMSE<3>::packed_bytes(d);
    std::vector<std::uint8_t> idxA(batch * pb), idxB(batch * pb);
    std::vector<float>        nA(batch), nB(batch);

    REQUIRE(a->quantize(x, batch, idxA, nA) == tq::Error::Ok);
    REQUIRE(b->quantize(x, batch, idxB, nB) == tq::Error::Ok);

    for (std::size_t i = 0; i < idxA.size(); ++i) {
        CAPTURE(i);
        REQUIRE(idxA[i] == idxB[i]);
    }
    for (std::size_t i = 0; i < nA.size(); ++i) {
        CAPTURE(i);
        REQUIRE(nA[i] == nB[i]);
    }
}

TEST_CASE("TurboQuantMSE<1> round-trip (1-bit)", "[quantizer][mse]") {
    // 1-bit: very lossy — relax the reconstruction bound.
    roundtrip_close<1>(/*d=*/128, /*batch=*/4, /*rel_mse<*/ 0.70);
}

TEST_CASE("TurboQuantMSE<2> round-trip (2-bit)", "[quantizer][mse]") {
    roundtrip_close<2>(128, 4, 0.25);
}

TEST_CASE("TurboQuantMSE<3> round-trip (3-bit)", "[quantizer][mse]") {
    roundtrip_close<3>(128, 4, 0.10);
}

TEST_CASE("TurboQuantMSE<4> round-trip (4-bit)", "[quantizer][mse]") {
    roundtrip_close<4>(128, 4, 0.04);
}

TEST_CASE("TurboQuantMSE: reconstruction MSE decreases with bits",
          "[quantizer][mse]") {
    const std::size_t d = 128, batch = 4;
    std::vector<float> x(batch * d);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(0.03f * static_cast<float>(i)) * 2.0f;
    }

    auto run = [&](auto& q_opt) {
        auto& q = *q_opt;
        using Q = std::decay_t<decltype(q)>;
        const std::size_t pb = Q::packed_bytes(d);
        std::vector<std::uint8_t> idx(batch * pb);
        std::vector<float>        nr(batch), xh(batch * d);
        REQUIRE(q.quantize(x, batch, idx, nr)       == tq::Error::Ok);
        REQUIRE(q.dequantize(idx, nr, batch, xh)    == tq::Error::Ok);
        double sse = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double d_ = static_cast<double>(x[i]) - static_cast<double>(xh[i]);
            sse += d_ * d_;
        }
        return sse;
    };

    auto q1 = tq::TurboQuantMSE<1>::make(d, 42);
    auto q2 = tq::TurboQuantMSE<2>::make(d, 42);
    auto q3 = tq::TurboQuantMSE<3>::make(d, 42);
    auto q4 = tq::TurboQuantMSE<4>::make(d, 42);
    REQUIRE(q1.has_value()); REQUIRE(q2.has_value());
    REQUIRE(q3.has_value()); REQUIRE(q4.has_value());

    const double e1 = run(q1);
    const double e2 = run(q2);
    const double e3 = run(q3);
    const double e4 = run(q4);
    CAPTURE(e1, e2, e3, e4);
    REQUIRE(e1 > e2);
    REQUIRE(e2 > e3);
    REQUIRE(e3 > e4);
}

TEST_CASE("TurboQuantMSE: shape mismatch surfaces as ShapeMismatch",
          "[quantizer][mse]") {
    const std::size_t d = 128, batch = 2;
    auto q = tq::TurboQuantMSE<3>::make(d, 42);
    REQUIRE(q.has_value());
    std::vector<float>        x(batch * d);
    std::vector<std::uint8_t> idx(batch * q->packed_bytes(d) + 1); // wrong size
    std::vector<float>        nr(batch);
    REQUIRE(q->quantize(x, batch, idx, nr) == tq::Error::ShapeMismatch);
}

TEST_CASE("TurboQuantMSE::from_matrix accepts externally supplied Pi",
          "[quantizer][mse][fixture]") {
    const std::size_t d = 64;
    auto r  = tq::Rotation::make(d, /*seed=*/7);
    REQUIRE(r.has_value());
    std::vector<float> pi_copy(r->matrix().begin(), r->matrix().end());

    auto q = tq::TurboQuantMSE<3>::from_matrix(pi_copy, d);
    REQUIRE(q.has_value());

    const std::size_t batch = 2;
    std::vector<float> x(batch * d);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.1f * i);

    const std::size_t pb = tq::TurboQuantMSE<3>::packed_bytes(d);
    std::vector<std::uint8_t> idx(batch * pb);
    std::vector<float>        nr(batch), xh(batch * d);
    REQUIRE(q->quantize(x, batch, idx, nr)    == tq::Error::Ok);
    REQUIRE(q->dequantize(idx, nr, batch, xh) == tq::Error::Ok);
    // Norm is preserved to a tight tolerance.
    for (std::size_t b = 0; b < batch; ++b) {
        double s = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            const double v = static_cast<double>(x[b * d + i]);
            s += v * v;
        }
        REQUIRE_THAT(nr[b], WithinRel(static_cast<float>(std::sqrt(s)), 1e-5f));
    }
}
