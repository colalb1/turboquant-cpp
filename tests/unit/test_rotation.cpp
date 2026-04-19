// Rotation unit tests — orthogonality, determinism, forward/backward round
// trip. Not a parity test against Python (see tests/parity/ for that, M3).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <vector>

#include "turboquant/rotation.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Compute ||Pi * Pi^T - I||_F^2. For an orthogonal matrix this is 0.
double orthogonality_error(std::span<const float> pi, std::size_t d) {
    double worst = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < d; ++k) {
                dot += static_cast<double>(pi[i * d + k]) * pi[j * d + k];
            }
            const double target = (i == j) ? 1.0 : 0.0;
            worst               = std::max(worst, std::fabs(dot - target));
        }
    }
    return worst;
}

}  // namespace

TEST_CASE("Rotation::make produces an orthogonal matrix", "[rotation]") {
    for (std::size_t d : {64u, 128u, 256u}) {
        auto r = tq::Rotation::make(d, /*seed=*/42);
        REQUIRE(r.has_value());
        const double err = orthogonality_error(r->matrix(), d);
        // LAPACK Householder QR is accurate to ~d * eps; allow 1e-4 slack
        // since we're working in float32.
        CAPTURE(d, err);
        REQUIRE(err < 1e-4);
    }
}

TEST_CASE("Rotation::make is deterministic for a fixed seed", "[rotation]") {
    const std::size_t d = 128;
    auto              a = tq::Rotation::make(d, 42);
    auto              b = tq::Rotation::make(d, 42);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    auto ma = a->matrix();
    auto mb = b->matrix();
    REQUIRE(ma.size() == mb.size());
    for (std::size_t i = 0; i < ma.size(); ++i) {
        // Bit-equal is the target; allow zero ulp tolerance.
        CAPTURE(i);
        REQUIRE(ma[i] == mb[i]);
    }
}

TEST_CASE("Rotation::make differs across seeds", "[rotation]") {
    const std::size_t d = 128;
    auto              a = tq::Rotation::make(d, 42);
    auto              b = tq::Rotation::make(d, 43);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    auto        ma   = a->matrix();
    auto        mb   = b->matrix();
    std::size_t diff = 0;
    for (std::size_t i = 0; i < ma.size(); ++i)
        if (ma[i] != mb[i]) ++diff;
    REQUIRE(diff > ma.size() / 2);
}

TEST_CASE("Rotation::forward / backward round-trip preserves the vector", "[rotation]") {
    const std::size_t d     = 128;
    const std::size_t batch = 4;
    auto              r     = tq::Rotation::make(d, 42);
    REQUIRE(r.has_value());

    std::vector<float> x(batch * d), y(batch * d), x2(batch * d);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((i * 7 + 3) % 23) / 23.0f - 0.5f;
    }

    REQUIRE(r->forward(x, y, batch) == tq::Error::Ok);
    REQUIRE(r->backward(y, x2, batch) == tq::Error::Ok);

    for (std::size_t i = 0; i < x.size(); ++i) {
        CAPTURE(i, x[i], x2[i]);
        REQUIRE_THAT(x2[i], WithinAbs(x[i], 1e-4));
    }
}

TEST_CASE("Rotation::from_matrix round-trips via forward/backward", "[rotation][fixture]") {
    const std::size_t d   = 64;
    auto              src = tq::Rotation::make(d, 7);
    REQUIRE(src.has_value());

    std::vector<float> pi_copy(src->matrix().begin(), src->matrix().end());
    auto               loaded = tq::Rotation::from_matrix(pi_copy, d);
    REQUIRE(loaded.has_value());

    std::vector<float> x(d), y_src(d), y_lo(d);
    for (std::size_t i = 0; i < d; ++i)
        x[i] = std::sin(0.1f * i);

    REQUIRE(src->forward(x, y_src, 1) == tq::Error::Ok);
    REQUIRE(loaded->forward(x, y_lo, 1) == tq::Error::Ok);
    for (std::size_t i = 0; i < d; ++i) {
        REQUIRE(y_src[i] == y_lo[i]);
    }
}

TEST_CASE("Rotation rejects invalid dims", "[rotation]") {
    REQUIRE_FALSE(tq::Rotation::make(0, 42).has_value());
    REQUIRE_FALSE(tq::Rotation::make(tq::kMaxDim + 1, 42).has_value());
    std::array<float, 4> buf{1, 0, 0, 1};
    REQUIRE_FALSE(tq::Rotation::from_matrix(buf, 3).has_value());
}
