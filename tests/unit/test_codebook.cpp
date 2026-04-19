// Codebook registry tests: embedded lookup + loader + cache semantics.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "turboquant/codebook.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("embedded_codebooks() returns a non-empty table", "[codebook]") {
    auto blobs = tq::embedded_codebooks();
    REQUIRE_FALSE(blobs.empty());
    // Every bundled blob should have size-consistent centroid / interior
    // arrays for its (d, bits).
    for (const auto& b : blobs) {
        const std::size_t n_cl = std::size_t{1} << b.bits;
        REQUIRE(n_cl >= 2);
        REQUIRE(b.centroids != nullptr);
        REQUIRE(b.decision_boundaries != nullptr);
        REQUIRE(b.dim >= 64);
        REQUIRE(b.bits >= 1);
        REQUIRE(b.bits <= 4);
    }
}

TEST_CASE("CodebookRegistry::find_embedded returns bundled blobs", "[codebook]") {
    auto& reg = tq::CodebookRegistry::instance();
    for (const auto& b : tq::embedded_codebooks()) {
        auto v = reg.find_embedded(b.dim, b.bits);
        REQUIRE(v.has_value());
        REQUIRE(v->dim == b.dim);
        REQUIRE(v->bits == b.bits);
        const std::size_t n_cl = std::size_t{1} << b.bits;
        REQUIRE(v->centroids.size() == n_cl);
        REQUIRE(v->decision_boundaries.size() == n_cl - 1);
        // Centroids must be strictly increasing.
        for (std::size_t i = 0; i + 1 < n_cl; ++i) {
            REQUIRE(v->centroids[i] < v->centroids[i + 1]);
        }
        // Interior boundaries must be strictly inside (-1, 1).
        for (float x : v->decision_boundaries) {
            REQUIRE(x > -1.0f);
            REQUIRE(x < 1.0f);
        }
    }
}

TEST_CASE("get() matches find_embedded() for bundled (d, bits)", "[codebook]") {
    auto& reg = tq::CodebookRegistry::instance();
    // Hit d=128, b=3 — present in the JSON bundle.
    auto e = reg.find_embedded(128, 3);
    REQUIRE(e.has_value());
    auto g = reg.get(128, 3);
    REQUIRE(g.has_value());
    REQUIRE(g->centroids.size() == e->centroids.size());
    for (std::size_t i = 0; i < e->centroids.size(); ++i) {
        REQUIRE(g->centroids[i] == e->centroids[i]);
    }
    for (std::size_t i = 0; i < e->decision_boundaries.size(); ++i) {
        REQUIRE(g->decision_boundaries[i] == e->decision_boundaries[i]);
    }
}

TEST_CASE("decision_boundaries match bundled JSON for d=128, b=3", "[codebook]") {
    // From codebook_d128_b3.json — the boundaries[1:-1] interior slice.
    const std::array<float, 7> expected = {
        -0.1532617987505358f, -0.09235678950747524f, -0.04409153199159803f, 0.0f,
        0.04409153199159803f, 0.09235678950747524f,  0.1532617987505358f,
    };
    auto v = tq::CodebookRegistry::instance().find_embedded(128, 3);
    REQUIRE(v.has_value());
    REQUIRE(v->decision_boundaries.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE_THAT(v->decision_boundaries[i], WithinAbs(expected[i], 1e-7));
    }
}

TEST_CASE("CodebookRegistry rejects invalid (d, bits)", "[codebook]") {
    auto& reg = tq::CodebookRegistry::instance();
    REQUIRE_FALSE(reg.get(0, 3).has_value());
    REQUIRE_FALSE(reg.get(128, 0).has_value());
    REQUIRE_FALSE(reg.get(128, 5).has_value());
}
