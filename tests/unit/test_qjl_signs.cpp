// QJL sign pack/unpack — LSB-first 8 bits per byte, unpack to {-1, +1}.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

#include "turboquant/qjl_signs.hpp"

TEST_CASE("QJLPack: packed_bytes rounds up to a multiple of 8", "[qjl]") {
    REQUIRE(tq::QJLPack::packed_bytes(0)   == 0);
    REQUIRE(tq::QJLPack::packed_bytes(1)   == 1);
    REQUIRE(tq::QJLPack::packed_bytes(8)   == 1);
    REQUIRE(tq::QJLPack::packed_bytes(9)   == 2);
    REQUIRE(tq::QJLPack::packed_bytes(128) == 16);
}

TEST_CASE("QJLPack: LSB-first sign packing (powers 1,2,4,...,128)", "[qjl]") {
    // projected values: signs = (>0) = 1 0 1 1 0 0 1 0  → bits 0,2,3,6 set
    //   byte = 1 + 4 + 8 + 64 = 77
    std::array<float, 8> projected = {  0.5f, -0.1f,  2.0f,  0.3f,
                                       -0.2f, -0.9f,  1.2f, -0.5f };
    std::array<std::uint8_t, 1> packed{};
    tq::QJLPack::pack(projected.data(), 8, packed.data());
    REQUIRE(packed[0] == 77);
}

TEST_CASE("QJLPack: zero is treated as non-positive (sign = -1)", "[qjl]") {
    std::array<float, 8> projected = { 0, 0, 0, 0, 0, 0, 0, 0 };
    std::array<std::uint8_t, 1> packed{};
    tq::QJLPack::pack(projected.data(), 8, packed.data());
    REQUIRE(packed[0] == 0);

    std::array<float, 8> unpacked{};
    tq::QJLPack::unpack_pm1(packed.data(), 8, unpacked.data());
    for (float v : unpacked) REQUIRE(v == -1.0f);
}

TEST_CASE("QJLPack: round-trip recovers ±1 for non-zero inputs", "[qjl]") {
    const std::size_t d = 37;   // odd, exercises padding
    std::vector<float> projected(d);
    for (std::size_t i = 0; i < d; ++i) {
        projected[i] = (i % 3 == 0) ? -0.5f : 0.7f;
    }
    std::vector<std::uint8_t> packed(tq::QJLPack::packed_bytes(d), 0);
    std::vector<float>        unpacked(d, 0.0f);
    tq::QJLPack::pack(projected.data(), d, packed.data());
    tq::QJLPack::unpack_pm1(packed.data(), d, unpacked.data());
    for (std::size_t i = 0; i < d; ++i) {
        CAPTURE(i);
        REQUIRE(unpacked[i] == (projected[i] > 0.0f ? 1.0f : -1.0f));
    }
}
