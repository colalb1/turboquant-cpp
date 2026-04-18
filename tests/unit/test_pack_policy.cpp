// Bit-pack / unpack round-trip + layout invariants.
//
// Python reference: turboquant/quantizer.py:38-90.
// We match the LSB-first layout exactly:
//   byte = Σ_k idx[k] << (k * effective_bits)
// with bits=3 storage aliasing bits=4 (quantizer.py:54-56).

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

#include "turboquant/pack_policy.hpp"

TEST_CASE("PackPolicy<1>: 8 vals/byte LSB-first round trip", "[pack]") {
    using P = tq::PackPolicy<1>;
    const std::size_t d = 17;  // non-multiple of 8 to exercise padding
    std::vector<std::uint8_t> raw(d), back(d, 0xFF);
    for (std::size_t i = 0; i < d; ++i) raw[i] = (i * 5 + 1) & 1;

    std::vector<std::uint8_t> packed(P::packed_bytes(d), 0);
    P::pack(raw.data(), d, packed.data());

    // Hand-verify byte 0: LSB-first.
    std::uint8_t expected_byte0 = 0;
    for (int k = 0; k < 8; ++k) {
        expected_byte0 = static_cast<std::uint8_t>(expected_byte0 | ((raw[k] & 0x1) << k));
    }
    REQUIRE(packed[0] == expected_byte0);

    P::unpack(packed.data(), d, back.data());
    for (std::size_t i = 0; i < d; ++i) {
        CAPTURE(i);
        REQUIRE(back[i] == raw[i]);
    }
}

TEST_CASE("PackPolicy<2>: 4 vals/byte LSB-first round trip", "[pack]") {
    using P = tq::PackPolicy<2>;
    const std::size_t d = 13;
    std::vector<std::uint8_t> raw(d), back(d, 0xFF);
    for (std::size_t i = 0; i < d; ++i) raw[i] = (i * 7 + 2) & 0x3;

    std::vector<std::uint8_t> packed(P::packed_bytes(d), 0);
    P::pack(raw.data(), d, packed.data());

    std::uint8_t expected_byte0 = 0;
    for (int k = 0; k < 4; ++k) {
        expected_byte0 = static_cast<std::uint8_t>(
            expected_byte0 | ((raw[k] & 0x3) << (k * 2)));
    }
    REQUIRE(packed[0] == expected_byte0);

    P::unpack(packed.data(), d, back.data());
    for (std::size_t i = 0; i < d; ++i) {
        CAPTURE(i);
        REQUIRE(back[i] == raw[i]);
    }
}

TEST_CASE("PackPolicy<4>: 2 vals/byte (low nibble first) round trip", "[pack]") {
    using P = tq::PackPolicy<4>;
    const std::size_t d = 5;
    std::vector<std::uint8_t> raw(d), back(d, 0xFF);
    for (std::size_t i = 0; i < d; ++i) raw[i] = (i * 3 + 1) & 0xF;

    std::vector<std::uint8_t> packed(P::packed_bytes(d), 0);
    P::pack(raw.data(), d, packed.data());

    // byte 0: low nibble = raw[0], high nibble = raw[1].
    REQUIRE((packed[0] & 0x0F) == raw[0]);
    REQUIRE(((packed[0] >> 4) & 0x0F) == raw[1]);

    P::unpack(packed.data(), d, back.data());
    for (std::size_t i = 0; i < d; ++i) {
        CAPTURE(i);
        REQUIRE(back[i] == raw[i]);
    }
}

TEST_CASE("PackPolicy<3> aliases 4-bit storage", "[pack]") {
    using P = tq::PackPolicy<3>;
    static_assert(P::effective_bits == 4);
    static_assert(P::vals_per_byte == 2);

    const std::size_t d = 11;
    std::vector<std::uint8_t> raw(d), back(d, 0xFF);
    // 3-bit values in [0, 7]
    for (std::size_t i = 0; i < d; ++i) raw[i] = (i * 2 + 3) & 0x7;

    std::vector<std::uint8_t> packed(P::packed_bytes(d), 0);
    P::pack(raw.data(), d, packed.data());

    // Must match 4-bit packing exactly.
    std::vector<std::uint8_t> packed4(tq::PackPolicy<4>::packed_bytes(d), 0);
    tq::PackPolicy<4>::pack(raw.data(), d, packed4.data());
    REQUIRE(packed.size() == packed4.size());
    for (std::size_t i = 0; i < packed.size(); ++i) REQUIRE(packed[i] == packed4[i]);

    P::unpack(packed.data(), d, back.data());
    for (std::size_t i = 0; i < d; ++i) {
        CAPTURE(i);
        REQUIRE(back[i] == raw[i]);
    }
}

TEST_CASE("PackPolicy: trailing pad is zero", "[pack]") {
    using P = tq::PackPolicy<1>;
    const std::size_t d = 3;  // 1 byte, 5 pad slots
    std::array<std::uint8_t, 3> raw = { 1, 0, 1 };
    std::array<std::uint8_t, P::packed_bytes(3)> packed{};
    P::pack(raw.data(), d, packed.data());
    // Byte = (1<<0) | (0<<1) | (1<<2) = 0b101 = 5. Pad bits must be zero.
    REQUIRE(packed[0] == 0b00000101);
}
