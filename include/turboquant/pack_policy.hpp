#pragma once

// LSB-first bit-packing policies for quantizer indices.
//
// Python reference: turboquant/quantizer.py:38-90
//   vals_per_byte: bits=1 → 8, bits=2 → 4, bits=3 → 2 (rounds up to 4-bit), bits=4 → 2
//   byte = Σ_k indices[k] << (k * effective_bits)
//   pad trailing slots with zero to fill the last byte
//
// Effective bits differ from the requested bits only for bits=3 (which stores
// as 4-bit nibbles, matching Python's "round up to 4-bit packing" comment).

#include <cstddef>
#include <cstdint>

#include "turboquant/types.hpp"

namespace tq {

template <int Bits> struct PackPolicy;

// bits = 1 — eight indices per byte, LSB-first.
template <>
struct PackPolicy<1> {
    static constexpr int  bits            = 1;
    static constexpr int  effective_bits  = 1;
    static constexpr int  vals_per_byte   = 8;
    static constexpr std::uint8_t mask    = 0x1;

    static constexpr std::size_t packed_bytes(std::size_t d) noexcept {
        return ceil_div(d, vals_per_byte);
    }

    // Pack `d` indices in [0, 1] into packed_bytes(d) bytes, LSB-first.
    static void pack(const std::uint8_t* indices, std::size_t d,
                     std::uint8_t* out) noexcept
    {
        const std::size_t nb = packed_bytes(d);
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t base = b * vals_per_byte;
            std::uint8_t byte = 0;
            for (int k = 0; k < vals_per_byte; ++k) {
                const std::size_t i = base + static_cast<std::size_t>(k);
                if (i < d) byte |= static_cast<std::uint8_t>((indices[i] & mask) << k);
            }
            out[b] = byte;
        }
    }

    static void unpack(const std::uint8_t* packed, std::size_t d,
                       std::uint8_t* out) noexcept
    {
        const std::size_t nb = packed_bytes(d);
        for (std::size_t b = 0; b < nb; ++b) {
            const std::uint8_t byte = packed[b];
            const std::size_t base = b * vals_per_byte;
            for (int k = 0; k < vals_per_byte; ++k) {
                const std::size_t i = base + static_cast<std::size_t>(k);
                if (i < d) out[i] = static_cast<std::uint8_t>((byte >> k) & mask);
            }
        }
    }
};

// bits = 2 — four indices per byte, LSB-first.
template <>
struct PackPolicy<2> {
    static constexpr int  bits            = 2;
    static constexpr int  effective_bits  = 2;
    static constexpr int  vals_per_byte   = 4;
    static constexpr std::uint8_t mask    = 0x3;

    static constexpr std::size_t packed_bytes(std::size_t d) noexcept {
        return ceil_div(d, vals_per_byte);
    }

    static void pack(const std::uint8_t* indices, std::size_t d,
                     std::uint8_t* out) noexcept
    {
        const std::size_t nb = packed_bytes(d);
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t base = b * vals_per_byte;
            std::uint8_t byte = 0;
            for (int k = 0; k < vals_per_byte; ++k) {
                const std::size_t i = base + static_cast<std::size_t>(k);
                if (i < d) byte |= static_cast<std::uint8_t>((indices[i] & mask) << (k * 2));
            }
            out[b] = byte;
        }
    }

    static void unpack(const std::uint8_t* packed, std::size_t d,
                       std::uint8_t* out) noexcept
    {
        const std::size_t nb = packed_bytes(d);
        for (std::size_t b = 0; b < nb; ++b) {
            const std::uint8_t byte = packed[b];
            const std::size_t base = b * vals_per_byte;
            for (int k = 0; k < vals_per_byte; ++k) {
                const std::size_t i = base + static_cast<std::size_t>(k);
                if (i < d) out[i] = static_cast<std::uint8_t>((byte >> (k * 2)) & mask);
            }
        }
    }
};

// bits = 4 — two indices per byte, LSB-first (low nibble first).
template <>
struct PackPolicy<4> {
    static constexpr int  bits            = 4;
    static constexpr int  effective_bits  = 4;
    static constexpr int  vals_per_byte   = 2;
    static constexpr std::uint8_t mask    = 0xF;

    static constexpr std::size_t packed_bytes(std::size_t d) noexcept {
        return ceil_div(d, vals_per_byte);
    }

    static void pack(const std::uint8_t* indices, std::size_t d,
                     std::uint8_t* out) noexcept
    {
        const std::size_t nb = packed_bytes(d);
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t i0 = b * 2;
            const std::size_t i1 = i0 + 1;
            std::uint8_t lo = (i0 < d) ? (indices[i0] & mask) : std::uint8_t{0};
            std::uint8_t hi = (i1 < d) ? (indices[i1] & mask) : std::uint8_t{0};
            out[b] = static_cast<std::uint8_t>(lo | (hi << 4));
        }
    }

    static void unpack(const std::uint8_t* packed, std::size_t d,
                       std::uint8_t* out) noexcept
    {
        const std::size_t nb = packed_bytes(d);
        for (std::size_t b = 0; b < nb; ++b) {
            const std::uint8_t byte = packed[b];
            const std::size_t i0 = b * 2;
            const std::size_t i1 = i0 + 1;
            if (i0 < d) out[i0] = static_cast<std::uint8_t>(byte & mask);
            if (i1 < d) out[i1] = static_cast<std::uint8_t>((byte >> 4) & mask);
        }
    }
};

// bits = 3 aliases bits=4 storage (matches quantizer.py:54-56). The index
// values stay in [0, 7] but occupy 4-bit nibbles.
template <>
struct PackPolicy<3> {
    static constexpr int  bits            = 3;
    static constexpr int  effective_bits  = 4;
    static constexpr int  vals_per_byte   = 2;
    static constexpr std::uint8_t mask    = 0x7;   // logical mask (index range)

    static constexpr std::size_t packed_bytes(std::size_t d) noexcept {
        return PackPolicy<4>::packed_bytes(d);
    }

    static void pack(const std::uint8_t* indices, std::size_t d,
                     std::uint8_t* out) noexcept
    {
        PackPolicy<4>::pack(indices, d, out);
    }

    static void unpack(const std::uint8_t* packed, std::size_t d,
                       std::uint8_t* out) noexcept
    {
        PackPolicy<4>::unpack(packed, d, out);
    }
};

} // namespace tq
