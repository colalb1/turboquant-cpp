#pragma once

// QJL sign pack / unpack helpers (LSB-first, 8 signs per byte).
//
// Python reference: turboquant/quantizer.py:214-229
//   pack:   byte = Σ_k (projected[k] > 0) * 2^k    (powers [1,2,4,...,128])
//   unpack: sign[k] = 2 * ((byte & 2^k) > 0) - 1    ∈ {-1, +1}
//
// Input dimension is padded with zeros on the right to a multiple of 8.

#include <cstddef>
#include <cstdint>

#include "turboquant/types.hpp"

namespace tq {

struct QJLPack {
    static constexpr std::size_t packed_bytes(std::size_t d) noexcept {
        return ceil_div(d, 8);
    }

    // Pack `d` floats: byte k has bit i set iff projected[8*k + i] > 0.
    static void pack(const float* projected, std::size_t d,
                     std::uint8_t* out) noexcept
    {
        const std::size_t nb = packed_bytes(d);
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t base = b * 8;
            std::uint8_t byte = 0;
            for (int k = 0; k < 8; ++k) {
                const std::size_t i = base + static_cast<std::size_t>(k);
                if (i < d && projected[i] > 0.0f) {
                    byte = static_cast<std::uint8_t>(byte | (1u << k));
                }
            }
            out[b] = byte;
        }
    }

    // Unpack to a dense float array of length d with values in {-1, +1}.
    // Trailing pad (d < 8*nb) is untouched; callers allocate exactly `d`.
    static void unpack_pm1(const std::uint8_t* packed, std::size_t d,
                           float* out) noexcept
    {
        const std::size_t nb = packed_bytes(d);
        for (std::size_t b = 0; b < nb; ++b) {
            const std::uint8_t byte = packed[b];
            const std::size_t base = b * 8;
            for (int k = 0; k < 8; ++k) {
                const std::size_t i = base + static_cast<std::size_t>(k);
                if (i < d) {
                    out[i] = ((byte >> k) & 1u) ? 1.0f : -1.0f;
                }
            }
        }
    }
};

} // namespace tq
