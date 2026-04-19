#pragma once

// Scalar reference implementations of every NEON kernel. Used:
//   (a) as a portable fallback on non-ARM hosts,
//   (b) as a ground-truth baseline for NEON parity tests.
//
// Everything here is header-only and callable with any float*.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "turboquant/pack_policy.hpp"
#include "turboquant/qjl_signs.hpp"

namespace tq::neon_scalar {

// L2 norm of d floats. Uses double-precision accumulation to match the
// TurboQuantMSE scalar path (and hence avoid spurious ulp drift when we
// compare against the NEON path which also accumulates pairwise in f32).
inline float l2norm(const float* x, std::size_t d) noexcept {
    double acc = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        const double v  = static_cast<double>(x[i]);
        acc            += v * v;
    }
    return static_cast<float>(std::sqrt(acc));
}

// y = x * inv_scale  (broadcast scalar * vector).
inline void scale(const float* x, float inv_scale, float* y, std::size_t d) noexcept {
    for (std::size_t i = 0; i < d; ++i)
        y[i] = x[i] * inv_scale;
}

// searchsorted with right=False:
//   result in [0, n], number of bounds strictly less than v.
inline std::uint8_t searchsorted(const float* bounds, std::size_t n, float v) noexcept {
    std::size_t i = 0;
    while (i < n && bounds[i] < v)
        ++i;
    return static_cast<std::uint8_t>(i);
}

// Quantize `d` floats by per-coord searchsorted, then LSB-first pack.
template <int Bits>
inline void searchsorted_and_pack(const float* rotated, const float* bounds, std::size_t n_bounds,
                                  std::uint8_t* packed_out, std::size_t d) noexcept {
    using Pack = PackPolicy<Bits>;
    // Local index buffer (stack-limited to kMaxDim ≤ 1024 per config.hpp).
    std::uint8_t raw[1024];
    for (std::size_t i = 0; i < d; ++i)
        raw[i] = searchsorted(bounds, n_bounds, rotated[i]);
    Pack::pack(raw, d, packed_out);
}

// Unpack `d` indices and gather centroids.
template <int Bits>
inline void unpack_and_gather(const std::uint8_t* packed, const float* centroids, float* y_out,
                              std::size_t d) noexcept {
    using Pack = PackPolicy<Bits>;
    std::uint8_t raw[1024];
    Pack::unpack(packed, d, raw);
    for (std::size_t i = 0; i < d; ++i)
        y_out[i] = centroids[raw[i]];
}

// QJL: pack signs of `projected` LSB-first, 8/byte.
inline void qjl_pack_signs(const float* projected, std::size_t d, std::uint8_t* out) noexcept {
    QJLPack::pack(projected, d, out);
}

// QJL: unpack to {-1, +1} floats.
inline void qjl_unpack_pm1(const std::uint8_t* packed, std::size_t d, float* out) noexcept {
    QJLPack::unpack_pm1(packed, d, out);
}

// Per-group asymmetric quant of one row (dim floats, n_groups = dim/gs).
// Writes dim uint8 indices in [0, n_levels] to `idx_raw`, and n_groups
// scales + zeros. scale = max((max-min)/n_levels, 1e-10), zero = min.
// Rounding: round-half-to-even (matches torch.round / vcvtnq_s32_f32).
inline void group_quant_row(const float* x, std::size_t dim, std::size_t gs, int n_levels,
                            std::uint8_t* idx_raw, float* scales, float* zeros) noexcept {
    const std::size_t ng         = dim / gs;
    const float       inv_levels = 1.0f / static_cast<float>(n_levels);
    for (std::size_t g = 0; g < ng; ++g) {
        const float* xp = x + g * gs;
        float        mn = xp[0], mx = xp[0];
        for (std::size_t i = 1; i < gs; ++i) {
            const float v = xp[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        float sc = (mx - mn) * inv_levels;
        if (sc < 1e-10f) sc = 1e-10f;
        scales[g]         = sc;
        zeros[g]          = mn;
        const float   inv = 1.0f / sc;
        std::uint8_t* out = idx_raw + g * gs;
        for (std::size_t i = 0; i < gs; ++i) {
            const float q  = (xp[i] - mn) * inv;
            long        qi = std::lrint(q);
            if (qi < 0) qi = 0;
            if (qi > n_levels) qi = n_levels;
            out[i] = static_cast<std::uint8_t>(qi);
        }
    }
}

// Per-group dequant of one row: x_out = idx * scale + zero.
inline void group_dequant_row(const std::uint8_t* idx_raw, std::size_t dim, std::size_t gs,
                              const float* scales, const float* zeros, float* x_out) noexcept {
    const std::size_t ng = dim / gs;
    for (std::size_t g = 0; g < ng; ++g) {
        const float         sc = scales[g];
        const float         z  = zeros[g];
        const std::uint8_t* ip = idx_raw + g * gs;
        float*              op = x_out + g * gs;
        for (std::size_t i = 0; i < gs; ++i) {
            op[i] = static_cast<float>(ip[i]) * sc + z;
        }
    }
}

}  // namespace tq::neon_scalar
