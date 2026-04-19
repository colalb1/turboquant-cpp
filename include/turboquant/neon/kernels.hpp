#pragma once

// NEON-accelerated kernels for the TurboQuant hot path. Every kernel has a
// scalar twin in `scalar_fallback.hpp` that must be bit-exact on identical
// inputs where possible, or match within a documented ULP budget where
// floating-point order of operations differs.
//
// All kernels are header-only and `always_inline` — the call sites are
// performance-critical and LTO is expected to fold them through multiple
// layers (CRTP + templated TurboQuantMSE).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "turboquant/config.hpp"
#include "turboquant/neon/scalar_fallback.hpp"
#include "turboquant/pack_policy.hpp"
#include "turboquant/qjl_signs.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace tq::neon {

// Stack scratch budget for per-row raw-index buffers in searchsorted_and_pack
// and unpack_and_gather. Must cover the whole supported dim range — the
// kernels have no runtime fallback for d > kMaxDim.
static_assert(kMaxDim <= 2048, "kMaxDim exceeds the NEON kernel stack budget");

// -----------------------------------------------------------------------------
// l2norm_f32: ||x||_2 over d floats.
//
// Uses pairwise float32 accumulation in 4 lanes to keep summation order
// deterministic vs the scalar path. We do NOT match the scalar (double-
// precision) result bit-for-bit — tests allow ~2 ulp drift for d ≤ 256.
// -----------------------------------------------------------------------------
[[gnu::always_inline]] inline float l2norm(const float* x, std::size_t d) noexcept {
#if defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    std::size_t i    = 0;
    for (; i + 16 <= d; i += 16) {
        const float32x4_t a = vld1q_f32(x + i + 0);
        const float32x4_t b = vld1q_f32(x + i + 4);
        const float32x4_t c = vld1q_f32(x + i + 8);
        const float32x4_t e = vld1q_f32(x + i + 12);
        acc0                = vfmaq_f32(acc0, a, a);
        acc1                = vfmaq_f32(acc1, b, b);
        acc2                = vfmaq_f32(acc2, c, c);
        acc3                = vfmaq_f32(acc3, e, e);
    }
    for (; i + 4 <= d; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        acc0                = vfmaq_f32(acc0, v, v);
    }
    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float       s   = vaddvq_f32(acc);
    for (; i < d; ++i)
        s += x[i] * x[i];
    return std::sqrt(s);
#else
    return neon_scalar::l2norm(x, d);
#endif
}

// -----------------------------------------------------------------------------
// scale: y = x * inv_scale   (broadcast scalar × vector).
// Bit-exact vs scalar twin because each element is exactly one multiply.
// -----------------------------------------------------------------------------
[[gnu::always_inline]] inline void scale(const float* x, float inv_scale, float* y,
                                         std::size_t d) noexcept {
#if defined(__ARM_NEON)
    const float32x4_t s = vdupq_n_f32(inv_scale);
    std::size_t       i = 0;
    for (; i + 16 <= d; i += 16) {
        vst1q_f32(y + i + 0, vmulq_f32(vld1q_f32(x + i + 0), s));
        vst1q_f32(y + i + 4, vmulq_f32(vld1q_f32(x + i + 4), s));
        vst1q_f32(y + i + 8, vmulq_f32(vld1q_f32(x + i + 8), s));
        vst1q_f32(y + i + 12, vmulq_f32(vld1q_f32(x + i + 12), s));
    }
    for (; i + 4 <= d; i += 4) {
        vst1q_f32(y + i, vmulq_f32(vld1q_f32(x + i), s));
    }
    for (; i < d; ++i)
        y[i] = x[i] * inv_scale;
#else
    neon_scalar::scale(x, inv_scale, y, d);
#endif
}

// -----------------------------------------------------------------------------
// searchsorted count (right=False): for a single value v, return the number
// of `bounds` strictly less than v. n_bounds ≤ 15 (max for bits=4).
//
// We vectorize across the 15 bounds, not across values, because the number
// of bounds is tiny and deterministic.
// -----------------------------------------------------------------------------
[[gnu::always_inline]] inline std::uint8_t searchsorted_one(const float* bounds, std::size_t n,
                                                            float v) noexcept {
#if defined(__ARM_NEON)
    if (n == 0) return 0;
    const float32x4_t vv = vdupq_n_f32(v);
    std::size_t       i  = 0;
    // Count lanes where bounds[i] < v.
    std::uint32_t count = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t b = vld1q_f32(bounds + i);
        // vcltq_f32 returns a mask per lane with all-ones when b < v.
        const uint32x4_t m = vcltq_f32(b, vv);
        // Reduce 4 masks to a population count of 1 bits.
        const uint32x4_t u  = vshrq_n_u32(m, 31);  // 0 or 1 per lane
        count              += vaddvq_u32(u);
    }
    for (; i < n; ++i)
        count += (bounds[i] < v) ? 1u : 0u;
    return static_cast<std::uint8_t>(count);
#else
    return neon_scalar::searchsorted(bounds, n, v);
#endif
}

// -----------------------------------------------------------------------------
// searchsorted_and_pack<Bits>: per-coord searchsorted over `rotated` +
// LSB-first bit-pack into `packed_out`.
// Bit-exact vs scalar twin: the packing is pure integer; searchsorted
// returns the same index value via a different reduction order.
// -----------------------------------------------------------------------------
template <int Bits>
[[gnu::always_inline]] inline void
searchsorted_and_pack(const float* rotated, const float* bounds, std::size_t n_bounds,
                      std::uint8_t* packed_out, std::size_t d) noexcept {
    using Pack = PackPolicy<Bits>;
    std::uint8_t raw[kMaxDim];
    for (std::size_t i = 0; i < d; ++i) {
        raw[i] = searchsorted_one(bounds, n_bounds, rotated[i]);
    }
    Pack::pack(raw, d, packed_out);
}

// -----------------------------------------------------------------------------
// unpack_and_gather<Bits>: unpack LSB-first packed indices, then gather
// centroids[idx[i]] into y_out[i].
// -----------------------------------------------------------------------------
template <int Bits>
[[gnu::always_inline]] inline void unpack_and_gather(const std::uint8_t* packed,
                                                     const float* centroids, float* y_out,
                                                     std::size_t d) noexcept {
    using Pack = PackPolicy<Bits>;
    std::uint8_t raw[kMaxDim];
    Pack::unpack(packed, d, raw);
    for (std::size_t i = 0; i < d; ++i)
        y_out[i] = centroids[raw[i]];
}

// -----------------------------------------------------------------------------
// qjl_pack_signs: byte k has bit i set iff projected[8*k + i] > 0.
// Uses vcgtq_f32 + a powers-of-two LUT.
// -----------------------------------------------------------------------------
[[gnu::always_inline]] inline void qjl_pack_signs(const float* projected, std::size_t d,
                                                  std::uint8_t* out) noexcept {
#if defined(__ARM_NEON)
    const std::size_t nb = QJLPack::packed_bytes(d);
    // Constant power-of-two lookup: bits 0..7 → 1,2,4,8,16,32,64,128.
    const uint32x4_t  pow_lo = {1u, 2u, 4u, 8u};
    const uint32x4_t  pow_hi = {16u, 32u, 64u, 128u};
    const float32x4_t zero   = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    for (std::size_t b = 0; b < nb; ++b) {
        std::uint32_t     byte = 0;
        const std::size_t base = b * 8;

        // First 4 lanes.
        if (base + 4 <= d) {
            const float32x4_t p     = vld1q_f32(projected + base);
            const uint32x4_t  m     = vcgtq_f32(p, zero);
            const uint32x4_t  bits  = vandq_u32(m, pow_lo);
            byte                   |= vaddvq_u32(bits);
        } else {
            for (int k = 0; k < 4; ++k) {
                const std::size_t idx = base + static_cast<std::size_t>(k);
                if (idx < d && projected[idx] > 0.0f) byte |= (1u << k);
            }
        }

        // Next 4 lanes.
        if (base + 8 <= d) {
            const float32x4_t p     = vld1q_f32(projected + base + 4);
            const uint32x4_t  m     = vcgtq_f32(p, zero);
            const uint32x4_t  bits  = vandq_u32(m, pow_hi);
            byte                   |= vaddvq_u32(bits);
        } else {
            for (int k = 4; k < 8; ++k) {
                const std::size_t idx = base + static_cast<std::size_t>(k);
                if (idx < d && projected[idx] > 0.0f) byte |= (1u << k);
            }
        }

        out[b] = static_cast<std::uint8_t>(byte);
        (void)i;
    }
#else
    neon_scalar::qjl_pack_signs(projected, d, out);
#endif
}

// -----------------------------------------------------------------------------
// qjl_unpack_pm1: byte b → 8 floats in {-1, +1} via shift-AND + vbslq_f32.
// -----------------------------------------------------------------------------
[[gnu::always_inline]] inline void qjl_unpack_pm1(const std::uint8_t* packed, std::size_t d,
                                                  float* out) noexcept {
#if defined(__ARM_NEON)
    const std::size_t nb     = QJLPack::packed_bytes(d);
    const uint32x4_t  pow_lo = {1u, 2u, 4u, 8u};
    const uint32x4_t  pow_hi = {16u, 32u, 64u, 128u};
    const float32x4_t pos    = vdupq_n_f32(1.0f);
    const float32x4_t neg    = vdupq_n_f32(-1.0f);

    for (std::size_t b = 0; b < nb; ++b) {
        const uint32x4_t  byte = vdupq_n_u32(packed[b]);
        const std::size_t base = b * 8;

        // Low 4 lanes.
        {
            const uint32x4_t  bits = vandq_u32(byte, pow_lo);
            const uint32x4_t  mask = vcgtq_u32(bits, vdupq_n_u32(0));
            const float32x4_t v    = vbslq_f32(mask, pos, neg);
            if (base + 4 <= d) {
                vst1q_f32(out + base, v);
            } else {
                float tmp[4];
                vst1q_f32(tmp, v);
                for (int k = 0; k < 4; ++k) {
                    if (base + k < d) out[base + k] = tmp[k];
                }
            }
        }

        // High 4 lanes.
        {
            const uint32x4_t  bits = vandq_u32(byte, pow_hi);
            const uint32x4_t  mask = vcgtq_u32(bits, vdupq_n_u32(0));
            const float32x4_t v    = vbslq_f32(mask, pos, neg);
            if (base + 8 <= d) {
                vst1q_f32(out + base + 4, v);
            } else {
                float tmp[4];
                vst1q_f32(tmp, v);
                for (int k = 0; k < 4; ++k) {
                    if (base + 4 + k < d) out[base + 4 + k] = tmp[k];
                }
            }
        }
    }
#else
    neon_scalar::qjl_unpack_pm1(packed, d, out);
#endif
}

// -----------------------------------------------------------------------------
// group_quant_row: per-group asymmetric quant (kv_cache.py:45-98).
// scale = max((max-min)/n_levels, 1e-10), zero = min. Rounding is
// round-half-to-even via vcvtnq_s32_f32 (matches torch.round()).
// Bit-exact vs scalar twin is NOT guaranteed: min/max across 4 NEON lanes
// uses vminq/vmaxq + horizontal reduce, which the scalar twin mirrors by
// scanning left-to-right. Both see identical min/max values.
// -----------------------------------------------------------------------------
[[gnu::always_inline]] inline void group_quant_row(const float* x, std::size_t dim, std::size_t gs,
                                                   int n_levels, std::uint8_t* idx_raw,
                                                   float* scales, float* zeros) noexcept {
#if defined(__ARM_NEON)
    const std::size_t ng         = dim / gs;
    const float       inv_levels = 1.0f / static_cast<float>(n_levels);
    for (std::size_t g = 0; g < ng; ++g) {
        const float* xp = x + g * gs;
        std::size_t  i  = 0;
        // min/max reduction.
        float mn, mx;
        if (gs >= 4) {
            float32x4_t vmin = vld1q_f32(xp);
            float32x4_t vmax = vmin;
            i                = 4;
            for (; i + 4 <= gs; i += 4) {
                const float32x4_t v = vld1q_f32(xp + i);
                vmin                = vminq_f32(vmin, v);
                vmax                = vmaxq_f32(vmax, v);
            }
            mn = vminvq_f32(vmin);
            mx = vmaxvq_f32(vmax);
            for (; i < gs; ++i) {
                if (xp[i] < mn) mn = xp[i];
                if (xp[i] > mx) mx = xp[i];
            }
        } else {
            mn = xp[0];
            mx = xp[0];
            for (i = 1; i < gs; ++i) {
                if (xp[i] < mn) mn = xp[i];
                if (xp[i] > mx) mx = xp[i];
            }
        }

        float sc = (mx - mn) * inv_levels;
        if (sc < 1e-10f) sc = 1e-10f;
        scales[g] = sc;
        zeros[g]  = mn;

        const float       inv   = 1.0f / sc;
        const float32x4_t vinv  = vdupq_n_f32(inv);
        const float32x4_t vzero = vdupq_n_f32(mn);
        const int32x4_t   vlo   = vdupq_n_s32(0);
        const int32x4_t   vhi   = vdupq_n_s32(n_levels);
        std::uint8_t*     op    = idx_raw + g * gs;

        i = 0;
        for (; i + 4 <= gs; i += 4) {
            const float32x4_t v  = vld1q_f32(xp + i);
            const float32x4_t q  = vmulq_f32(vsubq_f32(v, vzero), vinv);
            int32x4_t         qi = vcvtnq_s32_f32(q);
            qi                   = vmaxq_s32(vlo, vminq_s32(vhi, qi));
            // Narrow s32 → u8. n_levels ≤ 255 so saturation is safe.
            const uint16x4_t u16 = vqmovun_s32(qi);
            const uint8x8_t  u8  = vqmovn_u16(vcombine_u16(u16, u16));
            // Store 4 bytes from low lanes.
            std::uint32_t packed4;
            vst1_lane_u32(reinterpret_cast<std::uint32_t*>(&packed4), vreinterpret_u32_u8(u8), 0);
            std::memcpy(op + i, &packed4, 4);
        }
        for (; i < gs; ++i) {
            const float q  = (xp[i] - mn) * inv;
            long        qi = std::lrint(q);
            if (qi < 0) qi = 0;
            if (qi > n_levels) qi = n_levels;
            op[i] = static_cast<std::uint8_t>(qi);
        }
    }
#else
    neon_scalar::group_quant_row(x, dim, gs, n_levels, idx_raw, scales, zeros);
#endif
}

// -----------------------------------------------------------------------------
// group_dequant_row: x_out = idx * scale + zero, per group.
// -----------------------------------------------------------------------------
[[gnu::always_inline]] inline void group_dequant_row(const std::uint8_t* idx_raw, std::size_t dim,
                                                     std::size_t gs, const float* scales,
                                                     const float* zeros, float* x_out) noexcept {
#if defined(__ARM_NEON)
    const std::size_t ng = dim / gs;
    for (std::size_t g = 0; g < ng; ++g) {
        const float         sc  = scales[g];
        const float         z   = zeros[g];
        const float32x4_t   vsc = vdupq_n_f32(sc);
        const float32x4_t   vz  = vdupq_n_f32(z);
        const std::uint8_t* ip  = idx_raw + g * gs;
        float*              op  = x_out + g * gs;
        std::size_t         i   = 0;
        for (; i + 8 <= gs; i += 8) {
            const uint8x8_t   u8   = vld1_u8(ip + i);
            const uint16x8_t  u16  = vmovl_u8(u8);
            const uint32x4_t  lo32 = vmovl_u16(vget_low_u16(u16));
            const uint32x4_t  hi32 = vmovl_u16(vget_high_u16(u16));
            const float32x4_t f0   = vcvtq_f32_u32(lo32);
            const float32x4_t f1   = vcvtq_f32_u32(hi32);
            vst1q_f32(op + i, vfmaq_f32(vz, f0, vsc));
            vst1q_f32(op + i + 4, vfmaq_f32(vz, f1, vsc));
        }
        for (; i + 4 <= gs; i += 4) {
            const uint8x8_t   u8  = vld1_u8(ip + i);  // loads 8, we use 4
            const uint16x8_t  u16 = vmovl_u8(u8);
            const uint32x4_t  u32 = vmovl_u16(vget_low_u16(u16));
            const float32x4_t f   = vcvtq_f32_u32(u32);
            vst1q_f32(op + i, vfmaq_f32(vz, f, vsc));
        }
        for (; i < gs; ++i) {
            op[i] = static_cast<float>(ip[i]) * sc + z;
        }
    }
#else
    neon_scalar::group_dequant_row(idx_raw, dim, gs, scales, zeros, x_out);
#endif
}

}  // namespace tq::neon
