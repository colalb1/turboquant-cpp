#pragma once

// Per-group asymmetric quantization for KV-cache value vectors.
//
// Python reference: turboquant/kv_cache.py:45-116 (quantize_values /
// dequantize_values). Each row of length `dim` is split into
// `n_groups = dim / group_size` contiguous groups. For each group we
// compute:
//
//   scale_g = max((max_g - min_g) / n_levels, 1e-10)
//   zero_g  = min_g
//   idx_gi  = clamp(round((x_gi - zero_g) / scale_g), 0, n_levels)
//
// `idx` is then LSB-packed per PackPolicy<Bits>. For bits=8 the packing is
// a straight byte copy. `n_levels = 2^bits - 1` — NOT 2^bits (matches
// kv_cache.py:70).
//
// Rounding follows PyTorch `.round()` = round-half-to-even, matching
// NEON `vcvtnq_s32_f32` and C's `std::rint` under FE_TONEAREST.

#include "turboquant/error.hpp"
#include "turboquant/pack_policy.hpp"
#include "turboquant/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace tq {

namespace detail {

template <int Bits>
struct ValuePack;

template <>
struct ValuePack<2> {
    using Policy                  = PackPolicy<2>;
    static constexpr int n_levels = 3;
};
template <>
struct ValuePack<4> {
    using Policy                  = PackPolicy<4>;
    static constexpr int n_levels = 15;
};
// bits=8: no bit-packing, one byte per index.
template <>
struct ValuePack<8> {
    struct Policy {
        static constexpr std::size_t packed_bytes(std::size_t d) noexcept { return d; }
        static void pack(const std::uint8_t* in, std::size_t d, std::uint8_t* out) noexcept;
        static void unpack(const std::uint8_t* in, std::size_t d, std::uint8_t* out) noexcept;
    };
    static constexpr int n_levels = 255;
};

}  // namespace detail

template <int Bits>
struct ValueCodec {
    static_assert(Bits == 2 || Bits == 4 || Bits == 8, "ValueCodec supports bits in {2, 4, 8}");

    static constexpr int bits     = Bits;
    static constexpr int n_levels = detail::ValuePack<Bits>::n_levels;
    using Pack                    = typename detail::ValuePack<Bits>::Policy;

    static constexpr std::size_t packed_bytes(std::size_t dim) noexcept {
        return Pack::packed_bytes(dim);
    }

    static constexpr std::size_t n_groups(std::size_t dim, std::size_t group_size) noexcept {
        return dim / group_size;
    }

    // Quantize `batch` rows of length `dim`. `dim` must be divisible by
    // `group_size`. Output shapes:
    //   data_out   : batch * packed_bytes(dim)
    //   scales_out : batch * n_groups(dim, group_size)
    //   zeros_out  : batch * n_groups(dim, group_size)
    static Error quantize(std::span<const float> v, std::size_t batch, std::size_t dim,
                          std::size_t group_size, std::span<std::uint8_t> data_out,
                          std::span<float> scales_out, std::span<float> zeros_out) noexcept;

    static Error dequantize(std::span<const std::uint8_t> data, std::span<const float> scales,
                            std::span<const float> zeros, std::size_t batch, std::size_t dim,
                            std::size_t group_size, std::span<float> v_out) noexcept;
};

extern template struct ValueCodec<2>;
extern template struct ValueCodec<4>;
extern template struct ValueCodec<8>;

}  // namespace tq
