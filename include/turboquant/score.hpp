#pragma once

// Hybrid attention over a CompressedKVStore + recent ring contents.
//
// Python reference: turboquant/score.py:29-173.
//
// Three paths, selected by `has_history` / `has_recent`:
//   - compressed-only: store has >= MIN_HISTORY_FOR_TQ tokens, no recent
//   - recent-only:     store has < MIN_HISTORY_FOR_TQ tokens (or is empty),
//                      recent buffer non-empty
//   - hybrid:          both segments present, concatenated K/V before matmul
//   - (neither):       output is zero-filled
//
// Layouts (row-major float32):
//   query     : (n_q_tokens, num_query_heads, head_dim)
//   recent_k  : (n_recent,   num_kv_heads,     head_dim)
//   recent_v  : (n_recent,   num_kv_heads,     head_dim)
//   out       : (n_q_tokens, num_query_heads, head_dim)
//
// `scale` ≤ 0 → default to 1/sqrt(head_dim). `num_query_heads` must be a
// multiple of `store.num_kv_heads()` (GQA factor).

#include "turboquant/compressed_kv_store.hpp"
#include "turboquant/error.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace tq {

inline constexpr std::size_t MIN_HISTORY_FOR_TQ = 16;

template <int KeyBits, int ValBits>
Error compute_hybrid_attention(std::span<const float> query, std::size_t n_q_tokens,
                               std::size_t                          num_query_heads,
                               CompressedKVStore<KeyBits, ValBits>& store,
                               std::span<const float> recent_k, std::span<const float> recent_v,
                               std::size_t n_recent, float scale, std::span<float> out) noexcept;

extern template Error compute_hybrid_attention<2, 2>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<2, 2>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;
extern template Error compute_hybrid_attention<3, 2>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<3, 2>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;
extern template Error compute_hybrid_attention<4, 2>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<4, 2>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;
extern template Error compute_hybrid_attention<2, 4>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<2, 4>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;
extern template Error compute_hybrid_attention<3, 4>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<3, 4>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;
extern template Error compute_hybrid_attention<4, 4>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<4, 4>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;
extern template Error compute_hybrid_attention<2, 8>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<2, 8>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;
extern template Error compute_hybrid_attention<3, 8>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<3, 8>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;
extern template Error compute_hybrid_attention<4, 8>(std::span<const float>, std::size_t,
                                                     std::size_t, CompressedKVStore<4, 8>&,
                                                     std::span<const float>, std::span<const float>,
                                                     std::size_t, float, std::span<float>) noexcept;

}  // namespace tq
