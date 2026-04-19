#pragma once

#include <cstddef>
#include <cstdint>

namespace tq {

// Apple M-series L1 cache line (per CPP_IMPLEMENTATION_RULES.md §2).
inline constexpr std::size_t kCacheLine = 128;

// Hard cap on per-layer dimension; used for static buffer sizing.
// Covers all head_dim values in the Python reference (64, 128, 256, 576).
inline constexpr std::size_t kMaxDim = 1024;

// Upper bound on supported bit-widths in the quantizer.
inline constexpr int kMaxBits = 4;

// Public symbol visibility marker. Enables `-fvisibility=hidden` at the TU
// level while still exporting the small set of API types and factory funcs.
#if defined(_WIN32)
#define TQ_API
#else
#define TQ_API __attribute__((visibility("default")))
#endif

}  // namespace tq
