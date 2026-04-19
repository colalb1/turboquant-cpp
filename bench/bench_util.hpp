#pragma once

// Shared helpers for turboquant_bench.
//
// - deterministic fp32 fills
// - tiny Result<T> unwrap that abort()s on error (benches treat factory
//   failure as a configuration bug, not a runtime condition)

#include "turboquant/error.hpp"

#include <cstdint>
#include <cstdlib>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace tq::bench {

inline void fill_gaussian(std::span<float> buf, std::uint32_t seed) noexcept {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float& x : buf) x = dist(rng);
}

inline void fill_uniform(std::span<float> buf, float lo, float hi,
                         std::uint32_t seed) noexcept {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& x : buf) x = dist(rng);
}

template <class T>
T must(Result<T>&& r) {
    if (!r) std::abort();
    return std::move(*r);
}

} // namespace tq::bench
