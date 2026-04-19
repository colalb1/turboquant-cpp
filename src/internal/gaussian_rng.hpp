#pragma once

// Marsaglia polar Gaussian sampler, shared by rotation_accelerate.cpp and
// quantizer_prod.cpp. Declared `inline` so each TU sees a whole definition
// the optimizer can fold through the fill loop.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace tq::internal {

inline void next_pair(std::mt19937_64& eng, float& a, float& b) noexcept {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    float                                 u1 = 0.0f;
    float                                 u2 = 0.0f;
    float                                 s  = 0.0f;
    do {
        u1 = u(eng);
        u2 = u(eng);
        s  = u1 * u1 + u2 * u2;
    } while (s >= 1.0f || s == 0.0f);
    const float f = std::sqrt(-2.0f * std::log(s) / s);
    a             = u1 * f;
    b             = u2 * f;
}

inline void fill_gaussian(float* out, std::size_t n, std::uint32_t seed) noexcept {
    std::mt19937_64 eng(static_cast<std::uint64_t>(seed));
    std::size_t     i = 0;
    while (i + 1 < n) {
        next_pair(eng, out[i], out[i + 1]);
        i += 2;
    }
    if (i < n) {
        float a = 0.0f;
        float b = 0.0f;
        next_pair(eng, a, b);
        out[i] = a;
    }
}

}  // namespace tq::internal
