#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "turboquant/config.hpp"

namespace tq {

// Non-owning view over a batch of MSE-quantized vectors. All buffers are
// assumed 128-byte aligned; validity is a caller invariant (checked at
// factory time, not on every kernel invocation).
struct MSEQuantizedView {
    std::span<const std::uint8_t> indices;  // batch * packed_bytes(dim)
    std::span<const float>        norms;    // batch
    std::uint32_t                 batch = 0;
    std::uint32_t                 dim   = 0;
    std::uint32_t                 bits  = 0;
};

struct MSEQuantizedMut {
    std::span<std::uint8_t> indices;
    std::span<float>        norms;
    std::uint32_t           batch = 0;
    std::uint32_t           dim   = 0;
    std::uint32_t           bits  = 0;
};

// Non-owning view over a batch of Prod-quantized vectors.
struct ProdQuantizedView {
    std::span<const std::uint8_t> mse_indices;     // batch * packed_mse_bytes
    std::span<const std::uint8_t> qjl_signs;       // batch * ceil(dim/8)
    std::span<const float>        residual_norms;  // batch
    std::span<const float>        norms;           // batch
    std::uint32_t                 batch    = 0;
    std::uint32_t                 dim      = 0;
    std::uint32_t                 mse_bits = 0;
};

struct ProdQuantizedMut {
    std::span<std::uint8_t> mse_indices;
    std::span<std::uint8_t> qjl_signs;
    std::span<float>        residual_norms;
    std::span<float>        norms;
    std::uint32_t           batch    = 0;
    std::uint32_t           dim      = 0;
    std::uint32_t           mse_bits = 0;
};

// Non-owning view over a batch of group-quantized value vectors (KV cache).
struct ValueQuantizedView {
    std::span<const std::uint8_t> data;    // batch * packed_d
    std::span<const float>        scales;  // batch * n_groups
    std::span<const float>        zeros;   // batch * n_groups
    std::uint32_t                 batch      = 0;
    std::uint32_t                 dim        = 0;
    std::uint32_t                 bits       = 0;
    std::uint32_t                 group_size = 0;
};

struct ValueQuantizedMut {
    std::span<std::uint8_t> data;
    std::span<float>        scales;
    std::span<float>        zeros;
    std::uint32_t           batch      = 0;
    std::uint32_t           dim        = 0;
    std::uint32_t           bits       = 0;
    std::uint32_t           group_size = 0;
};

// Utility: round up a to the next multiple of b.
constexpr std::size_t round_up(std::size_t a, std::size_t b) noexcept {
    return (a + b - 1) / b * b;
}

// Utility: ceil(a / b).
constexpr std::size_t ceil_div(std::size_t a, std::size_t b) noexcept {
    return (a + b - 1) / b;
}

}  // namespace tq
