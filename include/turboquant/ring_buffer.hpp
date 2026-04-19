#pragma once

// Fixed-capacity recent-token buffer for the KV cache. Despite the Python
// name ("RingBuffer") this is a sequential buffer without wrap-around:
// `pos_` advances monotonically until the buffer is full, at which point
// the engine drains it (resetting pos_ to 0) and continues.
//
// Layout: row-major (n_tokens, num_kv_heads, head_dim) float32. Matches
// the Python reference in turboquant/capture.py:21-131.

#include "turboquant/aligned_buffer.hpp"
#include "turboquant/error.hpp"

#include <cstddef>
#include <span>

namespace tq {

class RingBuffer {
 public:
    static Result<RingBuffer> make(std::size_t capacity, std::size_t num_kv_heads,
                                   std::size_t head_dim) noexcept;

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t num_kv_heads() const noexcept { return num_kv_heads_; }
    std::size_t head_dim() const noexcept { return head_dim_; }
    std::size_t row_stride() const noexcept { return num_kv_heads_ * head_dim_; }

    std::size_t size() const noexcept { return pos_; }
    std::size_t space_left() const noexcept { return capacity_ - pos_; }
    bool        is_full() const noexcept { return pos_ >= capacity_; }
    std::size_t total_written() const noexcept { return total_written_; }

    // Append `n_tokens` tokens. Caller must ensure n_tokens <= space_left()
    // and that `keys` / `values` each hold at least n_tokens * row_stride()
    // floats. Returns ShapeMismatch if the span sizes are inconsistent.
    Error write(std::span<const float> keys, std::span<const float> values,
                std::size_t n_tokens) noexcept;

    // Copy all buffered tokens to `out_k` and `out_v`, then reset pos to 0.
    // `out_k` and `out_v` must each hold size() * row_stride() floats.
    // Returns the number of tokens drained.
    std::size_t drain(std::span<float> out_k, std::span<float> out_v) noexcept;

    // Read-only view over the currently buffered tokens.
    std::span<const float> keys_view() const noexcept;
    std::span<const float> values_view() const noexcept;

    void reset() noexcept {
        pos_           = 0;
        total_written_ = 0;
    }

    RingBuffer(RingBuffer&&) noexcept            = default;
    RingBuffer& operator=(RingBuffer&&) noexcept = default;
    RingBuffer(const RingBuffer&)                = delete;
    RingBuffer& operator=(const RingBuffer&)     = delete;

 private:
    RingBuffer(std::size_t cap, std::size_t nh, std::size_t d, AlignedBuffer<float> k,
               AlignedBuffer<float> v) noexcept
        : capacity_(cap), num_kv_heads_(nh), head_dim_(d), k_(std::move(k)), v_(std::move(v)) {}

    std::size_t          capacity_     = 0;
    std::size_t          num_kv_heads_ = 0;
    std::size_t          head_dim_     = 0;
    AlignedBuffer<float> k_;
    AlignedBuffer<float> v_;
    std::size_t          pos_           = 0;
    std::size_t          total_written_ = 0;
};

}  // namespace tq
