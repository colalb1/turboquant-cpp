// RingBuffer — fixed-capacity recent-token FIFO. See ring_buffer.hpp.

#include "turboquant/ring_buffer.hpp"

#include <cstring>

namespace tq {

Result<RingBuffer>
RingBuffer::make(std::size_t capacity,
                 std::size_t num_kv_heads,
                 std::size_t head_dim) noexcept
{
    if (capacity == 0 || num_kv_heads == 0 || head_dim == 0)
        return make_error<RingBuffer>(Error::InvalidDim);

    const std::size_t n = capacity * num_kv_heads * head_dim;
    AlignedBuffer<float> k, v;
    if (!k.resize(n)) return make_error<RingBuffer>(Error::RotationFailed);
    if (!v.resize(n)) return make_error<RingBuffer>(Error::RotationFailed);

    return Result<RingBuffer>(
        RingBuffer(capacity, num_kv_heads, head_dim, std::move(k), std::move(v)));
}

Error RingBuffer::write(std::span<const float> keys,
                        std::span<const float> values,
                        std::size_t            n_tokens) noexcept
{
    const std::size_t rs = row_stride();
    if (keys.size()   != n_tokens * rs) return Error::ShapeMismatch;
    if (values.size() != n_tokens * rs) return Error::ShapeMismatch;
    if (n_tokens > space_left())        return Error::BufferTooSmall;
    if (n_tokens == 0)                  return Error::Ok;

    std::memcpy(k_.data() + pos_ * rs, keys.data(),   n_tokens * rs * sizeof(float));
    std::memcpy(v_.data() + pos_ * rs, values.data(), n_tokens * rs * sizeof(float));
    pos_           += n_tokens;
    total_written_ += n_tokens;
    return Error::Ok;
}

std::size_t RingBuffer::drain(std::span<float> out_k, std::span<float> out_v) noexcept
{
    const std::size_t rs = row_stride();
    const std::size_t n  = pos_;
    if (out_k.size() < n * rs || out_v.size() < n * rs) return 0;
    if (n == 0) return 0;
    std::memcpy(out_k.data(), k_.data(), n * rs * sizeof(float));
    std::memcpy(out_v.data(), v_.data(), n * rs * sizeof(float));
    pos_ = 0;
    return n;
}

std::span<const float> RingBuffer::keys_view() const noexcept
{
    return { k_.data(), pos_ * row_stride() };
}

std::span<const float> RingBuffer::values_view() const noexcept
{
    return { v_.data(), pos_ * row_stride() };
}

} // namespace tq
