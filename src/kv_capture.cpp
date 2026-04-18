// KVCaptureEngine — orchestrates RingBuffer → CompressedKVStore.
// Python reference: turboquant/capture.py:134-241.

#include "turboquant/kv_capture.hpp"

#include <algorithm>
#include <utility>

namespace tq {

template <int KeyBits, int ValBits>
Result<KVCaptureEngine<KeyBits, ValBits>>
KVCaptureEngine<KeyBits, ValBits>::make(std::size_t head_dim,
                                        std::size_t num_kv_heads,
                                        std::size_t value_group_size,
                                        std::size_t ring_capacity,
                                        std::uint32_t seed) noexcept
{
    auto store = Store::make(head_dim, num_kv_heads, value_group_size, seed);
    if (!store) return make_error<KVCaptureEngine>(store.error());

    auto ring = RingBuffer::make(ring_capacity, num_kv_heads, head_dim);
    if (!ring) return make_error<KVCaptureEngine>(ring.error());

    return Result<KVCaptureEngine>(
        KVCaptureEngine(std::move(*store), std::move(*ring)));
}

template <int KeyBits, int ValBits>
Error KVCaptureEngine<KeyBits, ValBits>::drain_ring_to_store() noexcept
{
    const std::size_t n = ring_.size();
    if (n == 0) return Error::Ok;

    const std::size_t rs = ring_.row_stride();
    AlignedBuffer<float> kbuf, vbuf;
    if (!kbuf.resize(n * rs)) return Error::RotationFailed;
    if (!vbuf.resize(n * rs)) return Error::RotationFailed;
    const std::size_t drained = ring_.drain(
        std::span<float>(kbuf.data(), n * rs),
        std::span<float>(vbuf.data(), n * rs));
    if (drained != n) return Error::BufferTooSmall;

    return store_.append_chunk(
        std::span<const float>(kbuf.data(), n * rs),
        std::span<const float>(vbuf.data(), n * rs),
        n);
}

template <int KeyBits, int ValBits>
Error KVCaptureEngine<KeyBits, ValBits>::ingest_prefill(
    std::span<const float> keys,
    std::span<const float> values,
    std::size_t            n_tokens) noexcept
{
    const std::size_t rs = ring_.row_stride();
    if (keys.size()   != n_tokens * rs) return Error::ShapeMismatch;
    if (values.size() != n_tokens * rs) return Error::ShapeMismatch;

    const std::size_t cap = ring_.capacity();

    if (n_tokens <= cap) {
        // Everything fits in the ring. Drain whatever is buffered first if
        // the new batch would overflow.
        if (ring_.size() + n_tokens > cap) {
            const Error e = drain_ring_to_store();
            if (e != Error::Ok) return e;
        }
        return ring_.write(keys, values, n_tokens);
    }

    // n_tokens > capacity: compress leading (n_tokens - capacity) tokens,
    // keep the most recent `capacity` in the ring.
    // Drain any buffered content first so the compressed chunk is chronological.
    if (ring_.size() > 0) {
        const Error e = drain_ring_to_store();
        if (e != Error::Ok) return e;
    }
    const std::size_t n_compress = n_tokens - cap;
    const Error ec = store_.append_chunk(
        std::span<const float>(keys.data(),   n_compress * rs),
        std::span<const float>(values.data(), n_compress * rs),
        n_compress);
    if (ec != Error::Ok) return ec;

    return ring_.write(
        std::span<const float>(keys.data()   + n_compress * rs, cap * rs),
        std::span<const float>(values.data() + n_compress * rs, cap * rs),
        cap);
}

template <int KeyBits, int ValBits>
Error KVCaptureEngine<KeyBits, ValBits>::ingest_decode(
    std::span<const float> keys,
    std::span<const float> values,
    std::size_t            n_tokens) noexcept
{
    const std::size_t rs = ring_.row_stride();
    if (keys.size()   != n_tokens * rs) return Error::ShapeMismatch;
    if (values.size() != n_tokens * rs) return Error::ShapeMismatch;

    const float* k = keys.data();
    const float* v = values.data();
    std::size_t remaining = n_tokens;

    while (remaining > 0) {
        if (ring_.is_full()) {
            const Error e = drain_ring_to_store();
            if (e != Error::Ok) return e;
        }
        const std::size_t n = std::min(remaining, ring_.space_left());
        const Error ew = ring_.write(
            std::span<const float>(k, n * rs),
            std::span<const float>(v, n * rs), n);
        if (ew != Error::Ok) return ew;
        k         += n * rs;
        v         += n * rs;
        remaining -= n;
    }
    return Error::Ok;
}

template <int KeyBits, int ValBits>
Error KVCaptureEngine<KeyBits, ValBits>::flush() noexcept
{
    return drain_ring_to_store();
}

template class KVCaptureEngine<2, 2>;
template class KVCaptureEngine<3, 2>;
template class KVCaptureEngine<4, 2>;
template class KVCaptureEngine<2, 4>;
template class KVCaptureEngine<3, 4>;
template class KVCaptureEngine<4, 4>;
template class KVCaptureEngine<2, 8>;
template class KVCaptureEngine<3, 8>;
template class KVCaptureEngine<4, 8>;

} // namespace tq
