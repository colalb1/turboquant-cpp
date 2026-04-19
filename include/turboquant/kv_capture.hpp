#pragma once

// KVCaptureEngine — orchestrator between the recent-token ring buffer
// and the compressed KV store. Mirrors turboquant/capture.py:134-241.
//
// Write paths:
//   - ingest_prefill(keys, values, N): bulk-capture at prefill. If N fits
//     in the ring, write all to ring; otherwise compress the leading
//     (N - capacity) tokens into the store and keep the last `capacity`
//     in the ring.
//   - ingest_decode(keys, values, N):  append N tokens to the ring, draining
//     and compressing to the store whenever the ring fills.
//   - flush(): compress any buffered tokens immediately.

#include "turboquant/compressed_kv_store.hpp"
#include "turboquant/ring_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace tq {

template <int KeyBits, int ValBits>
class KVCaptureEngine {
 public:
    using Store = CompressedKVStore<KeyBits, ValBits>;

    static Result<KVCaptureEngine> make(std::size_t head_dim, std::size_t num_kv_heads,
                                        std::size_t value_group_size, std::size_t ring_capacity,
                                        std::uint32_t seed) noexcept;

    Error ingest_prefill(std::span<const float> keys, std::span<const float> values,
                         std::size_t n_tokens) noexcept;

    Error ingest_decode(std::span<const float> keys, std::span<const float> values,
                        std::size_t n_tokens) noexcept;

    Error flush() noexcept;

    std::size_t total_compressed_tokens() const noexcept { return store_.num_tokens(); }
    std::size_t total_buffered_tokens() const noexcept { return ring_.size(); }
    std::size_t total_tokens() const noexcept {
        return total_compressed_tokens() + total_buffered_tokens();
    }

    Store&            store() noexcept { return store_; }
    const Store&      store() const noexcept { return store_; }
    RingBuffer&       ring() noexcept { return ring_; }
    const RingBuffer& ring() const noexcept { return ring_; }

    void reset() noexcept {
        ring_.reset();
        store_.reset();
    }

    KVCaptureEngine(KVCaptureEngine&&) noexcept            = default;
    KVCaptureEngine& operator=(KVCaptureEngine&&) noexcept = default;
    KVCaptureEngine(const KVCaptureEngine&)                = delete;
    KVCaptureEngine& operator=(const KVCaptureEngine&)     = delete;

 private:
    KVCaptureEngine(Store s, RingBuffer r) noexcept : store_(std::move(s)), ring_(std::move(r)) {}

    // Copy ring buffer contents to the store as a single compressed chunk.
    Error drain_ring_to_store() noexcept;

    Store      store_;
    RingBuffer ring_;
};

extern template class KVCaptureEngine<2, 2>;
extern template class KVCaptureEngine<3, 2>;
extern template class KVCaptureEngine<4, 2>;
extern template class KVCaptureEngine<2, 4>;
extern template class KVCaptureEngine<3, 4>;
extern template class KVCaptureEngine<4, 4>;
extern template class KVCaptureEngine<2, 8>;
extern template class KVCaptureEngine<3, 8>;
extern template class KVCaptureEngine<4, 8>;

}  // namespace tq
