#pragma once

// Chunked compressed KV store: owns TurboQuantProd (keys) and ValueCodec
// (values) and accumulates quantized chunks. A lazy flat cache is
// materialized on first `get_flat()` and invalidated on the next write.
//
// Python reference: turboquant/store.py:26-178.
//
// Tensor layouts:
//   - append_chunk input:   keys/values are (n_tokens, num_kv_heads, head_dim)
//                           float row-major (matches the Python caller).
//   - internal chunk/flat:  (num_kv_heads, n_tokens, ...) per quantized field,
//                           which is what TurboQuantProd consumes.
//
// Bit widths are compile-time template parameters; runtime dispatch (e.g.
// for the ONNX op kernel) happens at a higher layer by switching on the
// requested bit tuple and calling the right instantiation.

#include "turboquant/aligned_buffer.hpp"
#include "turboquant/error.hpp"
#include "turboquant/quantizer_prod.hpp"
#include "turboquant/value_quant.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tq {

template <int KeyBits, int ValBits>
class CompressedKVStore {
 public:
    static_assert(KeyBits >= 2 && KeyBits <= 4, "CompressedKVStore: KeyBits must be in [2, 4]");
    static_assert(ValBits == 2 || ValBits == 4 || ValBits == 8,
                  "CompressedKVStore: ValBits must be in {2, 4, 8}");

    using Prod = TurboQuantProd<KeyBits>;
    using Val  = ValueCodec<ValBits>;

    static Result<CompressedKVStore> make(std::size_t head_dim, std::size_t num_kv_heads,
                                          std::size_t   value_group_size,
                                          std::uint32_t seed) noexcept;

    std::size_t head_dim() const noexcept { return head_dim_; }
    std::size_t num_kv_heads() const noexcept { return num_kv_heads_; }
    std::size_t value_group_size() const noexcept { return value_group_size_; }
    std::size_t num_tokens() const noexcept { return num_tokens_; }
    std::size_t num_chunks() const noexcept { return chunks_.size(); }
    const Prod& quantizer() const noexcept { return prod_; }

    // Per-token byte/float sizes.
    std::size_t mse_pb() const noexcept { return Prod::mse_packed_bytes(head_dim_); }
    std::size_t qjl_pb() const noexcept { return Prod::qjl_packed_bytes(head_dim_); }
    std::size_t val_pb() const noexcept { return Val::packed_bytes(head_dim_); }
    std::size_t val_ng() const noexcept { return Val::n_groups(head_dim_, value_group_size_); }

    // Quantize and append `n_tokens` worth of keys/values. Inputs are
    // (n_tokens, num_kv_heads, head_dim) row-major floats.
    Error append_chunk(std::span<const float> keys, std::span<const float> values,
                       std::size_t n_tokens) noexcept;

    // Lazy flat view. Spans are valid until the next append_chunk/reset.
    struct FlatView {
        std::span<const std::uint8_t> mse_indices;     // H * T * pb_mse
        std::span<const std::uint8_t> qjl_signs;       // H * T * pb_qjl
        std::span<const float>        residual_norms;  // H * T
        std::span<const float>        norms;           // H * T
        std::span<const std::uint8_t> val_data;        // H * T * pb_val
        std::span<const float>        val_scales;      // H * T * ng
        std::span<const float>        val_zeros;       // H * T * ng
        std::size_t                   total_tokens = 0;
    };

    Result<FlatView> get_flat() noexcept;

    std::size_t memory_bytes() const noexcept;
    void        reset() noexcept;

    CompressedKVStore(CompressedKVStore&&) noexcept            = default;
    CompressedKVStore& operator=(CompressedKVStore&&) noexcept = default;
    CompressedKVStore(const CompressedKVStore&)                = delete;
    CompressedKVStore& operator=(const CompressedKVStore&)     = delete;

 private:
    struct Chunk {
        AlignedBuffer<std::uint8_t> mse_indices;
        AlignedBuffer<std::uint8_t> qjl_signs;
        AlignedBuffer<float>        residual_norms;
        AlignedBuffer<float>        norms;
        AlignedBuffer<std::uint8_t> val_data;
        AlignedBuffer<float>        val_scales;
        AlignedBuffer<float>        val_zeros;
        std::size_t                 tokens = 0;
    };

    struct Flat {
        AlignedBuffer<std::uint8_t> mse_indices;
        AlignedBuffer<std::uint8_t> qjl_signs;
        AlignedBuffer<float>        residual_norms;
        AlignedBuffer<float>        norms;
        AlignedBuffer<std::uint8_t> val_data;
        AlignedBuffer<float>        val_scales;
        AlignedBuffer<float>        val_zeros;
        std::size_t                 total_tokens = 0;
    };

    CompressedKVStore(std::size_t head_dim, std::size_t num_kv_heads, std::size_t value_group_size,
                      Prod prod) noexcept
        : head_dim_(head_dim),
          num_kv_heads_(num_kv_heads),
          value_group_size_(value_group_size),
          prod_(std::move(prod)) {}

    std::size_t        head_dim_         = 0;
    std::size_t        num_kv_heads_     = 0;
    std::size_t        value_group_size_ = 0;
    std::size_t        num_tokens_       = 0;
    Prod               prod_;
    std::vector<Chunk> chunks_;
    Flat               flat_;
    bool               flat_valid_ = false;
};

extern template class CompressedKVStore<2, 2>;
extern template class CompressedKVStore<3, 2>;
extern template class CompressedKVStore<4, 2>;
extern template class CompressedKVStore<2, 4>;
extern template class CompressedKVStore<3, 4>;
extern template class CompressedKVStore<4, 4>;
extern template class CompressedKVStore<2, 8>;
extern template class CompressedKVStore<3, 8>;
extern template class CompressedKVStore<4, 8>;

}  // namespace tq
