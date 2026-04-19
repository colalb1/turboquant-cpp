// CompressedKVStore — chunked key+value store with lazy flat view.
// Python reference: turboquant/store.py:26-178.

#include "turboquant/compressed_kv_store.hpp"
#include "turboquant/qjl_signs.hpp"

#include <cstring>
#include <utility>

namespace tq {

namespace {

// Transpose (T, H, D) row-major floats → (H, T, D) row-major floats.
inline void transpose_thd_to_htd(const float* src, float* dst, std::size_t T, std::size_t H,
                                 std::size_t D) noexcept {
    const std::size_t row_bytes = D * sizeof(float);
    for (std::size_t h = 0; h < H; ++h) {
        for (std::size_t t = 0; t < T; ++t) {
            std::memcpy(dst + (h * T + t) * D, src + (t * H + h) * D, row_bytes);
        }
    }
}

}  // namespace

template <int KeyBits, int ValBits>
Result<CompressedKVStore<KeyBits, ValBits>>
CompressedKVStore<KeyBits, ValBits>::make(std::size_t head_dim, std::size_t num_kv_heads,
                                          std::size_t   value_group_size,
                                          std::uint32_t seed) noexcept {
    if (head_dim == 0 || num_kv_heads == 0 || value_group_size == 0)
        return make_error<CompressedKVStore>(Error::InvalidDim);
    if (head_dim % value_group_size != 0) return make_error<CompressedKVStore>(Error::InvalidDim);

    auto prod = Prod::make(head_dim, seed);
    if (!prod) return make_error<CompressedKVStore>(prod.error());

    return Result<CompressedKVStore>(
        CompressedKVStore(head_dim, num_kv_heads, value_group_size, std::move(*prod)));
}

template <int KeyBits, int ValBits>
Error CompressedKVStore<KeyBits, ValBits>::append_chunk(std::span<const float> keys,
                                                        std::span<const float> values,
                                                        std::size_t            n_tokens) noexcept {
    if (n_tokens == 0) return Error::Ok;

    const std::size_t H  = num_kv_heads_;
    const std::size_t D  = head_dim_;
    const std::size_t gs = value_group_size_;
    const std::size_t ng = val_ng();
    const std::size_t mb = mse_pb();
    const std::size_t qb = qjl_pb();
    const std::size_t vb = val_pb();
    const std::size_t HT = H * n_tokens;

    if (keys.size() != n_tokens * H * D) return Error::ShapeMismatch;
    if (values.size() != n_tokens * H * D) return Error::ShapeMismatch;

    // (T, H, D) → (H, T, D) scratch for the quantizer.
    AlignedBuffer<float> kt, vt;
    if (!kt.resize(HT * D)) return Error::RotationFailed;
    if (!vt.resize(HT * D)) return Error::RotationFailed;
    transpose_thd_to_htd(keys.data(), kt.data(), n_tokens, H, D);
    transpose_thd_to_htd(values.data(), vt.data(), n_tokens, H, D);

    Chunk ck;
    if (!ck.mse_indices.resize(HT * mb)) return Error::RotationFailed;
    if (!ck.qjl_signs.resize(HT * qb)) return Error::RotationFailed;
    if (!ck.residual_norms.resize(HT)) return Error::RotationFailed;
    if (!ck.norms.resize(HT)) return Error::RotationFailed;
    if (!ck.val_data.resize(HT * vb)) return Error::RotationFailed;
    if (!ck.val_scales.resize(HT * ng)) return Error::RotationFailed;
    if (!ck.val_zeros.resize(HT * ng)) return Error::RotationFailed;
    ck.tokens = n_tokens;

    const Error ek = prod_.quantize(std::span<const float>(kt.data(), HT * D), HT,
                                    std::span<std::uint8_t>(ck.mse_indices.data(), HT * mb),
                                    std::span<std::uint8_t>(ck.qjl_signs.data(), HT * qb),
                                    std::span<float>(ck.residual_norms.data(), HT),
                                    std::span<float>(ck.norms.data(), HT));
    if (ek != Error::Ok) return ek;

    const Error ev = Val::quantize(std::span<const float>(vt.data(), HT * D), HT, D, gs,
                                   std::span<std::uint8_t>(ck.val_data.data(), HT * vb),
                                   std::span<float>(ck.val_scales.data(), HT * ng),
                                   std::span<float>(ck.val_zeros.data(), HT * ng));
    if (ev != Error::Ok) return ev;

    chunks_.push_back(std::move(ck));
    num_tokens_ += n_tokens;
    flat_valid_  = false;
    return Error::Ok;
}

template <int KeyBits, int ValBits>
Result<typename CompressedKVStore<KeyBits, ValBits>::FlatView>
CompressedKVStore<KeyBits, ValBits>::get_flat() noexcept {
    if (chunks_.empty()) {
        FlatView v{};
        return Result<FlatView>(v);
    }
    if (flat_valid_) {
        const std::size_t H  = num_kv_heads_;
        const std::size_t T  = flat_.total_tokens;
        const std::size_t mb = mse_pb(), qb = qjl_pb(), vb = val_pb(), ng = val_ng();
        FlatView          v{};
        v.mse_indices    = {flat_.mse_indices.data(), H * T * mb};
        v.qjl_signs      = {flat_.qjl_signs.data(), H * T * qb};
        v.residual_norms = {flat_.residual_norms.data(), H * T};
        v.norms          = {flat_.norms.data(), H * T};
        v.val_data       = {flat_.val_data.data(), H * T * vb};
        v.val_scales     = {flat_.val_scales.data(), H * T * ng};
        v.val_zeros      = {flat_.val_zeros.data(), H * T * ng};
        v.total_tokens   = T;
        return Result<FlatView>(v);
    }

    const std::size_t H  = num_kv_heads_;
    const std::size_t T  = num_tokens_;
    const std::size_t mb = mse_pb(), qb = qjl_pb(), vb = val_pb(), ng = val_ng();

    if (!flat_.mse_indices.resize(H * T * mb)) return make_error<FlatView>(Error::RotationFailed);
    if (!flat_.qjl_signs.resize(H * T * qb)) return make_error<FlatView>(Error::RotationFailed);
    if (!flat_.residual_norms.resize(H * T)) return make_error<FlatView>(Error::RotationFailed);
    if (!flat_.norms.resize(H * T)) return make_error<FlatView>(Error::RotationFailed);
    if (!flat_.val_data.resize(H * T * vb)) return make_error<FlatView>(Error::RotationFailed);
    if (!flat_.val_scales.resize(H * T * ng)) return make_error<FlatView>(Error::RotationFailed);
    if (!flat_.val_zeros.resize(H * T * ng)) return make_error<FlatView>(Error::RotationFailed);

    // For each head h, concatenate chunks' head-h segments in chunk order.
    // Within a chunk, head h's bytes start at (h * T_chunk) rows.
    for (std::size_t h = 0; h < H; ++h) {
        std::size_t t_off = 0;
        for (const Chunk& ck : chunks_) {
            const std::size_t Tc      = ck.tokens;
            const std::size_t src_off = h * Tc;
            const std::size_t dst_off = h * T + t_off;

            std::memcpy(flat_.mse_indices.data() + dst_off * mb,
                        ck.mse_indices.data() + src_off * mb, Tc * mb);
            std::memcpy(flat_.qjl_signs.data() + dst_off * qb, ck.qjl_signs.data() + src_off * qb,
                        Tc * qb);
            std::memcpy(flat_.residual_norms.data() + dst_off, ck.residual_norms.data() + src_off,
                        Tc * sizeof(float));
            std::memcpy(flat_.norms.data() + dst_off, ck.norms.data() + src_off,
                        Tc * sizeof(float));
            std::memcpy(flat_.val_data.data() + dst_off * vb, ck.val_data.data() + src_off * vb,
                        Tc * vb);
            std::memcpy(flat_.val_scales.data() + dst_off * ng, ck.val_scales.data() + src_off * ng,
                        Tc * ng * sizeof(float));
            std::memcpy(flat_.val_zeros.data() + dst_off * ng, ck.val_zeros.data() + src_off * ng,
                        Tc * ng * sizeof(float));
            t_off += Tc;
        }
    }
    flat_.total_tokens = T;
    flat_valid_        = true;

    FlatView v{};
    v.mse_indices    = {flat_.mse_indices.data(), H * T * mb};
    v.qjl_signs      = {flat_.qjl_signs.data(), H * T * qb};
    v.residual_norms = {flat_.residual_norms.data(), H * T};
    v.norms          = {flat_.norms.data(), H * T};
    v.val_data       = {flat_.val_data.data(), H * T * vb};
    v.val_scales     = {flat_.val_scales.data(), H * T * ng};
    v.val_zeros      = {flat_.val_zeros.data(), H * T * ng};
    v.total_tokens   = T;
    return Result<FlatView>(v);
}

template <int KeyBits, int ValBits>
std::size_t CompressedKVStore<KeyBits, ValBits>::memory_bytes() const noexcept {
    std::size_t total = 0;
    for (const Chunk& ck : chunks_) {
        total += ck.mse_indices.size();
        total += ck.qjl_signs.size();
        total += ck.residual_norms.size() * sizeof(float);
        total += ck.norms.size() * sizeof(float);
        total += ck.val_data.size();
        total += ck.val_scales.size() * sizeof(float);
        total += ck.val_zeros.size() * sizeof(float);
    }
    return total;
}

template <int KeyBits, int ValBits>
void CompressedKVStore<KeyBits, ValBits>::reset() noexcept {
    chunks_.clear();
    num_tokens_        = 0;
    flat_valid_        = false;
    flat_.total_tokens = 0;
}

template class CompressedKVStore<2, 2>;
template class CompressedKVStore<3, 2>;
template class CompressedKVStore<4, 2>;
template class CompressedKVStore<2, 4>;
template class CompressedKVStore<3, 4>;
template class CompressedKVStore<4, 4>;
template class CompressedKVStore<2, 8>;
template class CompressedKVStore<3, 8>;
template class CompressedKVStore<4, 8>;

}  // namespace tq
