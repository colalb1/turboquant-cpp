// Per-group asymmetric quantization for KV-cache values.
// Python reference: turboquant/kv_cache.py:45-116.
//
// Compiled with -fno-exceptions -fno-rtti. Allocation uses AlignedBuffer
// for the per-row raw uint8 scratch when bits<8 (we pack after producing
// `dim` unpacked indices). For bits=8 we write indices directly to the
// output buffer and skip the scratch.

#include "turboquant/value_quant.hpp"
#include "turboquant/aligned_buffer.hpp"
#include "turboquant/neon/kernels.hpp"

#include <cstring>

namespace tq {

// Out-of-line bits=8 pack/unpack: straight byte copy (header forward-decls these).
namespace detail {
void ValuePack<8>::Policy::pack(const std::uint8_t* in, std::size_t d, std::uint8_t* out) noexcept {
    std::memcpy(out, in, d);
}
void ValuePack<8>::Policy::unpack(const std::uint8_t* in, std::size_t d,
                                  std::uint8_t* out) noexcept {
    std::memcpy(out, in, d);
}
}  // namespace detail

template <int Bits>
Error ValueCodec<Bits>::quantize(std::span<const float> v, std::size_t batch, std::size_t dim,
                                 std::size_t group_size, std::span<std::uint8_t> data_out,
                                 std::span<float> scales_out, std::span<float> zeros_out) noexcept {
    if (dim == 0 || group_size == 0) return Error::InvalidDim;
    if (dim % group_size != 0) return Error::InvalidDim;
    const std::size_t ng = dim / group_size;
    const std::size_t pb = Pack::packed_bytes(dim);

    if (v.size() != batch * dim) return Error::ShapeMismatch;
    if (data_out.size() != batch * pb) return Error::ShapeMismatch;
    if (scales_out.size() != batch * ng) return Error::ShapeMismatch;
    if (zeros_out.size() != batch * ng) return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

    AlignedBuffer<std::uint8_t> raw;
    if (!raw.resize(dim)) return Error::RotationFailed;

    for (std::size_t b = 0; b < batch; ++b) {
        neon::group_quant_row(v.data() + b * dim, dim, group_size, n_levels, raw.data(),
                              scales_out.data() + b * ng, zeros_out.data() + b * ng);
        Pack::pack(raw.data(), dim, data_out.data() + b * pb);
    }
    return Error::Ok;
}

template <int Bits>
Error ValueCodec<Bits>::dequantize(std::span<const std::uint8_t> data,
                                   std::span<const float> scales, std::span<const float> zeros,
                                   std::size_t batch, std::size_t dim, std::size_t group_size,
                                   std::span<float> v_out) noexcept {
    if (dim == 0 || group_size == 0) return Error::InvalidDim;
    if (dim % group_size != 0) return Error::InvalidDim;
    const std::size_t ng = dim / group_size;
    const std::size_t pb = Pack::packed_bytes(dim);

    if (data.size() != batch * pb) return Error::ShapeMismatch;
    if (scales.size() != batch * ng) return Error::ShapeMismatch;
    if (zeros.size() != batch * ng) return Error::ShapeMismatch;
    if (v_out.size() != batch * dim) return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

    AlignedBuffer<std::uint8_t> raw;
    if (!raw.resize(dim)) return Error::RotationFailed;

    for (std::size_t b = 0; b < batch; ++b) {
        Pack::unpack(data.data() + b * pb, dim, raw.data());
        neon::group_dequant_row(raw.data(), dim, group_size, scales.data() + b * ng,
                                zeros.data() + b * ng, v_out.data() + b * dim);
    }
    return Error::Ok;
}

template struct ValueCodec<2>;
template struct ValueCodec<4>;
template struct ValueCodec<8>;

}  // namespace tq
