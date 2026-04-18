#pragma once

// TurboQuant MSE quantizer (Algorithm 1).
//
// Python reference: turboquant/quantizer.py:93-169
//   quantize(x):
//     norms   = ||x||_2                 (per-row)
//     x_unit  = x / (norms + 1e-10)
//     y       = rotate_forward(x_unit, Pi)     # y = x_unit @ Pi.T
//     indices = searchsorted(decision_boundaries, y)   # right=False
//     packed  = bit_pack(indices, bits)
//   dequantize(packed, norms):
//     indices = bit_unpack(packed)
//     y_hat   = centroids[indices]
//     x_hat   = rotate_backward(y_hat, Pi)   # x_hat = y_hat @ Pi
//     x_hat  *= norms

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "turboquant/aligned_buffer.hpp"
#include "turboquant/arch.hpp"
#include "turboquant/codebook.hpp"
#include "turboquant/config.hpp"
#include "turboquant/error.hpp"
#include "turboquant/pack_policy.hpp"
#include "turboquant/rotation.hpp"

namespace tq {

template <int Bits, ArchTag Arch = arch::Scalar>
class TurboQuantMSE {
    static_assert(Bits >= 1 && Bits <= 4, "Bits must be in [1, 4]");

 public:
    using Pack = PackPolicy<Bits>;

    TurboQuantMSE() = default;
    TurboQuantMSE(const TurboQuantMSE&)            = delete;
    TurboQuantMSE& operator=(const TurboQuantMSE&) = delete;
    TurboQuantMSE(TurboQuantMSE&&) noexcept        = default;
    TurboQuantMSE& operator=(TurboQuantMSE&&) noexcept = default;

    // Factory: builds rotation + fetches codebook.
    [[nodiscard]] TQ_API static Result<TurboQuantMSE>
    make(std::size_t dim, std::uint32_t seed) noexcept;

    // Factory from an externally supplied rotation matrix (parity fixtures).
    [[nodiscard]] TQ_API static Result<TurboQuantMSE>
    from_matrix(std::span<const float> pi_row_major, std::size_t dim) noexcept;

    // Quantize `batch` row-major vectors of dim `dim()`.
    //   x            : batch * dim floats
    //   indices_out  : batch * packed_bytes(dim) uint8
    //   norms_out    : batch floats
    [[nodiscard]] TQ_API Error
    quantize(std::span<const float> x,
             std::size_t             batch,
             std::span<std::uint8_t> indices_out,
             std::span<float>        norms_out) const noexcept;

    // Dequantize back to `batch` row-major vectors.
    [[nodiscard]] TQ_API Error
    dequantize(std::span<const std::uint8_t> indices,
               std::span<const float>        norms,
               std::size_t                   batch,
               std::span<float>              x_out) const noexcept;

    static constexpr std::size_t packed_bytes(std::size_t d) noexcept {
        return Pack::packed_bytes(d);
    }

    std::size_t dim()  const noexcept { return dim_; }
    int         bits() const noexcept { return Bits; }

    // Access to underlying rotation (used by Prod quantizer, tests).
    const Rotation& rotation() const noexcept { return rotation_; }

    // Access to codebook view (centroids + interior decision boundaries).
    CodebookView codebook() const noexcept { return codebook_; }

 private:
    TurboQuantMSE(std::size_t d, Rotation rot, CodebookView cb) noexcept
        : dim_(static_cast<std::uint32_t>(d)), rotation_(std::move(rot)), codebook_(cb) {}

    std::uint32_t dim_ = 0;
    Rotation      rotation_{};
    CodebookView  codebook_{};
};

// Explicit instantiations live in src/quantizer_mse.cpp.
extern template class TurboQuantMSE<1, arch::Scalar>;
extern template class TurboQuantMSE<2, arch::Scalar>;
extern template class TurboQuantMSE<3, arch::Scalar>;
extern template class TurboQuantMSE<4, arch::Scalar>;

} // namespace tq
