// Scalar implementation of TurboQuantMSE<Bits, arch::Scalar>.
//
// Compiled with -fno-exceptions (see CMakeLists.txt). All control flow uses
// Error / Result codes; no throws.
//
// Layout is column-agnostic on the caller side: inputs/outputs are row-major
// (batch, dim). Internally we use an AlignedBuffer<float> scratch per call
// to hold the normalized-then-rotated vector for a single row, avoiding
// per-row heap traffic on the hot path.

#include "turboquant/quantizer_mse.hpp"
#include "turboquant/neon/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace tq {

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Result<TurboQuantMSE<Bits, Arch>> TurboQuantMSE<Bits, Arch>::make(std::size_t   dim,
                                                                  std::uint32_t seed) noexcept {
    if (dim == 0 || dim > kMaxDim) return make_error<TurboQuantMSE>(Error::InvalidDim);

    auto rot = Rotation::make(dim, seed);
    if (!rot) return make_error<TurboQuantMSE>(rot.error());

    auto cb = CodebookRegistry::instance().get(static_cast<std::uint32_t>(dim),
                                               static_cast<std::uint32_t>(Bits));
    if (!cb) return make_error<TurboQuantMSE>(cb.error());

    return Result<TurboQuantMSE>(TurboQuantMSE(dim, std::move(*rot), *cb));
}

template <int Bits, ArchTag Arch>
Result<TurboQuantMSE<Bits, Arch>>
TurboQuantMSE<Bits, Arch>::from_matrix(std::span<const float> pi_row_major,
                                       std::size_t            dim) noexcept {
    if (dim == 0 || dim > kMaxDim) return make_error<TurboQuantMSE>(Error::InvalidDim);

    auto rot = Rotation::from_matrix(pi_row_major, dim);
    if (!rot) return make_error<TurboQuantMSE>(rot.error());

    auto cb = CodebookRegistry::instance().get(static_cast<std::uint32_t>(dim),
                                               static_cast<std::uint32_t>(Bits));
    if (!cb) return make_error<TurboQuantMSE>(cb.error());

    return Result<TurboQuantMSE>(TurboQuantMSE(dim, std::move(*rot), *cb));
}

// -----------------------------------------------------------------------------
// quantize
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Error TurboQuantMSE<Bits, Arch>::quantize(std::span<const float> x, std::size_t batch,
                                          std::span<std::uint8_t> indices_out,
                                          std::span<float>        norms_out) const noexcept {
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d  = dim_;
    const std::size_t pb = Pack::packed_bytes(d);

    if (x.size() != batch * d) return Error::ShapeMismatch;
    if (indices_out.size() != batch * pb) return Error::ShapeMismatch;
    if (norms_out.size() != batch) return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

    if (!scratch_unit_.ensure_size(d)) return Error::RotationFailed;
    if (!scratch_rotated_.ensure_size(d)) return Error::RotationFailed;

    const float*      bounds   = codebook_.decision_boundaries.data();
    const std::size_t n_bounds = codebook_.decision_boundaries.size();

    for (std::size_t b = 0; b < batch; ++b) {
        const float* xb = x.data() + b * d;

        const float norm = neon::l2norm(xb, d);
        norms_out[b]     = norm;
        const float inv  = 1.0f / (norm + 1e-10f);
        neon::scale(xb, inv, scratch_unit_.data(), d);

        const Error fe = rotation_.forward(std::span<const float>(scratch_unit_.data(), d),
                                           std::span<float>(scratch_rotated_.data(), d),
                                           /*batch=*/1);
        if (fe != Error::Ok) return fe;

        neon::searchsorted_and_pack<Bits>(scratch_rotated_.data(), bounds, n_bounds,
                                          indices_out.data() + b * pb, d);
    }

    return Error::Ok;
}

// -----------------------------------------------------------------------------
// dequantize
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Error TurboQuantMSE<Bits, Arch>::dequantize(std::span<const std::uint8_t> indices,
                                            std::span<const float> norms, std::size_t batch,
                                            std::span<float> x_out) const noexcept {
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d  = dim_;
    const std::size_t pb = Pack::packed_bytes(d);

    if (indices.size() != batch * pb) return Error::ShapeMismatch;
    if (norms.size() != batch) return Error::ShapeMismatch;
    if (x_out.size() != batch * d) return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

    if (!scratch_yhat_.ensure_size(d)) return Error::RotationFailed;

    const float* centroids = codebook_.centroids.data();

    for (std::size_t b = 0; b < batch; ++b) {
        neon::unpack_and_gather<Bits>(indices.data() + b * pb, centroids, scratch_yhat_.data(), d);

        float*      xb = x_out.data() + b * d;
        const Error be = rotation_.backward(std::span<const float>(scratch_yhat_.data(), d),
                                            std::span<float>(xb, d),
                                            /*batch=*/1);
        if (be != Error::Ok) return be;

        neon::scale(xb, norms[b], xb, d);
    }

    return Error::Ok;
}

// -----------------------------------------------------------------------------
// Explicit instantiations — keep headers free of template bodies.
// -----------------------------------------------------------------------------

template class TurboQuantMSE<1, arch::Scalar>;
template class TurboQuantMSE<2, arch::Scalar>;
template class TurboQuantMSE<3, arch::Scalar>;
template class TurboQuantMSE<4, arch::Scalar>;

}  // namespace tq
