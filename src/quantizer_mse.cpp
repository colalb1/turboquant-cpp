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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace tq {

namespace {

// searchsorted with right=False semantics (Python default):
//   returns the count of boundaries that are strictly less than `v`.
// Equivalent to: index i in [0, n] such that bounds[i-1] < v <= bounds[i].
//
// For TurboQuant, `bounds` is the interior slice (size 2^bits - 1) and the
// returned index is the centroid index in [0, 2^bits - 1].
inline std::uint8_t searchsorted_right_false(const float* bounds,
                                             std::size_t n,
                                             float v) noexcept
{
    // Linear scan — codebooks are tiny (≤15 entries for bits ≤ 4). Binary
    // search would be slower due to branch mispredicts on such small n.
    std::size_t i = 0;
    while (i < n && bounds[i] < v) ++i;
    return static_cast<std::uint8_t>(i);
}

// L2 norm of d floats.
inline float l2norm(const float* x, std::size_t d) noexcept
{
    double acc = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        const double v = static_cast<double>(x[i]);
        acc += v * v;
    }
    return static_cast<float>(std::sqrt(acc));
}

} // namespace

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Result<TurboQuantMSE<Bits, Arch>>
TurboQuantMSE<Bits, Arch>::make(std::size_t dim, std::uint32_t seed) noexcept
{
    if (dim == 0 || dim > kMaxDim)
        return make_error<TurboQuantMSE>(Error::InvalidDim);

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
                                       std::size_t dim) noexcept
{
    if (dim == 0 || dim > kMaxDim)
        return make_error<TurboQuantMSE>(Error::InvalidDim);

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
Error TurboQuantMSE<Bits, Arch>::quantize(std::span<const float> x,
                                          std::size_t             batch,
                                          std::span<std::uint8_t> indices_out,
                                          std::span<float>        norms_out) const noexcept
{
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d  = dim_;
    const std::size_t pb = Pack::packed_bytes(d);

    if (x.size()           != batch * d)  return Error::ShapeMismatch;
    if (indices_out.size() != batch * pb) return Error::ShapeMismatch;
    if (norms_out.size()   != batch)       return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

    // Per-call scratch: unit vector (dim), rotated output (dim), raw indices (dim).
    AlignedBuffer<float> unit;
    AlignedBuffer<float> rotated;
    AlignedBuffer<std::uint8_t> raw;
    if (!unit.resize(d))    return Error::RotationFailed;
    if (!rotated.resize(d)) return Error::RotationFailed;
    if (!raw.resize(d))     return Error::RotationFailed;

    const float* bounds = codebook_.decision_boundaries.data();
    const std::size_t n_bounds = codebook_.decision_boundaries.size();

    for (std::size_t b = 0; b < batch; ++b) {
        const float* xb = x.data() + b * d;

        // 1. norm + unit-normalize with +1e-10 epsilon (quantizer.py:138).
        const float norm = l2norm(xb, d);
        norms_out[b] = norm;
        const float inv = 1.0f / (norm + 1e-10f);
        for (std::size_t i = 0; i < d; ++i) unit[i] = xb[i] * inv;

        // 2. y = x_unit @ Pi.T  (handled row-by-row via Rotation::forward).
        const Error fe = rotation_.forward(std::span<const float>(unit.data(), d),
                                           std::span<float>(rotated.data(), d),
                                           /*batch=*/1);
        if (fe != Error::Ok) return fe;

        // 3. Per-coordinate searchsorted (right=False).
        for (std::size_t i = 0; i < d; ++i) {
            raw[i] = searchsorted_right_false(bounds, n_bounds, rotated[i]);
        }

        // 4. Bit-pack.
        Pack::pack(raw.data(), d, indices_out.data() + b * pb);
    }

    return Error::Ok;
}

// -----------------------------------------------------------------------------
// dequantize
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Error TurboQuantMSE<Bits, Arch>::dequantize(std::span<const std::uint8_t> indices,
                                            std::span<const float>        norms,
                                            std::size_t                   batch,
                                            std::span<float>              x_out) const noexcept
{
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d  = dim_;
    const std::size_t pb = Pack::packed_bytes(d);

    if (indices.size() != batch * pb) return Error::ShapeMismatch;
    if (norms.size()   != batch)       return Error::ShapeMismatch;
    if (x_out.size()   != batch * d)  return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

    AlignedBuffer<std::uint8_t> raw;
    AlignedBuffer<float>        y_hat;
    if (!raw.resize(d))    return Error::RotationFailed;
    if (!y_hat.resize(d))  return Error::RotationFailed;

    const float* centroids = codebook_.centroids.data();

    for (std::size_t b = 0; b < batch; ++b) {
        Pack::unpack(indices.data() + b * pb, d, raw.data());
        for (std::size_t i = 0; i < d; ++i) {
            y_hat[i] = centroids[raw[i]];
        }

        float* xb = x_out.data() + b * d;
        const Error be = rotation_.backward(std::span<const float>(y_hat.data(), d),
                                            std::span<float>(xb, d),
                                            /*batch=*/1);
        if (be != Error::Ok) return be;

        const float s = norms[b];
        for (std::size_t i = 0; i < d; ++i) xb[i] *= s;
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

} // namespace tq
