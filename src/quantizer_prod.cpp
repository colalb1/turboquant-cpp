// TurboQuantProd scalar implementation.
//
// Compiled with -fno-exceptions. Uses Accelerate's cblas_sgemm for the QJL
// projection and dequantization matmuls; the rest is scalar loops.
//
// Python reference: turboquant/quantizer.py:172-306.

#include "turboquant/quantizer_prod.hpp"
#include "turboquant/neon/kernels.hpp"

#include "internal/gaussian_rng.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>

#define ACCELERATE_NEW_LAPACK   1
#define ACCELERATE_LAPACK_ILP64 0
#include <Accelerate/Accelerate.h>

namespace tq {

namespace {

using internal::fill_gaussian;

inline float qjl_scale_for_dim(std::size_t d) noexcept {
    // Python: math.sqrt(math.pi / 2.0) / dim  (quantizer.py:212)
    const double s = std::sqrt(M_PI / 2.0) / static_cast<double>(d);
    return static_cast<float>(s);
}

}  // namespace

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Result<TurboQuantProd<Bits, Arch>> TurboQuantProd<Bits, Arch>::make(std::size_t   dim,
                                                                    std::uint32_t seed) noexcept {
    if (dim == 0 || dim > kMaxDim) return make_error<TurboQuantProd>(Error::InvalidDim);

    auto mse = MSE::make(dim, seed);
    if (!mse) return make_error<TurboQuantProd>(mse.error());

    AlignedBuffer<float> s;
    if (!s.resize(dim * dim)) return make_error<TurboQuantProd>(Error::RotationFailed);

    // Python uses seed + 1000 for S (quantizer.py:208).
    fill_gaussian(s.data(), dim * dim, seed + 1000u);

    const float scale = qjl_scale_for_dim(dim);
    return Result<TurboQuantProd>(TurboQuantProd(dim, std::move(*mse), std::move(s), scale));
}

template <int Bits, ArchTag Arch>
Result<TurboQuantProd<Bits, Arch>>
TurboQuantProd<Bits, Arch>::from_matrices(std::span<const float> pi_row_major,
                                          std::span<const float> s_row_major,
                                          std::size_t            dim) noexcept {
    if (dim == 0 || dim > kMaxDim) return make_error<TurboQuantProd>(Error::InvalidDim);
    if (pi_row_major.size() != dim * dim || s_row_major.size() != dim * dim)
        return make_error<TurboQuantProd>(Error::ShapeMismatch);

    auto mse = MSE::from_matrix(pi_row_major, dim);
    if (!mse) return make_error<TurboQuantProd>(mse.error());

    AlignedBuffer<float> s;
    if (!s.resize(dim * dim)) return make_error<TurboQuantProd>(Error::RotationFailed);
    std::memcpy(s.data(), s_row_major.data(), dim * dim * sizeof(float));

    const float scale = qjl_scale_for_dim(dim);
    return Result<TurboQuantProd>(TurboQuantProd(dim, std::move(*mse), std::move(s), scale));
}

// -----------------------------------------------------------------------------
// quantize
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Error TurboQuantProd<Bits, Arch>::quantize(std::span<const float> x, std::size_t batch,
                                           std::span<std::uint8_t> mse_indices_out,
                                           std::span<std::uint8_t> qjl_signs_out,
                                           std::span<float>        residual_norms_out,
                                           std::span<float>        norms_out) const noexcept {
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d  = dim_;
    const std::size_t mb = Pack::packed_bytes(d);
    const std::size_t qb = QJLPack::packed_bytes(d);

    if (x.size() != batch * d) return Error::ShapeMismatch;
    if (mse_indices_out.size() != batch * mb) return Error::ShapeMismatch;
    if (qjl_signs_out.size() != batch * qb) return Error::ShapeMismatch;
    if (residual_norms_out.size() != batch) return Error::ShapeMismatch;
    if (norms_out.size() != batch) return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

    // Stage 1: MSE quantize (writes indices + norms directly).
    const Error mse_err = mse_.quantize(x, batch, mse_indices_out, norms_out);
    if (mse_err != Error::Ok) return mse_err;

    // Stage 2: reconstruct MSE, compute residual, project, pack signs.
    const std::size_t n = batch * d;
    if (!scratch_a_.ensure_size(n)) return Error::RotationFailed;  // x_mse
    if (!scratch_b_.ensure_size(n)) return Error::RotationFailed;  // residual
    if (!scratch_c_.ensure_size(n)) return Error::RotationFailed;  // projected

    float* x_mse     = scratch_a_.data();
    float* residual  = scratch_b_.data();
    float* projected = scratch_c_.data();

    const Error dq_err =
        mse_.dequantize(mse_indices_out, norms_out, batch, std::span<float>(x_mse, n));
    if (dq_err != Error::Ok) return dq_err;

    // residual = x - x_mse
    for (std::size_t i = 0; i < n; ++i) {
        residual[i] = x[i] - x_mse[i];
    }

    // residual_norms = ||residual||_2 per row
    for (std::size_t b = 0; b < batch; ++b) {
        double       acc = 0.0;
        const float* r   = residual + b * d;
        for (std::size_t i = 0; i < d; ++i) {
            const double v  = static_cast<double>(r[i]);
            acc            += v * v;
        }
        residual_norms_out[b] = static_cast<float>(std::sqrt(acc));
    }

    // projected = residual @ S.T  (row-major: projected = residual * S^T)
    //   shape: (batch, d) = (batch, d) * (d, d)^T
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(batch),
                    static_cast<int>(d), static_cast<int>(d), alpha, residual,
                    static_cast<int>(d), s_.data(), static_cast<int>(d), beta, projected,
                    static_cast<int>(d));
    }

    for (std::size_t b = 0; b < batch; ++b) {
        neon::qjl_pack_signs(projected + b * d, d, qjl_signs_out.data() + b * qb);
    }

    return Error::Ok;
}

// -----------------------------------------------------------------------------
// dequantize
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Error TurboQuantProd<Bits, Arch>::dequantize(std::span<const std::uint8_t> mse_indices,
                                             std::span<const std::uint8_t> qjl_signs,
                                             std::span<const float>        residual_norms,
                                             std::span<const float> norms, std::size_t batch,
                                             std::span<float> x_out) const noexcept {
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d  = dim_;
    const std::size_t mb = Pack::packed_bytes(d);
    const std::size_t qb = QJLPack::packed_bytes(d);

    if (mse_indices.size() != batch * mb) return Error::ShapeMismatch;
    if (qjl_signs.size() != batch * qb) return Error::ShapeMismatch;
    if (residual_norms.size() != batch) return Error::ShapeMismatch;
    if (norms.size() != batch) return Error::ShapeMismatch;
    if (x_out.size() != batch * d) return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

    // Stage 1: MSE dequantize into x_out.
    const Error dq = mse_.dequantize(mse_indices, norms, batch, x_out);
    if (dq != Error::Ok) return dq;

    // Stage 2: x_out += qjl_scale * residual_norms[b] * (signs @ S)
    const std::size_t n = batch * d;
    if (!scratch_a_.ensure_size(n)) return Error::RotationFailed;  // signs
    if (!scratch_b_.ensure_size(n)) return Error::RotationFailed;  // qjl_contrib

    float* signs       = scratch_a_.data();
    float* qjl_contrib = scratch_b_.data();

    for (std::size_t b = 0; b < batch; ++b) {
        neon::qjl_unpack_pm1(qjl_signs.data() + b * qb, d, signs + b * d);
    }

    // qjl_contrib = signs @ S  (row-major: (batch, d) = (batch, d) * (d, d))
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(batch),
                    static_cast<int>(d), static_cast<int>(d), alpha, signs, static_cast<int>(d),
                    s_.data(), static_cast<int>(d), beta, qjl_contrib, static_cast<int>(d));
    }

    for (std::size_t b = 0; b < batch; ++b) {
        const float  scale = qjl_scale_ * residual_norms[b];
        float*       xr    = x_out.data() + b * d;
        const float* qr    = qjl_contrib + b * d;
        for (std::size_t i = 0; i < d; ++i)
            xr[i] += scale * qr[i];
    }

    return Error::Ok;
}

// -----------------------------------------------------------------------------
// attention_score
// -----------------------------------------------------------------------------

template <int Bits, ArchTag Arch>
Error TurboQuantProd<Bits, Arch>::attention_score(std::span<const float> query, std::size_t n_q,
                                                  std::span<const std::uint8_t> key_mse_indices,
                                                  std::span<const std::uint8_t> key_qjl_signs,
                                                  std::span<const float>        key_residual_norms,
                                                  std::span<const float> key_norms, std::size_t n_k,
                                                  std::span<float> scores_out) const noexcept {
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d  = dim_;
    const std::size_t mb = Pack::packed_bytes(d);
    const std::size_t qb = QJLPack::packed_bytes(d);

    if (query.size() != n_q * d) return Error::ShapeMismatch;
    if (key_mse_indices.size() != n_k * mb) return Error::ShapeMismatch;
    if (key_qjl_signs.size() != n_k * qb) return Error::ShapeMismatch;
    if (key_residual_norms.size() != n_k) return Error::ShapeMismatch;
    if (key_norms.size() != n_k) return Error::ShapeMismatch;
    if (scores_out.size() != n_q * n_k) return Error::ShapeMismatch;
    if (n_q == 0 || n_k == 0) return Error::Ok;

    // 1. Reconstruct keys from MSE (n_k × d).
    AlignedBuffer<float> k_mse;
    if (!k_mse.resize(n_k * d)) return Error::RotationFailed;
    const Error dq =
        mse_.dequantize(key_mse_indices, key_norms, n_k, std::span<float>(k_mse.data(), n_k * d));
    if (dq != Error::Ok) return dq;

    // 2. Unpack QJL signs (n_k × d, values in {-1, +1}).
    AlignedBuffer<float> signs;
    if (!signs.resize(n_k * d)) return Error::RotationFailed;
    for (std::size_t k = 0; k < n_k; ++k) {
        neon::qjl_unpack_pm1(key_qjl_signs.data() + k * qb, d, signs.data() + k * d);
    }

    // 3. Sketch queries: q_sketched = query @ S.T   (n_q × d)
    AlignedBuffer<float> q_sketched;
    if (!q_sketched.resize(n_q * d)) return Error::RotationFailed;

    const float one  = 1.0f;
    const float zero = 0.0f;
    // scores_mse = query @ k_mse.T   (n_q × n_k)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(n_q),
                static_cast<int>(n_k), static_cast<int>(d), one, query.data(), static_cast<int>(d),
                k_mse.data(), static_cast<int>(d), zero, scores_out.data(), static_cast<int>(n_k));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(n_q), static_cast<int>(d),
                static_cast<int>(d), one, query.data(), static_cast<int>(d), s_.data(),
                static_cast<int>(d), zero, q_sketched.data(), static_cast<int>(d));

    // 4. scores += qjl_scale * residual_norms[k] * (q_sketched @ signs.T)
    //    accumulate into scores_out (beta=1) after scaling the QJL path.
    AlignedBuffer<float> scores_qjl;
    if (!scores_qjl.resize(n_q * n_k)) return Error::RotationFailed;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(n_q),
                static_cast<int>(n_k), static_cast<int>(d), one, q_sketched.data(),
                static_cast<int>(d), signs.data(), static_cast<int>(d), zero, scores_qjl.data(),
                static_cast<int>(n_k));

    for (std::size_t q = 0; q < n_q; ++q) {
        for (std::size_t k = 0; k < n_k; ++k) {
            scores_out[q * n_k + k] += qjl_scale_ * key_residual_norms[k] * scores_qjl[q * n_k + k];
        }
    }

    return Error::Ok;
}

// -----------------------------------------------------------------------------
// Explicit instantiations
// -----------------------------------------------------------------------------

template class TurboQuantProd<2, arch::Scalar>;
template class TurboQuantProd<3, arch::Scalar>;
template class TurboQuantProd<4, arch::Scalar>;

}  // namespace tq
