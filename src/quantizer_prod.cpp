// TurboQuantProd scalar implementation.
//
// Compiled with -fno-exceptions. Uses Accelerate's cblas_sgemm for the QJL
// projection and dequantization matmuls; the rest is scalar loops.
//
// Python reference: turboquant/quantizer.py:172-306.

#include "turboquant/quantizer_prod.hpp"
#include "turboquant/neon/kernels.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK   1
#define ACCELERATE_LAPACK_ILP64 0
#include <Accelerate/Accelerate.h>
#endif

namespace tq {

namespace {

// Marsaglia polar method — shared shape with rotation_accelerate.cpp. We
// duplicate it rather than expose a public symbol because both TUs are
// compiled with -fno-exceptions and we want the inliner to see it whole.
inline void next_pair(std::mt19937_64& eng, float& a, float& b) noexcept {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    float                                 u1, u2, s;
    do {
        u1 = u(eng);
        u2 = u(eng);
        s  = u1 * u1 + u2 * u2;
    } while (s >= 1.0f || s == 0.0f);
    const float f = std::sqrt(-2.0f * std::log(s) / s);
    a             = u1 * f;
    b             = u2 * f;
}

void fill_gaussian(float* out, std::size_t n, std::uint32_t seed) noexcept {
    std::mt19937_64 eng(static_cast<std::uint64_t>(seed));
    std::size_t     i = 0;
    while (i + 1 < n) {
        next_pair(eng, out[i], out[i + 1]);
        i += 2;
    }
    if (i < n) {
        float a, b;
        next_pair(eng, a, b);
        out[i] = a;
    }
}

inline float qjl_scale_for_dim(std::size_t d) noexcept {
    // Python: math.sqrt(math.pi / 2.0) / dim  (quantizer.py:212)
    const double s = std::sqrt(M_PI / 2.0) / static_cast<double>(d);
    return static_cast<float>(s);
}

#if !defined(__APPLE__)
// y = A * x  (A is row-major d×d, x is d, y is d). Non-Apple fallback path
// only — Apple always uses cblas_sgemm.
inline void gemv_rowmajor(const float* A, const float* x, float* y, std::size_t d,
                          bool accumulate) noexcept {
    for (std::size_t i = 0; i < d; ++i) {
        const float* row = A + i * d;
        float        acc = 0.0f;
        for (std::size_t j = 0; j < d; ++j)
            acc += row[j] * x[j];
        y[i] = accumulate ? y[i] + acc : acc;
    }
}
#endif  // !__APPLE__

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
    AlignedBuffer<float> x_mse;
    AlignedBuffer<float> residual;
    AlignedBuffer<float> projected;
    if (!x_mse.resize(batch * d)) return Error::RotationFailed;
    if (!residual.resize(batch * d)) return Error::RotationFailed;
    if (!projected.resize(batch * d)) return Error::RotationFailed;

    const Error dq_err = mse_.dequantize(mse_indices_out, norms_out, batch,
                                         std::span<float>(x_mse.data(), batch * d));
    if (dq_err != Error::Ok) return dq_err;

    // residual = x - x_mse
    for (std::size_t i = 0; i < batch * d; ++i) {
        residual[i] = x[i] - x_mse[i];
    }

    // residual_norms = ||residual||_2 per row
    for (std::size_t b = 0; b < batch; ++b) {
        double       acc = 0.0;
        const float* r   = residual.data() + b * d;
        for (std::size_t i = 0; i < d; ++i) {
            const double v  = static_cast<double>(r[i]);
            acc            += v * v;
        }
        residual_norms_out[b] = static_cast<float>(std::sqrt(acc));
    }

    // projected = residual @ S.T  (row-major: projected = residual * S^T)
    //   shape: (batch, d) = (batch, d) * (d, d)^T
#if defined(__APPLE__)
    const float alpha = 1.0f, beta = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(batch),
                static_cast<int>(d), static_cast<int>(d), alpha, residual.data(),
                static_cast<int>(d), s_.data(), static_cast<int>(d), beta, projected.data(),
                static_cast<int>(d));
#else
    for (std::size_t b = 0; b < batch; ++b) {
        // projected[b, i] = Σ_j residual[b, j] * S[i, j]
        gemv_rowmajor(s_.data(), residual.data() + b * d, projected.data() + b * d, d,
                      /*accumulate=*/false);
    }
#endif

    for (std::size_t b = 0; b < batch; ++b) {
        neon::qjl_pack_signs(projected.data() + b * d, d, qjl_signs_out.data() + b * qb);
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
    AlignedBuffer<float> signs;
    AlignedBuffer<float> qjl_contrib;
    if (!signs.resize(batch * d)) return Error::RotationFailed;
    if (!qjl_contrib.resize(batch * d)) return Error::RotationFailed;

    for (std::size_t b = 0; b < batch; ++b) {
        neon::qjl_unpack_pm1(qjl_signs.data() + b * qb, d, signs.data() + b * d);
    }

    // qjl_contrib = signs @ S  (row-major: (batch, d) = (batch, d) * (d, d))
#if defined(__APPLE__)
    const float alpha = 1.0f, beta = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(batch),
                static_cast<int>(d), static_cast<int>(d), alpha, signs.data(), static_cast<int>(d),
                s_.data(), static_cast<int>(d), beta, qjl_contrib.data(), static_cast<int>(d));
#else
    for (std::size_t b = 0; b < batch; ++b) {
        const float* sr = signs.data() + b * d;
        float*       yr = qjl_contrib.data() + b * d;
        for (std::size_t j = 0; j < d; ++j) {
            float acc = 0.0f;
            for (std::size_t i = 0; i < d; ++i)
                acc += sr[i] * s_[i * d + j];
            yr[j] = acc;
        }
    }
#endif

    for (std::size_t b = 0; b < batch; ++b) {
        const float  scale = qjl_scale_ * residual_norms[b];
        float*       xr    = x_out.data() + b * d;
        const float* qr    = qjl_contrib.data() + b * d;
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

#if defined(__APPLE__)
    // scores_mse = query @ k_mse.T   (n_q × n_k)
    const float one = 1.0f, zero = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(n_q),
                static_cast<int>(n_k), static_cast<int>(d), one, query.data(), static_cast<int>(d),
                k_mse.data(), static_cast<int>(d), zero, scores_out.data(), static_cast<int>(n_k));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(n_q), static_cast<int>(d),
                static_cast<int>(d), one, query.data(), static_cast<int>(d), s_.data(),
                static_cast<int>(d), zero, q_sketched.data(), static_cast<int>(d));
#else
    for (std::size_t q = 0; q < n_q; ++q) {
        for (std::size_t k = 0; k < n_k; ++k) {
            float acc = 0.0f;
            for (std::size_t i = 0; i < d; ++i)
                acc += query[q * d + i] * k_mse[k * d + i];
            scores_out[q * n_k + k] = acc;
        }
    }
    for (std::size_t q = 0; q < n_q; ++q) {
        gemv_rowmajor(s_.data(), query.data() + q * d, q_sketched.data() + q * d, d,
                      /*accumulate=*/false);
    }
#endif

    // 4. scores += qjl_scale * residual_norms[k] * (q_sketched @ signs.T)
    //    accumulate into scores_out (beta=1) after scaling the QJL path.
    AlignedBuffer<float> scores_qjl;
    if (!scores_qjl.resize(n_q * n_k)) return Error::RotationFailed;

#if defined(__APPLE__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(n_q),
                static_cast<int>(n_k), static_cast<int>(d), one, q_sketched.data(),
                static_cast<int>(d), signs.data(), static_cast<int>(d), zero, scores_qjl.data(),
                static_cast<int>(n_k));
#else
    for (std::size_t q = 0; q < n_q; ++q) {
        for (std::size_t k = 0; k < n_k; ++k) {
            float acc = 0.0f;
            for (std::size_t i = 0; i < d; ++i)
                acc += q_sketched[q * d + i] * signs[k * d + i];
            scores_qjl[q * n_k + k] = acc;
        }
    }
#endif

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
