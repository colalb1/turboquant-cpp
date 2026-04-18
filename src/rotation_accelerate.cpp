// Accelerate-backed Rotation implementation.
//
// Maps to Python reference turboquant/rotation.py:17-40:
//   G <- randn(d, d)  [seeded]
//   Q, R <- qr(G)
//   Q <- Q * sign(diag(R))   (column-wise; makes det(Q) = +1)
//
// RNG: PyTorch uses a seeded Mersenne-Twister + its internal normal sampler.
// We use mt19937_64 + Marsaglia polar method; the resulting Pi is valid and
// deterministic, but differs from Python's Pi bit-for-bit. Parity tests
// bypass this via Rotation::from_matrix().
//
// Compiled with -fno-exceptions (see CMakeLists.txt). All failure paths
// return Error codes.

#include "turboquant/rotation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

#if defined(__APPLE__)
#  define ACCELERATE_NEW_LAPACK 1
#  define ACCELERATE_LAPACK_ILP64 0
#  include <Accelerate/Accelerate.h>
#endif

namespace tq {

namespace {

// Marsaglia polar method — two normals per pair. Deterministic given the
// engine state.
inline void next_pair(std::mt19937_64& eng, float& a, float& b) noexcept {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    float u1, u2, s;
    do {
        u1 = u(eng);
        u2 = u(eng);
        s = u1*u1 + u2*u2;
    } while (s >= 1.0f || s == 0.0f);
    const float f = std::sqrt(-2.0f * std::log(s) / s);
    a = u1 * f;
    b = u2 * f;
}

void fill_gaussian(float* out, std::size_t n, std::uint32_t seed) noexcept {
    std::mt19937_64 eng(static_cast<std::uint64_t>(seed));
    std::size_t i = 0;
    while (i + 1 < n) {
        next_pair(eng, out[i], out[i+1]);
        i += 2;
    }
    if (i < n) {
        float a, b;
        next_pair(eng, a, b);
        out[i] = a;
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Rotation::make
// -----------------------------------------------------------------------------

Result<Rotation> Rotation::make(std::size_t dim, std::uint32_t seed) noexcept
{
    if (dim == 0 || dim > kMaxDim) return make_error<Rotation>(Error::InvalidDim);

    AlignedBuffer<float> pi;
    if (!pi.resize(dim * dim)) return make_error<Rotation>(Error::RotationFailed);

    // Fill Pi row-major with Gaussian samples. LAPACK expects column-major,
    // so we fill directly into column-major order: since the Gaussian
    // matrix is i.i.d., row-major vs column-major is statistically identical,
    // but we record the order we chose so the sign-fix stays consistent.
    fill_gaussian(pi.data(), dim * dim, seed);

#if defined(__APPLE__)
    // Save diag(R) sign after sgeqrf, then column-scale Q after sorgqr.
    AlignedBuffer<float> diag_sign;
    if (!diag_sign.resize(dim)) return make_error<Rotation>(Error::RotationFailed);

    AlignedBuffer<float> tau_buf;

    // Step 1: sgeqrf in place (pi treated as column-major d×d).
    __LAPACK_int m = static_cast<__LAPACK_int>(dim);
    __LAPACK_int n = static_cast<__LAPACK_int>(dim);
    __LAPACK_int lda = static_cast<__LAPACK_int>(dim);
    __LAPACK_int info = 0;

    if (!tau_buf.resize(dim)) return make_error<Rotation>(Error::RotationFailed);

    float wkopt = 0.0f;
    __LAPACK_int lwork = -1;
    sgeqrf_(&m, &n, pi.data(), &lda, tau_buf.data(), &wkopt, &lwork, &info);
    if (info != 0) return make_error<Rotation>(Error::LapackFailed);
    lwork = static_cast<__LAPACK_int>(wkopt);

    AlignedBuffer<float> work;
    if (!work.resize(static_cast<std::size_t>(lwork))) return make_error<Rotation>(Error::RotationFailed);

    sgeqrf_(&m, &n, pi.data(), &lda, tau_buf.data(), work.data(), &lwork, &info);
    if (info != 0) return make_error<Rotation>(Error::LapackFailed);

    // Capture sign(diag(R)) — column-major: element (i,i) is at i + i*dim.
    for (std::size_t i = 0; i < dim; ++i) {
        const float r_ii = pi[i + i * dim];
        diag_sign[i] = (r_ii >= 0.0f) ? 1.0f : -1.0f;
    }

    // Step 2: sorgqr — build Q explicitly.
    __LAPACK_int k = static_cast<__LAPACK_int>(dim);
    float wkopt2 = 0.0f;
    __LAPACK_int lwork2 = -1;
    sorgqr_(&m, &n, &k, pi.data(), &lda, tau_buf.data(), &wkopt2, &lwork2, &info);
    if (info != 0) return make_error<Rotation>(Error::LapackFailed);
    lwork2 = static_cast<__LAPACK_int>(wkopt2);
    if (lwork2 > lwork && !work.resize(static_cast<std::size_t>(lwork2))) {
        return make_error<Rotation>(Error::RotationFailed);
    }
    sorgqr_(&m, &n, &k, pi.data(), &lda, tau_buf.data(), work.data(), &lwork2, &info);
    if (info != 0) return make_error<Rotation>(Error::LapackFailed);

    // Step 3: sign-fix — column j of Q scaled by diag_sign[j].
    // Python does `Q * diag_sign.unsqueeze(0)` — row broadcast, i.e. column j
    // gets scaled by diag_sign[j]. Column-major here: column j occupies
    // contiguous indices [j*dim, j*dim + dim).
    for (std::size_t j = 0; j < dim; ++j) {
        const float s = diag_sign[j];
        if (s < 0.0f) {
            float* col = pi.data() + j * dim;
            for (std::size_t i = 0; i < dim; ++i) col[i] = -col[i];
        }
    }

    // Step 4: transpose column-major → row-major (so forward() can use a
    // straight row-major sgemm). For a square matrix this is a single
    // in-place transpose.
    // Use vDSP_mtrans for speed (out-of-place, then copy back).
    AlignedBuffer<float> rm;
    if (!rm.resize(dim * dim)) return make_error<Rotation>(Error::RotationFailed);
    vDSP_mtrans(pi.data(), 1, rm.data(), 1, static_cast<vDSP_Length>(dim), static_cast<vDSP_Length>(dim));
    std::memcpy(pi.data(), rm.data(), dim * dim * sizeof(float));

#else
    // Non-Apple host: leave pi as raw Gaussian (not orthogonal). This
    // branch only runs on CI linters — the production target is Apple
    // Silicon, where we always take the LAPACK path above.
    return make_error<Rotation>(Error::NotImplemented);
#endif

    return Result<Rotation>(Rotation(dim, std::move(pi)));
}

// -----------------------------------------------------------------------------
// Rotation::from_matrix
// -----------------------------------------------------------------------------

Result<Rotation> Rotation::from_matrix(std::span<const float> pi_row_major,
                                       std::size_t dim) noexcept
{
    if (dim == 0 || dim > kMaxDim) return make_error<Rotation>(Error::InvalidDim);
    if (pi_row_major.size() != dim * dim) return make_error<Rotation>(Error::ShapeMismatch);

    AlignedBuffer<float> pi;
    if (!pi.resize(dim * dim)) return make_error<Rotation>(Error::RotationFailed);
    std::memcpy(pi.data(), pi_row_major.data(), dim * dim * sizeof(float));

    return Result<Rotation>(Rotation(dim, std::move(pi)));
}

// -----------------------------------------------------------------------------
// Rotation::forward / backward
// -----------------------------------------------------------------------------

Error Rotation::forward(std::span<const float> x,
                        std::span<float>       y,
                        std::size_t            batch) const noexcept
{
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d = dim_;
    if (x.size() != batch * d || y.size() != batch * d) return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

#if defined(__APPLE__)
    // Python: y = x @ Pi.T   →   y[b, i] = Σ_j x[b, j] * Pi[i, j]
    // With row-major Pi (d×d), that's equivalent to:
    //   Y (batch × d) = X (batch × d) * Pi^T (d × d)  (row-major sgemm)
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(batch), static_cast<int>(d), static_cast<int>(d),
                alpha,
                x.data(), static_cast<int>(d),
                pi_.data(), static_cast<int>(d),
                beta,
                y.data(), static_cast<int>(d));
    return Error::Ok;
#else
    // Portable fallback (slow — used only on non-Apple CI builds).
    const float* pi = pi_.data();
    for (std::size_t b = 0; b < batch; ++b) {
        const float* xb = x.data() + b * d;
        float*       yb = y.data() + b * d;
        for (std::size_t i = 0; i < d; ++i) {
            float acc = 0.0f;
            const float* row = pi + i * d;
            for (std::size_t j = 0; j < d; ++j) acc += row[j] * xb[j];
            yb[i] = acc;
        }
    }
    return Error::Ok;
#endif
}

Error Rotation::backward(std::span<const float> y,
                         std::span<float>       x,
                         std::size_t            batch) const noexcept
{
    if (dim_ == 0) return Error::InvalidDim;
    const std::size_t d = dim_;
    if (y.size() != batch * d || x.size() != batch * d) return Error::ShapeMismatch;
    if (batch == 0) return Error::Ok;

#if defined(__APPLE__)
    // Python: x = y @ Pi   →   x[b, i] = Σ_j y[b, j] * Pi[j, i]
    //   X (batch × d) = Y (batch × d) * Pi (d × d)  (row-major sgemm, NoTrans)
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(batch), static_cast<int>(d), static_cast<int>(d),
                alpha,
                y.data(), static_cast<int>(d),
                pi_.data(), static_cast<int>(d),
                beta,
                x.data(), static_cast<int>(d));
    return Error::Ok;
#else
    const float* pi = pi_.data();
    for (std::size_t b = 0; b < batch; ++b) {
        const float* yb = y.data() + b * d;
        float*       xb = x.data() + b * d;
        for (std::size_t i = 0; i < d; ++i) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < d; ++j) acc += yb[j] * pi[j * d + i];
            xb[i] = acc;
        }
    }
    return Error::Ok;
#endif
}

} // namespace tq
