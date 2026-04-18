// Hybrid attention: compressed history + recent ring, GQA-aware matmul
// over FP32 dequantized K/V. Uses Accelerate's cblas_sgemm for the two
// per-(h_kv, g) gemms; softmax is scalar and row-wise.
//
// Python reference: turboquant/score.py:29-173.

#include "turboquant/score.hpp"

#include "turboquant/aligned_buffer.hpp"
#include "turboquant/value_quant.hpp"

#include <cmath>
#include <cstring>

#if defined(__APPLE__)
#  define ACCELERATE_NEW_LAPACK 1
#  define ACCELERATE_LAPACK_ILP64 0
#  include <Accelerate/Accelerate.h>
#endif

namespace tq {

namespace {

#if !defined(__APPLE__)
// Scalar fallback: C = alpha * A * B^op + beta * C  (row-major).
// Only used on non-Apple hosts for the tiny paths here.
inline void sgemm_fallback(bool trans_b, int M, int N, int K, float alpha,
                           const float* A, int lda, const float* B, int ldb,
                           float beta, float* C, int ldc) noexcept
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float a = A[i * lda + k];
                const float b = trans_b ? B[j * ldb + k] : B[k * ldb + j];
                acc += a * b;
            }
            float& c = C[i * ldc + j];
            c = alpha * acc + beta * c;
        }
    }
}
#endif

inline void gemm_row(bool trans_b, int M, int N, int K, float alpha,
                     const float* A, int lda, const float* B, int ldb,
                     float beta, float* C, int ldc) noexcept
{
#if defined(__APPLE__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans,
                trans_b ? CblasTrans : CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    sgemm_fallback(trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline void softmax_rowwise(float* s, std::size_t rows, std::size_t cols) noexcept
{
    for (std::size_t t = 0; t < rows; ++t) {
        float* row = s + t * cols;
        float mx = row[0];
        for (std::size_t i = 1; i < cols; ++i) {
            if (row[i] > mx) mx = row[i];
        }
        float sum = 0.0f;
        for (std::size_t i = 0; i < cols; ++i) {
            row[i] = std::exp(row[i] - mx);
            sum   += row[i];
        }
        const float inv = 1.0f / sum;
        for (std::size_t i = 0; i < cols; ++i) row[i] *= inv;
    }
}

} // namespace

template <int KeyBits, int ValBits>
Error compute_hybrid_attention(
    std::span<const float>               query,
    std::size_t                          n_q,
    std::size_t                          Q,
    CompressedKVStore<KeyBits, ValBits>& store,
    std::span<const float>               recent_k,
    std::span<const float>               recent_v,
    std::size_t                          n_recent,
    float                                scale,
    std::span<float>                     out) noexcept
{
    using Val = ValueCodec<ValBits>;

    const std::size_t D    = store.head_dim();
    const std::size_t H    = store.num_kv_heads();
    if (D == 0 || H == 0 || Q == 0) return Error::InvalidDim;
    if (Q % H != 0)                 return Error::InvalidArgument;
    const std::size_t G  = Q / H;
    const std::size_t gs = store.value_group_size();

    if (query.size() != n_q * Q * D) return Error::ShapeMismatch;
    if (out.size()   != n_q * Q * D) return Error::ShapeMismatch;
    if (n_recent > 0) {
        if (recent_k.size() != n_recent * H * D) return Error::ShapeMismatch;
        if (recent_v.size() != n_recent * H * D) return Error::ShapeMismatch;
    }

    if (scale <= 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto flat_res = store.get_flat();
    if (!flat_res) return flat_res.error();
    auto flat = *flat_res;

    const std::size_t N_hist    = flat.total_tokens;
    const bool        has_hist  = N_hist >= MIN_HISTORY_FOR_TQ;
    const bool        has_rec   = n_recent > 0;

    if (n_q == 0) return Error::Ok;

    if (!has_hist && !has_rec) {
        std::memset(out.data(), 0, out.size() * sizeof(float));
        return Error::Ok;
    }

    const std::size_t N_all = (has_hist ? N_hist : 0) + (has_rec ? n_recent : 0);

    // K_all, V_all laid out as (H, N_all, D).
    AlignedBuffer<float> k_all, v_all;
    if (!k_all.resize(H * N_all * D)) return Error::RotationFailed;
    if (!v_all.resize(H * N_all * D)) return Error::RotationFailed;

    if (has_hist) {
        // Prod::dequantize writes (H * N_hist, D) row-major = (H, N_hist, D).
        AlignedBuffer<float> khist, vhist;
        if (!khist.resize(H * N_hist * D)) return Error::RotationFailed;
        if (!vhist.resize(H * N_hist * D)) return Error::RotationFailed;

        const Error ek = store.quantizer().dequantize(
            flat.mse_indices, flat.qjl_signs, flat.residual_norms, flat.norms,
            H * N_hist,
            std::span<float>(khist.data(), H * N_hist * D));
        if (ek != Error::Ok) return ek;

        const Error ev = Val::dequantize(
            flat.val_data, flat.val_scales, flat.val_zeros,
            H * N_hist, D, gs,
            std::span<float>(vhist.data(), H * N_hist * D));
        if (ev != Error::Ok) return ev;

        // Copy per-head segment: (h, :N_hist, :) → (h, :N_hist, :) inside N_all stride.
        for (std::size_t h = 0; h < H; ++h) {
            std::memcpy(k_all.data() + h * N_all * D,
                        khist.data() + h * N_hist * D,
                        N_hist * D * sizeof(float));
            std::memcpy(v_all.data() + h * N_all * D,
                        vhist.data() + h * N_hist * D,
                        N_hist * D * sizeof(float));
        }
    }

    if (has_rec) {
        // (n_recent, H, D) → per-head append at offset (has_hist ? N_hist : 0).
        const std::size_t rec_off = has_hist ? N_hist : 0;
        for (std::size_t t = 0; t < n_recent; ++t) {
            const float* kt_src = recent_k.data() + t * H * D;
            const float* vt_src = recent_v.data() + t * H * D;
            for (std::size_t h = 0; h < H; ++h) {
                std::memcpy(k_all.data() + h * N_all * D + (rec_off + t) * D,
                            kt_src + h * D, D * sizeof(float));
                std::memcpy(v_all.data() + h * N_all * D + (rec_off + t) * D,
                            vt_src + h * D, D * sizeof(float));
            }
        }
    }

    // Transpose query (n_q, Q, D) → (H, G, n_q, D) so each (h, g) sub-block is
    // contiguous (n_q, D) for the per-pair gemm.
    AlignedBuffer<float> q_t, o_t;
    if (!q_t.resize(H * G * n_q * D)) return Error::RotationFailed;
    if (!o_t.resize(H * G * n_q * D)) return Error::RotationFailed;
    for (std::size_t t = 0; t < n_q; ++t) {
        for (std::size_t h = 0; h < H; ++h) {
            for (std::size_t g = 0; g < G; ++g) {
                std::memcpy(q_t.data() + ((h * G + g) * n_q + t) * D,
                            query.data() + t * Q * D + (h * G + g) * D,
                            D * sizeof(float));
            }
        }
    }

    AlignedBuffer<float> scores;
    if (!scores.resize(n_q * N_all)) return Error::RotationFailed;

    for (std::size_t h = 0; h < H; ++h) {
        const float* k_ptr = k_all.data() + h * N_all * D;
        const float* v_ptr = v_all.data() + h * N_all * D;
        for (std::size_t g = 0; g < G; ++g) {
            const float* q_ptr = q_t.data() + (h * G + g) * n_q * D;
            float*       o_ptr = o_t.data() + (h * G + g) * n_q * D;

            // scores (n_q, N_all) = scale * Q (n_q, D) * K^T (D, N_all)
            gemm_row(/*trans_b=*/true,
                     static_cast<int>(n_q), static_cast<int>(N_all),
                     static_cast<int>(D),
                     scale,
                     q_ptr, static_cast<int>(D),
                     k_ptr, static_cast<int>(D),
                     0.0f,
                     scores.data(), static_cast<int>(N_all));

            softmax_rowwise(scores.data(), n_q, N_all);

            // out (n_q, D) = weights (n_q, N_all) * V (N_all, D)
            gemm_row(/*trans_b=*/false,
                     static_cast<int>(n_q), static_cast<int>(D),
                     static_cast<int>(N_all),
                     1.0f,
                     scores.data(), static_cast<int>(N_all),
                     v_ptr, static_cast<int>(D),
                     0.0f,
                     o_ptr, static_cast<int>(D));
        }
    }

    // Un-transpose (H, G, n_q, D) → (n_q, Q, D).
    for (std::size_t t = 0; t < n_q; ++t) {
        for (std::size_t h = 0; h < H; ++h) {
            for (std::size_t g = 0; g < G; ++g) {
                std::memcpy(out.data() + t * Q * D + (h * G + g) * D,
                            o_t.data() + ((h * G + g) * n_q + t) * D,
                            D * sizeof(float));
            }
        }
    }

    return Error::Ok;
}

#define TQ_INSTANTIATE_SCORE(KB, VB)                                   \
    template Error compute_hybrid_attention<KB, VB>(                   \
        std::span<const float>, std::size_t, std::size_t,              \
        CompressedKVStore<KB, VB>&,                                    \
        std::span<const float>, std::span<const float>,                \
        std::size_t, float, std::span<float>) noexcept;

TQ_INSTANTIATE_SCORE(2, 2)
TQ_INSTANTIATE_SCORE(3, 2)
TQ_INSTANTIATE_SCORE(4, 2)
TQ_INSTANTIATE_SCORE(2, 4)
TQ_INSTANTIATE_SCORE(3, 4)
TQ_INSTANTIATE_SCORE(4, 4)
TQ_INSTANTIATE_SCORE(2, 8)
TQ_INSTANTIATE_SCORE(3, 8)
TQ_INSTANTIATE_SCORE(4, 8)

#undef TQ_INSTANTIATE_SCORE

} // namespace tq
