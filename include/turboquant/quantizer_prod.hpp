#pragma once

// TurboQuant inner-product quantizer (Algorithm 2).
//
// Python reference: turboquant/quantizer.py:172-306.
//
// Stage 1: MSE quantize at (Bits - 1) bits → residual r = x - x_hat.
// Stage 2: QJL project residual against S ∈ R^{d×d}, keep sign per coord.
// Dequantize:
//     x_hat_mse + sqrt(π/2)/d * ||r|| * (signs @ S)
// Attention score (asymmetric estimator):
//     <query, key> ≈ <query, x_mse> + sqrt(π/2)/d * ||r|| * <query @ S^T, signs>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "turboquant/aligned_buffer.hpp"
#include "turboquant/arch.hpp"
#include "turboquant/config.hpp"
#include "turboquant/error.hpp"
#include "turboquant/pack_policy.hpp"
#include "turboquant/qjl_signs.hpp"
#include "turboquant/quantizer_mse.hpp"

namespace tq {

template <int Bits, ArchTag Arch = arch::Scalar>
class TurboQuantProd {
    static_assert(Bits >= 2 && Bits <= 4,
                  "Prod quantizer needs >=2 bits (1 MSE + 1 QJL)");

 public:
    static constexpr int mse_bits = Bits - 1;
    using MSE  = TurboQuantMSE<mse_bits, Arch>;
    using Pack = typename MSE::Pack;

    TurboQuantProd() = default;
    TurboQuantProd(const TurboQuantProd&)            = delete;
    TurboQuantProd& operator=(const TurboQuantProd&) = delete;
    TurboQuantProd(TurboQuantProd&&) noexcept        = default;
    TurboQuantProd& operator=(TurboQuantProd&&) noexcept = default;

    [[nodiscard]] TQ_API static Result<TurboQuantProd>
    make(std::size_t dim, std::uint32_t seed) noexcept;

    // Stage-1 rotation supplied externally; Stage-2 QJL matrix S generated
    // from (seed + 1000). Used by parity fixtures.
    [[nodiscard]] TQ_API static Result<TurboQuantProd>
    from_matrices(std::span<const float> pi_row_major,
                  std::span<const float> s_row_major,
                  std::size_t dim) noexcept;

    // Layout of outputs:
    //   mse_indices_out   : batch * MSE::packed_bytes(dim)
    //   qjl_signs_out     : batch * QJLPack::packed_bytes(dim)
    //   residual_norms_out: batch
    //   norms_out         : batch
    [[nodiscard]] TQ_API Error
    quantize(std::span<const float> x, std::size_t batch,
             std::span<std::uint8_t> mse_indices_out,
             std::span<std::uint8_t> qjl_signs_out,
             std::span<float>        residual_norms_out,
             std::span<float>        norms_out) const noexcept;

    [[nodiscard]] TQ_API Error
    dequantize(std::span<const std::uint8_t> mse_indices,
               std::span<const std::uint8_t> qjl_signs,
               std::span<const float>        residual_norms,
               std::span<const float>        norms,
               std::size_t                   batch,
               std::span<float>              x_out) const noexcept;

    // Attention score: scores[q, k] = <query[q], key[k]> (asymmetric).
    //   query     : n_q * dim (row-major)
    //   keys      : ProdQuantized view (batch = n_k)
    //   scores_out: n_q * n_k (row-major)
    [[nodiscard]] TQ_API Error
    attention_score(std::span<const float>        query,
                    std::size_t                   n_q,
                    std::span<const std::uint8_t> key_mse_indices,
                    std::span<const std::uint8_t> key_qjl_signs,
                    std::span<const float>        key_residual_norms,
                    std::span<const float>        key_norms,
                    std::size_t                   n_k,
                    std::span<float>              scores_out) const noexcept;

    static constexpr std::size_t mse_packed_bytes(std::size_t d) noexcept {
        return Pack::packed_bytes(d);
    }
    static constexpr std::size_t qjl_packed_bytes(std::size_t d) noexcept {
        return QJLPack::packed_bytes(d);
    }

    std::size_t dim() const noexcept { return dim_; }

    // Accessors for tests.
    std::span<const float> s_matrix() const noexcept {
        return { s_.data(), static_cast<std::size_t>(dim_ * dim_) };
    }
    float qjl_scale() const noexcept { return qjl_scale_; }
    const MSE& mse() const noexcept { return mse_; }

 private:
    TurboQuantProd(std::size_t d, MSE mse, AlignedBuffer<float> s, float scale) noexcept
        : dim_(static_cast<std::uint32_t>(d)),
          mse_(std::move(mse)),
          s_(std::move(s)),
          qjl_scale_(scale) {}

    std::uint32_t        dim_ = 0;
    MSE                  mse_{};
    AlignedBuffer<float> s_{};
    float                qjl_scale_ = 0.0f;
};

extern template class TurboQuantProd<2, arch::Scalar>;
extern template class TurboQuantProd<3, arch::Scalar>;
extern template class TurboQuantProd<4, arch::Scalar>;

} // namespace tq
