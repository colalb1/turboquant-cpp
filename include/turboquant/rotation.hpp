#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "turboquant/aligned_buffer.hpp"
#include "turboquant/config.hpp"
#include "turboquant/error.hpp"

namespace tq {

// Random orthogonal rotation (Algorithm 1 of the TurboQuant paper).
//
// Python reference (authoritative): turboquant/rotation.py:17-40
//   - QR decomposition of a Gaussian d×d matrix, then sign-fix Q so that
//     diag(R) is non-negative (gives det(Q) = +1 in the generic case).
//   - Pi is stored row-major (d*d floats) and is 128-B aligned.
//
// RNG note (risk R1 from the plan): PyTorch's CPU `torch.randn` is not
// bit-compatible with any C++ standard generator. `make(d, seed)` here uses
// a deterministic C++ generator (mt19937_64 + Box-Muller) and is reproducible
// *within a C++ build*. To reproduce a Python-generated Pi exactly, load it
// via `from_matrix()` from a fixture.
class Rotation {
 public:
    Rotation() = default;

    Rotation(const Rotation&)            = delete;
    Rotation& operator=(const Rotation&) = delete;
    Rotation(Rotation&&) noexcept        = default;
    Rotation& operator=(Rotation&&) noexcept = default;

    // Generate Pi ∈ R^{d×d} from a 32-bit seed. Returns Error on failure.
    [[nodiscard]] TQ_API static Result<Rotation>
    make(std::size_t dim, std::uint32_t seed) noexcept;

    // Load Pi from a caller-supplied row-major d*d buffer. Does not
    // orthogonalize — caller must ensure the matrix is already orthogonal.
    // Used by parity tests to consume a Pi dumped from PyTorch.
    [[nodiscard]] TQ_API static Result<Rotation>
    from_matrix(std::span<const float> pi_row_major, std::size_t dim) noexcept;

    // Forward: y = x @ Pi^T (equivalent to y[i] = Σ_j Pi[i,j] * x[j]).
    // x, y are row-major (batch, dim). Writes exactly batch*dim floats to y.
    [[nodiscard]] TQ_API Error
    forward(std::span<const float> x, std::span<float> y, std::size_t batch) const noexcept;

    // Backward: x = y @ Pi (inverse of forward; Pi is orthogonal).
    [[nodiscard]] TQ_API Error
    backward(std::span<const float> y, std::span<float> x, std::size_t batch) const noexcept;

    std::size_t dim() const noexcept { return dim_; }

    // Row-major view of Pi (size dim*dim). 128-B aligned.
    std::span<const float> matrix() const noexcept {
        return { pi_.data(), static_cast<std::size_t>(dim_ * dim_) };
    }

 private:
    Rotation(std::size_t d, AlignedBuffer<float> pi) noexcept
        : dim_(static_cast<std::uint32_t>(d)), pi_(std::move(pi)) {}

    std::uint32_t        dim_ = 0;
    AlignedBuffer<float> pi_;
};

} // namespace tq
