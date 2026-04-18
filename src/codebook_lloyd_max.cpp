// Lloyd-Max codebook computation. Replicates the logic of Python reference
// turboquant/codebook.py:68-143 with two deliberate deviations:
//
//   1. Integration uses a fixed-grid trapezoidal rule instead of scipy's
//      adaptive Gauss-Kronrod. Deterministic across platforms; differences
//      from scipy are within 0.5% on the Beta PDF we integrate (smooth away
//      from ±1). For the bundled (d,b) pairs we treat the JSONs as ground
//      truth and never invoke this path at runtime.
//
//   2. PDF evaluation goes through `lgamma` for the log-normalizer (matches
//      scipy.special.gammaln); `log(1 - x*x)` is computed with `log1p(-x*x)`
//      for numerical stability near |x| = 1.
//
// Entry point is `compute_lloyd_max_internal` — consumed by the codebook
// registry. Compiled with -fno-exceptions.

#include "turboquant/codebook.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace tq {

namespace {

// log of the Beta-on-[-1,1] PDF:
//   f(x; d) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
inline double log_beta_pdf(double x, int d) noexcept
{
    // Clip to mirror Python's np.clip(x, -1+1e-15, 1-1e-15).
    const double eps = 1e-15;
    if (x >  1.0 - eps) x =  1.0 - eps;
    if (x < -1.0 + eps) x = -1.0 + eps;

    const double log_const = std::lgamma(d / 2.0)
                           - 0.5 * std::log(M_PI)
                           - std::lgamma((d - 1) / 2.0);
    const double exponent  = (d - 3) / 2.0;
    // log1p(-x*x) is more stable than log(1 - x*x) near |x| = 1.
    const double log_val = log_const + exponent * std::log1p(-x * x);
    return log_val;
}

inline double beta_pdf(double x, int d) noexcept
{
    return std::exp(log_beta_pdf(x, d));
}

// Fixed-grid trapezoidal integration on [lo, hi] with n subintervals.
template <class F>
inline double trapezoid(F&& f, double lo, double hi, int n) noexcept
{
    if (hi <= lo || n <= 0) return 0.0;
    const double h = (hi - lo) / n;
    double acc = 0.5 * (f(lo) + f(hi));
    for (int i = 1; i < n; ++i) {
        acc += f(lo + i * h);
    }
    return acc * h;
}

// Number of quadrature points per cell. Global opts.n_quadrature is the
// count over [-1, 1]; per-cell we use max(64, prop * n_quadrature / 2).
inline int cell_points(int total, double lo, double hi) noexcept
{
    const double frac = std::max(0.0, (hi - lo) / 2.0);
    const int n = static_cast<int>(frac * total);
    return std::max(64, n);
}

double conditional_mean(double lo, double hi, int d, int n_total) noexcept
{
    if (hi <= lo) return 0.5 * (lo + hi);
    const int n = cell_points(n_total, lo, hi);
    const double num = trapezoid([d](double x){ return x * beta_pdf(x, d); }, lo, hi, n);
    const double den = trapezoid([d](double x){ return beta_pdf(x, d); },     lo, hi, n);
    if (den < 1e-30) return 0.5 * (lo + hi);
    return num / den;
}

double mse_cost(const std::vector<double>& centroids,
                const std::vector<double>& boundaries,
                int d, int n_total) noexcept
{
    double cost = 0.0;
    const std::size_t n = centroids.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double lo = boundaries[i];
        const double hi = boundaries[i + 1];
        const double c  = centroids[i];
        const int np   = cell_points(n_total, lo, hi);
        cost += trapezoid(
            [d, c](double x) {
                const double dx = x - c;
                return dx * dx * beta_pdf(x, d);
            },
            lo, hi, np);
    }
    return cost;
}

} // namespace

Result<CodebookView>
compute_lloyd_max_internal(std::uint32_t dim, std::uint32_t bits,
                           const LloydMaxOpts& opts,
                           AlignedBuffer<float>& centroids_out,
                           AlignedBuffer<float>& interior_out,
                           float& mse_per_coord_out) noexcept
{
    if (dim < 3)              return make_error<CodebookView>(Error::InvalidDim);
    if (bits < 1 || bits > 4) return make_error<CodebookView>(Error::InvalidBits);

    const int d = static_cast<int>(dim);
    const std::size_t n_clusters = std::size_t{1} << bits;

    // 1. Initial centroids via quantile midpoints of the Beta CDF.
    const int grid_n = opts.n_quadrature;
    std::vector<double> x_grid(grid_n);
    std::vector<double> pdf_vals(grid_n);
    const double x_lo = -1.0 + 1e-10;
    const double x_hi =  1.0 - 1e-10;
    const double dx = (x_hi - x_lo) / (grid_n - 1);
    for (int i = 0; i < grid_n; ++i) {
        x_grid[i]   = x_lo + i * dx;
        pdf_vals[i] = beta_pdf(x_grid[i], d);
    }
    std::vector<double> cdf(grid_n);
    double acc = 0.0;
    for (int i = 0; i < grid_n; ++i) {
        acc += pdf_vals[i] * dx;
        cdf[i] = acc;
    }
    const double cdf_max = cdf.back() > 0.0 ? cdf.back() : 1.0;
    for (auto& v : cdf) v /= cdf_max;

    std::vector<double> centroids(n_clusters);
    for (std::size_t i = 0; i < n_clusters; ++i) {
        const double q_lo = static_cast<double>(i)     / n_clusters;
        const double q_hi = static_cast<double>(i + 1) / n_clusters;
        const double q_mid = 0.5 * (q_lo + q_hi);
        // searchsorted for q_mid in cdf.
        const auto it = std::lower_bound(cdf.begin(), cdf.end(), q_mid);
        std::size_t idx = static_cast<std::size_t>(it - cdf.begin());
        if (idx >= static_cast<std::size_t>(grid_n)) idx = grid_n - 1;
        centroids[i] = x_grid[idx];
    }

    // 2. Lloyd-Max iterations.
    std::vector<double> boundaries(n_clusters + 1);
    boundaries.front() = -1.0;
    boundaries.back()  =  1.0;

    double prev_cost = std::numeric_limits<double>::infinity();
    double cost = 0.0;
    for (int iter = 0; iter < opts.max_iter; ++iter) {
        for (std::size_t i = 0; i + 1 < n_clusters; ++i) {
            boundaries[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
        }
        std::vector<double> new_centroids(n_clusters);
        for (std::size_t i = 0; i < n_clusters; ++i) {
            new_centroids[i] = conditional_mean(boundaries[i], boundaries[i + 1],
                                                d, opts.n_quadrature);
        }
        cost = mse_cost(new_centroids, boundaries, d, opts.n_quadrature);
        centroids = std::move(new_centroids);
        if (std::fabs(prev_cost - cost) < opts.tol) break;
        prev_cost = cost;
    }

    // Final interior boundaries (same formula as in-loop, refreshed after
    // the last centroid update).
    for (std::size_t i = 0; i + 1 < n_clusters; ++i) {
        boundaries[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
    }

    // 3. Emit to caller buffers.
    if (!centroids_out.resize(n_clusters) ||
        !interior_out.resize(n_clusters - 1)) {
        return make_error<CodebookView>(Error::CodebookCorrupt);
    }
    for (std::size_t i = 0; i < n_clusters; ++i)
        centroids_out[i] = static_cast<float>(centroids[i]);
    for (std::size_t i = 0; i + 1 < n_clusters; ++i)
        interior_out[i] = static_cast<float>(boundaries[i + 1]);

    mse_per_coord_out = static_cast<float>(cost);

    // Caller (codebook_loader.cpp) builds the CodebookView from the
    // AlignedBuffers; returning any value with Error::Ok is fine here.
    CodebookView v;
    v.centroids = { centroids_out.data(), centroids_out.size() };
    v.decision_boundaries = { interior_out.data(), interior_out.size() };
    v.dim = dim;
    v.bits = bits;
    v.mse_per_coord = mse_per_coord_out;
    v.mse_total = mse_per_coord_out * static_cast<float>(dim);
    return Result<CodebookView>(v);
}

} // namespace tq
