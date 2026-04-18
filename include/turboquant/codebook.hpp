#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "turboquant/aligned_buffer.hpp"
#include "turboquant/config.hpp"
#include "turboquant/error.hpp"

namespace tq {

// Non-owning view of a codebook. Lifetime is tied to the registry that
// issued it — the registry keeps the owning storage alive for the process.
//
// `decision_boundaries` is the *interior* slice matching Python's
// `boundaries[1:-1]` (quantizer.py:127) — i.e. size 2^bits - 1, not 2^bits+1.
// This is what searchsorted operates on.
struct CodebookView {
    std::span<const float> centroids;            // size = 2^bits
    std::span<const float> decision_boundaries;  // size = 2^bits - 1
    std::uint32_t dim       = 0;
    std::uint32_t bits      = 0;
    float         mse_per_coord = 0.0f;
    float         mse_total     = 0.0f;
};

// Options for Lloyd-Max codebook computation (used by compute()).
struct LloydMaxOpts {
    int    max_iter     = 200;
    double tol          = 1e-12;
    int    n_quadrature = 10000;   // fixed-grid trapezoidal points
};

// Registry / cache of loaded codebooks. Thread-safe singleton.
//
// Lookup order on get(d, bits):
//   1. In-memory cache (hit → return view).
//   2. Embedded table (generated from bundled JSONs at build time).
//   3. $TURBOQUANT_CODEBOOK_DIR/codebook_d{D}_b{B}.json on disk (if set).
//   4. Compute Lloyd-Max in C++ and cache.
class TQ_API CodebookRegistry {
 public:
    static CodebookRegistry& instance() noexcept;

    // Get-or-compute. Returns a view valid for the lifetime of the process.
    [[nodiscard]] Result<CodebookView>
    get(std::uint32_t dim, std::uint32_t bits) noexcept;

    // Explicit compute — used by tests and for pre-warming. Writes the
    // result into the cache keyed by (dim, bits). If the key is already
    // cached, returns the cached view.
    [[nodiscard]] Result<CodebookView>
    compute(std::uint32_t dim, std::uint32_t bits,
            const LloydMaxOpts& opts = {}) noexcept;

    // Load from a JSON file on disk. Caches the result.
    [[nodiscard]] Result<CodebookView>
    load_json(std::string_view path) noexcept;

    // Lookup an embedded codebook (compiled-in from bundled JSONs).
    // Returns CodebookMissing if not embedded. Does NOT mutate the cache.
    [[nodiscard]] Result<CodebookView>
    find_embedded(std::uint32_t dim, std::uint32_t bits) noexcept;

 private:
    CodebookRegistry()  = default;
    ~CodebookRegistry() = default;
    CodebookRegistry(const CodebookRegistry&)            = delete;
    CodebookRegistry& operator=(const CodebookRegistry&) = delete;

    struct Impl;
    static Impl& impl() noexcept;
};

// -----------------------------------------------------------------------------
// Embedded codebook table (generated)
//
// `codebooks_to_cpp.py` emits a translation unit defining
// `tq::embedded_codebooks()` returning a span over CodebookBlob records.
// We expose the blob shape here so the loader can consume it without an
// extra header.
// -----------------------------------------------------------------------------
struct CodebookBlob {
    std::uint32_t dim;
    std::uint32_t bits;
    float         mse_per_coord;
    float         mse_total;
    const float*  centroids;            // size = 2^bits
    const float*  decision_boundaries;  // size = 2^bits - 1  (interior)
};

// Provided by codebook_embedded.cpp (generated at configure time).
std::span<const CodebookBlob> embedded_codebooks() noexcept;

} // namespace tq
