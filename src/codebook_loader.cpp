// Codebook registry: embedded blob lookup + JSON on-disk loader + in-memory
// cache. Lloyd-Max computation lives in codebook_lloyd_max.cpp — the
// registry calls it for (dim, bits) pairs that aren't in the embedded
// table or cached yet.
//
// Compiled with -fno-exceptions. Where nlohmann/json would normally throw
// on malformed input, we use `json::parse(..., nullptr, false)` which
// returns a value convertible to `false` on failure.

#include "turboquant/codebook.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// Codec TUs are compiled with -fno-exceptions. nlohmann/json is
// exception-oriented by default; JSON_NOEXCEPTION replaces `throw` with
// `std::abort()`. We guard every call with explicit `is_*()` / `contains()`
// checks, so abort paths are only reachable on genuinely corrupt inputs.
#define JSON_NOEXCEPTION
#include <nlohmann/json.hpp>

namespace tq {

// Forward decl — defined in codebook_lloyd_max.cpp.
Result<CodebookView> compute_lloyd_max_internal(std::uint32_t dim, std::uint32_t bits,
                                                const LloydMaxOpts&   opts,
                                                AlignedBuffer<float>& centroids_out,
                                                AlignedBuffer<float>& interior_out,
                                                float&                mse_per_coord_out) noexcept;

struct CodebookRegistry::Impl {
    struct Entry {
        AlignedBuffer<float> centroids;  // owning
        AlignedBuffer<float> interior;   // owning, size 2^b - 1
        std::uint32_t        dim           = 0;
        std::uint32_t        bits          = 0;
        float                mse_per_coord = 0.0f;
        float                mse_total     = 0.0f;
    };

    std::mutex                               mu;
    std::unordered_map<std::uint64_t, Entry> cache;

    static std::uint64_t key(std::uint32_t d, std::uint32_t b) noexcept {
        return (static_cast<std::uint64_t>(d) << 32) | static_cast<std::uint64_t>(b);
    }

    CodebookView view_of(const Entry& e) noexcept {
        CodebookView v;
        v.centroids           = {e.centroids.data(), e.centroids.size()};
        v.decision_boundaries = {e.interior.data(), e.interior.size()};
        v.dim                 = e.dim;
        v.bits                = e.bits;
        v.mse_per_coord       = e.mse_per_coord;
        v.mse_total           = e.mse_total;
        return v;
    }
};

CodebookRegistry::Impl& CodebookRegistry::impl() noexcept {
    static Impl s;
    return s;
}

CodebookRegistry& CodebookRegistry::instance() noexcept {
    static CodebookRegistry r;
    return r;
}

// -----------------------------------------------------------------------------
// Embedded lookup — no mutation, safe without the mutex.
// -----------------------------------------------------------------------------

Result<CodebookView> CodebookRegistry::find_embedded(std::uint32_t dim,
                                                     std::uint32_t bits) noexcept {
    auto blobs = embedded_codebooks();
    for (const auto& b : blobs) {
        if (b.dim == dim && b.bits == bits) {
            CodebookView      v;
            const std::size_t n_cl = std::size_t{1} << bits;
            v.centroids            = {b.centroids, n_cl};
            v.decision_boundaries  = {b.decision_boundaries, n_cl - 1};
            v.dim                  = b.dim;
            v.bits                 = b.bits;
            v.mse_per_coord        = b.mse_per_coord;
            v.mse_total            = b.mse_total;
            return Result<CodebookView>(v);
        }
    }
    return make_error<CodebookView>(Error::CodebookMissing);
}

// -----------------------------------------------------------------------------
// get(): cache → embedded → env-dir JSON → Lloyd-Max
// -----------------------------------------------------------------------------

Result<CodebookView> CodebookRegistry::get(std::uint32_t dim, std::uint32_t bits) noexcept {
    if (dim == 0 || dim > kMaxDim) return make_error<CodebookView>(Error::InvalidDim);
    if (bits < 1 || bits > 4) return make_error<CodebookView>(Error::InvalidBits);

    auto&               im = impl();
    const std::uint64_t k  = Impl::key(dim, bits);

    {
        std::lock_guard<std::mutex> lg(im.mu);
        auto                        it = im.cache.find(k);
        if (it != im.cache.end()) return Result<CodebookView>(im.view_of(it->second));
    }

    // Embedded table.
    if (auto r = find_embedded(dim, bits); r.has_value()) {
        // Copy embedded blob into owning cache storage so subsequent hits
        // are backed by aligned buffers (same treatment as other paths).
        Impl::Entry e;
        if (!e.centroids.resize(r->centroids.size()) ||
            !e.interior.resize(r->decision_boundaries.size())) {
            return make_error<CodebookView>(Error::CodebookCorrupt);
        }
        std::memcpy(e.centroids.data(), r->centroids.data(), r->centroids.size() * sizeof(float));
        std::memcpy(e.interior.data(), r->decision_boundaries.data(),
                    r->decision_boundaries.size() * sizeof(float));
        e.dim           = r->dim;
        e.bits          = r->bits;
        e.mse_per_coord = r->mse_per_coord;
        e.mse_total     = r->mse_total;

        std::lock_guard<std::mutex> lg(im.mu);
        auto [it, _] = im.cache.emplace(k, std::move(e));
        return Result<CodebookView>(im.view_of(it->second));
    }

    // $TURBOQUANT_CODEBOOK_DIR (optional).
    if (const char* dir = std::getenv("TURBOQUANT_CODEBOOK_DIR"); dir != nullptr) {
        std::string path;
        path.reserve(std::strlen(dir) + 64);
        path.append(dir);
        if (!path.empty() && path.back() != '/') path.push_back('/');
        path.append("codebook_d");
        path.append(std::to_string(dim));
        path.append("_b");
        path.append(std::to_string(bits));
        path.append(".json");

        if (auto r = load_json(path); r.has_value()) return r;
    }

    // Fall back to computing with Lloyd-Max.
    return compute(dim, bits);
}

// -----------------------------------------------------------------------------
// compute(): run Lloyd-Max and cache.
// -----------------------------------------------------------------------------

Result<CodebookView> CodebookRegistry::compute(std::uint32_t dim, std::uint32_t bits,
                                               const LloydMaxOpts& opts) noexcept {
    if (dim == 0 || dim > kMaxDim) return make_error<CodebookView>(Error::InvalidDim);
    if (bits < 1 || bits > 4) return make_error<CodebookView>(Error::InvalidBits);

    auto&               im = impl();
    const std::uint64_t k  = Impl::key(dim, bits);

    {
        std::lock_guard<std::mutex> lg(im.mu);
        auto                        it = im.cache.find(k);
        if (it != im.cache.end()) return Result<CodebookView>(im.view_of(it->second));
    }

    Impl::Entry e;
    float       mse_pc = 0.0f;
    if (auto err = compute_lloyd_max_internal(dim, bits, opts, e.centroids, e.interior, mse_pc);
        !err.has_value()) {
        return make_error<CodebookView>(err.error());
    }
    e.dim           = dim;
    e.bits          = bits;
    e.mse_per_coord = mse_pc;
    e.mse_total     = mse_pc * static_cast<float>(dim);

    std::lock_guard<std::mutex> lg(im.mu);
    auto [it, _] = im.cache.emplace(k, std::move(e));
    return Result<CodebookView>(im.view_of(it->second));
}

// -----------------------------------------------------------------------------
// load_json()
// -----------------------------------------------------------------------------

namespace {

bool read_file(std::string_view path, std::string& out) noexcept {
    std::ifstream f{std::string(path), std::ios::binary};
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

}  // namespace

Result<CodebookView> CodebookRegistry::load_json(std::string_view path) noexcept {
    std::string raw;
    if (!read_file(path, raw)) return make_error<CodebookView>(Error::IoError);

    // allow_exceptions=false; parse_error → discarded value.
    nlohmann::json j = nlohmann::json::parse(raw, /*cb=*/nullptr,
                                             /*allow_exceptions=*/false,
                                             /*ignore_comments=*/true);
    if (j.is_discarded()) return make_error<CodebookView>(Error::CodebookCorrupt);

    if (!j.is_object()) return make_error<CodebookView>(Error::CodebookCorrupt);
    for (const char* k : {"centroids", "boundaries", "d", "bits"}) {
        if (!j.contains(k)) return make_error<CodebookView>(Error::CodebookCorrupt);
    }

    const auto& jc = j["centroids"];
    const auto& jb = j["boundaries"];
    if (!jc.is_array() || !jb.is_array()) return make_error<CodebookView>(Error::CodebookCorrupt);

    std::uint32_t dim, bits;
    if (!j["d"].is_number_unsigned() && !j["d"].is_number_integer())
        return make_error<CodebookView>(Error::CodebookCorrupt);
    dim  = static_cast<std::uint32_t>(j["d"].get<long long>());
    bits = static_cast<std::uint32_t>(j["bits"].get<long long>());

    const std::size_t n_clusters = std::size_t{1} << bits;
    if (jc.size() != n_clusters || jb.size() != n_clusters + 1)
        return make_error<CodebookView>(Error::CodebookCorrupt);

    if (!jb.front().is_number() || !jb.back().is_number() || jb.front().get<double>() != -1.0 ||
        jb.back().get<double>() != 1.0) {
        return make_error<CodebookView>(Error::CodebookCorrupt);
    }

    Impl::Entry e;
    if (!e.centroids.resize(n_clusters) || !e.interior.resize(n_clusters - 1))
        return make_error<CodebookView>(Error::CodebookCorrupt);

    for (std::size_t i = 0; i < n_clusters; ++i) {
        e.centroids[i] = static_cast<float>(jc[i].get<double>());
    }
    // Interior = boundaries[1:-1]
    for (std::size_t i = 0; i < n_clusters - 1; ++i) {
        e.interior[i] = static_cast<float>(jb[i + 1].get<double>());
    }
    // Monotonicity on centroids.
    for (std::size_t i = 0; i + 1 < n_clusters; ++i) {
        if (!(e.centroids[i] < e.centroids[i + 1]))
            return make_error<CodebookView>(Error::CodebookCorrupt);
    }

    e.dim  = dim;
    e.bits = bits;
    if (j.contains("mse_per_coord") && j["mse_per_coord"].is_number())
        e.mse_per_coord = static_cast<float>(j["mse_per_coord"].get<double>());
    if (j.contains("mse_total") && j["mse_total"].is_number())
        e.mse_total = static_cast<float>(j["mse_total"].get<double>());

    auto&                       im = impl();
    const std::uint64_t         k  = Impl::key(dim, bits);
    std::lock_guard<std::mutex> lg(im.mu);
    auto [it, _] = im.cache.emplace(k, std::move(e));
    return Result<CodebookView>(im.view_of(it->second));
}

}  // namespace tq
