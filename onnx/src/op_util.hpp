#pragma once

// Small helpers shared across op_*.cpp translation units. ORT's C++ API
// (onnxruntime_cxx_api.h) uses exceptions internally; we contain them to
// Compute() and CreateKernel() boundaries by catching in ExceptionBoundary.

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace tq::onnx {

// Read an int attribute, return a default if missing.
inline std::int64_t read_int_attr(const OrtApi& api, const OrtKernelInfo* info, const char* name,
                                  std::int64_t default_value) {
    std::int64_t v = default_value;
    OrtStatus*   s = api.KernelInfoGetAttribute_int64(info, name, &v);
    if (s != nullptr) {
        api.ReleaseStatus(s);
        return default_value;
    }
    return v;
}

// Require an int attribute; throws Ort::Exception on failure.
inline std::int64_t require_int_attr(const OrtApi& api, const OrtKernelInfo* info,
                                     const char* name) {
    std::int64_t v = 0;
    Ort::ThrowOnError(api.KernelInfoGetAttribute_int64(info, name, &v));
    return v;
}

// Fixed-capacity shape buffer for ORT output shape construction. Avoids the
// heap allocation a std::vector would incur on every Compute() call.
struct ShapeBuf {
    static constexpr std::size_t         kMaxRank = 8;
    std::array<std::int64_t, kMaxRank> dims{};
    std::size_t                        rank = 0;

    const std::int64_t* data() const noexcept { return dims.data(); }
    std::size_t         size() const noexcept { return rank; }

    operator std::span<const std::int64_t>() const noexcept { return {dims.data(), rank}; }
};

// Count the product of a shape's dims (or 1 if empty).
inline std::size_t shape_numel(std::span<const std::int64_t> dims) noexcept {
    std::size_t n = 1;
    for (auto d : dims) n *= static_cast<std::size_t>(d);
    return n;
}

// Everything but the last dim.
inline ShapeBuf shape_leading(std::span<const std::int64_t> dims) {
    ShapeBuf out;
    if (dims.empty()) return out;
    out.rank = dims.size() - 1;
    if (out.rank > ShapeBuf::kMaxRank) {
        ORT_CXX_API_THROW("turboquant: tensor rank exceeds ShapeBuf::kMaxRank",
                          ORT_INVALID_ARGUMENT);
    }
    for (std::size_t i = 0; i < out.rank; ++i) out.dims[i] = dims[i];
    return out;
}

// A buffer of leading dims plus one trailing dim, e.g. x.shape[:-1] + [k].
inline ShapeBuf shape_with_last(std::span<const std::int64_t> leading, std::int64_t last) {
    ShapeBuf out;
    out.rank = leading.size() + 1;
    if (out.rank > ShapeBuf::kMaxRank) {
        ORT_CXX_API_THROW("turboquant: tensor rank exceeds ShapeBuf::kMaxRank",
                          ORT_INVALID_ARGUMENT);
    }
    for (std::size_t i = 0; i < leading.size(); ++i) out.dims[i] = leading[i];
    out.dims[leading.size()] = last;
    return out;
}

}  // namespace tq::onnx
