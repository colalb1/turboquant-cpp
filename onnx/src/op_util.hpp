#pragma once

// Small helpers shared across op_*.cpp translation units. ORT's C++ API
// (onnxruntime_cxx_api.h) uses exceptions internally; we contain them to
// Compute() and CreateKernel() boundaries by catching in ExceptionBoundary.

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

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

// Count the product of a shape's dims (or 1 if empty).
inline std::size_t shape_numel(const std::vector<std::int64_t>& dims) noexcept {
    std::size_t n = 1;
    for (auto d : dims)
        n *= static_cast<std::size_t>(d);
    return n;
}

// Everything but the last dim.
inline std::vector<std::int64_t> shape_leading(const std::vector<std::int64_t>& dims) {
    if (dims.empty()) return {};
    return {dims.begin(), dims.end() - 1};
}

// A buffer of leading dims plus one trailing dim, e.g. x.shape[:-1] + [k].
inline std::vector<std::int64_t> shape_with_last(const std::vector<std::int64_t>& leading,
                                                 std::int64_t                     last) {
    std::vector<std::int64_t> out(leading);
    out.push_back(last);
    return out;
}

}  // namespace tq::onnx
