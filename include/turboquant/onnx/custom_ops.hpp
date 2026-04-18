#pragma once

// Public entry point for ORT custom-op registration. Consumers (inference
// servers, Python loaders) call RegisterCustomOps on an OrtSessionOptions
// to make the com.turboquant domain available to that session.
//
// This is the ONLY symbol the dylib exports — everything else has hidden
// visibility. See onnx/CMakeLists.txt and the -exported_symbols_list file.

#include <onnxruntime_c_api.h>

#if defined(_WIN32)
#  define TQ_ORT_EXPORT __declspec(dllexport)
#else
#  define TQ_ORT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

TQ_ORT_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) noexcept;

#ifdef __cplusplus
}
#endif
