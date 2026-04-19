// RegisterCustomOps — sole exported symbol of libturboquant_onnx.dylib.
//
// Creates the com.turboquant domain and attaches every op's static
// OrtCustomOp descriptor. ORT copies the descriptor into its session
// state, so the static storage here is fine.
//
// Domain layout mirrors the plan:
//   TurboQuantMSE_Quantize / _Dequantize
//   TurboQuantProd_Quantize / _Dequantize / _AttentionScore
//   TurboQuantValue_Quantize / _Dequantize

#include "turboquant/onnx/custom_ops.hpp"

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

namespace tq::onnx {

// Op descriptors live in op_*.cpp translation units and surface as
// single static instances returned by get_*_op() getters. This keeps
// the registration table in one place without inlining each op's
// schema into this TU.
const OrtCustomOp* get_mse_quantize_op() noexcept;
const OrtCustomOp* get_mse_dequantize_op() noexcept;
const OrtCustomOp* get_prod_quantize_op() noexcept;
const OrtCustomOp* get_prod_dequantize_op() noexcept;
const OrtCustomOp* get_prod_attention_score_op() noexcept;
const OrtCustomOp* get_value_quantize_op() noexcept;
const OrtCustomOp* get_value_dequantize_op() noexcept;

}  // namespace tq::onnx

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                     const OrtApiBase*  api) noexcept {
    Ort::InitApi(api->GetApi(ORT_API_VERSION));

    OrtStatus*         status = nullptr;
    OrtCustomOpDomain* domain = nullptr;

    const OrtApi& ort = Ort::GetApi();

    status = ort.CreateCustomOpDomain("com.turboquant", &domain);
    if (status != nullptr) return status;

    const OrtCustomOp* ops[] = {
        tq::onnx::get_mse_quantize_op(),         tq::onnx::get_mse_dequantize_op(),
        tq::onnx::get_prod_quantize_op(),        tq::onnx::get_prod_dequantize_op(),
        tq::onnx::get_prod_attention_score_op(), tq::onnx::get_value_quantize_op(),
        tq::onnx::get_value_dequantize_op(),
    };

    for (const OrtCustomOp* op : ops) {
        status = ort.CustomOpDomain_Add(domain, op);
        if (status != nullptr) {
            ort.ReleaseCustomOpDomain(domain);
            return status;
        }
    }

    return ort.AddCustomOpDomain(options, domain);
}
