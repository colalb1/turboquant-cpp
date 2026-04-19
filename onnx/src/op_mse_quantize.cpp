// com.turboquant::TurboQuantMSE_Quantize
//   input  0: x        (float, [..., D])
//   output 0: indices  (uint8, [..., packed_bytes(D, bits)])
//   output 1: norms    (float, [...])
//   attrs   : dim (int), bits (int ∈ [1,4]), seed (int, default 42)

#include "op_util.hpp"
#include "shared_kernel_state.hpp"

#include "turboquant/error.hpp"
#include "turboquant/pack_policy.hpp"
#include "turboquant/quantizer_mse.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace tq::onnx {

namespace {

struct MseQuantKernel {
    int           bits;
    std::uint16_t dim;
    std::uint32_t seed;
    // Typed state ptrs — exactly one is non-null based on `bits`.
    std::shared_ptr<const TurboQuantMSE<1>> s1;
    std::shared_ptr<const TurboQuantMSE<2>> s2;
    std::shared_ptr<const TurboQuantMSE<3>> s3;
    std::shared_ptr<const TurboQuantMSE<4>> s4;

    MseQuantKernel(const OrtApi& api, const OrtKernelInfo* info) {
        dim  = static_cast<std::uint16_t>(require_int_attr(api, info, "dim"));
        bits = static_cast<int>         (require_int_attr(api, info, "bits"));
        seed = static_cast<std::uint32_t>(read_int_attr   (api, info, "seed", 42));
        if (bits < 1 || bits > 4)
            ORT_CXX_API_THROW("MSE_Quantize: bits must be in [1,4]", ORT_INVALID_ARGUMENT);

        const StateKey k{ CodecId::MSE, dim, static_cast<std::uint8_t>(bits), seed };
        switch (bits) {
            case 1: s1 = get_state<TurboQuantMSE<1>>(k); if (!s1) ORT_CXX_API_THROW("MSE_Quantize: state init failed", ORT_RUNTIME_EXCEPTION); break;
            case 2: s2 = get_state<TurboQuantMSE<2>>(k); if (!s2) ORT_CXX_API_THROW("MSE_Quantize: state init failed", ORT_RUNTIME_EXCEPTION); break;
            case 3: s3 = get_state<TurboQuantMSE<3>>(k); if (!s3) ORT_CXX_API_THROW("MSE_Quantize: state init failed", ORT_RUNTIME_EXCEPTION); break;
            case 4: s4 = get_state<TurboQuantMSE<4>>(k); if (!s4) ORT_CXX_API_THROW("MSE_Quantize: state init failed", ORT_RUNTIME_EXCEPTION); break;
        }
    }

    std::size_t packed_bytes() const noexcept {
        switch (bits) {
            case 1: return PackPolicy<1>::packed_bytes(dim);
            case 2: return PackPolicy<2>::packed_bytes(dim);
            case 3: return PackPolicy<3>::packed_bytes(dim);
            case 4: return PackPolicy<4>::packed_bytes(dim);
        }
        return 0;
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        auto x = ctx.GetInput(0);
        auto info = x.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        if (shape.empty() || shape.back() != static_cast<std::int64_t>(dim))
            ORT_CXX_API_THROW("MSE_Quantize: last dim must equal attribute `dim`",
                              ORT_INVALID_ARGUMENT);

        const std::size_t batch = shape_numel(shape_leading(shape));
        const float* x_data = x.GetTensorData<float>();

        const std::size_t pb = packed_bytes();
        auto idx_shape = shape_with_last(shape_leading(shape),
                                          static_cast<std::int64_t>(pb));
        auto nrm_shape = shape_leading(shape);

        auto idx_out = ctx.GetOutput(0, idx_shape.data(), idx_shape.size());
        auto nrm_out = ctx.GetOutput(1, nrm_shape.data(), nrm_shape.size());

        std::uint8_t* idx_p = idx_out.GetTensorMutableData<std::uint8_t>();
        float*        nrm_p = nrm_out.GetTensorMutableData<float>();

        Error e = Error::NotImplemented;
        switch (bits) {
            case 1: e = s1->quantize({ x_data, batch * dim }, batch, { idx_p, batch * pb }, { nrm_p, batch }); break;
            case 2: e = s2->quantize({ x_data, batch * dim }, batch, { idx_p, batch * pb }, { nrm_p, batch }); break;
            case 3: e = s3->quantize({ x_data, batch * dim }, batch, { idx_p, batch * pb }, { nrm_p, batch }); break;
            case 4: e = s4->quantize({ x_data, batch * dim }, batch, { idx_p, batch * pb }, { nrm_p, batch }); break;
        }
        if (e != Error::Ok)
            ORT_CXX_API_THROW("MSE_Quantize: core quantize failed", ORT_RUNTIME_EXCEPTION);
    }
};

struct MseQuantOp : Ort::CustomOpBase<MseQuantOp, MseQuantKernel> {
    const char* GetName() const noexcept                    { return "TurboQuantMSE_Quantize"; }
    const char* GetExecutionProviderType() const noexcept   { return "CPUExecutionProvider"; }
    std::size_t GetInputTypeCount() const noexcept          { return 1; }
    ONNXTensorElementDataType GetInputType(std::size_t) const noexcept {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    std::size_t GetOutputTypeCount() const noexcept         { return 2; }
    ONNXTensorElementDataType GetOutputType(std::size_t i) const noexcept {
        return i == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new MseQuantKernel(api, info);
    }
};

} // namespace

const OrtCustomOp* get_mse_quantize_op() noexcept {
    static const MseQuantOp op;
    return &op;
}

} // namespace tq::onnx
