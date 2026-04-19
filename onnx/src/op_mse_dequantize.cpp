// com.turboquant::TurboQuantMSE_Dequantize
//   input  0: indices (uint8, [..., packed_bytes(D, bits)])
//   input  1: norms   (float, [...])
//   output 0: x       (float, [..., D])
//   attrs   : dim (int), bits (int ∈ [1,4]), seed (int, default 42)

#include "op_util.hpp"
#include "shared_kernel_state.hpp"

#include "turboquant/error.hpp"
#include "turboquant/pack_policy.hpp"
#include "turboquant/quantizer_mse.hpp"

#include <cstdint>
#include <memory>

namespace tq::onnx {

namespace {

struct MseDequantKernel {
    int           bits;
    std::uint16_t dim;
    std::uint32_t seed;
    std::shared_ptr<const TurboQuantMSE<1>> s1;
    std::shared_ptr<const TurboQuantMSE<2>> s2;
    std::shared_ptr<const TurboQuantMSE<3>> s3;
    std::shared_ptr<const TurboQuantMSE<4>> s4;

    MseDequantKernel(const OrtApi& api, const OrtKernelInfo* info) {
        dim  = static_cast<std::uint16_t>(require_int_attr(api, info, "dim"));
        bits = static_cast<int>         (require_int_attr(api, info, "bits"));
        seed = static_cast<std::uint32_t>(read_int_attr   (api, info, "seed", 42));
        if (bits < 1 || bits > 4)
            ORT_CXX_API_THROW("MSE_Dequantize: bits must be in [1,4]", ORT_INVALID_ARGUMENT);

        const StateKey k{ CodecId::MSE, dim, static_cast<std::uint8_t>(bits), seed };
        switch (bits) {
            case 1: s1 = get_state<TurboQuantMSE<1>>(k); break;
            case 2: s2 = get_state<TurboQuantMSE<2>>(k); break;
            case 3: s3 = get_state<TurboQuantMSE<3>>(k); break;
            case 4: s4 = get_state<TurboQuantMSE<4>>(k); break;
        }
        if (!(s1 || s2 || s3 || s4))
            ORT_CXX_API_THROW("MSE_Dequantize: state init failed", ORT_RUNTIME_EXCEPTION);
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
        auto idx_t = ctx.GetInput(0);
        auto nrm_t = ctx.GetInput(1);
        auto nrm_shape = nrm_t.GetTensorTypeAndShapeInfo().GetShape();

        const std::size_t batch = shape_numel(nrm_shape);
        const std::size_t pb    = packed_bytes();

        auto out_shape = shape_with_last(nrm_shape, static_cast<std::int64_t>(dim));
        auto x_out     = ctx.GetOutput(0, out_shape.data(), out_shape.size());

        const std::uint8_t* idx_p = idx_t.GetTensorData<std::uint8_t>();
        const float*        nrm_p = nrm_t.GetTensorData<float>();
        float*              x_p   = x_out.GetTensorMutableData<float>();

        Error e = Error::NotImplemented;
        switch (bits) {
            case 1: e = s1->dequantize({ idx_p, batch * pb }, { nrm_p, batch }, batch, { x_p, batch * dim }); break;
            case 2: e = s2->dequantize({ idx_p, batch * pb }, { nrm_p, batch }, batch, { x_p, batch * dim }); break;
            case 3: e = s3->dequantize({ idx_p, batch * pb }, { nrm_p, batch }, batch, { x_p, batch * dim }); break;
            case 4: e = s4->dequantize({ idx_p, batch * pb }, { nrm_p, batch }, batch, { x_p, batch * dim }); break;
        }
        if (e != Error::Ok)
            ORT_CXX_API_THROW("MSE_Dequantize: core dequantize failed", ORT_RUNTIME_EXCEPTION);
    }
};

struct MseDequantOp : Ort::CustomOpBase<MseDequantOp, MseDequantKernel> {
    const char* GetName() const noexcept                    { return "TurboQuantMSE_Dequantize"; }
    const char* GetExecutionProviderType() const noexcept   { return "CPUExecutionProvider"; }
    std::size_t GetInputTypeCount() const noexcept          { return 2; }
    ONNXTensorElementDataType GetInputType(std::size_t i) const noexcept {
        return i == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    std::size_t GetOutputTypeCount() const noexcept         { return 1; }
    ONNXTensorElementDataType GetOutputType(std::size_t) const noexcept {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new MseDequantKernel(api, info);
    }
};

} // namespace

const OrtCustomOp* get_mse_dequantize_op() noexcept {
    static const MseDequantOp op;
    return &op;
}

} // namespace tq::onnx
