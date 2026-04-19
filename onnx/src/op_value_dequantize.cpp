// com.turboquant::TurboQuantValue_Dequantize
//   input  0: data     (uint8, [..., packed_bytes(D)])
//   input  1: scales   (float, [..., n_groups])
//   input  2: zeros    (float, [..., n_groups])
//   output 0: v        (float, [..., D])
//   attrs   : bits ∈ {2,4,8}, group_size, head_dim

#include "op_util.hpp"

#include "turboquant/error.hpp"
#include "turboquant/value_quant.hpp"

#include <cstdint>

namespace tq::onnx {

namespace {

struct ValueDequantKernel {
    int           bits;
    std::uint16_t dim;
    std::uint16_t group_size;

    ValueDequantKernel(const OrtApi& api, const OrtKernelInfo* info) {
        dim        = static_cast<std::uint16_t>(require_int_attr(api, info, "head_dim"));
        bits       = static_cast<int>          (require_int_attr(api, info, "bits"));
        group_size = static_cast<std::uint16_t>(require_int_attr(api, info, "group_size"));
        if (bits != 2 && bits != 4 && bits != 8)
            ORT_CXX_API_THROW("Value_Dequantize: bits must be 2, 4, or 8", ORT_INVALID_ARGUMENT);
        if (dim % group_size != 0)
            ORT_CXX_API_THROW("Value_Dequantize: head_dim must be multiple of group_size",
                              ORT_INVALID_ARGUMENT);
    }

    std::size_t packed_bytes() const noexcept {
        switch (bits) {
            case 2: return ValueCodec<2>::packed_bytes(dim);
            case 4: return ValueCodec<4>::packed_bytes(dim);
            case 8: return ValueCodec<8>::packed_bytes(dim);
        }
        return 0;
    }
    std::size_t n_groups() const noexcept { return static_cast<std::size_t>(dim / group_size); }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        auto data   = ctx.GetInput(0);
        auto scales = ctx.GetInput(1);
        auto zeros  = ctx.GetInput(2);

        auto s_shape = scales.GetTensorTypeAndShapeInfo().GetShape();
        // scales has leading dims then [n_groups]. Strip the last.
        auto lead = shape_leading(s_shape);
        const std::size_t batch = shape_numel(lead);
        const std::size_t pb    = packed_bytes();
        const std::size_t ng    = n_groups();

        auto v_shape = shape_with_last(lead, static_cast<std::int64_t>(dim));
        auto v_out   = ctx.GetOutput(0, v_shape.data(), v_shape.size());

        const std::uint8_t* d_p = data.GetTensorData<std::uint8_t>();
        const float*        s_p = scales.GetTensorData<float>();
        const float*        z_p = zeros.GetTensorData<float>();
        float*              v_p = v_out.GetTensorMutableData<float>();

        Error e = Error::NotImplemented;
        switch (bits) {
            case 2: e = ValueCodec<2>::dequantize({d_p,batch*pb},{s_p,batch*ng},{z_p,batch*ng},batch,dim,group_size,{v_p,batch*dim}); break;
            case 4: e = ValueCodec<4>::dequantize({d_p,batch*pb},{s_p,batch*ng},{z_p,batch*ng},batch,dim,group_size,{v_p,batch*dim}); break;
            case 8: e = ValueCodec<8>::dequantize({d_p,batch*pb},{s_p,batch*ng},{z_p,batch*ng},batch,dim,group_size,{v_p,batch*dim}); break;
        }
        if (e != Error::Ok)
            ORT_CXX_API_THROW("Value_Dequantize: core dequantize failed", ORT_RUNTIME_EXCEPTION);
    }
};

struct ValueDequantOp : Ort::CustomOpBase<ValueDequantOp, ValueDequantKernel> {
    const char* GetName() const noexcept                    { return "TurboQuantValue_Dequantize"; }
    const char* GetExecutionProviderType() const noexcept   { return "CPUExecutionProvider"; }
    std::size_t GetInputTypeCount() const noexcept          { return 3; }
    ONNXTensorElementDataType GetInputType(std::size_t i) const noexcept {
        return i == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    std::size_t GetOutputTypeCount() const noexcept         { return 1; }
    ONNXTensorElementDataType GetOutputType(std::size_t) const noexcept {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new ValueDequantKernel(api, info);
    }
};

} // namespace

const OrtCustomOp* get_value_dequantize_op() noexcept {
    static const ValueDequantOp op;
    return &op;
}

} // namespace tq::onnx
