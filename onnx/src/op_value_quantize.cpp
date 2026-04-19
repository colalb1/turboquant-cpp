// com.turboquant::TurboQuantValue_Quantize
//   input  0: v        (float, [..., D])
//   output 0: data     (uint8, [..., packed_bytes(D)])
//   output 1: scales   (float, [..., n_groups])
//   output 2: zeros    (float, [..., n_groups])
//   attrs   : bits ∈ {2,4,8}, group_size, head_dim
//
// Stateless — min/max per group computed on-the-fly.

#include "op_util.hpp"

#include "turboquant/error.hpp"
#include "turboquant/value_quant.hpp"

#include <cstdint>

namespace tq::onnx {

namespace {

struct ValueQuantKernel {
    int           bits;
    std::uint16_t dim;
    std::uint16_t group_size;

    ValueQuantKernel(const OrtApi& api, const OrtKernelInfo* info) {
        dim        = static_cast<std::uint16_t>(require_int_attr(api, info, "head_dim"));
        bits       = static_cast<int>          (require_int_attr(api, info, "bits"));
        group_size = static_cast<std::uint16_t>(require_int_attr(api, info, "group_size"));
        if (bits != 2 && bits != 4 && bits != 8)
            ORT_CXX_API_THROW("Value_Quantize: bits must be 2, 4, or 8", ORT_INVALID_ARGUMENT);
        if (dim % group_size != 0)
            ORT_CXX_API_THROW("Value_Quantize: head_dim must be multiple of group_size",
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
        auto v = ctx.GetInput(0);
        auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.empty() || shape.back() != static_cast<std::int64_t>(dim))
            ORT_CXX_API_THROW("Value_Quantize: last dim must equal head_dim",
                              ORT_INVALID_ARGUMENT);

        const std::size_t batch = shape_numel(shape_leading(shape));
        const std::size_t pb    = packed_bytes();
        const std::size_t ng    = n_groups();

        auto lead = shape_leading(shape);
        auto d_shape = shape_with_last(lead, static_cast<std::int64_t>(pb));
        auto s_shape = shape_with_last(lead, static_cast<std::int64_t>(ng));
        auto z_shape = shape_with_last(lead, static_cast<std::int64_t>(ng));

        auto d_out = ctx.GetOutput(0, d_shape.data(), d_shape.size());
        auto s_out = ctx.GetOutput(1, s_shape.data(), s_shape.size());
        auto z_out = ctx.GetOutput(2, z_shape.data(), z_shape.size());

        const float*  v_p = v.GetTensorData<float>();
        std::uint8_t* d_p = d_out.GetTensorMutableData<std::uint8_t>();
        float*        s_p = s_out.GetTensorMutableData<float>();
        float*        z_p = z_out.GetTensorMutableData<float>();

        Error e = Error::NotImplemented;
        switch (bits) {
            case 2: e = ValueCodec<2>::quantize({v_p,batch*dim},batch,dim,group_size,{d_p,batch*pb},{s_p,batch*ng},{z_p,batch*ng}); break;
            case 4: e = ValueCodec<4>::quantize({v_p,batch*dim},batch,dim,group_size,{d_p,batch*pb},{s_p,batch*ng},{z_p,batch*ng}); break;
            case 8: e = ValueCodec<8>::quantize({v_p,batch*dim},batch,dim,group_size,{d_p,batch*pb},{s_p,batch*ng},{z_p,batch*ng}); break;
        }
        if (e != Error::Ok)
            ORT_CXX_API_THROW("Value_Quantize: core quantize failed", ORT_RUNTIME_EXCEPTION);
    }
};

struct ValueQuantOp : Ort::CustomOpBase<ValueQuantOp, ValueQuantKernel> {
    const char* GetName() const noexcept                    { return "TurboQuantValue_Quantize"; }
    const char* GetExecutionProviderType() const noexcept   { return "CPUExecutionProvider"; }
    std::size_t GetInputTypeCount() const noexcept          { return 1; }
    ONNXTensorElementDataType GetInputType(std::size_t) const noexcept {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    std::size_t GetOutputTypeCount() const noexcept         { return 3; }
    ONNXTensorElementDataType GetOutputType(std::size_t i) const noexcept {
        return i == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new ValueQuantKernel(api, info);
    }
};

} // namespace

const OrtCustomOp* get_value_quantize_op() noexcept {
    static const ValueQuantOp op;
    return &op;
}

} // namespace tq::onnx
