// com.turboquant::TurboQuantProd_Quantize
//   input  0: x               (float, [..., D])
//   output 0: mse_indices     (uint8, [..., mse_pb])
//   output 1: qjl_signs       (uint8, [..., qjl_pb])
//   output 2: residual_norms  (float, [...])
//   output 3: norms           (float, [...])
//   attrs   : dim, bits ∈ [2,4], seed=42

#include "op_util.hpp"
#include "shared_kernel_state.hpp"

#include "turboquant/error.hpp"
#include "turboquant/pack_policy.hpp"
#include "turboquant/qjl_signs.hpp"
#include "turboquant/quantizer_prod.hpp"

#include <cstdint>
#include <memory>

namespace tq::onnx {

namespace {

struct ProdQuantKernel {
    int           bits;
    std::uint16_t dim;
    std::uint32_t seed;
    std::shared_ptr<const TurboQuantProd<2>> s2;
    std::shared_ptr<const TurboQuantProd<3>> s3;
    std::shared_ptr<const TurboQuantProd<4>> s4;

    ProdQuantKernel(const OrtApi& api, const OrtKernelInfo* info) {
        dim  = static_cast<std::uint16_t>(require_int_attr(api, info, "dim"));
        bits = static_cast<int>         (require_int_attr(api, info, "bits"));
        seed = static_cast<std::uint32_t>(read_int_attr   (api, info, "seed", 42));
        if (bits < 2 || bits > 4)
            ORT_CXX_API_THROW("Prod_Quantize: bits must be in [2,4]", ORT_INVALID_ARGUMENT);

        const StateKey k{ CodecId::Prod, dim, static_cast<std::uint8_t>(bits), seed };
        switch (bits) {
            case 2: s2 = get_state<TurboQuantProd<2>>(k); break;
            case 3: s3 = get_state<TurboQuantProd<3>>(k); break;
            case 4: s4 = get_state<TurboQuantProd<4>>(k); break;
        }
        if (!(s2 || s3 || s4))
            ORT_CXX_API_THROW("Prod_Quantize: state init failed", ORT_RUNTIME_EXCEPTION);
    }

    std::size_t mse_pb() const noexcept {
        switch (bits) {
            case 2: return PackPolicy<1>::packed_bytes(dim);   // Prod uses Bits-1 for MSE
            case 3: return PackPolicy<2>::packed_bytes(dim);
            case 4: return PackPolicy<3>::packed_bytes(dim);
        }
        return 0;
    }
    std::size_t qjl_pb() const noexcept { return QJLPack::packed_bytes(dim); }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        auto x = ctx.GetInput(0);
        auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.empty() || shape.back() != static_cast<std::int64_t>(dim))
            ORT_CXX_API_THROW("Prod_Quantize: last dim mismatch", ORT_INVALID_ARGUMENT);

        const std::size_t batch = shape_numel(shape_leading(shape));
        const float* x_data = x.GetTensorData<float>();

        const std::size_t mb = mse_pb();
        const std::size_t qb = qjl_pb();
        auto lead = shape_leading(shape);
        auto mse_shape = shape_with_last(lead, static_cast<std::int64_t>(mb));
        auto qjl_shape = shape_with_last(lead, static_cast<std::int64_t>(qb));
        auto rnm_shape = lead;
        auto nrm_shape = lead;

        auto mse_out = ctx.GetOutput(0, mse_shape.data(), mse_shape.size());
        auto qjl_out = ctx.GetOutput(1, qjl_shape.data(), qjl_shape.size());
        auto rnm_out = ctx.GetOutput(2, rnm_shape.data(), rnm_shape.size());
        auto nrm_out = ctx.GetOutput(3, nrm_shape.data(), nrm_shape.size());

        std::uint8_t* m_p = mse_out.GetTensorMutableData<std::uint8_t>();
        std::uint8_t* q_p = qjl_out.GetTensorMutableData<std::uint8_t>();
        float*        r_p = rnm_out.GetTensorMutableData<float>();
        float*        n_p = nrm_out.GetTensorMutableData<float>();

        Error e = Error::NotImplemented;
        switch (bits) {
            case 2: e = s2->quantize({x_data,batch*dim},batch,{m_p,batch*mb},{q_p,batch*qb},{r_p,batch},{n_p,batch}); break;
            case 3: e = s3->quantize({x_data,batch*dim},batch,{m_p,batch*mb},{q_p,batch*qb},{r_p,batch},{n_p,batch}); break;
            case 4: e = s4->quantize({x_data,batch*dim},batch,{m_p,batch*mb},{q_p,batch*qb},{r_p,batch},{n_p,batch}); break;
        }
        if (e != Error::Ok)
            ORT_CXX_API_THROW("Prod_Quantize: core quantize failed", ORT_RUNTIME_EXCEPTION);
    }
};

struct ProdQuantOp : Ort::CustomOpBase<ProdQuantOp, ProdQuantKernel> {
    const char* GetName() const noexcept                    { return "TurboQuantProd_Quantize"; }
    const char* GetExecutionProviderType() const noexcept   { return "CPUExecutionProvider"; }
    std::size_t GetInputTypeCount() const noexcept          { return 1; }
    ONNXTensorElementDataType GetInputType(std::size_t) const noexcept {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    std::size_t GetOutputTypeCount() const noexcept         { return 4; }
    ONNXTensorElementDataType GetOutputType(std::size_t i) const noexcept {
        return (i == 0 || i == 1) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                                   : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new ProdQuantKernel(api, info);
    }
};

} // namespace

const OrtCustomOp* get_prod_quantize_op() noexcept {
    static const ProdQuantOp op;
    return &op;
}

} // namespace tq::onnx
