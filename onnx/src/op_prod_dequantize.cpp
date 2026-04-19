// com.turboquant::TurboQuantProd_Dequantize
//   input  0: mse_indices     (uint8, [..., mse_pb])
//   input  1: qjl_signs       (uint8, [..., qjl_pb])
//   input  2: residual_norms  (float, [...])
//   input  3: norms           (float, [...])
//   output 0: x               (float, [..., D])

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

struct ProdDequantKernel {
    int                                      bits;
    std::uint16_t                            dim;
    std::uint32_t                            seed;
    std::shared_ptr<const TurboQuantProd<2>> s2;
    std::shared_ptr<const TurboQuantProd<3>> s3;
    std::shared_ptr<const TurboQuantProd<4>> s4;

    ProdDequantKernel(const OrtApi& api, const OrtKernelInfo* info) {
        dim  = static_cast<std::uint16_t>(require_int_attr(api, info, "dim"));
        bits = static_cast<int>(require_int_attr(api, info, "bits"));
        seed = static_cast<std::uint32_t>(read_int_attr(api, info, "seed", 42));
        if (bits < 2 || bits > 4)
            ORT_CXX_API_THROW("Prod_Dequantize: bits must be in [2,4]", ORT_INVALID_ARGUMENT);

        const StateKey k{CodecId::Prod, dim, static_cast<std::uint8_t>(bits), seed};
        switch (bits) {
        case 2: s2 = get_state<TurboQuantProd<2>>(k); break;
        case 3: s3 = get_state<TurboQuantProd<3>>(k); break;
        case 4: s4 = get_state<TurboQuantProd<4>>(k); break;
        }
        if (!(s2 || s3 || s4))
            ORT_CXX_API_THROW("Prod_Dequantize: state init failed", ORT_RUNTIME_EXCEPTION);
    }

    std::size_t mse_pb() const noexcept {
        switch (bits) {
        case 2: return PackPolicy<1>::packed_bytes(dim);
        case 3: return PackPolicy<2>::packed_bytes(dim);
        case 4: return PackPolicy<3>::packed_bytes(dim);
        }
        return 0;
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        auto               mse = ctx.GetInput(0);
        auto               qjl = ctx.GetInput(1);
        auto               rnm = ctx.GetInput(2);
        auto               nrm = ctx.GetInput(3);

        auto              nrm_shape = nrm.GetTensorTypeAndShapeInfo().GetShape();
        const std::size_t batch     = shape_numel(nrm_shape);
        const std::size_t mb        = mse_pb();
        const std::size_t qb        = QJLPack::packed_bytes(dim);

        auto out_shape = shape_with_last(nrm_shape, static_cast<std::int64_t>(dim));
        auto x_out     = ctx.GetOutput(0, out_shape.data(), out_shape.size());

        const std::uint8_t* m_p = mse.GetTensorData<std::uint8_t>();
        const std::uint8_t* q_p = qjl.GetTensorData<std::uint8_t>();
        const float*        r_p = rnm.GetTensorData<float>();
        const float*        n_p = nrm.GetTensorData<float>();
        float*              x_p = x_out.GetTensorMutableData<float>();

        Error e = Error::NotImplemented;
        switch (bits) {
        case 2:
            e = s2->dequantize({m_p, batch * mb}, {q_p, batch * qb}, {r_p, batch}, {n_p, batch},
                               batch, {x_p, batch * dim});
            break;
        case 3:
            e = s3->dequantize({m_p, batch * mb}, {q_p, batch * qb}, {r_p, batch}, {n_p, batch},
                               batch, {x_p, batch * dim});
            break;
        case 4:
            e = s4->dequantize({m_p, batch * mb}, {q_p, batch * qb}, {r_p, batch}, {n_p, batch},
                               batch, {x_p, batch * dim});
            break;
        }
        if (e != Error::Ok)
            ORT_CXX_API_THROW("Prod_Dequantize: core dequantize failed", ORT_RUNTIME_EXCEPTION);
    }
};

struct ProdDequantOp : Ort::CustomOpBase<ProdDequantOp, ProdDequantKernel> {
    const char* GetName() const noexcept { return "TurboQuantProd_Dequantize"; }
    const char* GetExecutionProviderType() const noexcept { return "CPUExecutionProvider"; }
    std::size_t GetInputTypeCount() const noexcept { return 4; }
    ONNXTensorElementDataType GetInputType(std::size_t i) const noexcept {
        return (i == 0 || i == 1) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                                  : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    std::size_t               GetOutputTypeCount() const noexcept { return 1; }
    ONNXTensorElementDataType GetOutputType(std::size_t) const noexcept {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new ProdDequantKernel(api, info);
    }
};

}  // namespace

const OrtCustomOp* get_prod_dequantize_op() noexcept {
    static const ProdDequantOp op;
    return &op;
}

}  // namespace tq::onnx
