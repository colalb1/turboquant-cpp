// com.turboquant::TurboQuantProd_AttentionScore
//   input  0: query            (float, [..., Nq, D])
//   input  1: mse_indices      (uint8, [..., Nk, mse_pb])
//   input  2: qjl_signs        (uint8, [..., Nk, qjl_pb])
//   input  3: residual_norms   (float, [..., Nk])
//   input  4: norms            (float, [..., Nk])
//   output 0: scores           (float, [..., Nq, Nk])
//
// Leading dims of all inputs must match. Nq, Nk may differ. This op
// implements only the "single batch" flavor used by tests — leading
// dims must be empty or match via explicit broadcast at graph level.

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

struct ProdScoreKernel {
    int                                      bits;
    std::uint16_t                            dim;
    std::uint32_t                            seed;
    std::shared_ptr<const TurboQuantProd<2>> s2;
    std::shared_ptr<const TurboQuantProd<3>> s3;
    std::shared_ptr<const TurboQuantProd<4>> s4;

    ProdScoreKernel(const OrtApi& api, const OrtKernelInfo* info) {
        dim  = static_cast<std::uint16_t>(require_int_attr(api, info, "dim"));
        bits = static_cast<int>(require_int_attr(api, info, "bits"));
        seed = static_cast<std::uint32_t>(read_int_attr(api, info, "seed", 42));
        if (bits < 2 || bits > 4)
            ORT_CXX_API_THROW("Prod_AttnScore: bits must be in [2,4]", ORT_INVALID_ARGUMENT);

        const StateKey k{CodecId::Prod, dim, static_cast<std::uint8_t>(bits), seed};
        switch (bits) {
        case 2: s2 = get_state<TurboQuantProd<2>>(k); break;
        case 3: s3 = get_state<TurboQuantProd<3>>(k); break;
        case 4: s4 = get_state<TurboQuantProd<4>>(k); break;
        }
        if (!(s2 || s3 || s4))
            ORT_CXX_API_THROW("Prod_AttnScore: state init failed", ORT_RUNTIME_EXCEPTION);
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
        auto               q   = ctx.GetInput(0);
        auto               mse = ctx.GetInput(1);
        auto               qjl = ctx.GetInput(2);
        auto               rnm = ctx.GetInput(3);
        auto               nrm = ctx.GetInput(4);

        auto q_shape = q.GetTensorTypeAndShapeInfo().GetShape();
        auto n_shape = nrm.GetTensorTypeAndShapeInfo().GetShape();
        if (q_shape.empty() || n_shape.empty())
            ORT_CXX_API_THROW("Prod_AttnScore: inputs must be rank-2 or higher",
                              ORT_INVALID_ARGUMENT);
        if (q_shape.back() != static_cast<std::int64_t>(dim))
            ORT_CXX_API_THROW("Prod_AttnScore: query last dim != attr dim", ORT_INVALID_ARGUMENT);

        const std::size_t n_q = static_cast<std::size_t>(q_shape[q_shape.size() - 2]);
        const std::size_t n_k = static_cast<std::size_t>(n_shape.back());

        // Output shape: drop trailing D from q_shape, replace the new last with n_q, append n_k.
        ShapeBuf lead   = shape_leading(q_shape);
        if (lead.rank == 0)
            ORT_CXX_API_THROW("Prod_AttnScore: query must have rank >= 2", ORT_INVALID_ARGUMENT);
        lead.dims[lead.rank - 1] = static_cast<std::int64_t>(n_q);
        ShapeBuf out_shape = shape_with_last(lead, static_cast<std::int64_t>(n_k));
        auto     scores_t  = ctx.GetOutput(0, out_shape.data(), out_shape.size());

        const float*        q_p  = q.GetTensorData<float>();
        const std::uint8_t* m_p  = mse.GetTensorData<std::uint8_t>();
        const std::uint8_t* qj_p = qjl.GetTensorData<std::uint8_t>();
        const float*        r_p  = rnm.GetTensorData<float>();
        const float*        n_p  = nrm.GetTensorData<float>();
        float*              sc_p = scores_t.GetTensorMutableData<float>();

        const std::size_t mb = mse_pb();
        const std::size_t qb = QJLPack::packed_bytes(dim);

        Error e = Error::NotImplemented;
        switch (bits) {
        case 2:
            e = s2->attention_score({q_p, n_q * dim}, n_q, {m_p, n_k * mb}, {qj_p, n_k * qb},
                                    {r_p, n_k}, {n_p, n_k}, n_k, {sc_p, n_q * n_k});
            break;
        case 3:
            e = s3->attention_score({q_p, n_q * dim}, n_q, {m_p, n_k * mb}, {qj_p, n_k * qb},
                                    {r_p, n_k}, {n_p, n_k}, n_k, {sc_p, n_q * n_k});
            break;
        case 4:
            e = s4->attention_score({q_p, n_q * dim}, n_q, {m_p, n_k * mb}, {qj_p, n_k * qb},
                                    {r_p, n_k}, {n_p, n_k}, n_k, {sc_p, n_q * n_k});
            break;
        }
        if (e != Error::Ok)
            ORT_CXX_API_THROW("Prod_AttnScore: core score failed", ORT_RUNTIME_EXCEPTION);
    }
};

struct ProdScoreOp : Ort::CustomOpBase<ProdScoreOp, ProdScoreKernel> {
    const char* GetName() const noexcept { return "TurboQuantProd_AttentionScore"; }
    const char* GetExecutionProviderType() const noexcept { return "CPUExecutionProvider"; }
    std::size_t GetInputTypeCount() const noexcept { return 5; }
    ONNXTensorElementDataType GetInputType(std::size_t i) const noexcept {
        if (i == 0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        if (i == 1 || i == 2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    std::size_t               GetOutputTypeCount() const noexcept { return 1; }
    ONNXTensorElementDataType GetOutputType(std::size_t) const noexcept {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new ProdScoreKernel(api, info);
    }
};

}  // namespace

const OrtCustomOp* get_prod_attention_score_op() noexcept {
    static const ProdScoreOp op;
    return &op;
}

}  // namespace tq::onnx
