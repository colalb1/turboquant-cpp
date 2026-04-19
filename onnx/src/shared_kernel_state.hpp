#pragma once

// Process-wide cache of TurboQuantMSE<Bits> / TurboQuantProd<Bits> state.
//
// The rotation matrix Pi and QJL matrix S depend only on (dim, seed). If
// we put them as graph initializers we'd bloat a typical 80-layer LLM by
// ~20 MB. Instead we reconstruct them once per (codec, dim, bits, seed)
// tuple and share via shared_ptr. All Compute() paths hit a thread-local
// 2-entry cache first (covers prefill + decode in the same layer without
// touching the global mutex).

#include <cstddef>
#include <cstdint>
#include <memory>

#include "turboquant/quantizer_mse.hpp"
#include "turboquant/quantizer_prod.hpp"

namespace tq::onnx {

enum class CodecId : std::uint8_t {
    MSE  = 1,
    Prod = 2,
};

struct StateKey {
    CodecId       codec;
    std::uint16_t dim;
    std::uint8_t  bits;
    std::uint32_t seed;

    bool operator==(const StateKey& o) const noexcept {
        return codec == o.codec && dim == o.dim
            && bits == o.bits  && seed == o.seed;
    }
};

struct StateKeyHash {
    std::size_t operator()(const StateKey& k) const noexcept {
        std::uint64_t h = static_cast<std::uint64_t>(k.codec) << 56
                        ^ static_cast<std::uint64_t>(k.dim)   << 32
                        ^ static_cast<std::uint64_t>(k.bits)  << 24
                        ^ static_cast<std::uint64_t>(k.seed);
        // splitmix64 finalizer
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return static_cast<std::size_t>(h);
    }
};

// Retrieve (or lazily construct) a cached state. `T` must be one of
// TurboQuantMSE<Bits> or TurboQuantProd<Bits>, whose `make(dim, seed)`
// factory is used to construct fresh instances. Returns nullptr if
// factory fails.
template <class T>
std::shared_ptr<const T>
get_state(const StateKey& key) noexcept;

// Explicit specializations live in shared_kernel_state.cpp — keeps the
// global unordered_map and its mutex out of every TU that includes this
// header.
extern template std::shared_ptr<const TurboQuantMSE<1>> get_state<TurboQuantMSE<1>>(const StateKey&) noexcept;
extern template std::shared_ptr<const TurboQuantMSE<2>> get_state<TurboQuantMSE<2>>(const StateKey&) noexcept;
extern template std::shared_ptr<const TurboQuantMSE<3>> get_state<TurboQuantMSE<3>>(const StateKey&) noexcept;
extern template std::shared_ptr<const TurboQuantMSE<4>> get_state<TurboQuantMSE<4>>(const StateKey&) noexcept;
extern template std::shared_ptr<const TurboQuantProd<2>> get_state<TurboQuantProd<2>>(const StateKey&) noexcept;
extern template std::shared_ptr<const TurboQuantProd<3>> get_state<TurboQuantProd<3>>(const StateKey&) noexcept;
extern template std::shared_ptr<const TurboQuantProd<4>> get_state<TurboQuantProd<4>>(const StateKey&) noexcept;

} // namespace tq::onnx
