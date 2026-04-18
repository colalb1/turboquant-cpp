#pragma once

#include <concepts>
#include <type_traits>

namespace tq::arch {

struct Neon {};
struct Scalar {};

} // namespace tq::arch

namespace tq {

template <class A>
concept ArchTag = std::same_as<A, arch::Neon> || std::same_as<A, arch::Scalar>;

// Default architecture tag for this build.
#if defined(__ARM_NEON)
using DefaultArch = arch::Neon;
#else
using DefaultArch = arch::Scalar;
#endif

} // namespace tq
