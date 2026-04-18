// Portable scalar fallback for Rotation.
//
// On Apple Silicon we always take the Accelerate path in
// rotation_accelerate.cpp; this TU is a no-op there (guarded out). On non-
// Apple hosts it would contain a pure-C++ QR via modified Gram-Schmidt. For
// now we leave it empty — non-Apple builds get Error::NotImplemented from
// Rotation::make, which is acceptable because the production target is
// macOS arm64.

#if !defined(__APPLE__)
// Future: implement modified Gram-Schmidt QR here so the library builds on
// Linux arm64 / x86_64 CI runners without Accelerate. Left as a deliberate
// milestone deferral — see IMPLEMENTATION_PLAN.md R1 risk mitigation.
#endif
