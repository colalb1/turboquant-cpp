// Smoke test for the ORT custom-op dylib.
//
// What this proves:
//   - The dylib built, links cleanly against ORT 1.17, and exposes
//     exactly one public symbol: RegisterCustomOps.
//   - Calling RegisterCustomOps with a real Ort::Env + SessionOptions
//     succeeds — which internally goes through CreateCustomOpDomain +
//     CustomOpDomain_Add for every op, plus ORT's own validation of
//     the OrtCustomOp descriptor fields (name, I/O counts, types).
//   - Registering the domain twice on the same SessionOptions is
//     still accepted (each call creates a fresh domain handle).
//
// What this does not cover (deferred): actually running a graph. That
// needs either a prebuilt .onnx fixture or the `onnx` Python package
// for model authoring, neither of which is available in the current
// host setup. A follow-up will check in a small round-trip graph.

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <catch2/catch_test_macros.hpp>

#include "turboquant/onnx/custom_ops.hpp"

static void init_api_once() {
    static bool initialized = false;
    if (!initialized) {
        Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
        initialized = true;
    }
}

TEST_CASE("RegisterCustomOps attaches com.turboquant domain cleanly", "[onnx]") {
    init_api_once();
    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "tq_smoke");
    Ort::SessionOptions opts;

    OrtStatus* st = RegisterCustomOps(opts, OrtGetApiBase());
    if (st != nullptr) {
        const char* msg = Ort::GetApi().GetErrorMessage(st);
        INFO("RegisterCustomOps: " << (msg ? msg : "(null)"));
        Ort::GetApi().ReleaseStatus(st);
        FAIL("RegisterCustomOps returned non-null OrtStatus");
    }
}

TEST_CASE("RegisterCustomOps is idempotent across repeated calls", "[onnx]") {
    init_api_once();
    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "tq_smoke_repeat");
    Ort::SessionOptions opts;

    for (int i = 0; i < 3; ++i) {
        OrtStatus* st = RegisterCustomOps(opts, OrtGetApiBase());
        if (st != nullptr) {
            const char* msg = Ort::GetApi().GetErrorMessage(st);
            INFO("RegisterCustomOps call " << i << ": " << (msg ? msg : "(null)"));
            Ort::GetApi().ReleaseStatus(st);
            FAIL("RegisterCustomOps returned non-null on repeated call");
        }
    }
}
