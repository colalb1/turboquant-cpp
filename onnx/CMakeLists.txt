# libturboquant_onnx.dylib — ORT custom-op shared library.
#
# Only RegisterCustomOps is a public symbol; everything else has hidden
# visibility. ORT's C++ helper header (onnxruntime_cxx_api.h) uses
# exceptions internally, so we do NOT pass -fno-exceptions here — but we
# keep the turboquant_core link which has them off for its codec TUs.

set(TQ_ONNX_SOURCES
    src/op_registry.cpp
    src/shared_kernel_state.cpp
    src/op_mse_quantize.cpp
    src/op_mse_dequantize.cpp
    src/op_prod_quantize.cpp
    src/op_prod_dequantize.cpp
    src/op_prod_attention_score.cpp
    src/op_value_quantize.cpp
    src/op_value_dequantize.cpp
)

add_library(turboquant_onnx SHARED ${TQ_ONNX_SOURCES})

target_include_directories(turboquant_onnx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/onnx/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/onnx/src
)

target_link_libraries(turboquant_onnx
    PRIVATE turboquant_core
    PRIVATE onnxruntime::onnxruntime
)

target_compile_options(turboquant_onnx PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-gnu-zero-variadic-macro-arguments
)

set_target_properties(turboquant_onnx PROPERTIES
    OUTPUT_NAME   turboquant_onnx
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Re-export only RegisterCustomOps. Everything else stays hidden even
# though ORT's CustomOpBase generates heavy template metadata.
if(APPLE)
    set(_exports_file "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols.txt")
    target_link_options(turboquant_onnx PRIVATE
        "-Wl,-exported_symbols_list,${_exports_file}")
    set_property(TARGET turboquant_onnx APPEND PROPERTY LINK_DEPENDS
        "${_exports_file}")
endif()

if(TURBOQUANT_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.cpp")
    add_executable(turboquant_onnx_smoke tests/smoke.cpp)
    target_link_libraries(turboquant_onnx_smoke
        PRIVATE turboquant_onnx
        PRIVATE turboquant_core
        PRIVATE onnxruntime::onnxruntime
        PRIVATE Catch2::Catch2WithMain)
    target_compile_options(turboquant_onnx_smoke PRIVATE
        -Wall -Wextra -Wpedantic)
    include(Catch)
    catch_discover_tests(turboquant_onnx_smoke)
endif()
