cmake_minimum_required(VERSION 3.24)
project(turboquant
    VERSION 0.1.0
    DESCRIPTION "C++20 / ONNX port of TurboQuant quantization for Apple Silicon"
    LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Policy / toolchain guardrails
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

option(TURBOQUANT_BUILD_TESTS     "Build unit and parity tests"  ON)
option(TURBOQUANT_BUILD_BENCH     "Build microbenchmarks"         ON)
option(TURBOQUANT_BUILD_ONNX      "Build ORT custom-op dylib"     ON)
option(TURBOQUANT_ENABLE_LTO      "Enable thin LTO in Release"    ON)

# Only supported host: Apple Silicon. Warn if we're elsewhere.
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"))
    message(WARNING
        "turboquant-cpp primary target is macOS arm64. "
        "Current host: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
        "Build may proceed with scalar fallback paths only.")
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

include(cmake/FetchJson.cmake)

if(TURBOQUANT_BUILD_TESTS)
    include(cmake/FetchCatch2.cmake)
endif()

if(TURBOQUANT_BUILD_BENCH)
    include(cmake/FetchBenchmark.cmake)
endif()

if(TURBOQUANT_BUILD_ONNX)
    include(cmake/FetchOnnxRuntime.cmake)
endif()

# Accelerate (BLAS + LAPACK + vDSP + vImage) — macOS only.
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()

# -----------------------------------------------------------------------------
# Generated sources (embedded codebooks)
# -----------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TURBOQUANT_CODEBOOK_SRC_DIR
    "${CMAKE_SOURCE_DIR}/codebooks"
    CACHE PATH "Directory of bundled TurboQuant codebook JSONs")
set(TURBOQUANT_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
set(TURBOQUANT_EMBEDDED_CODEBOOKS "${TURBOQUANT_GENERATED_DIR}/codebook_embedded.cpp")

file(GLOB TURBOQUANT_CODEBOOK_JSONS
    "${TURBOQUANT_CODEBOOK_SRC_DIR}/codebook_d*_b*.json")

file(MAKE_DIRECTORY "${TURBOQUANT_GENERATED_DIR}")

add_custom_command(
    OUTPUT "${TURBOQUANT_EMBEDDED_CODEBOOKS}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tools/codebooks_to_cpp.py"
            --in "${TURBOQUANT_CODEBOOK_SRC_DIR}"
            --out "${TURBOQUANT_EMBEDDED_CODEBOOKS}"
    DEPENDS ${TURBOQUANT_CODEBOOK_JSONS}
            "${CMAKE_SOURCE_DIR}/tools/codebooks_to_cpp.py"
    COMMENT "Generating embedded codebook table from bundled JSONs"
    VERBATIM)

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------

add_library(turboquant_core STATIC
    src/rotation_accelerate.cpp
    src/rotation_scalar.cpp
    src/codebook_loader.cpp
    src/codebook_lloyd_max.cpp
    src/quantizer_mse.cpp
    src/quantizer_prod.cpp
    src/value_quant.cpp
    src/ring_buffer.cpp
    src/compressed_kv_store.cpp
    src/kv_capture.cpp
    src/score.cpp
    "${TURBOQUANT_EMBEDDED_CODEBOOKS}"
)

target_include_directories(turboquant_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        "${TURBOQUANT_GENERATED_DIR}"
)

target_link_libraries(turboquant_core PRIVATE nlohmann_json::nlohmann_json)

if(APPLE)
    target_link_libraries(turboquant_core PUBLIC ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(turboquant_core PRIVATE TURBOQUANT_HAS_ACCELERATE=1)
endif()

# Per CPP_IMPLEMENTATION_RULES.md §15:
#   -O3 -march=armv8.5-a -mtune=apple-m1 -fno-exceptions -fno-rtti -flto=thin -DNDEBUG
target_compile_options(turboquant_core PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<AND:$<CONFIG:Release>,$<BOOL:${TURBOQUANT_ENABLE_LTO}>>:-flto=thin>
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
    -Wall -Wextra -Wpedantic
    -Wno-gnu-zero-variadic-macro-arguments
)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(turboquant_core PRIVATE
        -march=armv8.5-a
        -mtune=apple-m1
    )
endif()

# Codec translation units — exceptions/RTTI off.
set(TURBOQUANT_CODEC_TUS
    src/rotation_accelerate.cpp
    src/rotation_scalar.cpp
    src/codebook_loader.cpp
    src/codebook_lloyd_max.cpp
    src/quantizer_mse.cpp
    src/quantizer_prod.cpp
    src/value_quant.cpp
    src/ring_buffer.cpp
    src/compressed_kv_store.cpp
    src/kv_capture.cpp
    src/score.cpp
)
set_source_files_properties(${TURBOQUANT_CODEC_TUS}
    PROPERTIES COMPILE_OPTIONS "-fno-exceptions;-fno-rtti")

# LTO link flag in Release
if(TURBOQUANT_ENABLE_LTO)
    target_link_options(turboquant_core PUBLIC $<$<CONFIG:Release>:-flto=thin>)
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

if(TURBOQUANT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Bench (stub — filled in M9)
# -----------------------------------------------------------------------------

if(TURBOQUANT_BUILD_BENCH AND EXISTS "${CMAKE_SOURCE_DIR}/bench/CMakeLists.txt")
    add_subdirectory(bench)
endif()

# -----------------------------------------------------------------------------
# ONNX custom-op dylib (stub — filled in M10)
# -----------------------------------------------------------------------------

if(TURBOQUANT_BUILD_ONNX AND EXISTS "${CMAKE_SOURCE_DIR}/onnx/CMakeLists.txt")
    add_subdirectory(onnx)
endif()

message(STATUS "turboquant config:")
message(STATUS "  build type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "  tests:       ${TURBOQUANT_BUILD_TESTS}")
message(STATUS "  bench:       ${TURBOQUANT_BUILD_BENCH}")
message(STATUS "  onnx:        ${TURBOQUANT_BUILD_ONNX}")
message(STATUS "  lto:         ${TURBOQUANT_ENABLE_LTO}")
message(STATUS "  codebook dir: ${TURBOQUANT_CODEBOOK_SRC_DIR}")
