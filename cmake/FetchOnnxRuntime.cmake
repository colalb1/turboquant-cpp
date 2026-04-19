# Fetches a prebuilt onnxruntime tarball for macOS arm64 (CPU-only).
# Override with -DORT_ROOT=/path/to/onnxruntime-install for a system install.
#
# Produces imported target `onnxruntime::onnxruntime`.

if(DEFINED ORT_ROOT)
    message(STATUS "Using system ONNX Runtime at ${ORT_ROOT}")
    set(_ort_root "${ORT_ROOT}")
else()
    include(FetchContent)
    set(_ort_version "1.17.3")
    set(_ort_url
        "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_version}/onnxruntime-osx-arm64-${_ort_version}.tgz")
    message(STATUS "Fetching ONNX Runtime ${_ort_version} (osx-arm64) from ${_ort_url}")

    FetchContent_Declare(
        onnxruntime_bin
        URL "${_ort_url}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime_bin)
    set(_ort_root "${onnxruntime_bin_SOURCE_DIR}")
endif()

find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    HINTS "${_ort_root}/include"
    NO_DEFAULT_PATH)

find_library(ORT_LIBRARY
    NAMES onnxruntime
    HINTS "${_ort_root}/lib"
    NO_DEFAULT_PATH)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR
        "Failed to locate ONNX Runtime under ${_ort_root}. "
        "Ensure the tarball contains include/ and lib/ subdirectories.")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")

message(STATUS "ONNX Runtime headers: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ORT_LIBRARY}")
