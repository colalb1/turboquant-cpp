# turboquant_bench — Google Benchmark microbenchmarks over the codec + KV
# cache hot paths. One executable links all bench TUs + benchmark_main.
#
# Bench TUs compile with full exceptions/RTTI (Google Benchmark uses them)
# and link against turboquant_core, which enforces -fno-exceptions on its
# own codec sources.

set(TQ_BENCH_SOURCES
    bench_quantize.cpp
    bench_dequantize.cpp
    bench_prod_score.cpp
    bench_rotation.cpp
    bench_kv_cycle.cpp
)

add_executable(turboquant_bench ${TQ_BENCH_SOURCES})

target_link_libraries(turboquant_bench
    PRIVATE turboquant_core
    PRIVATE benchmark::benchmark_main)

target_compile_options(turboquant_bench PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-gnu-zero-variadic-macro-arguments
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(turboquant_bench PRIVATE
        -march=armv8.5-a
        -mtune=apple-m1
    )
endif()
