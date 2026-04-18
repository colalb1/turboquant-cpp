# Tests share Catch2 v3 and link to turboquant_core.

add_executable(turboquant_tests
    unit/test_rotation.cpp
    unit/test_codebook.cpp
    unit/test_lloyd_max.cpp
    unit/test_pack_policy.cpp
    unit/test_quantizer_mse.cpp
    unit/test_qjl_signs.cpp
    unit/test_quantizer_prod.cpp
    unit/test_neon_kernels.cpp
    unit/test_value_quant.cpp
)

target_link_libraries(turboquant_tests
    PRIVATE turboquant_core
    PRIVATE Catch2::Catch2WithMain)

target_compile_options(turboquant_tests PRIVATE -Wall -Wextra -Wpedantic)

include(Catch)
catch_discover_tests(turboquant_tests)
