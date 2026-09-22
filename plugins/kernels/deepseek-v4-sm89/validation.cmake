# Host-side launch contracts are shared with the model's plan builders.
# They contain no CUDA calls; the model must not import C++ symbols from a pack.
include_guard(GLOBAL)
set(pih_deepseek_launch_contract_sources
    deepseek_attention_dense_primitives.cpp
    deepseek_compressor_bf16_projection.cpp
    deepseek_compressor_bf16_store.cpp
    deepseek_compressor_pooling.cpp
    deepseek_endpoint.cpp
    deepseek_expert_swiglu.cpp
    deepseek_fp4_gemm.cpp
    deepseek_fp8_activation_quant.cpp
    deepseek_fp8_gemm.cpp
    deepseek_index_score.cpp
    deepseek_indexer_projection.cpp
    deepseek_mhc.cpp
    deepseek_rms_norm.cpp
    deepseek_rope_table.cpp
    deepseek_route_gather.cpp
    deepseek_router_bf16_gemm.cpp
    deepseek_sparse_attention.cpp)
list(TRANSFORM pih_deepseek_launch_contract_sources
    PREPEND "${CMAKE_CURRENT_LIST_DIR}/native/")
add_library(pih_deepseek_launch_contracts STATIC EXCLUDE_FROM_ALL
    ${pih_deepseek_launch_contract_sources})
target_compile_features(pih_deepseek_launch_contracts PRIVATE cxx_std_20)
target_include_directories(pih_deepseek_launch_contracts
    PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_launch_contracts PRIVATE pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_launch_contracts pih_core_primitives)
set_target_properties(pih_deepseek_launch_contracts PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
