# Implementation owned by the SM89 Kernel Pack. Included after CUDA admission.
include_guard(GLOBAL)
set(
    PIH_DEEPSEEK_SM89_KERNEL_SOURCES
    plugins/kernels/deepseek-v4-sm89/native/deepseek_attention_dense_primitives.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_compressor_bf16_projection.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_compressor_bf16_store.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_compressor_pooling.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_endpoint.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_expert_accumulate.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_expert_swiglu.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_fp4_gemm.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_fp8_activation_quant.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_fp8_gemm.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_index_score.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_indexer_projection.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_mhc.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_rms_norm.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_route_gather.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_router_bf16_gemm.cu
    plugins/kernels/deepseek-v4-sm89/native/deepseek_sparse_attention.cu
)
pih_assert_deepseek_pp1_source_closure(
    PIH_DEEPSEEK_SM89_KERNEL_SOURCES
)
set(pih_sm89_kernel_files ${PIH_DEEPSEEK_SM89_KERNEL_SOURCES})
list(TRANSFORM pih_sm89_kernel_files PREPEND "${PROJECT_SOURCE_DIR}/")
add_library(
    pih_deepseek_sm89_kernels_impl OBJECT
    ${pih_sm89_kernel_files}
)
set(
    PIH_DEEPSEEK_SM89_KERNEL_CONTRACT_SOURCES
    plugins/kernels/deepseek-v4-sm89/native/deepseek_attention_dense_primitives.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_compressor_bf16_projection.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_compressor_bf16_store.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_compressor_pooling.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_endpoint.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_expert_swiglu.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_fp4_gemm.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_fp8_activation_quant.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_fp8_gemm.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_index_score.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_indexer_projection.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_mhc.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_rms_norm.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_rope_table.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_route_gather.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_router_bf16_gemm.cpp
    plugins/kernels/deepseek-v4-sm89/native/deepseek_sparse_attention.cpp
    src/backend/cuda/cuda_status.cpp
)
set(pih_sm89_contract_files ${PIH_DEEPSEEK_SM89_KERNEL_CONTRACT_SOURCES})
list(TRANSFORM pih_sm89_contract_files PREPEND "${PROJECT_SOURCE_DIR}/")
add_library(
    pih_deepseek_sm89_kernel_contract_impl OBJECT
    ${pih_sm89_contract_files}
)
target_include_directories(
    pih_deepseek_sm89_kernels_impl
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(
    pih_deepseek_sm89_kernel_contract_impl PUBLIC cxx_std_20
)
target_include_directories(
    pih_deepseek_sm89_kernel_contract_impl
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_definitions(
    pih_deepseek_sm89_kernels_impl
    PRIVATE PIH_DEEPSEEK_DSPARK_DISABLED=1
)
target_compile_definitions(
    pih_deepseek_sm89_kernel_contract_impl
    PRIVATE PIH_DEEPSEEK_DSPARK_DISABLED=1
)
target_link_libraries(
    pih_deepseek_sm89_kernel_contract_impl
    PUBLIC CUDA::cudart
)
pih_assert_direct_link_allowlist(
    pih_deepseek_sm89_kernel_contract_impl
    CUDA::cudart
)
target_link_libraries(
    pih_deepseek_sm89_kernels_impl
    PUBLIC CUDA::cuda_driver CUDA::cudart
)
pih_assert_direct_link_allowlist(
    pih_deepseek_sm89_kernels_impl
    CUDA::cuda_driver
    CUDA::cudart
)
set_target_properties(
    pih_deepseek_sm89_kernels_impl
    PROPERTIES
        CUDA_ARCHITECTURES "89-real"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
)
set_target_properties(
    pih_deepseek_sm89_kernel_contract_impl
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

pih_assert_deepseek_pp1_source_closure(PIH_DEEPSEEK_SM89_KERNEL_CONTRACT_SOURCES)
set_target_properties(pih_deepseek_sm89_kernels_impl pih_deepseek_sm89_kernel_contract_impl
    PROPERTIES EXCLUDE_FROM_ALL TRUE CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(pih_deepseek_sm89_kernels_impl PROPERTIES CUDA_VISIBILITY_PRESET hidden)
