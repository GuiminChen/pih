# Build the existing DeepSeek CUDA operations for Hopper as an independent
# binary closure. The source lists are explicit in the SM89 owner and contain
# no architecture-specific source substitution.
include_guard(GLOBAL)
if(NOT DEFINED PIH_DEEPSEEK_SM89_KERNEL_SOURCES OR
   NOT DEFINED PIH_DEEPSEEK_SM89_KERNEL_CONTRACT_SOURCES)
    message(FATAL_ERROR "DeepSeek SM90 requires the shared native kernel source inventory")
endif()
set(pih_sm90_kernel_files ${PIH_DEEPSEEK_SM89_KERNEL_SOURCES})
list(TRANSFORM pih_sm90_kernel_files PREPEND "${PROJECT_SOURCE_DIR}/")
add_library(pih_deepseek_sm90_kernels_impl OBJECT ${pih_sm90_kernel_files})
target_include_directories(pih_deepseek_sm90_kernels_impl PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_sm90_kernels_impl PUBLIC CUDA::cuda_driver CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_sm90_kernels_impl CUDA::cuda_driver CUDA::cudart)
target_compile_definitions(pih_deepseek_sm90_kernels_impl PRIVATE PIH_DEEPSEEK_DSPARK_DISABLED=1)
set_target_properties(pih_deepseek_sm90_kernels_impl PROPERTIES
    CUDA_ARCHITECTURES "90-real" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON EXCLUDE_FROM_ALL TRUE
    CUDA_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

set(pih_sm90_contract_files ${PIH_DEEPSEEK_SM89_KERNEL_CONTRACT_SOURCES})
list(TRANSFORM pih_sm90_contract_files PREPEND "${PROJECT_SOURCE_DIR}/")
add_library(pih_deepseek_sm90_kernel_contract_impl OBJECT ${pih_sm90_contract_files})
target_include_directories(pih_deepseek_sm90_kernel_contract_impl PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(pih_deepseek_sm90_kernel_contract_impl PUBLIC cxx_std_20)
target_link_libraries(pih_deepseek_sm90_kernel_contract_impl PUBLIC CUDA::cudart)
pih_assert_direct_link_allowlist(pih_deepseek_sm90_kernel_contract_impl CUDA::cudart)
target_compile_definitions(pih_deepseek_sm90_kernel_contract_impl PRIVATE PIH_DEEPSEEK_DSPARK_DISABLED=1)
set_target_properties(pih_deepseek_sm90_kernel_contract_impl PROPERTIES
    POSITION_INDEPENDENT_CODE ON EXCLUDE_FROM_ALL TRUE
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
