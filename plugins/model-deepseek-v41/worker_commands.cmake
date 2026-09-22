# Included after native worker/microkernel targets exist. Keep command creation
# in the root binary directory so documented executable paths do not change.
if(TARGET pih_deepseek_v41_rank_worker_loop AND
   "pih.kernels.deepseek-v41.sm103" IN_LIST PIH_ENABLED_PLUGINS)
    if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "103-real")
        message(FATAL_ERROR "V4.1 rank worker requires exactly 103-real")
    endif()
    add_executable(pih-v41-nccl-bootstrap EXCLUDE_FROM_ALL
        "${PROJECT_SOURCE_DIR}/plugins/transport-nccl/nccl_bootstrap_main.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/nccl_bootstrap_channel.cpp")
    target_include_directories(pih-v41-nccl-bootstrap PRIVATE "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_LIST_DIR}" "${PIH_NATIVE_ENGRAM_NCCL_INCLUDE_DIR}")
    target_compile_features(pih-v41-nccl-bootstrap PRIVATE cxx_std_20)
    target_link_libraries(pih-v41-nccl-bootstrap PRIVATE pih_core_primitives CUDA::cudart "${PIH_NATIVE_ENGRAM_NCCL_LIBRARY}")
    pih_assert_direct_link_allowlist(pih-v41-nccl-bootstrap pih_core_primitives CUDA::cudart "${PIH_NATIVE_ENGRAM_NCCL_LIBRARY}")
    add_executable(pih-v41-rank-worker EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/worker_main.cpp" "${PROJECT_SOURCE_DIR}/src/worker/worker_bootstrap.cpp")
    target_include_directories(pih-v41-rank-worker PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src")
    target_compile_features(pih-v41-rank-worker PRIVATE cxx_std_20)
    target_compile_definitions(pih-v41-rank-worker PRIVATE PIH_V41_COMPILED_SM=103)
    target_link_libraries(pih-v41-rank-worker PRIVATE PIH::microkernel pih_development_lock
        pih_deepseek_v41_rank_worker_loop pih_deepseek_v41_rank_channel
        pih_deepseek_v41_weight_files pih_deepseek_v41_inference_operation
        pih_deepseek_v41_expert_workspace pih_core_primitives)
    pih_assert_direct_link_allowlist(pih-v41-rank-worker PIH::microkernel pih_development_lock
        pih_deepseek_v41_rank_worker_loop pih_deepseek_v41_rank_channel
        pih_deepseek_v41_weight_files pih_deepseek_v41_inference_operation
        pih_deepseek_v41_expert_workspace pih_core_primitives)
endif()
