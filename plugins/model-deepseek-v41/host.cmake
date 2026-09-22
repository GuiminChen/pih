# Header admission and placement are CPU work, also needed by offline converters.
# Do not hide this target behind CUDA compiler availability.
add_library(pih_deepseek_v41_engram_hash STATIC EXCLUDE_FROM_ALL
    engram_hash.cpp)
target_compile_features(pih_deepseek_v41_engram_hash PUBLIC cxx_std_20)
target_link_libraries(pih_deepseek_v41_engram_hash PUBLIC pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_engram_hash pih_core_primitives)
set_target_properties(pih_deepseek_v41_engram_hash PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_weight_inventory STATIC EXCLUDE_FROM_ALL
    config.cpp engram_weights.cpp
    weight_inventory.cpp weight_catalog.cpp)
target_include_directories(pih_deepseek_v41_weight_inventory PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_weight_inventory PRIVATE pih_native_model_support pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_weight_inventory pih_native_model_support pih_core_primitives)
target_compile_features(pih_deepseek_v41_weight_inventory PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_weight_inventory PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_partition_copy STATIC EXCLUDE_FROM_ALL
    weight_partition_copy.cpp
    weight_dequantize.cpp
    weight_expert_convert.cpp
    weight_scalar_convert.cpp
    weight_source_wo_a.cpp
    weight_output_layout.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pih_deepseek_v41_partition_copy PRIVATE -fno-fast-math -ffp-contract=off)
endif()
target_compile_features(pih_deepseek_v41_partition_copy PUBLIC cxx_std_20)
target_link_libraries(pih_deepseek_v41_partition_copy PUBLIC pih_core_primitives pih_deepseek_v41_weight_inventory pih_native_model_support)
pih_assert_direct_link_allowlist(pih_deepseek_v41_partition_copy pih_core_primitives pih_deepseek_v41_weight_inventory pih_native_model_support)
add_library(pih_deepseek_v41_sequence_cache STATIC EXCLUDE_FROM_ALL sequence_cache.cpp)
target_include_directories(pih_deepseek_v41_sequence_cache PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_features(pih_deepseek_v41_sequence_cache PRIVATE cxx_std_20)
target_link_libraries(pih_deepseek_v41_sequence_cache PRIVATE pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_sequence_cache pih_core_primitives)
set_target_properties(pih_deepseek_v41_sequence_cache PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_boundary_plan STATIC EXCLUDE_FROM_ALL boundary_plan.cpp)
target_include_directories(pih_deepseek_v41_boundary_plan PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_features(pih_deepseek_v41_boundary_plan PRIVATE cxx_std_20)
target_link_libraries(pih_deepseek_v41_boundary_plan PRIVATE pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_boundary_plan pih_core_primitives)
set_target_properties(pih_deepseek_v41_boundary_plan PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(pih_deepseek_v41_weight_files STATIC EXCLUDE_FROM_ALL
        weight_files.cpp weight_manifest.cpp)
    target_include_directories(pih_deepseek_v41_weight_files PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_features(pih_deepseek_v41_weight_files PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_weight_files PRIVATE pih_deepseek_v41_weight_inventory pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_weight_files pih_deepseek_v41_weight_inventory pih_core_primitives)
    set_target_properties(pih_deepseek_v41_weight_files PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(pih_deepseek_v41_materialize STATIC EXCLUDE_FROM_ALL
        weight_materialize.cpp
        weight_source_inventory.cpp
        weight_source_checkpoint.cpp
        weight_source_file.cpp)
    target_compile_features(pih_deepseek_v41_materialize PUBLIC cxx_std_20)
    target_link_libraries(pih_deepseek_v41_materialize PUBLIC pih_deepseek_v41_weight_files pih_deepseek_v41_partition_copy pih_deepseek_v41_engram_hash)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_materialize pih_deepseek_v41_weight_files pih_deepseek_v41_partition_copy pih_deepseek_v41_engram_hash)
    add_executable(pih-v41-reshard EXCLUDE_FROM_ALL reshard_main.cpp)
    target_compile_features(pih-v41-reshard PRIVATE cxx_std_20)
    target_link_libraries(pih-v41-reshard PRIVATE pih_deepseek_v41_materialize)
    pih_assert_direct_link_allowlist(pih-v41-reshard pih_deepseek_v41_materialize)
    add_executable(pih-v41-convert EXCLUDE_FROM_ALL reshard_main.cpp)
    target_compile_definitions(pih-v41-convert PRIVATE PIH_V41_CONVERT_COMMAND=1)
    target_compile_features(pih-v41-convert PRIVATE cxx_std_20)
    target_link_libraries(pih-v41-convert PRIVATE pih_deepseek_v41_materialize)
    pih_assert_direct_link_allowlist(pih-v41-convert pih_deepseek_v41_materialize)
endif()
# Supervisor code must not acquire CUDA/NCCL dependencies through validation.
add_library(pih_deepseek_v41_sampling_host STATIC EXCLUDE_FROM_ALL sampling.cpp)
target_compile_features(pih_deepseek_v41_sampling_host PUBLIC cxx_std_20)
target_link_libraries(pih_deepseek_v41_sampling_host PRIVATE pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_sampling_host pih_core_primitives)
set_target_properties(pih_deepseek_v41_sampling_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_token_output STATIC EXCLUDE_FROM_ALL token_output.cpp)
target_include_directories(pih_deepseek_v41_token_output PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_token_output PRIVATE pih_output_burst_credits pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_token_output pih_output_burst_credits pih_core_primitives)
target_compile_features(pih_deepseek_v41_token_output PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_token_output PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_token_ledger STATIC EXCLUDE_FROM_ALL token_ledger.cpp
    rank_commit.cpp inference_request.cpp supervisor_output.cpp)
target_include_directories(pih_deepseek_v41_token_ledger PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_token_ledger PRIVATE pih_deepseek_v41_token_output pih_deepseek_v41_token_stop pih_deepseek_v41_sampling_host pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_token_ledger pih_deepseek_v41_token_output pih_deepseek_v41_token_stop pih_deepseek_v41_sampling_host pih_core_primitives)
target_compile_features(pih_deepseek_v41_token_ledger PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_token_ledger PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(pih_deepseek_v41_request_policy STATIC EXCLUDE_FROM_ALL supervisor_request_policy.cpp)
target_compile_features(pih_deepseek_v41_request_policy PUBLIC cxx_std_20)
target_include_directories(pih_deepseek_v41_request_policy PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_request_policy PRIVATE pih_deepseek_v41_token_ledger pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_request_policy pih_deepseek_v41_token_ledger pih_core_primitives)
set_target_properties(pih_deepseek_v41_request_policy PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(pih_deepseek_v41_rank_channel STATIC EXCLUDE_FROM_ALL rank_channel.cpp
      rank_collector.cpp request_channel.cpp
      generation_loop.cpp rank_process_watch.cpp
      generation_session.cpp
      lifecycle_channel.cpp
      rank_socket.cpp
      rank_channels.cpp
      worker_spawn.cpp
      worker_executable.cpp
      worker_environment.cpp
      nccl_bootstrap_channel.cpp
      nccl_bootstrap_broker.cpp
      worker_bootstrap.cpp
      worker_group.cpp
      worker_cgroup.cpp)
  target_include_directories(pih_deepseek_v41_rank_channel PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_compile_features(pih_deepseek_v41_rank_channel PRIVATE cxx_std_20)
  target_link_libraries(pih_deepseek_v41_rank_channel PRIVATE pih_deepseek_v41_token_ledger pih_core_primitives)
  pih_assert_direct_link_allowlist(pih_deepseek_v41_rank_channel pih_deepseek_v41_token_ledger pih_core_primitives)
    set_target_properties(pih_deepseek_v41_rank_channel PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(TARGET pih_deepseek_v41_tokenizer)
        add_library(pih_deepseek_v41_supervisor_request STATIC EXCLUDE_FROM_ALL
            supervisor_request.cpp
            supervisor_deployment.cpp
            supervisor_config.cpp
            supervisor_operation.cpp)
        target_compile_features(pih_deepseek_v41_supervisor_request PUBLIC cxx_std_20)
        set_target_properties(pih_deepseek_v41_supervisor_request PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(pih_deepseek_v41_supervisor_request PRIVATE
            pih_deepseek_v41_tokenizer pih_deepseek_v41_rank_channel pih_deepseek_v41_request_policy pih_core_primitives)
        pih_assert_direct_link_allowlist(pih_deepseek_v41_supervisor_request
            pih_deepseek_v41_tokenizer pih_deepseek_v41_rank_channel pih_deepseek_v41_request_policy pih_core_primitives)
        add_executable(pih-v41-supervisor EXCLUDE_FROM_ALL supervisor_main.cpp)
        target_compile_features(pih-v41-supervisor PRIVATE cxx_std_20)
        target_link_libraries(pih-v41-supervisor PRIVATE pih_deepseek_v41_supervisor_request
            pih_deepseek_v41_tokenizer pih_deepseek_v41_rank_channel pih_core_primitives)
        pih_assert_direct_link_allowlist(pih-v41-supervisor pih_deepseek_v41_supervisor_request
            pih_deepseek_v41_tokenizer pih_deepseek_v41_rank_channel pih_core_primitives)
    endif()
endif()
add_library(pih_deepseek_v41_token_stop STATIC EXCLUDE_FROM_ALL token_stop.cpp)
target_include_directories(pih_deepseek_v41_token_stop PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pih_deepseek_v41_token_stop PRIVATE pih_deepseek_v41_sampling_host pih_core_primitives)
pih_assert_direct_link_allowlist(pih_deepseek_v41_token_stop pih_deepseek_v41_sampling_host pih_core_primitives)
target_compile_features(pih_deepseek_v41_token_stop PRIVATE cxx_std_20)
set_target_properties(pih_deepseek_v41_token_stop PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Preserve documented CLI locations when target ownership moves to this directory.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(command pih-v41-reshard pih-v41-convert pih-v41-supervisor)
        if(TARGET ${command})
            set_target_properties(${command} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
        endif()
    endforeach()
    if(TARGET pih-v41-reshard AND TARGET pih-v41-convert AND TARGET pih-v41-token-map)
        add_custom_target(pih-v41-artifact-link-check)
        add_dependencies(pih-v41-artifact-link-check
            pih-v41-reshard pih-v41-convert pih-v41-token-map)
    endif()
endif()
