if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(pih_platform_linux_impl OBJECT
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_process_driver.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_artifact_transfer_controller_operations.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_artifact_transfer_receiver_operations.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_artifact_metadata_transfer_controller_operations.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_spawn_authority_probe.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_post_exec_resource_probe.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_post_mapping_resource_operations.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_materialization_operations.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_artifact_prefault_operations.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_spawn_controller.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_worker_handshake_operations.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_worker_bootstrap_runtime.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_deepseek_rank_worker_startup.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_cgroup_v2_generation_probe.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_allocation_lease_probe.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_cgroup_v2_generation_controller.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_termination_signal_source.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_supervisor_shutdown_channel_driver.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_supervisor_shutdown_server_driver.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_pidfd_reap_operations.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_artifact_descriptor_stat_backend.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_shm_namespace_stat_backend.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_listener_kernel_backend.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_engine_inet_diag_namespace_handle_probe.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_runtime_profile_reference_lease_probe.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/linux_runtime_profile_authority_lease_probe.cpp"
    )
    target_compile_features(pih_platform_linux_impl PUBLIC cxx_std_20)
    target_include_directories(
        pih_platform_linux_impl
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    )
    set_target_properties(
        pih_platform_linux_impl PROPERTIES POSITION_INDEPENDENT_CODE ON
    )
        set_target_properties(
            pih_platform_linux_impl PROPERTIES EXCLUDE_FROM_ALL TRUE
        )
endif()
