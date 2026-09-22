add_library(
    pih_storage_verified_artifact_impl STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_LIST_DIR}/mapped_file.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controller_file_lease.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dm_verity_snapshot_receipt.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/canonical_extent_writer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/durable_file_sink.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/exclusive_file_publisher.cpp"
)
target_compile_features(pih_storage_verified_artifact_impl PUBLIC cxx_std_20)
set_target_properties(
    pih_storage_verified_artifact_impl
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
    set_target_properties(
        pih_storage_verified_artifact_impl
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )
target_include_directories(
    pih_storage_verified_artifact_impl
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    pih_storage_verified_artifact_impl
    PUBLIC pih_core_primitives
)
pih_assert_direct_link_allowlist(
    pih_storage_verified_artifact_impl
    pih_core_primitives
)
