add_library(pih_plugin_storage_verified_artifact SHARED entrypoint.cpp)
target_link_libraries(
    pih_plugin_storage_verified_artifact
    PRIVATE
        PIH::plugin_sdk
        pih_storage_verified_artifact_impl
)
pih_assert_direct_link_allowlist(
    pih_plugin_storage_verified_artifact
    PIH::plugin_sdk
    pih_storage_verified_artifact_impl
)
target_compile_features(pih_plugin_storage_verified_artifact PRIVATE cxx_std_20)
set_target_properties(
    pih_plugin_storage_verified_artifact
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
)
install(TARGETS pih_plugin_storage_verified_artifact
        LIBRARY DESTINATION lib COMPONENT pih-plugin-storage-verified-artifact
        RUNTIME DESTINATION lib COMPONENT pih-plugin-storage-verified-artifact)
