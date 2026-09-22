add_library(pih_test_minimal_plugin SHARED minimal_plugin.cpp)
target_link_libraries(pih_test_minimal_plugin PRIVATE PIH::plugin_sdk)
pih_assert_direct_link_allowlist(
    pih_test_minimal_plugin
    PIH::plugin_sdk
)
target_compile_features(pih_test_minimal_plugin PRIVATE cxx_std_20)
set_target_properties(
    pih_test_minimal_plugin
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
install(
    TARGETS pih_test_minimal_plugin
    LIBRARY DESTINATION lib COMPONENT pih-plugin-testing-minimal
    RUNTIME DESTINATION lib COMPONENT pih-plugin-testing-minimal
)

file(
    GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/pih-minimal-development.lock"
    CONTENT "{\"schema\":\"pih.development-lock.v1\",\"support_status\":\"unsupported\",\"kernel_packs\":[],\"plugins\":[{\"plugin_id\":\"pih.testing.minimal\",\"plugin_version\":\"1.0.0\",\"entrypoint\":\"lib/$<TARGET_FILE_NAME:pih_test_minimal_plugin>\"}],\"capabilities\":[{\"capability_id\":\"testing.minimal.v1\",\"provider_id\":\"pih.testing.minimal\",\"contract_id\":\"pih.testing.minimal.v1\",\"threading_model\":1,\"scope\":2,\"cardinality\":1}]}"
)

if(TARGET pih-worker)
    add_custom_target(
        pih_phase1_smoke
        COMMAND
            "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/lib"
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different
            $<TARGET_FILE:pih_test_minimal_plugin>
            "${CMAKE_BINARY_DIR}/lib"
        COMMAND
            $<TARGET_FILE:pih-worker>
            --lock
            "${CMAKE_BINARY_DIR}/pih-minimal-development.lock"
        DEPENDS pih-worker pih_test_minimal_plugin
        USES_TERMINAL
        VERBATIM
    )
endif()
