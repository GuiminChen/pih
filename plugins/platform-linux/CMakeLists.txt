add_library(
    pih_plugin_platform_linux SHARED
    entrypoint.cpp
)
target_link_libraries(
    pih_plugin_platform_linux
    PRIVATE PIH::plugin_sdk
)
pih_assert_direct_link_allowlist(
    pih_plugin_platform_linux
    PIH::plugin_sdk
)
target_compile_features(pih_plugin_platform_linux PRIVATE cxx_std_20)
set_target_properties(
    pih_plugin_platform_linux
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
)
install(TARGETS pih_plugin_platform_linux
        LIBRARY DESTINATION lib COMPONENT pih-plugin-platform-linux
        RUNTIME DESTINATION lib COMPONENT pih-plugin-platform-linux)
