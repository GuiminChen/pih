cmake_minimum_required(VERSION 3.26)
project(PIHExternalPlugin LANGUAGES C CXX)
find_package(PIHPluginSDK 0.1.0 EXACT CONFIG REQUIRED)
add_library(sdk_c_header_probe OBJECT sdk_c_header_probe.c)
target_link_libraries(sdk_c_header_probe PRIVATE PIH::plugin_sdk)
set_target_properties(sdk_c_header_probe PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
add_library(hello_plugin SHARED hello_plugin.cpp)
add_dependencies(hello_plugin sdk_c_header_probe)
target_link_libraries(hello_plugin PRIVATE PIH::plugin_sdk)
set_target_properties(hello_plugin PROPERTIES
    PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/$<CONFIG>/hello.lock" CONTENT
"{\"schema\":\"pih.development-lock.v1\",\"support_status\":\"unsupported\",\"kernel_packs\":[],\"plugins\":[{\"plugin_id\":\"example.hello\",\"plugin_version\":\"0.1.0\",\"entrypoint\":\"lib/$<TARGET_FILE_NAME:hello_plugin>\"}],\"capabilities\":[{\"capability_id\":\"example.hello.v1\",\"provider_id\":\"example.hello\",\"contract_id\":\"example.hello.v1\",\"threading_model\":1,\"scope\":2,\"cardinality\":1}]}")
set(PIH_WORKER "" CACHE FILEPATH "Path to the matching pih-worker executable")
if(PIH_WORKER)
    add_custom_target(smoke
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/$<CONFIG>/lib"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:hello_plugin>"
                "${CMAKE_BINARY_DIR}/$<CONFIG>/lib"
        COMMAND "${PIH_WORKER}" --lock "${CMAKE_BINARY_DIR}/$<CONFIG>/hello.lock"
        DEPENDS hello_plugin VERBATIM)
endif()
include(CTest)
if(BUILD_TESTING)
    add_executable(hello_contract_test hello_contract_test.cpp hello_plugin.cpp)
    target_link_libraries(hello_contract_test PRIVATE PIH::plugin_sdk)
    add_test(NAME hello_contract COMMAND hello_contract_test)
endif()
