find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# No PIH::core, GPU backend, model engine or Python extension dependency.
add_executable(pih_native_http_framing_tests
    ../cpp/plugin/native_http_framing_test.cpp
    ../../plugins/surface-openai-http/http_request_framing.cpp)
target_compile_features(pih_native_http_framing_tests PRIVATE cxx_std_20)
target_link_libraries(pih_native_http_framing_tests PRIVATE GTest::gtest_main)
pih_assert_direct_link_allowlist(pih_native_http_framing_tests GTest::gtest_main)
# Building must not execute the test binary, including test discovery.
gtest_discover_tests(pih_native_http_framing_tests DISCOVERY_MODE PRE_TEST)

add_executable(pih_native_qwen_text_stop_tests
    ../cpp/model/qwen_native_text_stop_test.cpp
    ../../src/core/bounded_json.cpp
    ../../src/core/status.cpp)
target_compile_features(pih_native_qwen_text_stop_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_qwen_text_stop_tests PRIVATE ../../include)
target_link_libraries(pih_native_qwen_text_stop_tests PRIVATE GTest::gtest_main)
pih_assert_direct_link_allowlist(pih_native_qwen_text_stop_tests GTest::gtest_main)
gtest_discover_tests(pih_native_qwen_text_stop_tests DISCOVERY_MODE PRE_TEST)

add_executable(pih_native_v41_output_tests
    ../cpp/plugin/native_v41_output_test.cpp
    ../../plugins/model-deepseek-v41/token_output.cpp
    ../../plugins/execution-default/output_burst_credit_pool.cpp
    ../../src/core/status.cpp)
target_compile_features(pih_native_v41_output_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_v41_output_tests PRIVATE ../../include)
target_link_libraries(pih_native_v41_output_tests PRIVATE GTest::gtest_main)
pih_assert_direct_link_allowlist(pih_native_v41_output_tests GTest::gtest_main)
gtest_discover_tests(pih_native_v41_output_tests DISCOVERY_MODE PRE_TEST)

add_custom_target(pih-native-contracts
    DEPENDS pih_native_http_framing_tests pih_native_qwen_text_stop_tests
            pih_native_v41_output_tests)

add_executable(pih_native_execution_context_tests
    ../cpp/plugin/native_execution_context_test.cpp
    ../../plugins/execution-default/controller_provider.cpp
    $<TARGET_OBJECTS:pih_execution_default_impl>)
target_compile_features(pih_native_execution_context_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_execution_context_tests PRIVATE ../../include)
target_link_libraries(pih_native_execution_context_tests PRIVATE
    GTest::gtest_main pih_core_primitives pih_packed_plan_values)
pih_assert_direct_link_allowlist(pih_native_execution_context_tests
    GTest::gtest_main pih_core_primitives pih_packed_plan_values)
gtest_discover_tests(pih_native_execution_context_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_execution_context_tests)

add_executable(pih_native_qwen_execution_binding_tests
    ../cpp/plugin/native_qwen_execution_binding_test.cpp)
target_compile_features(pih_native_qwen_execution_binding_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_qwen_execution_binding_tests PRIVATE ../../include)
target_link_libraries(pih_native_qwen_execution_binding_tests PRIVATE GTest::gtest_main)
pih_assert_direct_link_allowlist(pih_native_qwen_execution_binding_tests GTest::gtest_main)
gtest_discover_tests(pih_native_qwen_execution_binding_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_qwen_execution_binding_tests)

add_executable(pih_native_qwen_memory_binding_tests
    ../cpp/plugin/native_qwen_memory_binding_test.cpp ../../src/core/status.cpp)
target_compile_features(pih_native_qwen_memory_binding_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_qwen_memory_binding_tests PRIVATE ../../include)
target_link_libraries(pih_native_qwen_memory_binding_tests PRIVATE GTest::gtest_main)
pih_assert_direct_link_allowlist(pih_native_qwen_memory_binding_tests GTest::gtest_main)
gtest_discover_tests(pih_native_qwen_memory_binding_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_qwen_memory_binding_tests)

add_executable(pih_native_qwen_resource_binding_tests
    ../cpp/plugin/native_qwen_resource_binding_test.cpp
    ../../src/backend/cuda/cuda_runtime_resources.cpp ../../src/core/status.cpp)
target_compile_features(pih_native_qwen_resource_binding_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_qwen_resource_binding_tests PRIVATE ../../include)
target_link_libraries(pih_native_qwen_resource_binding_tests PRIVATE GTest::gtest_main)
pih_assert_direct_link_allowlist(pih_native_qwen_resource_binding_tests GTest::gtest_main)
gtest_discover_tests(pih_native_qwen_resource_binding_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_qwen_resource_binding_tests)

add_executable(pih_native_qwen_async_binding_tests
    ../cpp/plugin/native_qwen_async_binding_test.cpp ../../src/core/status.cpp
    ../../src/core/tensor_view.cpp
    ../../plugins/model-qwen3/nvidia_qwen3_bf16_kv_scrub_driver.cpp)
target_compile_features(pih_native_qwen_async_binding_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_qwen_async_binding_tests PRIVATE ../../include)
target_link_libraries(pih_native_qwen_async_binding_tests PRIVATE GTest::gtest_main)
pih_assert_direct_link_allowlist(pih_native_qwen_async_binding_tests GTest::gtest_main)
gtest_discover_tests(pih_native_qwen_async_binding_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_qwen_async_binding_tests)

add_executable(pih_native_deepseek_artifact_epoch_tests
    ../cpp/model/deepseek_artifact_epoch_guard_test.cpp
    ../../plugins/model-deepseek-v4-flash/deepseek_artifact_epoch_guard.cpp
    ../../src/core/status.cpp)
target_compile_features(pih_native_deepseek_artifact_epoch_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_deepseek_artifact_epoch_tests PRIVATE ../../include)
target_link_libraries(pih_native_deepseek_artifact_epoch_tests PRIVATE GTest::gtest_main)
pih_assert_direct_link_allowlist(pih_native_deepseek_artifact_epoch_tests GTest::gtest_main)
gtest_discover_tests(pih_native_deepseek_artifact_epoch_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_deepseek_artifact_epoch_tests)

add_executable(pih_native_v41_supervisor_policy_tests
    ../cpp/plugin/native_v41_supervisor_policy_test.cpp)
target_compile_features(pih_native_v41_supervisor_policy_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_v41_supervisor_policy_tests PRIVATE ../../include)
target_link_libraries(pih_native_v41_supervisor_policy_tests PRIVATE
    GTest::gtest_main pih_deepseek_v41_request_policy)
pih_assert_direct_link_allowlist(pih_native_v41_supervisor_policy_tests
    GTest::gtest_main pih_deepseek_v41_request_policy)
gtest_discover_tests(pih_native_v41_supervisor_policy_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_v41_supervisor_policy_tests)

add_executable(pih_native_v41_supervisor_output_tests
    ../cpp/plugin/native_v41_supervisor_output_test.cpp)
target_compile_features(pih_native_v41_supervisor_output_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_v41_supervisor_output_tests PRIVATE ../../include)
target_link_libraries(pih_native_v41_supervisor_output_tests PRIVATE
    GTest::gtest_main pih_deepseek_v41_token_ledger)
pih_assert_direct_link_allowlist(pih_native_v41_supervisor_output_tests
    GTest::gtest_main pih_deepseek_v41_token_ledger)
gtest_discover_tests(pih_native_v41_supervisor_output_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_v41_supervisor_output_tests)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(pih_native_text_output_contract_tests
        ../cpp/plugin/native_text_output_contract_test.cpp
        ../../plugins/surface-openai-http/http_request_framing.cpp)
    target_compile_features(pih_native_text_output_contract_tests PRIVATE cxx_std_20)
    target_include_directories(pih_native_text_output_contract_tests PRIVATE ../../include)
    target_link_libraries(pih_native_text_output_contract_tests PRIVATE GTest::gtest_main pih_core_primitives)
    pih_assert_direct_link_allowlist(pih_native_text_output_contract_tests GTest::gtest_main pih_core_primitives)
    gtest_discover_tests(pih_native_text_output_contract_tests DISCOVERY_MODE PRE_TEST)
    add_dependencies(pih-native-contracts pih_native_text_output_contract_tests)
endif()
