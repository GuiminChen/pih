find_package(GTest CONFIG REQUIRED)
find_package(Threads REQUIRED)
include(GoogleTest)

add_executable(pih_native_bounded_workers_test bounded_workers_test.cpp)
target_compile_features(pih_native_bounded_workers_test PRIVATE cxx_std_20)
target_link_libraries(pih_native_bounded_workers_test PRIVATE Threads::Threads)
add_test(NAME native_bounded_workers COMMAND pih_native_bounded_workers_test)

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
            pih_native_v41_output_tests pih_native_bounded_workers_test)

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

add_executable(pih_native_deepseek_batch_test deepseek_v4_batch_test.cpp
    ../../plugins/model-deepseek-v4-flash/batch_generation.cpp ../../src/core/status.cpp)
target_compile_features(pih_native_deepseek_batch_test PRIVATE cxx_std_20)
target_include_directories(pih_native_deepseek_batch_test PRIVATE ../../include)
target_link_libraries(pih_native_deepseek_batch_test PRIVATE Threads::Threads)
pih_assert_direct_link_allowlist(pih_native_deepseek_batch_test Threads::Threads)
add_test(NAME native_deepseek_batch COMMAND pih_native_deepseek_batch_test)
add_executable(pih_native_deepseek_batch_resources_test deepseek_v4_batch_resources_test.cpp
    ../../plugins/model-deepseek-v4-flash/deepseek_endpoint_device_resources.cpp
    ../../src/core/status.cpp ../../src/core/buffer.cpp ../../src/core/tensor_view.cpp)
target_compile_features(pih_native_deepseek_batch_resources_test PRIVATE cxx_std_20)
target_include_directories(pih_native_deepseek_batch_resources_test PRIVATE ../../include)
add_test(NAME native_deepseek_batch_resources COMMAND pih_native_deepseek_batch_resources_test)
add_dependencies(pih-native-contracts pih_native_deepseek_batch_test pih_native_deepseek_batch_resources_test)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET pih_deepseek_v41_supervisor_request)
    add_executable(pih_native_v41_packed_generation_test
        ../../plugins/model-deepseek-v41/tests/packed_generation_session_test.cpp)
    target_compile_features(pih_native_v41_packed_generation_test PRIVATE cxx_std_20)
    target_include_directories(pih_native_v41_packed_generation_test PRIVATE ../../include)
    target_link_libraries(pih_native_v41_packed_generation_test PRIVATE pih_deepseek_v41_supervisor_request)
    add_test(NAME native_v41_packed_generation COMMAND pih_native_v41_packed_generation_test)
    add_dependencies(pih-native-contracts pih_native_v41_packed_generation_test)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET pih_deepseek_v41_supervisor_request)
    add_executable(pih_native_v41_batch_config_test
        ../../plugins/model-deepseek-v41/tests/batch_supervisor_config_test.cpp)
    target_compile_features(pih_native_v41_batch_config_test PRIVATE cxx_std_20)
    target_include_directories(pih_native_v41_batch_config_test PRIVATE ../../include)
    target_link_libraries(pih_native_v41_batch_config_test PRIVATE pih_deepseek_v41_supervisor_request)
    add_test(NAME native_v41_batch_config COMMAND pih_native_v41_batch_config_test)
    # Imported ICU targets from model/tokenizer directories are not visible here.
    find_package(ICU REQUIRED COMPONENTS uc i18n)
    add_executable(pih_native_v41_batch_ipc_test
        ../../plugins/model-deepseek-v41/tests/batch_ipc_test.cpp)
    target_compile_features(pih_native_v41_batch_ipc_test PRIVATE cxx_std_20)
    target_link_libraries(pih_native_v41_batch_ipc_test PRIVATE pih_core_primitives ICU::i18n)
    add_test(NAME native_v41_batch_ipc COMMAND pih_native_v41_batch_ipc_test)
    add_executable(pih_native_v41_batch_transport_test
        ../../plugins/model-deepseek-v41/tests/packed_generation_transport_test.cpp
        ../../plugins/model-deepseek-v41/packed_generation_session.cpp
        ../../plugins/model-deepseek-v41/rank_collector.cpp
        ../../plugins/model-deepseek-v41/rank_commit.cpp
        ../../plugins/model-deepseek-v41/token_ledger.cpp
        ../../plugins/model-deepseek-v41/token_output.cpp
        ../../plugins/model-deepseek-v41/token_stop.cpp
        ../../plugins/model-deepseek-v41/inference_request.cpp
        ../../plugins/model-deepseek-v41/sampling.cpp
        ../../plugins/execution-default/output_burst_credit_pool.cpp
        ../../src/core/status.cpp)
    target_compile_features(pih_native_v41_batch_transport_test PRIVATE cxx_std_20)
    target_include_directories(pih_native_v41_batch_transport_test PRIVATE ../../include)
    target_link_libraries(pih_native_v41_batch_transport_test PRIVATE ICU::i18n ICU::uc Threads::Threads)
    pih_assert_direct_link_allowlist(pih_native_v41_batch_transport_test ICU::i18n ICU::uc Threads::Threads)
    add_test(NAME native_v41_batch_transport COMMAND pih_native_v41_batch_transport_test)
    add_dependencies(pih-native-contracts pih_native_v41_batch_config_test pih_native_v41_batch_ipc_test pih_native_v41_batch_transport_test)
endif()
foreach(packed_test pih_v41_packed_request_test pih_v41_packed_cache_test)
    if(TARGET ${packed_test})
        add_dependencies(pih-native-contracts ${packed_test})
        if(packed_test STREQUAL "pih_v41_packed_cache_test")
            add_test(NAME ${packed_test} COMMAND ${packed_test}
                "${PROJECT_SOURCE_DIR}/plugins/model-deepseek-v41/tests/frozen_config.json")
        else()
            add_test(NAME ${packed_test} COMMAND ${packed_test})
        endif()
    endif()
endforeach()

add_executable(pih_native_qwen_batch_tests
    ../cpp/model/qwen_batch_generation_test.cpp
    $<TARGET_OBJECTS:pih_execution_default_impl>)
target_compile_features(pih_native_qwen_batch_tests PRIVATE cxx_std_20)
target_include_directories(pih_native_qwen_batch_tests PRIVATE ../../include)
target_link_libraries(pih_native_qwen_batch_tests PRIVATE GTest::gtest_main Threads::Threads
    pih_core_primitives pih_packed_plan_values)
pih_assert_direct_link_allowlist(pih_native_qwen_batch_tests GTest::gtest_main Threads::Threads
    pih_core_primitives pih_packed_plan_values)
gtest_discover_tests(pih_native_qwen_batch_tests DISCOVERY_MODE PRE_TEST)
add_dependencies(pih-native-contracts pih_native_qwen_batch_tests)
