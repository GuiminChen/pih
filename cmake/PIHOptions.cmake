option(PIH_ENABLE_CUDA "Build the CUDA backend" OFF)
option(PIH_ENABLE_NCCL "Removed legacy NCCL aggregate (ON is rejected)" OFF)
option(PIH_BUILD_NATIVE_ENGRAM_NCCL "Enable pinned NCCL for the native transport plugin, bootstrap command and V4.1 model adapters" OFF)
option(PIH_BUILD_TESTS "Removed monolithic test aggregate (ON is rejected)" OFF)
option(PIH_BUILD_NATIVE_CONTRACT_TESTS "Build standalone native plugin contract tests without the monolith" OFF)
option(PIH_BUILD_WORKER "Build the native plugin worker" ON)
option(PIH_BUILD_PLUGINS "Build native runtime plugins" ON)
option(PIH_BUILD_MONOLITH "Removed monolithic inference build (ON is rejected)" OFF)
option(PIH_BUILD_TOKENIZER_TOOLS "Build CPU-native text and semantic artifact tools (requires ICU)" OFF)
option(PIH_BUILD_QWEN_ARTIFACT_TOOLS "Build CPU-only native Qwen weight conversion tools" OFF)
option(PIH_BUILD_QWEN_QUALIFICATION_TOOLS "Build separate native Qwen CUDA qualification commands (never installed in production)" OFF)
include("${CMAKE_CURRENT_LIST_DIR}/PIHQwenArchitectures.cmake")
option(PIH_BUILD_ARTIFACT_TOOLS "Build CPU-native offline weight artifact tools" OFF)
set(
    PIH_ENABLED_PLUGINS
    "pih.testing.minimal"
    CACHE STRING
    "Semicolon-separated native plugin IDs to configure, or 'all'"
)
set(
    PIH_DEPLOYMENT_PROFILE
    "phase1-native"
    CACHE STRING
    "Plugin-set contract: phase1-native, deepseek-4090d-pp1, or custom"
)
set_property(
    CACHE PIH_DEPLOYMENT_PROFILE PROPERTY STRINGS
    phase1-native deepseek-4090d-pp1 custom
)
if(NOT PIH_DEPLOYMENT_PROFILE STREQUAL "phase1-native" AND
   NOT PIH_DEPLOYMENT_PROFILE STREQUAL "deepseek-4090d-pp1" AND
   NOT PIH_DEPLOYMENT_PROFILE STREQUAL "custom")
    message(FATAL_ERROR "Unknown PIH_DEPLOYMENT_PROFILE: ${PIH_DEPLOYMENT_PROFILE}")
endif()
set(
    PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256
    "0000000000000000000000000000000000000000000000000000000000000000"
    CACHE STRING
    "Exact DeepSeek V4 Flash 0731 target-generation artifact root"
)
set(
    PIH_DEEPSEEK_4090D_ARTIFACT_DIRECTORY
    ""
    CACHE PATH
    "Existing verified DeepSeek V4 Flash 0731 generation used by the PP1 run target"
)
string(LENGTH "${PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256}" PIH_DEEPSEEK_ROOT_LENGTH)
if(NOT PIH_DEEPSEEK_ROOT_LENGTH EQUAL 64 OR
   NOT PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256 MATCHES "^[0-9a-f]+$")
    message(
        FATAL_ERROR
        "PIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256 must be 64 lowercase hex characters"
    )
endif()
