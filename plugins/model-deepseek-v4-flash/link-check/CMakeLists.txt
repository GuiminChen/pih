cmake_minimum_required(VERSION 3.26)
project(DeepSeekPp1LinkCheck LANGUAGES CXX)

# Build the real model DSO and artifact executables without loading providers,
# opening a checkpoint, or requiring a CUDA toolkit. Runtime CUDA operations
# are supplied through capability tables, not unresolved ELF symbols.
get_filename_component(PIH_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(PIH_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(PIH_ENABLE_NCCL OFF CACHE BOOL "" FORCE)
set(PIH_BUILD_NATIVE_ENGRAM_NCCL OFF CACHE BOOL "" FORCE)
set(PIH_BUILD_WORKER OFF CACHE BOOL "" FORCE)
set(PIH_BUILD_PLUGINS OFF CACHE BOOL "" FORCE)
set(PIH_BUILD_NATIVE_CONTRACT_TESTS OFF CACHE BOOL "" FORCE)
set(PIH_BUILD_TOKENIZER_TOOLS OFF CACHE BOOL "" FORCE)
set(PIH_BUILD_QWEN_ARTIFACT_TOOLS OFF CACHE BOOL "" FORCE)
set(PIH_BUILD_QWEN_QUALIFICATION_TOOLS OFF CACHE BOOL "" FORCE)
set(PIH_BUILD_ARTIFACT_TOOLS ON CACHE BOOL "" FORCE)
set(PIH_DEPLOYMENT_PROFILE custom CACHE STRING "" FORCE)
add_subdirectory("${PIH_REPOSITORY_ROOT}" core)

# Owner fragments intentionally resolve shared headers from the PIH root.
set(PROJECT_SOURCE_DIR "${PIH_REPOSITORY_ROOT}")
add_subdirectory("${PIH_REPOSITORY_ROOT}/plugins/model-deepseek-v4-flash" model)
add_custom_target(pih-deepseek-pp1-link-check
    DEPENDS pih_plugin_model_deepseek_v4_flash
            pih-deepseek-artifact-prepare
            pih-deepseek-artifact-verify
            pih-deepseek-generation-store)
