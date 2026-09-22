# Model-owned native host/offline targets. This is not registration of a model
# execution capability. Never link the vendored Python reference.
# Host supervision conditionally consumes the tokenizer target below.
include(tokenizer.cmake)
include(host.cmake)

if(PIH_BUILD_ARTIFACT_TOOLS)
    add_library(pih_deepseek_v41_config STATIC config.cpp attention_step.cpp engram_hash.cpp engram_weights.cpp engram_launch.cpp)
    target_compile_features(pih_deepseek_v41_config PRIVATE cxx_std_20)
    target_link_libraries(pih_deepseek_v41_config PRIVATE pih_native_model_support)
    pih_assert_direct_link_allowlist(pih_deepseek_v41_config pih_native_model_support)
endif()
