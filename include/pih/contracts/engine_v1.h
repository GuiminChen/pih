#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_ENGINE_ABI_VERSION_V1 1U
#define PIH_ENGINE_HANDLE_GENERATION_BOUND_READ_ONLY_V1 1U
#define PIH_ENGINE_MODEL_ID_CAPACITY_V1 129U

typedef struct pih_engine_create_request_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t activation_epoch;
  const char* deployment_root;
  const char* configuration_json;
  uint64_t configuration_size;
} pih_engine_create_request_v1;

typedef struct pih_engine_handle_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t activation_epoch;
  uint64_t generation;
  void* instance;
  char model_id[PIH_ENGINE_MODEL_ID_CAPACITY_V1];
} pih_engine_handle_v1;

static inline int pih_engine_model_id_is_valid_v1(
    const char model_id[PIH_ENGINE_MODEL_ID_CAPACITY_V1]) {
  if (model_id == 0 || model_id[0] == '\0') return 0;
  int terminated = 0;
  for (uint32_t index = 0; index < PIH_ENGINE_MODEL_ID_CAPACITY_V1; ++index) {
    const char byte = model_id[index];
    if (terminated) {
      if (byte != '\0') return 0;
    } else if (byte == '\0') {
      terminated = 1;
    } else if (!((byte >= 'A' && byte <= 'Z') ||
                 (byte >= 'a' && byte <= 'z') ||
                 (byte >= '0' && byte <= '9') || byte == '.' ||
                 byte == '-' || byte == '_' || byte == '/')) {
      return 0;
    }
  }
  return terminated;
}

static inline int pih_engine_model_id_is_empty_v1(
    const char model_id[PIH_ENGINE_MODEL_ID_CAPACITY_V1]) {
  if (model_id == 0) return 0;
  for (uint32_t index = 0; index < PIH_ENGINE_MODEL_ID_CAPACITY_V1; ++index) {
    if (model_id[index] != '\0') return 0;
  }
  return 1;
}

static inline int pih_engine_handle_is_live_v1(
    const pih_engine_handle_v1* engine) {
  return engine != 0 && engine->struct_size == sizeof(pih_engine_handle_v1) &&
         engine->abi_version == PIH_ENGINE_ABI_VERSION_V1 &&
         engine->activation_epoch != 0 && engine->generation != 0 &&
         engine->instance != 0 &&
         pih_engine_model_id_is_valid_v1(engine->model_id);
}

static inline int pih_engine_handle_is_empty_v1(
    const pih_engine_handle_v1* engine) {
  return engine != 0 && engine->struct_size == sizeof(pih_engine_handle_v1) &&
         engine->abi_version == PIH_ENGINE_ABI_VERSION_V1 &&
         engine->activation_epoch == 0 && engine->generation == 0 &&
         engine->instance == 0 &&
         pih_engine_model_id_is_empty_v1(engine->model_id);
}

typedef pih_status_v1 (*pih_create_engine_v1)(
    void* context, const pih_engine_create_request_v1* request,
    pih_engine_handle_v1* engine);
typedef pih_status_v1 (*pih_engine_read_operation_v1)(
    void* context, const pih_engine_handle_v1* engine);
typedef pih_status_v1 (*pih_destroy_engine_v1)(
    void* context, pih_engine_handle_v1* engine);

#define PIH_ENGINE_SMOKE_TOP_LOGPROBS_MAX_V1 2U

typedef struct pih_engine_logprob_v1 {
  uint32_t token_id;
  float logprob;
} pih_engine_logprob_v1;

typedef struct pih_engine_smoke_request_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t deadline_monotonic_ns;
  uint32_t prompt_count;
  const char* prompt_bytes;
  uint64_t prompt_size;
  uint32_t maximum_completion_tokens;
  float temperature;
  float top_p;
  uint64_t seed;
  uint32_t seed_present;
  uint32_t logprobs_enabled;
  uint32_t top_logprobs_count;
  uint32_t stop_count;
  const char* stop_bytes;
  uint64_t stop_size;
} pih_engine_smoke_request_v1;

typedef struct pih_engine_smoke_result_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t token_id;
  float selected_logprob;
  uint32_t top_logprobs_count;
  pih_engine_logprob_v1
      top_logprobs[PIH_ENGINE_SMOKE_TOP_LOGPROBS_MAX_V1];
  uint32_t finish_reason_length;
} pih_engine_smoke_result_v1;

typedef pih_status_v1 (*pih_execute_engine_smoke_request_v1)(
    void* context, const pih_engine_handle_v1* engine,
    const pih_engine_smoke_request_v1* request,
    pih_engine_smoke_result_v1* result);

typedef struct pih_engine_factory_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  uint32_t handle_generation_semantics;
  void* context;
  pih_create_engine_v1 create;
  pih_engine_read_operation_v1 admit_request;
  pih_execute_engine_smoke_request_v1 execute_smoke_request;
  pih_destroy_engine_v1 destroy;
} pih_engine_factory_api_v1;

#ifdef __cplusplus
}
#endif
