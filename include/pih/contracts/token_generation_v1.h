#pragma once

#include "pih/contracts/engine_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_TOKEN_GENERATION_ABI_V1 1U
#define PIH_TOKEN_FINISH_STOP_V1 1U
#define PIH_TOKEN_FINISH_LENGTH_V1 2U

/* Token-native generation: no tokenizer, chat formatting, or Python bridge.
 * All buffers are caller-owned and must remain live for this synchronous call.
 * Calls are serialized with create/destroy on the same engine. A failed call
 * publishes no partial result. Capacity must cover maximum_completion_tokens;
 * retrying a request after a resource/internal error is not safe. */
typedef struct pih_token_generation_request_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  const uint32_t* prompt_tokens;
  uint32_t prompt_token_count;
  uint32_t maximum_completion_tokens;
  uint32_t minimum_completion_tokens;
  float temperature;
  float top_p;
  uint64_t seed;
  uint64_t deadline_monotonic_ns;
  const uint32_t* stop_tokens;
  uint32_t stop_token_count;
} pih_token_generation_request_v1;

typedef struct pih_token_generation_result_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  uint32_t* tokens;
  uint32_t token_capacity;
  uint32_t token_count;
  uint32_t finish_reason;
  uint64_t prompt_token_count;
  uint64_t completion_token_count;
} pih_token_generation_result_v1;

typedef struct pih_token_generation_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_status_v1 (*generate)(void* context, const pih_engine_handle_v1* engine,
      const pih_token_generation_request_v1* request,
      pih_token_generation_result_v1* result);
} pih_token_generation_api_v1;

#ifdef __cplusplus
}
#endif
