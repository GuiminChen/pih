#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#define PIH_EXECUTION_DEFAULT_ABI_VERSION_V1 1U

typedef struct pih_execution_capacity_request_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t maximum_prefill_chunk_tokens;
  uint32_t maximum_decode_sequences;
  uint32_t maximum_verify_sequences;
  uint32_t speculative_tokens_per_sequence;
} pih_execution_capacity_request_v1;

typedef struct pih_execution_capacity_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t max_pipeline_tokens;
  uint32_t maximum_sequences;
  uint32_t expert_tokens;
} pih_execution_capacity_v1;

typedef pih_status_v1 (*pih_execution_compile_capacity_v1)(
    void* context, const pih_execution_capacity_request_v1* request,
    pih_execution_capacity_v1* capacity);

typedef struct pih_execution_default_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_execution_compile_capacity_v1 compile_capacity;
} pih_execution_default_api_v1;
