#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_CAPABILITY_ABI_VERSION_V1 1U

enum {
  PIH_CAPABILITY_THREADING_SINGLE_THREADED_V1 = 1,
  PIH_CAPABILITY_THREADING_SERIALIZED_V1 = 2,
  PIH_CAPABILITY_THREADING_CONCURRENT_V1 = 3,
};

enum {
  PIH_CAPABILITY_SCOPE_PROCESS_V1 = 1,
  PIH_CAPABILITY_SCOPE_ACTIVATION_V1 = 2,
  PIH_CAPABILITY_SCOPE_ENGINE_V1 = 3,
  PIH_CAPABILITY_SCOPE_REQUEST_V1 = 4,
};

enum {
  PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1 = 1,
  PIH_CAPABILITY_CARDINALITY_ZERO_OR_ONE_V1 = 2,
  PIH_CAPABILITY_CARDINALITY_ONE_OR_MORE_V1 = 3,
  PIH_CAPABILITY_CARDINALITY_MANY_V1 = 4,
};

typedef struct pih_capability_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* capability_id;
  const char* contract_id;
  const void* api;
  uint32_t threading_model;
  uint32_t scope;
  uint32_t cardinality;
} pih_capability_v1;

typedef pih_status_v1 (*pih_register_capability_callback_v1)(
    void* context, const pih_capability_v1* capability);
typedef pih_status_v1 (*pih_resolve_capability_callback_v1)(
    void* context, const char* capability_id, const char* contract_id,
    uint32_t required_scope, uint32_t required_cardinality, const void** api);

#ifdef __cplusplus
}
#endif
