#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1 1U

typedef pih_status_v1 (*pih_lifecycle_callback_v1)(void* context);

typedef struct pih_plugin_lifecycle_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  pih_lifecycle_callback_v1 register_plugin;
  pih_lifecycle_callback_v1 configure;
  pih_lifecycle_callback_v1 start;
  pih_lifecycle_callback_v1 ready;
  pih_lifecycle_callback_v1 drain;
  pih_lifecycle_callback_v1 stop;
  pih_lifecycle_callback_v1 dispose;
} pih_plugin_lifecycle_v1;

#ifdef __cplusplus
}
#endif
