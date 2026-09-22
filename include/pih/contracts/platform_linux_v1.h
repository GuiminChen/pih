#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#define PIH_PLATFORM_LINUX_ABI_VERSION_V1 1U

typedef pih_status_v1 (*pih_platform_linux_validate_runtime_v1)(void* context);

typedef struct pih_platform_linux_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_platform_linux_validate_runtime_v1 validate_runtime;
} pih_platform_linux_api_v1;
