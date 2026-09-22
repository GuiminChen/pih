#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/capability.h"
#include "pih/plugin_sdk/lifecycle.h"

#if defined(_WIN32)
#define PIH_PLUGIN_EXPORT __declspec(dllexport)
#else
#define PIH_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_PLUGIN_ABI_VERSION_V1 1U

typedef void (*pih_log_callback_v1)(void* context, uint32_t severity,
                                          const char* code,
                                          const char* message);

typedef struct pih_host_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  void* context;
  pih_log_callback_v1 log;
  pih_register_capability_callback_v1 register_capability;
  pih_resolve_capability_callback_v1 resolve_capability;
  uint64_t activation_epoch;
} pih_host_api_v1;

static inline int pih_host_api_is_valid_v1(const pih_host_api_v1* host) {
  return host != 0 && host->struct_size == sizeof(pih_host_api_v1) &&
         host->abi_version == PIH_PLUGIN_ABI_VERSION_V1 &&
         host->context != 0 && host->log != 0 &&
         host->register_capability != 0 && host->resolve_capability != 0 &&
         host->activation_epoch != 0;
}

typedef struct pih_plugin_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* plugin_id;
  const char* plugin_version;
  void* context;
  pih_plugin_lifecycle_v1 lifecycle;
} pih_plugin_api_v1;

static inline int pih_plugin_api_accepts_v1(
    const pih_plugin_api_v1* plugin) {
  return plugin != 0 && plugin->struct_size == sizeof(pih_plugin_api_v1) &&
         plugin->abi_version == PIH_PLUGIN_ABI_VERSION_V1;
}

typedef pih_status_v1 (*pih_plugin_entry_fn_v1)(
    const pih_host_api_v1* host, pih_plugin_api_v1* plugin);

#ifdef __cplusplus
}
#endif
