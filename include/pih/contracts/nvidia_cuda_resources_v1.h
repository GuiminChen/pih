#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1 1U
#define PIH_CUDA_CONTEXT_SCHED_YIELD_V1 0x02U

typedef pih_status_v1 (*pih_cuda_retain_primary_context_v1)(
    void* context, int32_t device_ordinal, uint32_t context_flags,
    uintptr_t* retained_context);
typedef pih_status_v1 (*pih_cuda_bind_runtime_v1)(
    void* context, int32_t device_ordinal, uintptr_t retained_context);
typedef pih_status_v1 (*pih_cuda_create_context_resource_v1)(
    void* context, uintptr_t retained_context, uintptr_t* resource);
typedef pih_status_v1 (*pih_cuda_destroy_resource_v1)(
    void* context, uintptr_t resource);
typedef pih_status_v1 (*pih_cuda_release_primary_context_v1)(
    void* context, int32_t device_ordinal, uintptr_t retained_context);

typedef struct pih_nvidia_cuda_resources_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_cuda_retain_primary_context_v1 retain_primary_context;
  pih_cuda_bind_runtime_v1 bind_runtime;
  pih_cuda_create_context_resource_v1 create_nonblocking_stream;
  pih_cuda_create_context_resource_v1 create_disable_timing_event;
  pih_cuda_destroy_resource_v1 destroy_event;
  pih_cuda_destroy_resource_v1 destroy_stream;
  pih_cuda_release_primary_context_v1 release_primary_context;
} pih_nvidia_cuda_resources_api_v1;

#ifdef __cplusplus
}
#endif
