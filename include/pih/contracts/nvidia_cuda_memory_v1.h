#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 1U

enum pih_cuda_memory_kind_v1 {
  PIH_CUDA_MEMORY_DEVICE_V1 = 1,
  PIH_CUDA_MEMORY_PINNED_HOST_V1 = 2,
};

typedef struct pih_cuda_allocation_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uintptr_t address;
  uint64_t bytes;
  uint64_t alignment;
  uint64_t generation;
  int32_t device_ordinal;
  uint32_t memory_kind;
} pih_cuda_allocation_v1;

typedef pih_status_v1 (*pih_cuda_allocate_v1)(
    void* context, int32_t device_ordinal, uint64_t bytes,
    uint64_t alignment, pih_cuda_allocation_v1* allocation);
typedef pih_status_v1 (*pih_cuda_deallocate_v1)(
    void* context, const pih_cuda_allocation_v1* allocation);
typedef pih_status_v1 (*pih_cuda_copy_h2d_v1)(
    void* context, int32_t device_ordinal, uintptr_t destination,
    const void* source, uint64_t bytes);

typedef struct pih_nvidia_cuda_memory_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_cuda_allocate_v1 allocate_device;
  pih_cuda_deallocate_v1 deallocate_device;
  pih_cuda_allocate_v1 allocate_pinned_host;
  pih_cuda_deallocate_v1 deallocate_pinned_host;
  pih_cuda_copy_h2d_v1 copy_h2d;
} pih_nvidia_cuda_memory_api_v1;

#ifdef __cplusplus
}
#endif
