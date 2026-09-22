#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/status.h"

#define PIH_NVIDIA_CUDA_ABI_VERSION_V1 1U

typedef pih_status_v1 (*pih_nvidia_cuda_prepare_device_v1)(
    void* context, int32_t device_ordinal, uint32_t expected_sm_major,
    uint32_t expected_sm_minor);

typedef struct pih_nvidia_cuda_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_nvidia_cuda_prepare_device_v1 prepare_device;
} pih_nvidia_cuda_api_v1;
