#pragma once

#include <stdint.h>

#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include "pih/plugin_sdk/status.h"

#define PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 1U

typedef struct pih_memory_host_spill_plan_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  uint32_t enabled;
  uint32_t expert_slot_count;
  uint32_t staging_extent_count;
  uint32_t transfer_reservation_window;
  uint32_t reserved;
  uint64_t expert_bundle_bytes;
  uint64_t pinned_bytes;
  uint64_t device_bytes;
} pih_memory_host_spill_plan_v1;

typedef struct pih_memory_host_spill_snapshot_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  uint32_t enabled;
  int32_t device_ordinal;
  uint32_t expert_slot_count;
  uint32_t live_expert_slot_count;
  uint32_t staging_extent_count;
  uint32_t live_staging_extent_count;
  uint32_t completion_event_count;
  uint64_t pinned_budget_bytes;
  uint64_t pinned_live_bytes;
  uint64_t device_budget_bytes;
  uint64_t device_live_bytes;
} pih_memory_host_spill_snapshot_v1;

typedef pih_status_v1 (*pih_memory_host_spill_compile_plan_v1)(
    void* context, uint32_t enabled, uint32_t expert_slot_count,
    uint32_t staging_extent_count, uint64_t expert_bundle_bytes,
    pih_memory_host_spill_plan_v1* plan);
typedef pih_status_v1 (*pih_memory_host_spill_allocate_pinned_v1)(
    void* context, uint64_t bytes, uint64_t alignment,
    pih_cuda_allocation_v1* allocation);
typedef pih_status_v1 (*pih_memory_host_spill_deallocate_pinned_v1)(
    void* context, const pih_cuda_allocation_v1* allocation);
typedef pih_status_v1 (*pih_memory_host_spill_copy_h2d_async_v1)(
    void* context, uintptr_t retained_context, uintptr_t destination,
    uintptr_t source, uint64_t bytes, uintptr_t stream);
typedef pih_status_v1 (*pih_memory_host_spill_record_event_v1)(
    void* context, uintptr_t retained_context, uintptr_t event,
    uintptr_t stream);
typedef pih_status_v1 (*pih_memory_host_spill_query_event_v1)(
    void* context, uintptr_t retained_context, uintptr_t event,
    uint32_t* event_status);
typedef pih_status_v1 (*pih_memory_host_spill_create_event_v1)(
    void* context, uintptr_t retained_context, uintptr_t* event);
typedef pih_status_v1 (*pih_memory_host_spill_destroy_event_v1)(
    void* context, uintptr_t event);
typedef pih_status_v1 (*pih_memory_host_spill_allocate_device_slot_v1)(
    void* context, int32_t device_ordinal, uint64_t bytes,
    uint64_t alignment, pih_cuda_allocation_v1* allocation);
typedef pih_status_v1 (*pih_memory_host_spill_deallocate_device_slot_v1)(
    void* context, const pih_cuda_allocation_v1* allocation);
typedef pih_status_v1 (*pih_memory_host_spill_inspect_v1)(
    void* context, pih_memory_host_spill_snapshot_v1* snapshot);

typedef struct pih_memory_host_spill_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  void* context;
  pih_memory_host_spill_compile_plan_v1 compile_plan;
  pih_memory_host_spill_allocate_pinned_v1 allocate_pinned;
  pih_memory_host_spill_deallocate_pinned_v1 deallocate_pinned;
  pih_memory_host_spill_copy_h2d_async_v1 copy_h2d_async;
  pih_memory_host_spill_record_event_v1 record_event;
  pih_memory_host_spill_query_event_v1 query_event;
  pih_memory_host_spill_create_event_v1 create_completion_event;
  pih_memory_host_spill_destroy_event_v1 destroy_completion_event;
  pih_memory_host_spill_allocate_device_slot_v1 allocate_device_slot;
  pih_memory_host_spill_deallocate_device_slot_v1 deallocate_device_slot;
  pih_memory_host_spill_inspect_v1 inspect;
} pih_memory_host_spill_api_v1;
