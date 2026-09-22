#include "pih/contracts/memory_host_spill_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/contracts/nvidia_cuda_resources_v1.h"
#include "pih/plugin_sdk/abi.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

namespace {

constexpr uint32_t kPp1TransferReservationWindow = 2U;

struct State final {
  std::mutex mutex;
  uint32_t phase{};
  const pih_host_api_v1* host{};
  const pih_nvidia_cuda_memory_api_v1* memory{};
  const pih_nvidia_cuda_async_api_v1* async{};
  const pih_nvidia_cuda_resources_api_v1* resources{};
  bool plan_compiled{};
  bool enabled{};
  bool ownership_corrupted{};
  uint32_t expert_slot_count{};
  uint32_t staging_extent_count{};
  uint64_t expert_bundle_bytes{};
  uint64_t pinned_budget_bytes{};
  uint64_t device_budget_bytes{};
  uint64_t live_pinned_bytes{};
  uint64_t live_device_bytes{};
  int32_t device_ordinal{-1};
  uintptr_t retained_context{};
  uintptr_t pinned_address{};
  uint64_t pinned_alignment{};
  uint64_t pinned_generation{};
  std::array<uintptr_t, kPp1TransferReservationWindow> device_addresses{};
  std::array<uint64_t, kPp1TransferReservationWindow> device_alignments{};
  std::array<uint64_t, kPp1TransferReservationWindow> device_generations{};
  std::array<uintptr_t, kPp1TransferReservationWindow> completion_events{};
};

void ClearResolvedCapabilities(State& state) noexcept {
  state.memory = nullptr;
  state.async = nullptr;
  state.resources = nullptr;
}

class CapabilityConfigurationTransaction final {
 public:
  explicit CapabilityConfigurationTransaction(State& state) noexcept
      : state_(state) {}
  CapabilityConfigurationTransaction(
      const CapabilityConfigurationTransaction&) = delete;
  CapabilityConfigurationTransaction& operator=(
      const CapabilityConfigurationTransaction&) = delete;
  ~CapabilityConfigurationTransaction() {
    if (!committed_) ClearResolvedCapabilities(state_);
  }
  void Commit() noexcept { committed_ = true; }

 private:
  State& state_;
  bool committed_ = false;
};

pih_status_v1 Status(uint32_t code, const char* message = "") {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  std::strncpy(value.message, message, sizeof(value.message) - 1);
  return value;
}

pih_status_v1 CheckedProviderStatus(pih_status_v1 status,
                                    const char* invalid_message) {
  return pih_status_is_valid_v1(&status)
      ? status
      : Status(PIH_STATUS_INTERNAL_V1, invalid_message);
}

bool ValidAlignment(uint64_t alignment) noexcept {
  return alignment != 0 && (alignment & (alignment - 1)) == 0 &&
         alignment <= 256;
}

bool ValidAllocationOutput(const pih_cuda_allocation_v1* allocation) noexcept {
  return allocation != nullptr &&
         allocation->struct_size == sizeof(*allocation) &&
         allocation->abi_version == PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 &&
         allocation->address == 0 && allocation->bytes == 0 &&
         allocation->generation == 0;
}

bool ValidAllocation(const pih_cuda_allocation_v1& allocation,
                     uint64_t bytes, uint64_t requested_alignment,
                     uint32_t memory_kind, int32_t device_ordinal) noexcept {
  return allocation.struct_size == sizeof(allocation) &&
         allocation.abi_version == PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 &&
         allocation.memory_kind == memory_kind &&
         allocation.device_ordinal == device_ordinal &&
         allocation.address != 0 && allocation.bytes == bytes &&
         allocation.generation != 0 &&
         allocation.alignment != 0 &&
         allocation.alignment >= requested_alignment &&
         (allocation.alignment & (allocation.alignment - 1)) == 0 &&
         allocation.address % allocation.alignment == 0;
}

template <typename Operation>
pih_status_v1 ContainAbi(Operation operation, const char* failure) noexcept {
  try {
    return operation();
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "host_spill_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}

pih_status_v1 Advance(void* context, uint32_t expected) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.phase != expected) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  ++state.phase;
  return Status(PIH_STATUS_OK_V1);
}

State state;

pih_status_v1 CompilePlanCore(void* context, uint32_t enabled,
                          uint32_t expert_slot_count,
                          uint32_t staging_extent_count,
                          uint64_t expert_bundle_bytes,
                          pih_memory_host_spill_plan_v1* plan) {
  if (plan == nullptr || plan->struct_size != sizeof(*plan) ||
      plan->contract_version != PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_plan_invalid");
  }
  *plan = {sizeof(*plan), PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1};
  if (context == nullptr || enabled > 1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_plan_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.phase != 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_provider_not_ready");
  }
  const bool valid = enabled != 0
      ? expert_slot_count == kPp1TransferReservationWindow &&
            staging_extent_count == kPp1TransferReservationWindow &&
            expert_bundle_bytes != 0 &&
            expert_bundle_bytes <=
                std::numeric_limits<uint64_t>::max() /
                    kPp1TransferReservationWindow
      : expert_slot_count == 0 && staging_extent_count == 0 &&
            expert_bundle_bytes == 0;
  if (!valid) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_capacity_invalid");
  }
  if (state.live_pinned_bytes != 0 || state.live_device_bytes != 0 ||
      std::ranges::any_of(state.completion_events,
                          [](uintptr_t event) { return event != 0; })) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_plan_has_live_resources");
  }
  if (state.plan_compiled &&
      (state.enabled != (enabled != 0) ||
       state.expert_slot_count != expert_slot_count ||
       state.staging_extent_count != staging_extent_count ||
       state.expert_bundle_bytes != expert_bundle_bytes)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_plan_already_compiled");
  }
  plan->enabled = enabled;
  plan->expert_slot_count = expert_slot_count;
  plan->staging_extent_count = staging_extent_count;
  plan->transfer_reservation_window =
      enabled != 0 ? kPp1TransferReservationWindow : 0U;
  plan->reserved = 0;
  plan->expert_bundle_bytes = expert_bundle_bytes;
  plan->pinned_bytes = expert_bundle_bytes * staging_extent_count;
  plan->device_bytes = expert_bundle_bytes * expert_slot_count;
  state.plan_compiled = true;
  state.enabled = enabled != 0;
  state.expert_slot_count = expert_slot_count;
  state.staging_extent_count = staging_extent_count;
  state.expert_bundle_bytes = expert_bundle_bytes;
  state.pinned_budget_bytes = plan->pinned_bytes;
  state.device_budget_bytes = plan->device_bytes;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 AllocatePinnedCore(void* context, uint64_t bytes,
                             uint64_t alignment,
                             pih_cuda_allocation_v1* allocation) {
  if (context == nullptr || !ValidAllocationOutput(allocation) ||
      !ValidAlignment(alignment)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_pinned_allocation_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.phase != 4 || state.memory == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_provider_not_ready");
  }
  if (!state.plan_compiled || !state.enabled || bytes == 0 ||
      bytes != state.pinned_budget_bytes || state.live_pinned_bytes != 0 ||
      state.live_pinned_bytes > state.pinned_budget_bytes ||
      bytes > state.pinned_budget_bytes - state.live_pinned_bytes) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "host_spill_pinned_budget_exhausted");
  }
  pih_cuda_allocation_v1 created{
      sizeof(created), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  auto status = CheckedProviderStatus(
      state.memory->allocate_pinned_host(
          state.memory->context, -1, bytes, alignment, &created),
      "host_spill_pinned_provider_status_invalid");
  if (!pih_status_is_ok_v1(&status)) {
    if (!ValidAllocationOutput(&created)) state.ownership_corrupted = true;
    return status;
  }
  const bool aliases_owned_address =
      created.address != 0 &&
      (created.address == state.pinned_address ||
       std::find(state.device_addresses.begin(), state.device_addresses.end(),
                 created.address) != state.device_addresses.end());
  if (aliases_owned_address) {
    state.ownership_corrupted = true;
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_pinned_provider_result_invalid");
  }
  if (!ValidAllocation(created, bytes, alignment,
                       PIH_CUDA_MEMORY_PINNED_HOST_V1, -1)) {
    const auto cleanup = CheckedProviderStatus(
        state.memory->deallocate_pinned_host(state.memory->context, &created),
        "host_spill_pinned_cleanup_status_invalid");
    if (!pih_status_is_ok_v1(&cleanup)) state.ownership_corrupted = true;
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_pinned_provider_result_invalid");
  }
  state.live_pinned_bytes += bytes;
  state.pinned_address = created.address;
  state.pinned_alignment = created.alignment;
  state.pinned_generation = created.generation;
  *allocation = created;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 DeallocatePinnedCore(
    void* context, const pih_cuda_allocation_v1* allocation) {
  if (context == nullptr || allocation == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_pinned_deallocation_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.memory == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_memory_provider_missing");
  }
  if (!ValidAllocation(*allocation, state.pinned_budget_bytes,
                       state.pinned_alignment,
                       PIH_CUDA_MEMORY_PINNED_HOST_V1, -1) ||
      allocation->bytes != state.live_pinned_bytes ||
      allocation->address != state.pinned_address ||
      allocation->alignment != state.pinned_alignment ||
      allocation->generation != state.pinned_generation) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_pinned_allocation_not_owned");
  }
  auto status = CheckedProviderStatus(
      state.memory->deallocate_pinned_host(state.memory->context, allocation),
      "host_spill_pinned_provider_status_invalid");
  if (pih_status_is_ok_v1(&status)) {
    state.live_pinned_bytes -= allocation->bytes;
    state.pinned_address = 0;
    state.pinned_alignment = 0;
    state.pinned_generation = 0;
  }
  return status;
}

pih_status_v1 CopyH2dAsyncCore(void* context, uintptr_t retained_context,
                           uintptr_t destination, uintptr_t source,
                           uint64_t bytes, uintptr_t stream) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_h2d_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.phase != 4 || state.async == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_provider_not_ready");
  }
  if (!state.plan_compiled || !state.enabled ||
      state.retained_context == 0 ||
      retained_context != state.retained_context ||
      bytes != state.expert_bundle_bytes ||
      std::find(state.device_addresses.begin(), state.device_addresses.end(),
                destination) == state.device_addresses.end() ||
      state.pinned_address == 0 ||
      state.pinned_address >
          std::numeric_limits<uintptr_t>::max() -
              state.pinned_budget_bytes ||
      source < state.pinned_address ||
      source > state.pinned_address + state.pinned_budget_bytes - bytes ||
      (source - state.pinned_address) % state.expert_bundle_bytes != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_h2d_allocation_not_owned");
  }
  return CheckedProviderStatus(
      state.async->copy_async(
          state.async->context, retained_context, destination, source, bytes,
          PIH_CUDA_COPY_H2D_V1, stream),
      "host_spill_async_provider_status_invalid");
}

pih_status_v1 RecordEventCore(void* context, uintptr_t retained_context,
                          uintptr_t event, uintptr_t stream) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_event_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.phase != 4 || state.async == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_provider_not_ready");
  }
  const auto owned = std::find(state.completion_events.begin(),
                               state.completion_events.end(), event);
  if (owned == state.completion_events.end() ||
      retained_context != state.retained_context) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_event_not_owned");
  }
  return CheckedProviderStatus(
      state.async->record_event(state.async->context, retained_context,
                                event, stream),
      "host_spill_async_provider_status_invalid");
}

pih_status_v1 QueryEventCore(void* context, uintptr_t retained_context,
                         uintptr_t event, uint32_t* event_status) {
  if (event_status != nullptr) *event_status = 0;
  if (context == nullptr || event_status == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_event_query_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.phase != 4 || state.async == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_provider_not_ready");
  }
  const auto owned = std::find(state.completion_events.begin(),
                               state.completion_events.end(), event);
  if (owned == state.completion_events.end() ||
      retained_context != state.retained_context) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_event_not_owned");
  }
  auto status = CheckedProviderStatus(
      state.async->query_event(state.async->context, retained_context,
                               event, event_status),
      "host_spill_async_provider_status_invalid");
  if (!pih_status_is_ok_v1(&status)) {
    *event_status = 0;
    return status;
  }
  if (*event_status != PIH_CUDA_EVENT_COMPLETE_V1 &&
      *event_status != PIH_CUDA_EVENT_PENDING_V1) {
    *event_status = 0;
    state.ownership_corrupted = true;
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_event_provider_result_invalid");
  }
  return status;
}

pih_status_v1 CreateCompletionEventCore(void* context,
                                    uintptr_t retained_context,
                                    uintptr_t* event) {
  if (context == nullptr || retained_context == 0 || event == nullptr ||
      *event != 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_event_create_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.phase != 4 || state.resources == nullptr ||
      !state.plan_compiled || !state.enabled) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_provider_not_ready");
  }
  const auto empty = std::find(state.completion_events.begin(),
                               state.completion_events.end(), 0);
  if (empty == state.completion_events.end()) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "host_spill_event_table_full");
  }
  if (state.retained_context != 0 &&
      state.retained_context != retained_context) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_context_not_owned");
  }
  uintptr_t created = 0;
  auto status = CheckedProviderStatus(
      state.resources->create_disable_timing_event(
          state.resources->context, retained_context, &created),
      "host_spill_resource_provider_status_invalid");
  if (!pih_status_is_ok_v1(&status)) {
    if (created != 0) state.ownership_corrupted = true;
    return status;
  }
  if (created == 0 ||
      std::find(state.completion_events.begin(),
                state.completion_events.end(), created) !=
          state.completion_events.end()) {
    state.ownership_corrupted = true;
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_event_provider_result_invalid");
  }
  *empty = created;
  state.retained_context = retained_context;
  *event = created;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 DestroyCompletionEventCore(void* context, uintptr_t event) {
  if (context == nullptr || event == 0) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_event_destroy_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.resources == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_provider_missing");
  }
  const auto owned = std::find(state.completion_events.begin(),
                               state.completion_events.end(), event);
  if (owned == state.completion_events.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_event_not_owned");
  }
  auto status = CheckedProviderStatus(
      state.resources->destroy_event(state.resources->context, event),
      "host_spill_resource_provider_status_invalid");
  if (pih_status_is_ok_v1(&status)) {
    *owned = 0;
    if (std::ranges::none_of(state.completion_events,
                             [](uintptr_t value) { return value != 0; })) {
      state.retained_context = 0;
    }
  }
  return status;
}

pih_status_v1 AllocateDeviceSlotCore(void* context, int32_t device_ordinal,
                                 uint64_t bytes, uint64_t alignment,
                                 pih_cuda_allocation_v1* allocation) {
  if (context == nullptr || device_ordinal < 0 ||
      !ValidAllocationOutput(allocation) ||
      !ValidAlignment(alignment)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_device_slot_allocation_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.phase != 4 || state.memory == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_provider_not_ready");
  }
  if (state.device_ordinal >= 0 &&
      state.device_ordinal != device_ordinal) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_device_not_owned");
  }
  if (!state.plan_compiled || !state.enabled || bytes == 0 ||
      bytes != state.expert_bundle_bytes ||
      state.live_device_bytes > state.device_budget_bytes ||
      bytes > state.device_budget_bytes - state.live_device_bytes) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "host_spill_device_budget_exhausted");
  }
  pih_cuda_allocation_v1 created{
      sizeof(created), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  auto status = CheckedProviderStatus(
      state.memory->allocate_device(
          state.memory->context, device_ordinal, bytes, alignment, &created),
      "host_spill_device_provider_status_invalid");
  if (!pih_status_is_ok_v1(&status)) {
    if (!ValidAllocationOutput(&created)) state.ownership_corrupted = true;
    return status;
  }
  const bool aliases_owned_address =
      created.address != 0 &&
      (created.address == state.pinned_address ||
       std::find(state.device_addresses.begin(), state.device_addresses.end(),
                 created.address) != state.device_addresses.end());
  if (aliases_owned_address) {
    state.ownership_corrupted = true;
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_device_provider_result_invalid");
  }
  if (!ValidAllocation(created, bytes, alignment,
                       PIH_CUDA_MEMORY_DEVICE_V1, device_ordinal)) {
    const auto cleanup = CheckedProviderStatus(
        state.memory->deallocate_device(state.memory->context, &created),
        "host_spill_device_cleanup_status_invalid");
    if (!pih_status_is_ok_v1(&cleanup)) state.ownership_corrupted = true;
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_device_provider_result_invalid");
  }
  const auto empty = std::find(state.device_addresses.begin(),
                               state.device_addresses.end(), 0);
  if (empty == state.device_addresses.end()) {
    const auto cleanup = CheckedProviderStatus(
        state.memory->deallocate_device(state.memory->context, &created),
        "host_spill_device_cleanup_status_invalid");
    if (!pih_status_is_ok_v1(&cleanup)) state.ownership_corrupted = true;
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "host_spill_device_slot_table_full");
  }
  const auto index = static_cast<std::size_t>(
      empty - state.device_addresses.begin());
  state.device_addresses[index] = created.address;
  state.device_alignments[index] = created.alignment;
  state.device_generations[index] = created.generation;
  state.device_ordinal = device_ordinal;
  state.live_device_bytes += bytes;
  *allocation = created;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 DeallocateDeviceSlotCore(
    void* context, const pih_cuda_allocation_v1* allocation) {
  if (context == nullptr || allocation == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_device_slot_deallocation_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.memory == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_memory_provider_missing");
  }
  const auto owned = std::find(state.device_addresses.begin(),
                               state.device_addresses.end(),
                               allocation->address);
  if (allocation->memory_kind != PIH_CUDA_MEMORY_DEVICE_V1 ||
      allocation->device_ordinal != state.device_ordinal ||
      allocation->bytes == 0 || allocation->bytes > state.live_device_bytes ||
      owned == state.device_addresses.end()) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_device_allocation_not_owned");
  }
  const auto owned_index = static_cast<std::size_t>(
      owned - state.device_addresses.begin());
  if (!ValidAllocation(*allocation, state.expert_bundle_bytes,
                       state.device_alignments[owned_index],
                       PIH_CUDA_MEMORY_DEVICE_V1,
                       state.device_ordinal) ||
      state.device_alignments[owned_index] != allocation->alignment ||
      state.device_generations[owned_index] != allocation->generation) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_device_allocation_not_owned");
  }
  auto status = CheckedProviderStatus(
      state.memory->deallocate_device(state.memory->context, allocation),
      "host_spill_device_provider_status_invalid");
  if (pih_status_is_ok_v1(&status)) {
    state.live_device_bytes -= allocation->bytes;
    state.device_addresses[owned_index] = 0;
    state.device_alignments[owned_index] = 0;
    state.device_generations[owned_index] = 0;
    if (state.live_device_bytes == 0) state.device_ordinal = -1;
  }
  return status;
}

pih_status_v1 InspectCore(void* context,
                      pih_memory_host_spill_snapshot_v1* snapshot) {
  if (snapshot == nullptr || snapshot->struct_size != sizeof(*snapshot) ||
      snapshot->contract_version != PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_snapshot_invalid");
  }
  *snapshot = {sizeof(*snapshot), PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1};
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_snapshot_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.phase != 4 || !state.plan_compiled) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_plan_not_compiled");
  }
  snapshot->enabled = state.enabled ? 1U : 0U;
  snapshot->device_ordinal = state.device_ordinal;
  snapshot->expert_slot_count = state.expert_slot_count;
  snapshot->live_expert_slot_count = static_cast<uint32_t>(std::count_if(
      state.device_addresses.begin(), state.device_addresses.end(),
      [](uintptr_t address) { return address != 0; }));
  snapshot->staging_extent_count = state.staging_extent_count;
  snapshot->live_staging_extent_count =
      state.expert_bundle_bytes == 0
          ? 0U
          : static_cast<uint32_t>(state.live_pinned_bytes /
                                  state.expert_bundle_bytes);
  snapshot->completion_event_count = static_cast<uint32_t>(std::count_if(
      state.completion_events.begin(), state.completion_events.end(),
      [](uintptr_t event) { return event != 0; }));
  snapshot->pinned_budget_bytes = state.pinned_budget_bytes;
  snapshot->pinned_live_bytes = state.live_pinned_bytes;
  snapshot->device_budget_bytes = state.device_budget_bytes;
  snapshot->device_live_bytes = state.live_device_bytes;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 CompilePlan(void* context, uint32_t enabled,
                          uint32_t expert_slot_count,
                          uint32_t staging_extent_count,
                          uint64_t expert_bundle_bytes,
                          pih_memory_host_spill_plan_v1* plan) noexcept {
  return ContainAbi(
      [&] {
        return CompilePlanCore(context, enabled, expert_slot_count,
                               staging_extent_count, expert_bundle_bytes,
                               plan);
      },
      "host_spill_plan_compilation_failed");
}

pih_status_v1 AllocatePinned(void* context, uint64_t bytes,
                             uint64_t alignment,
                             pih_cuda_allocation_v1* allocation) noexcept {
  return ContainAbi(
      [&] { return AllocatePinnedCore(context, bytes, alignment, allocation); },
      "host_spill_pinned_allocation_failed");
}

pih_status_v1 DeallocatePinned(
    void* context, const pih_cuda_allocation_v1* allocation) noexcept {
  return ContainAbi(
      [&] { return DeallocatePinnedCore(context, allocation); },
      "host_spill_pinned_deallocation_failed");
}

pih_status_v1 CopyH2dAsync(void* context, uintptr_t retained_context,
                           uintptr_t destination, uintptr_t source,
                           uint64_t bytes, uintptr_t stream) noexcept {
  return ContainAbi(
      [&] {
        return CopyH2dAsyncCore(context, retained_context, destination,
                                source, bytes, stream);
      },
      "host_spill_h2d_copy_failed");
}

pih_status_v1 RecordEvent(void* context, uintptr_t retained_context,
                          uintptr_t event, uintptr_t stream) noexcept {
  return ContainAbi(
      [&] { return RecordEventCore(context, retained_context, event, stream); },
      "host_spill_event_record_failed");
}

pih_status_v1 QueryEvent(void* context, uintptr_t retained_context,
                         uintptr_t event, uint32_t* event_status) noexcept {
  return ContainAbi(
      [&] {
        return QueryEventCore(context, retained_context, event, event_status);
      },
      "host_spill_event_query_failed");
}

pih_status_v1 CreateCompletionEvent(void* context,
                                    uintptr_t retained_context,
                                    uintptr_t* event) noexcept {
  return ContainAbi(
      [&] { return CreateCompletionEventCore(context, retained_context, event); },
      "host_spill_event_create_failed");
}

pih_status_v1 DestroyCompletionEvent(void* context,
                                     uintptr_t event) noexcept {
  return ContainAbi(
      [&] { return DestroyCompletionEventCore(context, event); },
      "host_spill_event_destroy_failed");
}

pih_status_v1 AllocateDeviceSlot(
    void* context, int32_t device_ordinal, uint64_t bytes,
    uint64_t alignment, pih_cuda_allocation_v1* allocation) noexcept {
  return ContainAbi(
      [&] {
        return AllocateDeviceSlotCore(context, device_ordinal, bytes,
                                      alignment, allocation);
      },
      "host_spill_device_allocation_failed");
}

pih_status_v1 DeallocateDeviceSlot(
    void* context, const pih_cuda_allocation_v1* allocation) noexcept {
  return ContainAbi(
      [&] { return DeallocateDeviceSlotCore(context, allocation); },
      "host_spill_device_deallocation_failed");
}

pih_status_v1 Inspect(
    void* context, pih_memory_host_spill_snapshot_v1* snapshot) noexcept {
  return ContainAbi([&] { return InspectCore(context, snapshot); },
                    "host_spill_inspection_failed");
}

pih_memory_host_spill_api_v1 spill_api{
    sizeof(pih_memory_host_spill_api_v1),
    PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1, &state, &CompilePlan,
    &AllocatePinned, &DeallocatePinned, &CopyH2dAsync, &RecordEvent,
    &QueryEvent, &CreateCompletionEvent, &DestroyCompletionEvent,
    &AllocateDeviceSlot, &DeallocateDeviceSlot, &Inspect};

pih_status_v1 RegisterCore(void* context) {
  auto& state = *static_cast<State*>(context);
  if (state.host->register_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registry_missing");
  }
  pih_capability_v1 capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  capability.capability_id = "memory.host-spill.v1";
  capability.contract_id = "pih.memory.host-spill.v1";
  capability.api = &spill_api;
  capability.threading_model = PIH_CAPABILITY_THREADING_SERIALIZED_V1;
  capability.scope = PIH_CAPABILITY_SCOPE_ACTIVATION_V1;
  capability.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  const auto status = CheckedProviderStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  return Advance(context, 0);
}

pih_status_v1 ConfigureCore(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.phase != 1) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  CapabilityConfigurationTransaction transaction(state);
  if (state.host->resolve_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_resolver_missing");
  }
  const void* api = nullptr;
  const auto memory_resolved = CheckedProviderStatus(
      state.host->resolve_capability(
          state.host->context, "device.cuda-memory.v1",
          "pih.device.cuda-memory.v1", PIH_CAPABILITY_SCOPE_PROCESS_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api),
      "capability_resolver_status_invalid");
  if (!pih_status_is_ok_v1(&memory_resolved)) return memory_resolved;
  state.memory = static_cast<const pih_nvidia_cuda_memory_api_v1*>(api);
  if (state.memory == nullptr ||
      state.memory->struct_size != sizeof(*state.memory) ||
      state.memory->contract_version !=
          PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      state.memory->context == nullptr ||
      state.memory->allocate_device == nullptr ||
      state.memory->deallocate_device == nullptr ||
      state.memory->allocate_pinned_host == nullptr ||
      state.memory->deallocate_pinned_host == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_memory_capability_invalid");
  }
  api = nullptr;
  const auto async_resolved = CheckedProviderStatus(
      state.host->resolve_capability(
          state.host->context, "device.cuda-async.v1",
          "pih.device.cuda-async.v1", PIH_CAPABILITY_SCOPE_PROCESS_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api),
      "capability_resolver_status_invalid");
  if (!pih_status_is_ok_v1(&async_resolved)) return async_resolved;
  state.async = static_cast<const pih_nvidia_cuda_async_api_v1*>(api);
  if (state.async == nullptr ||
      state.async->struct_size != sizeof(*state.async) ||
      state.async->contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      state.async->context == nullptr || state.async->copy_async == nullptr ||
      state.async->record_event == nullptr ||
      state.async->query_event == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_async_capability_invalid");
  }
  api = nullptr;
  const auto resources_resolved = CheckedProviderStatus(
      state.host->resolve_capability(
          state.host->context, "device.cuda-resources.v1",
          "pih.device.cuda-resources.v1", PIH_CAPABILITY_SCOPE_PROCESS_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api),
      "capability_resolver_status_invalid");
  if (!pih_status_is_ok_v1(&resources_resolved)) return resources_resolved;
  state.resources =
      static_cast<const pih_nvidia_cuda_resources_api_v1*>(api);
  if (state.resources == nullptr ||
      state.resources->struct_size != sizeof(*state.resources) ||
      state.resources->contract_version !=
          PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1 ||
      state.resources->context == nullptr ||
      state.resources->create_disable_timing_event == nullptr ||
      state.resources->destroy_event == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "cuda_resources_capability_invalid");
  }
  const auto advanced = Advance(context, 1);
  if (pih_status_is_ok_v1(&advanced)) transaction.Commit();
  return advanced;
}
pih_status_v1 StartCore(void* context) { return Advance(context, 2); }
pih_status_v1 ReadyCore(void* context) { return Advance(context, 3); }
pih_status_v1 DrainCore(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.live_pinned_bytes != 0 || state.live_device_bytes != 0 ||
      state.device_ordinal != -1) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_allocations_still_live");
  }
  if (std::ranges::any_of(state.completion_events,
                          [](uintptr_t event) { return event != 0; }) ||
      state.retained_context != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_events_still_live");
  }
  return Advance(context, 4);
}
pih_status_v1 StopCore(void* context) {
  if (context == nullptr) return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                                        "lifecycle_context_invalid");
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase == 3) {
    state.phase = 6;
    return Status(PIH_STATUS_OK_V1);
  }
  return Advance(context, 5);
}
pih_status_v1 DisposeCore(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "host_spill_lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase != 1 && state.phase != 2 && state.phase != 6) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  if (state.ownership_corrupted) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_resource_ownership_corrupted");
  }
  if (state.live_pinned_bytes != 0 || state.live_device_bytes != 0 ||
      state.device_ordinal != -1) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_allocations_still_live");
  }
  if (std::ranges::any_of(state.completion_events,
                          [](uintptr_t event) { return event != 0; }) ||
      state.retained_context != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "host_spill_events_still_live");
  }
  ClearResolvedCapabilities(state);
  state.plan_compiled = false;
  state.enabled = false;
  state.ownership_corrupted = false;
  state.expert_slot_count = 0;
  state.staging_extent_count = 0;
  state.expert_bundle_bytes = 0;
  state.pinned_budget_bytes = 0;
  state.device_budget_bytes = 0;
  state.live_pinned_bytes = 0;
  state.live_device_bytes = 0;
  state.device_ordinal = -1;
  state.retained_context = 0;
  state.pinned_address = 0;
  state.pinned_alignment = 0;
  state.pinned_generation = 0;
  state.device_addresses.fill(0);
  state.device_alignments.fill(0);
  state.device_generations.fill(0);
  state.completion_events.fill(0);
  state.host = nullptr;
  state.phase = 7;
  return Status(PIH_STATUS_OK_V1);
}

#define PIH_HOST_SPILL_LIFECYCLE_WRAPPER(name, core, failure) \
  pih_status_v1 name(void* context) noexcept {                 \
    return ContainAbi([&] { return core(context); }, failure); \
  }

PIH_HOST_SPILL_LIFECYCLE_WRAPPER(Register, RegisterCore,
                                 "host_spill_registration_failed")
PIH_HOST_SPILL_LIFECYCLE_WRAPPER(Configure, ConfigureCore,
                                 "host_spill_configuration_failed")
PIH_HOST_SPILL_LIFECYCLE_WRAPPER(Start, StartCore,
                                 "host_spill_start_failed")
PIH_HOST_SPILL_LIFECYCLE_WRAPPER(Ready, ReadyCore,
                                 "host_spill_ready_failed")
PIH_HOST_SPILL_LIFECYCLE_WRAPPER(Drain, DrainCore,
                                 "host_spill_drain_failed")
PIH_HOST_SPILL_LIFECYCLE_WRAPPER(Stop, StopCore,
                                 "host_spill_stop_failed")
PIH_HOST_SPILL_LIFECYCLE_WRAPPER(Dispose, DisposeCore,
                                 "host_spill_dispose_failed")

#undef PIH_HOST_SPILL_LIFECYCLE_WRAPPER

}  // namespace

extern "C" PIH_PLUGIN_EXPORT pih_status_v1 pih_plugin_entry_v1(
    const pih_host_api_v1* host, pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "host_api_invalid");
  }
  if (!pih_plugin_api_accepts_v1(plugin)) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "plugin_api_invalid");
  }
  if (state.host != nullptr || state.phase != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "plugin_entry_already_bound");
  }
  state.host = host;
  plugin->abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  plugin->plugin_id = "pih.memory.host-spill";
  plugin->plugin_version = "1.0.0";
  plugin->context = &state;
  plugin->lifecycle.struct_size = sizeof(pih_plugin_lifecycle_v1);
  plugin->lifecycle.abi_version = PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1;
  plugin->lifecycle.register_plugin = &Register;
  plugin->lifecycle.configure = &Configure;
  plugin->lifecycle.start = &Start;
  plugin->lifecycle.ready = &Ready;
  plugin->lifecycle.drain = &Drain;
  plugin->lifecycle.stop = &Stop;
  plugin->lifecycle.dispose = &Dispose;
  return Status(PIH_STATUS_OK_V1);
}
