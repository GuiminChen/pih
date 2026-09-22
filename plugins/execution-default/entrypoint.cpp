#include "pih/contracts/execution_default_v1.h"
#include "pih/plugin_sdk/abi.h"
#include "controller_provider.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

namespace {

struct State final {
  std::mutex mutex;
  uint32_t phase{};
  const pih_host_api_v1* host{};
  bool capacity_compiled{};
  uint32_t maximum_prefill_chunk_tokens{};
  uint32_t maximum_decode_sequences{};
  uint32_t maximum_verify_sequences{};
  uint32_t speculative_tokens_per_sequence{};
};

State state;

pih_status_v1 Status(uint32_t code, const char* message = "") {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  std::strncpy(value.message, message, sizeof(value.message) - 1);
  return value;
}

pih_status_v1 CheckedHostStatus(pih_status_v1 status,
                                const char* invalid_message) {
  return pih_status_is_valid_v1(&status)
      ? status
      : Status(PIH_STATUS_INTERNAL_V1, invalid_message);
}

template <typename Operation>
pih_status_v1 ContainAbi(Operation operation, const char* failure) noexcept {
  try {
    return operation();
  } catch (const std::bad_alloc&) {
    return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1,
                  "execution_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}

pih_status_v1 Advance(void* context, uint32_t expected) {
  if (context != &state) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "execution_lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.phase != expected) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  ++state.phase;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 CompileCapacityCore(
    void* context, const pih_execution_capacity_request_v1* request,
    pih_execution_capacity_v1* capacity) {
  if (capacity != nullptr && capacity->struct_size == sizeof(*capacity) &&
      capacity->abi_version == PIH_EXECUTION_DEFAULT_ABI_VERSION_V1) {
    *capacity = {};
    capacity->struct_size = sizeof(*capacity);
    capacity->abi_version = PIH_EXECUTION_DEFAULT_ABI_VERSION_V1;
  }
  if (context != &state || request == nullptr || capacity == nullptr ||
      request->struct_size != sizeof(*request) ||
      request->abi_version != PIH_EXECUTION_DEFAULT_ABI_VERSION_V1 ||
      capacity->struct_size != sizeof(*capacity) ||
      capacity->abi_version != PIH_EXECUTION_DEFAULT_ABI_VERSION_V1 ||
      request->maximum_prefill_chunk_tokens == 0 ||
      request->maximum_decode_sequences == 0 ||
      request->maximum_verify_sequences == 0 ||
      request->speculative_tokens_per_sequence > 5) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "execution_capacity_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase != 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "execution_provider_not_ready");
  }
  if (request->speculative_tokens_per_sequence != 0 &&
      request->maximum_verify_sequences >
          std::numeric_limits<uint32_t>::max() /
              request->speculative_tokens_per_sequence) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "execution_capacity_overflow");
  }
  if (state.capacity_compiled &&
      (state.maximum_prefill_chunk_tokens !=
           request->maximum_prefill_chunk_tokens ||
       state.maximum_decode_sequences != request->maximum_decode_sequences ||
       state.maximum_verify_sequences != request->maximum_verify_sequences ||
       state.speculative_tokens_per_sequence !=
           request->speculative_tokens_per_sequence)) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "execution_capacity_already_compiled");
  }
  const auto verify_tokens = request->maximum_verify_sequences *
                             request->speculative_tokens_per_sequence;
  const auto max_pipeline_tokens =
      std::max({request->maximum_prefill_chunk_tokens,
                request->maximum_decode_sequences, verify_tokens});
  const auto maximum_sequences =
      std::max(request->maximum_decode_sequences,
               request->maximum_verify_sequences);
  if (maximum_sequences > max_pipeline_tokens) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "execution_sequence_ceiling_exceeds_token_ceiling");
  }
  capacity->max_pipeline_tokens = max_pipeline_tokens;
  capacity->maximum_sequences = maximum_sequences;
  capacity->expert_tokens = max_pipeline_tokens;
  state.capacity_compiled = true;
  state.maximum_prefill_chunk_tokens =
      request->maximum_prefill_chunk_tokens;
  state.maximum_decode_sequences = request->maximum_decode_sequences;
  state.maximum_verify_sequences = request->maximum_verify_sequences;
  state.speculative_tokens_per_sequence =
      request->speculative_tokens_per_sequence;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 CompileCapacity(
    void* context, const pih_execution_capacity_request_v1* request,
    pih_execution_capacity_v1* capacity) noexcept {
  return ContainAbi(
      [&] { return CompileCapacityCore(context, request, capacity); },
      "execution_capacity_compilation_failed");
}

pih_execution_default_api_v1 execution_api{
    sizeof(pih_execution_default_api_v1), PIH_EXECUTION_DEFAULT_ABI_VERSION_V1,
    &state, &CompileCapacity};

pih_status_v1 RegisterCore(void* context) {
  if (context != &state || state.host == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "execution_registration_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.host->register_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registry_missing");
  }
  pih_capability_v1 capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  capability.capability_id = "execution.default.v1";
  capability.contract_id = "pih.execution.default.v1";
  capability.api = &execution_api;
  capability.threading_model = PIH_CAPABILITY_THREADING_CONCURRENT_V1;
  capability.scope = PIH_CAPABILITY_SCOPE_ACTIVATION_V1;
  capability.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  const auto status = CheckedHostStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  pih_capability_v1 controller{};
  controller.struct_size = sizeof(controller);
  controller.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  controller.capability_id = "execution.controller.v1";
  controller.contract_id = "pih.execution.controller.v1";
  controller.api = pih::execution_plugin::ControllerApi();
  controller.threading_model = PIH_CAPABILITY_THREADING_CONCURRENT_V1;
  controller.scope = PIH_CAPABILITY_SCOPE_ACTIVATION_V1;
  controller.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  const auto controller_status = CheckedHostStatus(
      state.host->register_capability(state.host->context, &controller),
      "controller_registry_status_invalid");
  if (!pih_status_is_ok_v1(&controller_status)) return controller_status;
  return Advance(context, 0);
}

pih_status_v1 ConfigureCore(void* context) { return Advance(context, 1); }
pih_status_v1 StartCore(void* context) { return Advance(context, 2); }
pih_status_v1 ReadyCore(void* context) {
  const auto result = Advance(context, 3);
  if (pih_status_is_ok_v1(&result)) pih::execution_plugin::SetControllerReady(true);
  return result;
}
pih_status_v1 DrainCore(void* context) {
  if (context != &state) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "execution_lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase != 4)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "lifecycle_order_invalid");
  // Close creation before observing handles: the provider uses its own mutex.
  pih::execution_plugin::SetControllerReady(false);
  if (!pih::execution_plugin::ControllerHandlesEmpty())
    return Status(PIH_STATUS_UNAVAILABLE_V1,
                  "controller_handles_not_retired");
  return Advance(context, 4);
}
pih_status_v1 StopCore(void* context) {
  if (context != &state) return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                                        "lifecycle_context_invalid");
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase != 3 && state.phase != 5)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "lifecycle_order_invalid");
  pih::execution_plugin::SetControllerReady(false);
  if (!pih::execution_plugin::ControllerHandlesEmpty())
    return Status(PIH_STATUS_UNAVAILABLE_V1,
                  "controller_handles_not_retired");
  if (state.phase == 3) {
    state.phase = 6;
    return Status(PIH_STATUS_OK_V1);
  }
  return Advance(context, 5);
}
pih_status_v1 DisposeCore(void* context) {
  if (context != &state) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "execution_lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  std::lock_guard lock(state.mutex);
  if (state.phase != 1 && state.phase != 2 && state.phase != 6) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  if (!pih::execution_plugin::ControllerHandlesEmpty())
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "controller_handles_not_retired");
  state.capacity_compiled = false;
  state.maximum_prefill_chunk_tokens = 0;
  state.maximum_decode_sequences = 0;
  state.maximum_verify_sequences = 0;
  state.speculative_tokens_per_sequence = 0;
  state.host = nullptr;
  state.phase = 7;
  return Status(PIH_STATUS_OK_V1);
}

#define PIH_EXECUTION_LIFECYCLE_WRAPPER(name, core, failure) \
  pih_status_v1 name(void* context) noexcept {                \
    return ContainAbi([&] { return core(context); }, failure); \
  }

PIH_EXECUTION_LIFECYCLE_WRAPPER(Register, RegisterCore,
                                "execution_registration_failed")
PIH_EXECUTION_LIFECYCLE_WRAPPER(Configure, ConfigureCore,
                                "execution_configuration_failed")
PIH_EXECUTION_LIFECYCLE_WRAPPER(Start, StartCore,
                                "execution_start_failed")
PIH_EXECUTION_LIFECYCLE_WRAPPER(Ready, ReadyCore,
                                "execution_ready_failed")
PIH_EXECUTION_LIFECYCLE_WRAPPER(Drain, DrainCore,
                                "execution_drain_failed")
PIH_EXECUTION_LIFECYCLE_WRAPPER(Stop, StopCore, "execution_stop_failed")
PIH_EXECUTION_LIFECYCLE_WRAPPER(Dispose, DisposeCore,
                                "execution_dispose_failed")

#undef PIH_EXECUTION_LIFECYCLE_WRAPPER

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
  plugin->plugin_id = "pih.execution.default";
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
