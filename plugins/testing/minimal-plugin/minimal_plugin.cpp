#include "pih/plugin_sdk/abi.h"

#include <cstring>
#include <new>

namespace {

struct State {
  uint32_t phase{};
  const pih_host_api_v1* host{};
};

uint32_t capability_marker = 0x50494831U;

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
                  "minimal_plugin_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}

pih_status_v1 Advance(void* context, uint32_t expected) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.phase != expected) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  ++state.phase;
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 RegisterCore(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.host == nullptr || state.host->register_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registry_missing");
  }
  pih_capability_v1 capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  capability.capability_id = "testing.minimal.v1";
  capability.contract_id = "pih.testing.minimal.v1";
  capability.api = &capability_marker;
  capability.threading_model =
      PIH_CAPABILITY_THREADING_SINGLE_THREADED_V1;
  capability.scope = PIH_CAPABILITY_SCOPE_ACTIVATION_V1;
  capability.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  const auto status = CheckedHostStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  return Advance(context, 0);
}
pih_status_v1 ConfigureCore(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.host == nullptr || state.host->resolve_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_resolver_missing");
  }
  const void* api = nullptr;
  const auto resolved = CheckedHostStatus(
      state.host->resolve_capability(
          state.host->context, "testing.minimal.v1",
          "pih.testing.minimal.v1", PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api),
      "capability_resolver_status_invalid");
  if (!pih_status_is_ok_v1(&resolved)) return resolved;
  if (api != &capability_marker) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_binding_mismatch");
  }
  return Advance(context, 1);
}
pih_status_v1 StartCore(void* context) { return Advance(context, 2); }
pih_status_v1 ReadyCore(void* context) { return Advance(context, 3); }
pih_status_v1 DrainCore(void* context) { return Advance(context, 4); }
pih_status_v1 StopCore(void* context) {
  if (context == nullptr) return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                                        "lifecycle_context_invalid");
  auto& state = *static_cast<State*>(context);
  if (state.phase == 3) {
    state.phase = 6;
    return Status(PIH_STATUS_OK_V1);
  }
  return Advance(context, 5);
}
pih_status_v1 DisposeCore(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "lifecycle_context_invalid");
  }
  auto& state = *static_cast<State*>(context);
  if (state.phase != 1 && state.phase != 2 && state.phase != 6) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "lifecycle_order_invalid");
  }
  state.host = nullptr;
  state.phase = 7;
  return Status(PIH_STATUS_OK_V1);
}

#define PIH_MINIMAL_LIFECYCLE_WRAPPER(name, core, failure) \
  pih_status_v1 name(void* context) noexcept {               \
    return ContainAbi([&] { return core(context); }, failure); \
  }

PIH_MINIMAL_LIFECYCLE_WRAPPER(Register, RegisterCore,
                              "minimal_registration_failed")
PIH_MINIMAL_LIFECYCLE_WRAPPER(Configure, ConfigureCore,
                              "minimal_configuration_failed")
PIH_MINIMAL_LIFECYCLE_WRAPPER(Start, StartCore, "minimal_start_failed")
PIH_MINIMAL_LIFECYCLE_WRAPPER(Ready, ReadyCore, "minimal_ready_failed")
PIH_MINIMAL_LIFECYCLE_WRAPPER(Drain, DrainCore, "minimal_drain_failed")
PIH_MINIMAL_LIFECYCLE_WRAPPER(Stop, StopCore, "minimal_stop_failed")
PIH_MINIMAL_LIFECYCLE_WRAPPER(Dispose, DisposeCore,
                              "minimal_dispose_failed")

#undef PIH_MINIMAL_LIFECYCLE_WRAPPER

State state;

}  // namespace

extern "C" PIH_PLUGIN_EXPORT pih_status_v1
pih_plugin_entry_v1(const pih_host_api_v1* host,
                    pih_plugin_api_v1* plugin) noexcept {
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
  plugin->abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  plugin->plugin_id = "pih.testing.minimal";
  plugin->plugin_version = "1.0.0";
  state.host = host;
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
