#include "pih/plugin_sdk/abi.h"
#include "pih/contracts/platform_linux_v1.h"

#include <sys/utsname.h>

#include <cstring>
#include <new>

namespace {

struct State final {
  uint32_t phase{};
  const pih_host_api_v1* host{};
};

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
                  "platform_allocation_failed");
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, failure);
  }
}

pih_status_v1 Advance(void* context, uint32_t expected) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "platform_lifecycle_context_invalid");
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

pih_status_v1 ValidateRuntimeCore(void* context) {
  if (context == nullptr) {
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1,
                  "platform_context_invalid");
  }
  const auto& state = *static_cast<State*>(context);
  if (state.phase < 2 || state.phase > 4) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "platform_provider_not_configured");
  }
  struct utsname identity {};
  if (uname(&identity) != 0 || std::strcmp(identity.sysname, "Linux") != 0 ||
      std::strcmp(identity.machine, "x86_64") != 0) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "platform_identity_mismatch");
  }
  return Status(PIH_STATUS_OK_V1);
}

pih_status_v1 ValidateRuntime(void* context) noexcept {
  return ContainAbi([&] { return ValidateRuntimeCore(context); },
                    "platform_runtime_validation_failed");
}

pih_platform_linux_api_v1 platform_api{
    sizeof(pih_platform_linux_api_v1), PIH_PLATFORM_LINUX_ABI_VERSION_V1,
    &state, &ValidateRuntime};

pih_status_v1 RegisterCore(void* context) {
  auto& state = *static_cast<State*>(context);
  if (state.host->register_capability == nullptr) {
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1,
                  "capability_registry_missing");
  }
  pih_capability_v1 capability{};
  capability.struct_size = sizeof(capability);
  capability.abi_version = PIH_CAPABILITY_ABI_VERSION_V1;
  capability.capability_id = "platform.linux.v1";
  capability.contract_id = "pih.platform.linux.v1";
  capability.api = &platform_api;
  capability.threading_model = PIH_CAPABILITY_THREADING_SERIALIZED_V1;
  capability.scope = PIH_CAPABILITY_SCOPE_PROCESS_V1;
  capability.cardinality = PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1;
  const auto status = CheckedHostStatus(
      state.host->register_capability(state.host->context, &capability),
      "capability_registry_status_invalid");
  if (!pih_status_is_ok_v1(&status)) return status;
  return Advance(context, 0);
}

pih_status_v1 ConfigureCore(void* context) { return Advance(context, 1); }
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
                  "platform_lifecycle_context_invalid");
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

#define PIH_PLATFORM_LIFECYCLE_WRAPPER(name, core, failure) \
  pih_status_v1 name(void* context) noexcept {               \
    return ContainAbi([&] { return core(context); }, failure); \
  }

PIH_PLATFORM_LIFECYCLE_WRAPPER(Register, RegisterCore,
                               "platform_registration_failed")
PIH_PLATFORM_LIFECYCLE_WRAPPER(Configure, ConfigureCore,
                               "platform_configuration_failed")
PIH_PLATFORM_LIFECYCLE_WRAPPER(Start, StartCore, "platform_start_failed")
PIH_PLATFORM_LIFECYCLE_WRAPPER(Ready, ReadyCore, "platform_ready_failed")
PIH_PLATFORM_LIFECYCLE_WRAPPER(Drain, DrainCore, "platform_drain_failed")
PIH_PLATFORM_LIFECYCLE_WRAPPER(Stop, StopCore, "platform_stop_failed")
PIH_PLATFORM_LIFECYCLE_WRAPPER(Dispose, DisposeCore,
                               "platform_dispose_failed")

#undef PIH_PLATFORM_LIFECYCLE_WRAPPER

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
  state.host = host;
  plugin->abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  plugin->plugin_id = "pih.platform.linux";
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
