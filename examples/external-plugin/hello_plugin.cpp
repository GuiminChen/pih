#include "pih/plugin_sdk/abi.h"

#include <cstring>

namespace {
struct State {
  const pih_host_api_v1* host{};
  unsigned phase{};
} state;
const unsigned hello_value = 42;

pih_status_v1 Status(uint32_t code, const char* message = "") noexcept {
  pih_status_v1 result{};
  result.struct_size = sizeof(result);
  result.abi_version = PIH_STATUS_ABI_VERSION_V1;
  result.code = code;
  std::strncpy(result.message, message, sizeof(result.message) - 1);
  return result;
}

// All callbacks contain exceptions before returning across the C ABI.
template <unsigned Phase>
pih_status_v1 Step(void* context) noexcept {
  if (context != &state || !state.host || state.phase != Phase)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "hello_lifecycle_order");
  try {
    if constexpr (Phase == 0) {
      const pih_capability_v1 capability{
          sizeof(pih_capability_v1), PIH_CAPABILITY_ABI_VERSION_V1,
          "example.hello.v1", "example.hello.v1", &hello_value,
          PIH_CAPABILITY_THREADING_SINGLE_THREADED_V1,
          PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1};
      const auto result = state.host->register_capability(state.host->context, &capability);
      if (!pih_status_is_valid_v1(&result))
        return Status(PIH_STATUS_INTERNAL_V1, "hello_invalid_host_status");
      if (!pih_status_is_ok_v1(&result)) return result;
    }
    if constexpr (Phase == 1) {
      const void* api = nullptr;
      const auto result = state.host->resolve_capability(
          state.host->context, "example.hello.v1", "example.hello.v1",
          PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
          PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1, &api);
      if (!pih_status_is_valid_v1(&result))
        return Status(PIH_STATUS_INTERNAL_V1, "hello_invalid_host_status");
      if (!pih_status_is_ok_v1(&result)) return result;
      if (api != &hello_value)
        return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "hello_binding_mismatch");
    }
    ++state.phase;
    return Status(PIH_STATUS_OK_V1);
  } catch (...) {
    return Status(PIH_STATUS_INTERNAL_V1, "hello_callback_failed");
  }
}

pih_status_v1 Stop(void* context) noexcept {
  // Startup rollback may stop a started plugin before Ready succeeds.
  if (context == &state && state.host && state.phase == 3) {
    state.phase = 6;
    return Status(PIH_STATUS_OK_V1);
  }
  return Step<5>(context);
}
pih_status_v1 Dispose(void* context) noexcept {
  if (context != &state || !state.host ||
      (state.phase != 1 && state.phase != 2 && state.phase != 6))
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "hello_dispose_order");
  state.host = nullptr;
  state.phase = 7;  // One activation per library load.
  return Status(PIH_STATUS_OK_V1);
}
}  // namespace

extern "C" PIH_PLUGIN_EXPORT pih_status_v1 pih_plugin_entry_v1(
    const pih_host_api_v1* host, pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host) || !pih_plugin_api_accepts_v1(plugin))
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "hello_abi_mismatch");
  if (state.host || state.phase != 0)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "hello_already_bound");
  plugin->plugin_id = "example.hello";
  plugin->plugin_version = "0.1.0";
  plugin->context = &state;
  plugin->lifecycle = {sizeof(pih_plugin_lifecycle_v1),
      PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1, Step<0>, Step<1>, Step<2>,
      Step<3>, Step<4>, Stop, Dispose};
  state.host = host;
  return Status(PIH_STATUS_OK_V1);
}
