#include "pih/plugin_sdk/abi.h"
#include <iostream>
#include <stdexcept>

extern "C" pih_status_v1 pih_plugin_entry_v1(
    const pih_host_api_v1*, pih_plugin_api_v1*) noexcept;

namespace {
const void* registered = nullptr;
bool reject_registration = true;
bool throw_resolution = false;
pih_status_v1 Result(uint32_t code) {
  return {sizeof(pih_status_v1), PIH_STATUS_ABI_VERSION_V1, code, {}};
}
void Log(void*, uint32_t, const char*, const char*) {}
pih_status_v1 Register(void*, const pih_capability_v1* capability) {
  if (reject_registration) return Result(PIH_STATUS_RESOURCE_EXHAUSTED_V1);
  registered = capability->api;
  return Result(PIH_STATUS_OK_V1);
}
pih_status_v1 Resolve(void*, const char*, const char*, uint32_t, uint32_t,
                      const void** result) {
  if (throw_resolution) throw std::runtime_error("host failed");
  *result = registered;
  return Result(PIH_STATUS_OK_V1);
}
void Expect(pih_status_v1 result, uint32_t code) {
  if (!pih_status_is_valid_v1(&result) || result.code != code)
    throw std::runtime_error("unexpected status");
}
}
int main() {
  try {
    int context = 0;
    pih_host_api_v1 host{sizeof(host), PIH_PLUGIN_ABI_VERSION_V1,
                         &context, Log, Register, Resolve, 1};
    pih_plugin_api_v1 plugin{};
    plugin.struct_size = sizeof(plugin);
    plugin.abi_version = PIH_PLUGIN_ABI_VERSION_V1;
    Expect(pih_plugin_entry_v1(nullptr, &plugin), PIH_STATUS_INVALID_ARGUMENT_V1);
    host.abi_version = 999;
    Expect(pih_plugin_entry_v1(&host, &plugin), PIH_STATUS_INVALID_ARGUMENT_V1);
    host.abi_version = PIH_PLUGIN_ABI_VERSION_V1;
    Expect(pih_plugin_entry_v1(&host, &plugin), PIH_STATUS_OK_V1);
    Expect(pih_plugin_entry_v1(&host, &plugin), PIH_STATUS_FAILED_PRECONDITION_V1);
    auto& ops = plugin.lifecycle;
    Expect(ops.start(plugin.context), PIH_STATUS_FAILED_PRECONDITION_V1);
    Expect(ops.register_plugin(plugin.context), PIH_STATUS_RESOURCE_EXHAUSTED_V1);
    reject_registration = false;
    Expect(ops.register_plugin(plugin.context), PIH_STATUS_OK_V1);
    throw_resolution = true;
    Expect(ops.configure(plugin.context), PIH_STATUS_INTERNAL_V1);
    throw_resolution = false;
    Expect(ops.configure(plugin.context), PIH_STATUS_OK_V1);
    Expect(ops.start(plugin.context), PIH_STATUS_OK_V1);
    Expect(ops.ready(plugin.context), PIH_STATUS_OK_V1);
    Expect(ops.drain(plugin.context), PIH_STATUS_OK_V1);
    Expect(ops.stop(plugin.context), PIH_STATUS_OK_V1);
    Expect(ops.dispose(plugin.context), PIH_STATUS_OK_V1);
    Expect(ops.start(plugin.context), PIH_STATUS_FAILED_PRECONDITION_V1);
    std::cout << "external plugin contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
