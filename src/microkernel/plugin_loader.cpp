#include "microkernel/plugin_loader.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "microkernel/verified_native_library.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pih::microkernel {
namespace {

pih_plugin_entry_fn_v1 ResolveEntry(void* handle) {
#if defined(_WIN32)
  const auto symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle),
                                     "pih_plugin_entry_v1");
#else
  void* symbol = dlsym(handle, "pih_plugin_entry_v1");
#endif
  if (symbol == nullptr) {
    throw std::runtime_error("plugin_entry_missing");
  }
  pih_plugin_entry_fn_v1 entry{};
  static_assert(sizeof(entry) == sizeof(symbol));
  std::memcpy(&entry, &symbol, sizeof(entry));
  return entry;
}

bool ValidCanonicalId(const char* value, std::size_t maximum_chars) {
  if (value == nullptr || value[0] == '\0') return false;
  const auto* end = static_cast<const char*>(
      std::memchr(value, '\0', maximum_chars + 1));
  if (end == nullptr || value[0] == '.' || end[-1] == '.') return false;
  char previous = '\0';
  for (auto cursor = value; cursor != end; ++cursor) {
    const char byte = *cursor;
    if (!((byte >= 'a' && byte <= 'z') ||
          (byte >= '0' && byte <= '9') || byte == '.' || byte == '-') ||
        (byte == '.' && previous == '.')) {
      return false;
    }
    previous = byte;
  }
  return true;
}

std::string PluginEntryFailure(const pih_status_v1& status) {
  std::string failure = "plugin_entry_rejected";
  if (!pih_status_is_valid_v1(&status)) {
    failure += ":status_abi_invalid";
    return failure;
  }
  failure += ":status=";
  failure += std::to_string(status.code);
  const auto* message_end = static_cast<const char*>(
      std::memchr(status.message, '\0', sizeof(status.message)));
  const auto message_size = message_end == nullptr
                                ? sizeof(status.message)
                                : static_cast<std::size_t>(message_end -
                                                           status.message);
  if (message_size != 0) {
    failure += ':';
    failure.append(status.message, message_size);
  }
  return failure;
}

void ValidateApi(const pih_plugin_api_v1& api) {
  if (api.struct_size != sizeof(pih_plugin_api_v1) ||
      api.abi_version != PIH_PLUGIN_ABI_VERSION_V1 ||
      !ValidCanonicalId(api.plugin_id, 256) ||
      !ValidCanonicalId(api.plugin_version, 64) || api.context == nullptr) {
    throw std::runtime_error("plugin_api_invalid");
  }
  const auto& lifecycle = api.lifecycle;
  if (lifecycle.struct_size != sizeof(pih_plugin_lifecycle_v1) ||
      lifecycle.abi_version != PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1 ||
      lifecycle.register_plugin == nullptr || lifecycle.configure == nullptr ||
      lifecycle.start == nullptr || lifecycle.ready == nullptr ||
      lifecycle.drain == nullptr || lifecycle.stop == nullptr ||
      lifecycle.dispose == nullptr) {
    throw std::runtime_error("plugin_lifecycle_invalid");
  }
}

}  // namespace

PluginInstance::PluginInstance(void* native_handle, pih_plugin_api_v1 api)
    : native_handle_(native_handle),
      api_(api),
      id_(api.plugin_id),
      version_(api.plugin_version) {}

PluginInstance::PluginInstance(PluginInstance&& other) noexcept
    : native_handle_(std::exchange(other.native_handle_, nullptr)),
      api_(other.api_),
      id_(std::move(other.id_)),
      version_(std::move(other.version_)) {}

namespace {

PluginInstance LoadPluginUnsealed(const std::string& path,
                                  const pih_host_api_v1& host,
                                  std::string_view expected_sha256_hex) {
  if (!pih_host_api_is_valid_v1(&host)) {
    throw std::invalid_argument("plugin_host_api_invalid");
  }
  void* handle = OpenVerifiedNativeLibrary(
      path, expected_sha256_hex, "plugin_load_failed");
  const auto entry = ResolveEntry(handle);
  pih_plugin_api_v1 api{};
  api.struct_size = sizeof(api);
  api.abi_version = PIH_PLUGIN_ABI_VERSION_V1;
  const auto status = entry(&host, &api);
  if (!pih_status_is_ok_v1(&status)) {
    // A mapped native plugin may own TLS, static objects or driver state even
    // when entry negotiation fails. The worker is fail-stop from here, so keep
    // the object mapped until process exit instead of running unload logic.
    throw std::runtime_error(PluginEntryFailure(status));
  }
  ValidateApi(api);
  return PluginInstance(handle, api);
}

}  // namespace

PluginInstance PluginLoader::Load(const std::string& path,
                                  const pih_host_api_v1& host,
                                  std::string_view expected_sha256_hex) {
  if (sealed_) {
    throw std::logic_error("plugin_loader_sealed");
  }
  return LoadPluginUnsealed(path, host, expected_sha256_hex);
}

void PluginLoader::Seal() {
  if (sealed_) {
    throw std::logic_error("plugin_loader_already_sealed");
  }
  sealed_ = true;
}

}  // namespace pih::microkernel
