#include "microkernel/kernel_pack_loader.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "microkernel/verified_native_library.h"
#include "pih/plugin_sdk/kernel_pack.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pih::microkernel {
namespace {

pih_kernel_pack_identity_fn_v1 ResolveIdentity(void* handle) {
#if defined(_WIN32)
  const auto symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle),
                                     PIH_KERNEL_PACK_IDENTITY_SYMBOL_V1);
#else
  void* symbol = dlsym(handle, PIH_KERNEL_PACK_IDENTITY_SYMBOL_V1);
#endif
  if (symbol == nullptr) throw std::runtime_error("kernel_pack_identity_missing");
  pih_kernel_pack_identity_fn_v1 identity{};
  static_assert(sizeof(identity) == sizeof(symbol));
  std::memcpy(&identity, &symbol, sizeof(identity));
  return identity;
}

pih_kernel_pack_api_fn_v1 ResolveApi(void* handle) {
#if defined(_WIN32)
  const auto symbol = GetProcAddress(reinterpret_cast<HMODULE>(handle),
                                     PIH_KERNEL_PACK_API_SYMBOL_V1);
#else
  void* symbol = dlsym(handle, PIH_KERNEL_PACK_API_SYMBOL_V1);
#endif
  if (symbol == nullptr) throw std::runtime_error("kernel_pack_api_missing");
  pih_kernel_pack_api_fn_v1 api{};
  static_assert(sizeof(api) == sizeof(symbol));
  std::memcpy(&api, &symbol, sizeof(api));
  return api;
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

}  // namespace

KernelPackInstance::KernelPackInstance(void* native_handle, std::string id,
                                       std::string version,
                                       std::string pack_abi,
                                       std::string architecture,
                                       const void* contract_api)
    : native_handle_(native_handle),
      id_(std::move(id)),
      version_(std::move(version)),
      pack_abi_(std::move(pack_abi)),
      architecture_(std::move(architecture)), contract_api_(contract_api) {}

KernelPackInstance::KernelPackInstance(KernelPackInstance&& other) noexcept
    : native_handle_(std::exchange(other.native_handle_, nullptr)),
      id_(std::move(other.id_)),
      version_(std::move(other.version_)),
      pack_abi_(std::move(other.pack_abi_)),
      architecture_(std::move(other.architecture_)),
      contract_api_(std::exchange(other.contract_api_, nullptr)) {}

namespace {

KernelPackInstance LoadKernelPackUnsealed(
    const std::string& path, std::string_view expected_sha256_hex) {
  void* handle = OpenVerifiedNativeLibrary(
      path, expected_sha256_hex, "kernel_pack_load_failed");
  const auto get_identity = ResolveIdentity(handle);
  const auto get_api = ResolveApi(handle);
  const auto* identity = get_identity();
  if (identity == nullptr || identity->struct_size != sizeof(*identity) ||
      identity->abi_version != PIH_KERNEL_PACK_ABI_VERSION_V1 ||
      !ValidCanonicalId(identity->pack_id, 256) ||
      !ValidCanonicalId(identity->pack_version, 64) ||
      !ValidCanonicalId(identity->pack_abi, 256) ||
      !ValidCanonicalId(identity->architecture, 64)) {
    throw std::runtime_error("kernel_pack_identity_invalid");
  }
  const auto* api = get_api();
  if (api == nullptr || api->struct_size != sizeof(*api) ||
      api->abi_version != PIH_KERNEL_PACK_ABI_VERSION_V1 ||
      api->identity != identity || api->contract_api == nullptr || api->bind_origin == nullptr) {
    throw std::runtime_error("kernel_pack_api_invalid");
  }
  const auto bound = api->bind_origin(path.c_str());
  if (bound.struct_size != sizeof(pih_status_v1) ||
      bound.abi_version != PIH_STATUS_ABI_VERSION_V1 ||
      bound.code != PIH_STATUS_OK_V1) {
    throw std::runtime_error("kernel_pack_origin_binding_failed");
  }
  // Kernel Packs may have initialized driver/TLS/static state before their
  // exported ABI is validated. All failures are process-fatal, so never run
  // native unload logic; the mapping is reclaimed when the worker exits.
  return KernelPackInstance(handle, identity->pack_id,
                            identity->pack_version, identity->pack_abi,
                            identity->architecture, api->contract_api);
}

}  // namespace

KernelPackInstance KernelPackLoader::Load(
    const std::string& path, std::string_view expected_sha256_hex) {
  if (sealed_) {
    throw std::logic_error("kernel_pack_loader_sealed");
  }
  return LoadKernelPackUnsealed(path, expected_sha256_hex);
}

void KernelPackLoader::Seal() {
  if (sealed_) {
    throw std::logic_error("kernel_pack_loader_already_sealed");
  }
  sealed_ = true;
}

}  // namespace pih::microkernel
