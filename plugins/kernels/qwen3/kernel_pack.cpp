#include "pih/contracts/qwen_kernels_v1.h"
#include "pih/plugin_sdk/abi.h"
#include "pih/plugin_sdk/kernel_pack.h"
#include <filesystem>
#include <string>
#include <stdexcept>

namespace {
std::string installed_root;
pih_status_v1 BindOrigin(const char* path) noexcept {
  pih_status_v1 result{};
  result.struct_size = sizeof(result); result.abi_version = PIH_STATUS_ABI_VERSION_V1;
  result.code = PIH_STATUS_INVALID_ARGUMENT_V1;
  try {
    if (!path || !std::filesystem::path(path).is_absolute()) return result;
    const auto directory = std::filesystem::canonical(path).parent_path() / PIH_QWEN_PACK_DIRECTORY;
    const auto architecture = directory / PIH_QWEN_PACK_ARCH;
    if (!std::filesystem::is_regular_file(architecture / "qwen_bf16_primitives.cubin") ||
        !std::filesystem::is_regular_file(architecture / "qwen_bf16_primitives.cubin.json")) return result;
    if (!installed_root.empty() && installed_root != directory.string()) return result;
    installed_root = directory.string();
    result.code = PIH_STATUS_OK_V1;
  } catch (...) { result.code = PIH_STATUS_INTERNAL_V1; }
  return result;
}
pih_status_v1 Root(const char** path) noexcept {
  pih_status_v1 result{};
  result.struct_size = sizeof(result); result.abi_version = PIH_STATUS_ABI_VERSION_V1;
  result.code = PIH_STATUS_FAILED_PRECONDITION_V1;
  if (!path) { result.code = PIH_STATUS_INVALID_ARGUMENT_V1; return result; }
  *path = nullptr;
  if (installed_root.empty()) return result;
  *path = installed_root.c_str(); result.code = PIH_STATUS_OK_V1;
  return result;
}
const pih_kernel_pack_identity_v1 identity{sizeof(identity), PIH_KERNEL_PACK_ABI_VERSION_V1,
    PIH_QWEN_PACK_ID, "1.0.0", "pih.qwen-kernels.v1", PIH_QWEN_PACK_ARCH};
const pih_qwen_kernels_api_v1 contract{sizeof(contract), 1, PIH_QWEN_PACK_SM, Root};
const pih_kernel_pack_api_v1 api{sizeof(api), PIH_KERNEL_PACK_ABI_VERSION_V1, &identity, &contract, BindOrigin};
}
extern "C" PIH_PLUGIN_EXPORT const pih_kernel_pack_identity_v1* pih_kernel_pack_identity_v1_get() { return &identity; }
extern "C" PIH_PLUGIN_EXPORT const pih_kernel_pack_api_v1* pih_kernel_pack_api_v1_get() { return &api; }
