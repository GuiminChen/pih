#include "pih/backend/cuda/kernel_signature_manifest.h"

#include <algorithm>
#include <cctype>
#include <unordered_set>
#include <utility>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool is_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isdigit(character) != 0 ||
                  (character >= static_cast<unsigned char>('a') &&
                   character <= static_cast<unsigned char>('f'));
         });
}

}  // namespace

Result<std::uint32_t> kernel_wire_size(KernelWireType type) {
  switch (type) {
    case KernelWireType::kU8:
      return 1;
    case KernelWireType::kU16:
    case KernelWireType::kFloat16Bits:
    case KernelWireType::kBFloat16Bits:
      return 2;
    case KernelWireType::kU32:
    case KernelWireType::kI32:
    case KernelWireType::kFloat32:
      return 4;
    case KernelWireType::kU64:
    case KernelWireType::kI64:
    case KernelWireType::kFloat64:
    case KernelWireType::kDevicePointerU64:
      return 8;
  }
  return Status::InvalidArgument("unknown kernel wire type");
}

Result<std::uint32_t> kernel_wire_alignment(KernelWireType type) {
  return kernel_wire_size(type);
}

Result<KernelSignatureManifest> KernelSignatureManifest::Create(
    std::string logical_id, std::string selected_cubin_sha256,
    std::string parameter_abi_sha256,
    std::span<const KernelParameterSpec> parameters,
    std::uint32_t total_device_parameter_bytes) {
  if (logical_id.empty() || logical_id.size() > kMaximumLogicalIdBytes) {
    return Status::InvalidArgument("kernel logical id is empty or oversized");
  }
  if (!is_sha256(selected_cubin_sha256) || !is_sha256(parameter_abi_sha256)) {
    return Status::InvalidArgument("kernel identities must be lowercase SHA-256");
  }
  if (parameters.empty() || parameters.size() > kMaximumParameters) {
    return Status::InvalidArgument("kernel parameter count is outside manifest bounds");
  }
  if (total_device_parameter_bytes == 0 ||
      total_device_parameter_bytes > kMaximumDeviceParameterBytes) {
    return Status::InvalidArgument("kernel device parameter bytes exceed ABI bounds");
  }

  std::unordered_set<std::string_view> roles;
  roles.reserve(parameters.size());
  std::uint64_t previous_end = 0;
  for (const auto& parameter : parameters) {
    if (parameter.role.empty() || parameter.role.size() > kMaximumRoleBytes ||
        !roles.insert(parameter.role).second) {
      return Status::InvalidArgument("kernel parameter roles must be bounded and unique");
    }
    if (parameter.contract_id.empty() ||
        parameter.contract_id.size() > kMaximumContractIdBytes) {
      return Status::InvalidArgument(
          "kernel parameter requires a bounded scalar or pointer contract id");
    }
    auto size = kernel_wire_size(parameter.wire_type);
    auto alignment = kernel_wire_alignment(parameter.wire_type);
    if (!size.ok()) return size.status();
    if (!alignment.ok()) return alignment.status();
    if (parameter.device_layout_offset % alignment.value() != 0) {
      return Status::InvalidArgument("kernel parameter offset is misaligned");
    }
    if (parameter.device_layout_offset < previous_end) {
      return Status::InvalidArgument("kernel parameter offsets overlap or reorder");
    }
    auto end = checked_add_u64(parameter.device_layout_offset, size.value());
    if (!end.ok()) return end.status();
    if (end.value() > total_device_parameter_bytes) {
      return Status::InvalidArgument("kernel parameter exceeds declared layout bytes");
    }
    previous_end = end.value();
  }

  return KernelSignatureManifest(
      std::move(logical_id), std::move(selected_cubin_sha256),
      std::move(parameter_abi_sha256),
      std::vector<KernelParameterSpec>(parameters.begin(), parameters.end()),
      total_device_parameter_bytes);
}

}  // namespace pih
