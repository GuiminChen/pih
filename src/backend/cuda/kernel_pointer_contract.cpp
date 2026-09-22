#include "pih/backend/cuda/kernel_pointer_contract.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace pih {
namespace {

bool known_access(KernelPointerAccess value) {
  switch (value) {
    case KernelPointerAccess::kRead:
    case KernelPointerAccess::kWrite:
    case KernelPointerAccess::kReadWrite:
    case KernelPointerAccess::kAtomic:
      return true;
  }
  return false;
}

bool known_owner(KernelPointerOwnerClass value) {
  switch (value) {
    case KernelPointerOwnerClass::kWeight:
    case KernelPointerOwnerClass::kActivation:
    case KernelPointerOwnerClass::kWorkspace:
    case KernelPointerOwnerClass::kKvState:
      return true;
  }
  return false;
}

}  // namespace

Result<KernelPointerContract> KernelPointerContract::Create(
    std::string contract_id, KernelPointerAccess access,
    KernelPointerOwnerClass owner_class, std::int32_t owning_rank,
    std::int32_t device_index, std::uint64_t required_bytes,
    std::uint64_t required_alignment, std::uint32_t alias_group,
    bool allow_exact_alias) {
  if (contract_id.empty() || contract_id.size() > kMaximumIdBytes) {
    return Status::InvalidArgument("kernel pointer contract id is empty or oversized");
  }
  if (!known_access(access) || !known_owner(owner_class) || owning_rank < 0 ||
      device_index < 0 || required_bytes == 0) {
    return Status::InvalidArgument("kernel pointer contract metadata is invalid");
  }
  if (required_alignment == 0 ||
      (required_alignment & (required_alignment - 1)) != 0 ||
      required_alignment > 4096) {
    return Status::InvalidArgument("kernel pointer contract alignment is invalid");
  }
  if (allow_exact_alias != (alias_group != 0)) {
    return Status::InvalidArgument(
        "exact alias permission requires one explicit nonzero alias group");
  }
  return KernelPointerContract(
      std::move(contract_id), access, owner_class, owning_rank, device_index,
      required_bytes, required_alignment, alias_group, allow_exact_alias);
}

VerifiedDevicePointer::VerifiedDevicePointer(
    std::uint64_t bits, Device device, std::uint64_t generation,
    std::uint64_t required_bytes, std::string_view contract_id,
    KernelPointerAccess access, std::uint32_t alias_group,
    bool allow_exact_alias)
    : bits_(bits),
      device_(device),
      generation_(generation),
      required_bytes_(required_bytes),
      contract_id_size_(static_cast<std::uint8_t>(contract_id.size())),
      access_(access),
      alias_group_(alias_group),
      allow_exact_alias_(allow_exact_alias) {
  std::copy(contract_id.begin(), contract_id.end(), contract_id_.begin());
}

Result<VerifiedDevicePointer> VerifiedDevicePointer::Create(
    const TensorView& view, const KernelPointerContract& contract,
    KernelPointerOwnerClass actual_owner_class, std::int32_t actual_owner_rank,
    std::uint64_t actual_generation) {
  if (view.device().type() != DeviceType::kCuda ||
      view.device().index() != contract.device_index() ||
      actual_owner_rank != contract.owning_rank() ||
      actual_owner_class != contract.owner_class() || actual_generation == 0 ||
      actual_generation != view.generation()) {
    return Status::InvalidArgument(
        "kernel pointer owner, rank, device or generation does not match contract");
  }
  if (contract.required_bytes() > view.byte_span()) {
    return Status::InvalidArgument("kernel pointer required span exceeds tensor view");
  }
  const auto bits = static_cast<std::uint64_t>(
      reinterpret_cast<std::uintptr_t>(view.data()));
  if (bits == 0 || bits % contract.required_alignment() != 0 ||
      contract.required_bytes() >
          std::numeric_limits<std::uint64_t>::max() - bits) {
    return Status::InvalidArgument(
        "kernel pointer is null, misaligned or has an overflowing span");
  }
  return VerifiedDevicePointer(bits, view.device(), actual_generation,
                               contract.required_bytes(), contract.contract_id(),
                               contract.access(), contract.alias_group(),
                               contract.allow_exact_alias());
}

}  // namespace pih
