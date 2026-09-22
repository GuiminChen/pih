#include "pih/backend/cuda/kernel_argument_packet.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace pih {

Result<KernelArgumentPacket> KernelArgumentPacket::Create(
    const KernelSignatureManifest& signature,
    std::span<const KernelPointerContract> pointer_contracts) {
  std::vector<std::string_view> required_contract_ids;
  for (const auto& parameter : signature.parameters()) {
    if (parameter.wire_type == KernelWireType::kDevicePointerU64) {
      required_contract_ids.push_back(parameter.contract_id);
    }
  }
  if (required_contract_ids.size() != pointer_contracts.size()) {
    return Status::InvalidArgument(
        "kernel pointer contract table does not cover signature pointers");
  }
  for (const auto required_id : required_contract_ids) {
    const auto matches = std::count_if(
        pointer_contracts.begin(), pointer_contracts.end(),
        [required_id](const KernelPointerContract& contract) {
          return contract.contract_id() == required_id;
        });
    if (matches != 1) {
      return Status::InvalidArgument(
          "kernel pointer contract id is missing or duplicated");
    }
  }
  return KernelArgumentPacket(
      std::string(signature.logical_id()),
      std::string(signature.parameter_abi_sha256()),
      std::vector<KernelParameterSpec>(signature.parameters().begin(),
                                       signature.parameters().end()),
      signature.total_device_parameter_bytes(),
      std::vector<KernelPointerContract>(pointer_contracts.begin(),
                                         pointer_contracts.end()));
}

KernelArgumentPacket::KernelArgumentPacket(
    std::string logical_id, std::string parameter_abi_sha256,
    std::vector<KernelParameterSpec> parameters, std::uint32_t total_bytes,
    std::vector<KernelPointerContract> contracts)
    : logical_id_(std::move(logical_id)),
      parameter_abi_sha256_(std::move(parameter_abi_sha256)),
      parameters_(std::move(parameters)),
      cells_(total_bytes),
      kernel_params_(parameters_.size()),
      written_(parameters_.size()),
      pointer_contracts_(std::move(contracts)),
      pointer_bindings_(parameters_.size()) {
  rebuild_parameter_pointers();
}

KernelArgumentPacket::KernelArgumentPacket(KernelArgumentPacket&& other) noexcept
    : logical_id_(std::move(other.logical_id_)),
      parameter_abi_sha256_(std::move(other.parameter_abi_sha256_)),
      parameters_(std::move(other.parameters_)),
      cells_(std::move(other.cells_)),
      kernel_params_(std::move(other.kernel_params_)),
      written_(std::move(other.written_)),
      pointer_contracts_(std::move(other.pointer_contracts_)),
      pointer_bindings_(std::move(other.pointer_bindings_)) {
  rebuild_parameter_pointers();
  other.kernel_params_.clear();
}

KernelArgumentPacket& KernelArgumentPacket::operator=(
    KernelArgumentPacket&& other) noexcept {
  if (this != &other) {
    logical_id_ = std::move(other.logical_id_);
    parameter_abi_sha256_ = std::move(other.parameter_abi_sha256_);
    parameters_ = std::move(other.parameters_);
    cells_ = std::move(other.cells_);
    kernel_params_ = std::move(other.kernel_params_);
    written_ = std::move(other.written_);
    pointer_contracts_ = std::move(other.pointer_contracts_);
    pointer_bindings_ = std::move(other.pointer_bindings_);
    rebuild_parameter_pointers();
    other.kernel_params_.clear();
  }
  return *this;
}

void KernelArgumentPacket::rebuild_parameter_pointers() noexcept {
  if (kernel_params_.size() != parameters_.size()) return;
  for (std::size_t ordinal = 0; ordinal < parameters_.size(); ++ordinal) {
    kernel_params_[ordinal] =
        cells_.data() + parameters_[ordinal].device_layout_offset;
  }
}

Status KernelArgumentPacket::set_value(std::size_t ordinal,
                                       KernelWireType expected,
                                       const void* value, std::size_t bytes) {
  if (ordinal >= parameters_.size()) {
    return Status::InvalidArgument("kernel parameter ordinal is out of range");
  }
  if (parameters_[ordinal].wire_type != expected) {
    return Status::InvalidArgument("kernel parameter setter wire type mismatch");
  }
  if (written_[ordinal] != 0) {
    return Status::FailedPrecondition("kernel parameter cell was already written");
  }
  auto expected_bytes = kernel_wire_size(expected);
  if (!expected_bytes.ok()) return expected_bytes.status();
  if (bytes != expected_bytes.value()) {
    return Status::Internal("kernel parameter setter width mismatch");
  }
  std::memcpy(kernel_params_[ordinal], value, bytes);
  written_[ordinal] = 1;
  return Status::Ok();
}

#define PIH_DEFINE_PACKET_SETTER(name, cpp_type, wire_type)             \
  Status KernelArgumentPacket::name(std::size_t ordinal, cpp_type value) {    \
    return set_value(ordinal, wire_type, &value, sizeof(value));              \
  }

PIH_DEFINE_PACKET_SETTER(set_u8, std::uint8_t, KernelWireType::kU8)
PIH_DEFINE_PACKET_SETTER(set_u16, std::uint16_t, KernelWireType::kU16)
PIH_DEFINE_PACKET_SETTER(set_u32, std::uint32_t, KernelWireType::kU32)
PIH_DEFINE_PACKET_SETTER(set_u64, std::uint64_t, KernelWireType::kU64)
PIH_DEFINE_PACKET_SETTER(set_i32, std::int32_t, KernelWireType::kI32)
PIH_DEFINE_PACKET_SETTER(set_i64, std::int64_t, KernelWireType::kI64)
PIH_DEFINE_PACKET_SETTER(set_float16_bits, std::uint16_t,
                               KernelWireType::kFloat16Bits)
PIH_DEFINE_PACKET_SETTER(set_bfloat16_bits, std::uint16_t,
                               KernelWireType::kBFloat16Bits)

#undef PIH_DEFINE_PACKET_SETTER

Status KernelArgumentPacket::set_float32(std::size_t ordinal, float value) {
  if (!std::isfinite(value)) {
    return Status::InvalidArgument("kernel float32 parameter must be finite");
  }
  return set_value(ordinal, KernelWireType::kFloat32, &value, sizeof(value));
}

Status KernelArgumentPacket::set_float64(std::size_t ordinal, double value) {
  if (!std::isfinite(value)) {
    return Status::InvalidArgument("kernel float64 parameter must be finite");
  }
  return set_value(ordinal, KernelWireType::kFloat64, &value, sizeof(value));
}

Status KernelArgumentPacket::set_device_pointer(std::size_t ordinal,
                                                const VerifiedDevicePointer& value) {
  if (ordinal >= parameters_.size() ||
      parameters_[ordinal].wire_type != KernelWireType::kDevicePointerU64 ||
      parameters_[ordinal].contract_id != value.contract_id()) {
    return Status::InvalidArgument(
        "kernel device pointer does not match parameter contract");
  }
  for (const auto& existing : pointer_bindings_) {
    if (!existing.has_value()) continue;
    if (existing->device() != value.device()) {
      return Status::InvalidArgument("kernel pointers span more than one CUDA device");
    }
    const std::uint64_t existing_end =
        existing->bits() + existing->required_bytes();
    const std::uint64_t value_end = value.bits() + value.required_bytes();
    const bool overlaps =
        existing->bits() < value_end && value.bits() < existing_end;
    if (overlaps) {
      const bool exact = existing->bits() == value.bits() &&
                         existing->required_bytes() == value.required_bytes();
      const bool permitted = exact && existing->allow_exact_alias() &&
                             value.allow_exact_alias() &&
                             existing->generation() == value.generation() &&
                             existing->alias_group() != 0 &&
                             existing->alias_group() == value.alias_group();
      if (!permitted) {
        return Status::InvalidArgument("kernel pointer overlap is not permitted");
      }
    }
  }
  const std::uint64_t bits = value.bits();
  const Status written = set_value(
      ordinal, KernelWireType::kDevicePointerU64, &bits, sizeof(bits));
  if (written.ok()) pointer_bindings_[ordinal] = value;
  return written;
}

bool KernelArgumentPacket::ready() const noexcept {
  return !written_.empty() &&
         std::all_of(written_.begin(), written_.end(),
                     [](std::uint8_t value) { return value != 0; });
}

Result<void**> KernelArgumentPacket::ready_kernel_params() {
  if (!ready()) {
    return Status::FailedPrecondition("kernel argument packet is incomplete");
  }
  return kernel_params_.data();
}

void KernelArgumentPacket::reset() noexcept {
  std::fill(cells_.begin(), cells_.end(), std::byte{0});
  std::fill(written_.begin(), written_.end(), 0);
  std::fill(pointer_bindings_.begin(), pointer_bindings_.end(), std::nullopt);
}

const void* KernelArgumentPacket::argument_cell(std::size_t ordinal) const {
  return ordinal < kernel_params_.size() ? kernel_params_[ordinal] : nullptr;
}

}  // namespace pih
