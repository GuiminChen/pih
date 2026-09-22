#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pih/core/result.h"

namespace pih {

enum class KernelWireType : std::uint8_t {
  kU8 = 0,
  kU16,
  kU32,
  kU64,
  kI32,
  kI64,
  kFloat32,
  kFloat64,
  kFloat16Bits,
  kBFloat16Bits,
  kDevicePointerU64,
};

Result<std::uint32_t> kernel_wire_size(KernelWireType type);
Result<std::uint32_t> kernel_wire_alignment(KernelWireType type);

struct KernelParameterSpec final {
  std::string role;
  KernelWireType wire_type;
  std::uint32_t device_layout_offset;
  std::string contract_id;
};

class KernelSignatureManifest final {
 public:
  static constexpr std::uint32_t kMaximumDeviceParameterBytes = 32764;
  static constexpr std::size_t kMaximumParameters = 256;
  static constexpr std::size_t kMaximumLogicalIdBytes = 128;
  static constexpr std::size_t kMaximumRoleBytes = 96;
  static constexpr std::size_t kMaximumContractIdBytes = 128;

  static Result<KernelSignatureManifest> Create(
      std::string logical_id, std::string selected_cubin_sha256,
      std::string parameter_abi_sha256,
      std::span<const KernelParameterSpec> parameters,
      std::uint32_t total_device_parameter_bytes);

  [[nodiscard]] std::string_view logical_id() const noexcept {
    return logical_id_;
  }
  [[nodiscard]] std::string_view selected_cubin_sha256() const noexcept {
    return selected_cubin_sha256_;
  }
  [[nodiscard]] std::string_view parameter_abi_sha256() const noexcept {
    return parameter_abi_sha256_;
  }
  [[nodiscard]] std::size_t parameter_count() const noexcept {
    return parameters_.size();
  }
  [[nodiscard]] std::uint32_t total_device_parameter_bytes() const noexcept {
    return total_device_parameter_bytes_;
  }
  [[nodiscard]] const KernelParameterSpec& parameter(std::size_t ordinal) const {
    return parameters_.at(ordinal);
  }
  [[nodiscard]] std::span<const KernelParameterSpec> parameters() const noexcept {
    return parameters_;
  }

 private:
  KernelSignatureManifest(std::string logical_id,
                          std::string selected_cubin_sha256,
                          std::string parameter_abi_sha256,
                          std::vector<KernelParameterSpec> parameters,
                          std::uint32_t total_device_parameter_bytes)
      : logical_id_(std::move(logical_id)),
        selected_cubin_sha256_(std::move(selected_cubin_sha256)),
        parameter_abi_sha256_(std::move(parameter_abi_sha256)),
        parameters_(std::move(parameters)),
        total_device_parameter_bytes_(total_device_parameter_bytes) {}

  std::string logical_id_;
  std::string selected_cubin_sha256_;
  std::string parameter_abi_sha256_;
  std::vector<KernelParameterSpec> parameters_;
  std::uint32_t total_device_parameter_bytes_;
};

}  // namespace pih
