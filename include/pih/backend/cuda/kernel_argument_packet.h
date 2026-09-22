#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/backend/cuda/kernel_pointer_contract.h"
#include "pih/backend/cuda/kernel_signature_manifest.h"

namespace pih {

class KernelArgumentPacket final {
 public:
  static Result<KernelArgumentPacket> Create(
      const KernelSignatureManifest& signature,
      std::span<const KernelPointerContract> pointer_contracts = {});

  KernelArgumentPacket(const KernelArgumentPacket&) = delete;
  KernelArgumentPacket& operator=(const KernelArgumentPacket&) = delete;
  KernelArgumentPacket(KernelArgumentPacket&& other) noexcept;
  KernelArgumentPacket& operator=(KernelArgumentPacket&& other) noexcept;

  Status set_u8(std::size_t ordinal, std::uint8_t value);
  Status set_u16(std::size_t ordinal, std::uint16_t value);
  Status set_u32(std::size_t ordinal, std::uint32_t value);
  Status set_u64(std::size_t ordinal, std::uint64_t value);
  Status set_i32(std::size_t ordinal, std::int32_t value);
  Status set_i64(std::size_t ordinal, std::int64_t value);
  Status set_float32(std::size_t ordinal, float value);
  Status set_float64(std::size_t ordinal, double value);
  Status set_float16_bits(std::size_t ordinal, std::uint16_t value);
  Status set_bfloat16_bits(std::size_t ordinal, std::uint16_t value);
  Status set_device_pointer(std::size_t ordinal,
                            const VerifiedDevicePointer& value);

  Result<void**> ready_kernel_params();
  void reset() noexcept;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::size_t parameter_count() const noexcept {
    return parameters_.size();
  }
  [[nodiscard]] std::string_view logical_id() const noexcept {
    return logical_id_;
  }
  [[nodiscard]] std::string_view parameter_abi_sha256() const noexcept {
    return parameter_abi_sha256_;
  }
  [[nodiscard]] const void* argument_cell(std::size_t ordinal) const;

 private:
  explicit KernelArgumentPacket(std::string logical_id,
                                std::string parameter_abi_sha256,
                                std::vector<KernelParameterSpec> parameters,
                                std::uint32_t total_bytes,
                                std::vector<KernelPointerContract> contracts);
  void rebuild_parameter_pointers() noexcept;
  Status set_value(std::size_t ordinal, KernelWireType expected,
                   const void* value, std::size_t bytes);

  std::string logical_id_;
  std::string parameter_abi_sha256_;
  std::vector<KernelParameterSpec> parameters_;
  std::vector<std::byte> cells_;
  std::vector<void*> kernel_params_;
  std::vector<std::uint8_t> written_;
  std::vector<KernelPointerContract> pointer_contracts_;
  std::vector<std::optional<VerifiedDevicePointer>> pointer_bindings_;
};

}  // namespace pih
