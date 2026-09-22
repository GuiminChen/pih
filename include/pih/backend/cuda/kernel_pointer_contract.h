#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "pih/core/tensor_view.h"

namespace pih {

enum class KernelPointerAccess : std::uint8_t {
  kRead = 0,
  kWrite,
  kReadWrite,
  kAtomic,
};

enum class KernelPointerOwnerClass : std::uint8_t {
  kWeight = 0,
  kActivation,
  kWorkspace,
  kKvState,
};

class KernelPointerContract final {
 public:
  static constexpr std::size_t kMaximumIdBytes = 128;

  static Result<KernelPointerContract> Create(
      std::string contract_id, KernelPointerAccess access,
      KernelPointerOwnerClass owner_class, std::int32_t owning_rank,
      std::int32_t device_index, std::uint64_t required_bytes,
      std::uint64_t required_alignment, std::uint32_t alias_group,
      bool allow_exact_alias);

  [[nodiscard]] std::string_view contract_id() const noexcept {
    return contract_id_;
  }
  [[nodiscard]] KernelPointerAccess access() const noexcept { return access_; }
  [[nodiscard]] KernelPointerOwnerClass owner_class() const noexcept {
    return owner_class_;
  }
  [[nodiscard]] std::int32_t owning_rank() const noexcept { return owning_rank_; }
  [[nodiscard]] std::int32_t device_index() const noexcept {
    return device_index_;
  }
  [[nodiscard]] std::uint64_t required_bytes() const noexcept {
    return required_bytes_;
  }
  [[nodiscard]] std::uint64_t required_alignment() const noexcept {
    return required_alignment_;
  }
  [[nodiscard]] std::uint32_t alias_group() const noexcept {
    return alias_group_;
  }
  [[nodiscard]] bool allow_exact_alias() const noexcept {
    return allow_exact_alias_;
  }

 private:
  KernelPointerContract(std::string contract_id, KernelPointerAccess access,
                        KernelPointerOwnerClass owner_class,
                        std::int32_t owning_rank, std::int32_t device_index,
                        std::uint64_t required_bytes,
                        std::uint64_t required_alignment,
                        std::uint32_t alias_group, bool allow_exact_alias)
      : contract_id_(std::move(contract_id)),
        access_(access),
        owner_class_(owner_class),
        owning_rank_(owning_rank),
        device_index_(device_index),
        required_bytes_(required_bytes),
        required_alignment_(required_alignment),
        alias_group_(alias_group),
        allow_exact_alias_(allow_exact_alias) {}

  std::string contract_id_;
  KernelPointerAccess access_;
  KernelPointerOwnerClass owner_class_;
  std::int32_t owning_rank_;
  std::int32_t device_index_;
  std::uint64_t required_bytes_;
  std::uint64_t required_alignment_;
  std::uint32_t alias_group_;
  bool allow_exact_alias_;
};

class VerifiedDevicePointer final {
 public:
  static Result<VerifiedDevicePointer> Create(
      const TensorView& view, const KernelPointerContract& contract,
      KernelPointerOwnerClass actual_owner_class, std::int32_t actual_owner_rank,
      std::uint64_t actual_generation);

  [[nodiscard]] std::uint64_t bits() const noexcept { return bits_; }
  [[nodiscard]] Device device() const noexcept { return device_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] std::uint64_t required_bytes() const noexcept {
    return required_bytes_;
  }
  [[nodiscard]] std::string_view contract_id() const noexcept {
    return {contract_id_.data(), contract_id_size_};
  }
  [[nodiscard]] KernelPointerAccess access() const noexcept { return access_; }
  [[nodiscard]] std::uint32_t alias_group() const noexcept { return alias_group_; }
  [[nodiscard]] bool allow_exact_alias() const noexcept {
    return allow_exact_alias_;
  }

 private:
  VerifiedDevicePointer(std::uint64_t bits, Device device,
                        std::uint64_t generation, std::uint64_t required_bytes,
                        std::string_view contract_id, KernelPointerAccess access,
                        std::uint32_t alias_group, bool allow_exact_alias);

  std::uint64_t bits_;
  Device device_;
  std::uint64_t generation_;
  std::uint64_t required_bytes_;
  std::array<char, KernelPointerContract::kMaximumIdBytes> contract_id_{};
  std::uint8_t contract_id_size_ = 0;
  KernelPointerAccess access_;
  std::uint32_t alias_group_;
  bool allow_exact_alias_;
};

}  // namespace pih
