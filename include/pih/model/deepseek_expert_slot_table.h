#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_expert_bundle_layout.h"
#include "pih/model/deepseek_expert_pager.h"

namespace pih {

struct DeepSeekExpertLeaseDeviceView final {
  DeepSeekExpertIdentity identity;
  std::uint32_t slot = 0;
  std::uint64_t generation = 0;
  std::uint64_t context_identity = 0;
  std::uint32_t device_ordinal = 0;
  std::string_view layout_id;
  DeepSeekExpertBundleDeviceView bundle;
};

class DeepSeekExpertSlotTable final {
 public:
  static constexpr std::string_view kCanonicalLayoutId =
      "deepseek-0731-canonical-expert-v1";

  static Result<DeepSeekExpertSlotTable> Create(
      std::vector<std::uintptr_t> slot_bases,
      std::uint64_t context_identity, std::uint32_t device_ordinal);
  static Result<DeepSeekExpertSlotTable> CreateResidentOnly(
      std::uint64_t context_identity, std::uint32_t device_ordinal);
  Result<DeepSeekExpertLeaseDeviceView> bind(
      const DeepSeekExpertLease& lease) const;
  Result<DeepSeekExpertLeaseDeviceView> bind_resident(
      DeepSeekExpertIdentity identity, std::uint64_t generation,
      DeepSeekExpertBundleDeviceView bundle) const;
  [[nodiscard]] std::uint32_t slot_count() const noexcept {
    return static_cast<std::uint32_t>(slot_bases_.size());
  }

 private:
  std::vector<std::uintptr_t> slot_bases_;
  std::uint64_t context_identity_ = 0;
  std::uint32_t device_ordinal_ = 0;
};

}  // namespace pih
