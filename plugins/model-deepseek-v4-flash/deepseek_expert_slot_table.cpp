#include "pih/model/deepseek_expert_slot_table.h"

#include <algorithm>

namespace pih {

Result<DeepSeekExpertSlotTable> DeepSeekExpertSlotTable::Create(
    std::vector<std::uintptr_t> slot_bases,
    std::uint64_t context_identity, std::uint32_t device_ordinal) {
  if (slot_bases.empty() || context_identity == 0 || device_ordinal >= 32) {
    return Status::InvalidArgument(
        "DeepSeek expert slot table identity is invalid");
  }
  for (const auto base : slot_bases) {
    if (!DeepSeekExpertBundleLayout::Bind(base).ok()) {
      return Status::InvalidArgument(
          "DeepSeek expert slot backing is invalid");
    }
  }
  auto ordered = slot_bases;
  std::sort(ordered.begin(), ordered.end());
  for (std::size_t index = 1; index < ordered.size(); ++index) {
    if (ordered[index - 1] + DeepSeekExpertBundleLayout::kBundleBytes >
        ordered[index]) {
      return Status::InvalidArgument(
          "DeepSeek expert slot backing overlaps");
    }
  }
  DeepSeekExpertSlotTable table;
  table.slot_bases_ = std::move(slot_bases);
  table.context_identity_ = context_identity;
  table.device_ordinal_ = device_ordinal;
  return table;
}

Result<DeepSeekExpertSlotTable> DeepSeekExpertSlotTable::CreateResidentOnly(
    std::uint64_t context_identity, std::uint32_t device_ordinal) {
  if (context_identity == 0 || device_ordinal >= 32) {
    return Status::InvalidArgument(
        "DeepSeek resident-only slot table identity is invalid");
  }
  DeepSeekExpertSlotTable table;
  table.context_identity_ = context_identity;
  table.device_ordinal_ = device_ordinal;
  return table;
}

Result<DeepSeekExpertLeaseDeviceView> DeepSeekExpertSlotTable::bind(
    const DeepSeekExpertLease& lease) const {
  if (lease.identity.layer > 42 || lease.identity.expert >= 256 ||
      lease.slot >= slot_bases_.size() || lease.generation == 0) {
    return Status::FailedPrecondition(
        "DeepSeek expert lease cannot bind a device slot");
  }
  auto bundle = DeepSeekExpertBundleLayout::Bind(slot_bases_[lease.slot]);
  if (!bundle.ok()) return bundle.status();
  return DeepSeekExpertLeaseDeviceView{
      .identity = lease.identity,
      .slot = lease.slot,
      .generation = lease.generation,
      .context_identity = context_identity_,
      .device_ordinal = device_ordinal_,
      .layout_id = kCanonicalLayoutId,
      .bundle = *bundle,
  };
}

Result<DeepSeekExpertLeaseDeviceView> DeepSeekExpertSlotTable::bind_resident(
    DeepSeekExpertIdentity identity, std::uint64_t generation,
    DeepSeekExpertBundleDeviceView bundle) const {
  const auto valid_matrix = [](const DeepSeekExpertMatrixDeviceView& matrix) {
    return matrix.packed.address != 0 &&
           matrix.packed.bytes ==
               DeepSeekExpertBundleLayout::kPackedBytesPerMatrix &&
           matrix.scales.address != 0 &&
           matrix.scales.bytes ==
               DeepSeekExpertBundleLayout::kScaleBytesPerMatrix;
  };
  if (identity.layer > 42 || identity.expert >= 256 ||
      generation == 0 ||
      !valid_matrix(bundle.w1) || !valid_matrix(bundle.w2) ||
      !valid_matrix(bundle.w3)) {
    return Status::FailedPrecondition(
        "DeepSeek resident expert bundle cannot bind the compute lane");
  }
  return DeepSeekExpertLeaseDeviceView{
      identity, UINT32_MAX, generation, context_identity_, device_ordinal_,
      std::string_view{"deepseek-0731-main-resident-expert-v1"},
      bundle};
}

}  // namespace pih
