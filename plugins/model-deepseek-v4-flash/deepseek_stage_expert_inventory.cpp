#include "pih/model/deepseek_stage_expert_inventory.h"

#include <algorithm>
#include <limits>

namespace pih {
namespace {

constexpr std::uint32_t kExpertsPerLayer = 256;

std::uint32_t inventory_index(DeepSeekStageRange range,
                              DeepSeekExpertIdentity identity) {
  return (static_cast<std::uint32_t>(identity.layer) - range.first_layer) *
             kExpertsPerLayer +
         identity.expert;
}

}  // namespace

Result<DeepSeekStageExpertInventory> DeepSeekStageExpertInventory::Create(
    DeepSeekStageRange owned_layers,
    std::vector<DeepSeekStageExpertExtentRecord> records) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer >= 43) {
    return Status::InvalidArgument(
        "DeepSeek stage expert inventory ownership is invalid");
  }
  const auto layer_count = owned_layers.last_layer -
                           owned_layers.first_layer + 1U;
  const auto expected = layer_count * kExpertsPerLayer;
  if (records.size() != expected) {
    return Status::InvalidArgument(
        "DeepSeek stage expert inventory is not total");
  }
  DeepSeekStageExpertInventory inventory;
  inventory.owned_layers_ = owned_layers;
  inventory.extents_.resize(expected);
  std::vector<bool> present(expected, false);
  for (const auto& record : records) {
    if (record.identity.layer < owned_layers.first_layer ||
        record.identity.layer > owned_layers.last_layer ||
        record.identity.expert >= kExpertsPerLayer ||
        record.extent.address == 0 ||
        record.extent.address % 256U != 0 ||
        record.extent.bytes != DeepSeekExpertPager::kBundleBytes ||
        record.extent.registration_identity == 0 ||
        record.extent.address >
            std::numeric_limits<std::uintptr_t>::max() - record.extent.bytes) {
      return Status::InvalidArgument(
          "DeepSeek stage expert extent record is invalid");
    }
    const auto index = inventory_index(owned_layers, record.identity);
    if (present[index]) {
      return Status::InvalidArgument(
          "DeepSeek stage expert inventory contains a duplicate");
    }
    present[index] = true;
    inventory.extents_[index] = record.extent;
  }
  if (!std::all_of(present.begin(), present.end(), [](bool value) {
        return value;
      })) {
    return Status::InvalidArgument(
        "DeepSeek stage expert inventory has a missing record");
  }
  auto ordered = inventory.extents_;
  std::sort(ordered.begin(), ordered.end(), [](const auto& left,
                                                const auto& right) {
    return left.address < right.address;
  });
  for (std::size_t index = 1; index < ordered.size(); ++index) {
    if (ordered[index - 1].address + ordered[index - 1].bytes >
        ordered[index].address) {
      return Status::InvalidArgument(
          "DeepSeek stage expert extents overlap");
    }
  }
  return inventory;
}

Result<DeepSeekPinnedExpertExtent> DeepSeekStageExpertInventory::resolve(
    DeepSeekExpertIdentity identity, std::uint64_t bytes) {
  if (identity.layer < owned_layers_.first_layer ||
      identity.layer > owned_layers_.last_layer || identity.expert >= 256 ||
      bytes != DeepSeekExpertPager::kBundleBytes) {
    return Status::InvalidArgument(
        "DeepSeek expert request is outside stage inventory");
  }
  return extents_[inventory_index(owned_layers_, identity)];
}

}  // namespace pih
