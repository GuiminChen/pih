#include "pih/model/deepseek_mapped_expert_source.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace pih {
namespace {
constexpr std::array<std::uint64_t, 6> kSegmentBytes = {
    DeepSeekExpertBundleLayout::kPackedBytesPerMatrix,
    DeepSeekExpertBundleLayout::kScaleBytesPerMatrix,
    DeepSeekExpertBundleLayout::kPackedBytesPerMatrix,
    DeepSeekExpertBundleLayout::kScaleBytesPerMatrix,
    DeepSeekExpertBundleLayout::kPackedBytesPerMatrix,
    DeepSeekExpertBundleLayout::kScaleBytesPerMatrix};
constexpr std::uint32_t kExpertsPerLayer = 256;
bool identity_less(DeepSeekExpertIdentity left, DeepSeekExpertIdentity right) {
  return left.layer < right.layer ||
         (left.layer == right.layer && left.expert < right.expert);
}
}

DeepSeekMappedExpertSource::DeepSeekMappedExpertSource(
    DeepSeekMappedExpertSource&& other) noexcept
    : inventory_(other.inventory_), owned_layers_(other.owned_layers_),
      records_(std::move(other.records_)), extents_(std::move(other.extents_)) {}

Result<DeepSeekMappedExpertSource> DeepSeekMappedExpertSource::Create(
    const DeepSeekStageMappedInventory& inventory,
    DeepSeekStageRange owned_layers,
    std::vector<DeepSeekExpertMappedBundleRecord> records,
    std::vector<DeepSeekPinnedExpertExtent> staging_extents,
    std::uint32_t transfer_reservation_window) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer >= 43 ||
      transfer_reservation_window == 0 ||
      staging_extents.size() < transfer_reservation_window) {
    return Status::InvalidArgument("DeepSeek mapped expert source topology is invalid");
  }
  const auto expected = static_cast<std::uint64_t>(
                            owned_layers.last_layer - owned_layers.first_layer + 1U) *
                        kExpertsPerLayer;
  if (records.size() != expected) {
    return Status::InvalidArgument("DeepSeek mapped expert manifest is incomplete");
  }
  std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
    return identity_less(a.identity, b.identity);
  });
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto& record = records[i];
    if (record.identity.layer < owned_layers.first_layer ||
        record.identity.layer > owned_layers.last_layer ||
        record.identity.expert >= kExpertsPerLayer ||
        (i != 0 && records[i - 1].identity == record.identity)) {
      return Status::InvalidArgument("DeepSeek mapped expert identity is invalid");
    }
    for (std::size_t segment = 0; segment < record.segments.size(); ++segment) {
      if (record.segments[segment].shard_name.empty() ||
          record.segments[segment].bytes != kSegmentBytes[segment]) {
        return Status::InvalidArgument("DeepSeek expert segment layout is non-canonical");
      }
      auto mapped = inventory.bytes(record.segments[segment].shard_name,
                                    record.segments[segment].file_offset,
                                    record.segments[segment].bytes);
      if (!mapped.ok()) return mapped.status();
    }
  }
  std::vector<ExtentState> extents;
  extents.reserve(staging_extents.size());
  for (const auto& extent : staging_extents) {
    if (extent.address == 0 || extent.bytes != DeepSeekExpertBundleLayout::kBundleBytes ||
        extent.registration_identity == 0 ||
        extent.address % DeepSeekExpertBundleLayout::kAlignment != 0 ||
        extent.address > std::numeric_limits<std::uintptr_t>::max() - extent.bytes) {
      return Status::InvalidArgument("DeepSeek staging extent is not canonical or registered");
    }
    const auto duplicate = std::find_if(extents.begin(), extents.end(), [&](const auto& prior) {
      return prior.extent.address + prior.extent.bytes > extent.address &&
             extent.address + extent.bytes > prior.extent.address;
    });
    if (duplicate != extents.end()) {
      return Status::InvalidArgument("DeepSeek staging extents overlap");
    }
    extents.push_back({extent, false, {}});
  }
  return DeepSeekMappedExpertSource(inventory, owned_layers, std::move(records),
                                    std::move(extents));
}

Result<DeepSeekPinnedExpertExtent> DeepSeekMappedExpertSource::resolve(
    DeepSeekExpertIdentity identity, std::uint64_t bytes) {
  if (bytes != DeepSeekExpertBundleLayout::kBundleBytes) {
    return Status::InvalidArgument("DeepSeek expert bundle byte request is invalid");
  }
  std::scoped_lock lock(mutex_);
  const auto record = std::lower_bound(records_.begin(), records_.end(), identity,
                                       [](const auto& value, const auto& key) {
                                         return identity_less(value.identity, key);
                                       });
  if (record == records_.end() || record->identity != identity) {
    return Status::InvalidArgument("DeepSeek expert is outside the mapped manifest");
  }
  auto extent = std::find_if(extents_.begin(), extents_.end(),
                             [](const auto& value) { return !value.leased; });
  if (extent == extents_.end()) {
    return Status::ResourceExhausted("DeepSeek expert staging pool is exhausted");
  }
  std::uint64_t destination_offset = 0;
  for (const auto& segment : record->segments) {
    auto source = inventory_->bytes(segment.shard_name, segment.file_offset, segment.bytes);
    if (!source.ok()) return source.status();
    std::memcpy(reinterpret_cast<void*>(extent->extent.address + destination_offset),
                source->data(), static_cast<std::size_t>(segment.bytes));
    destination_offset += segment.bytes;
  }
  extent->leased = true;
  extent->identity = identity;
  return extent->extent;
}

Status DeepSeekMappedExpertSource::release(
    DeepSeekExpertIdentity identity, DeepSeekPinnedExpertExtent released) {
  std::scoped_lock lock(mutex_);
  auto extent = std::find_if(extents_.begin(), extents_.end(), [&](const auto& value) {
    return value.extent.address == released.address &&
           value.extent.bytes == released.bytes &&
           value.extent.registration_identity == released.registration_identity;
  });
  if (extent == extents_.end() || !extent->leased || extent->identity != identity) {
    return Status::FailedPrecondition("DeepSeek staging extent lease is invalid");
  }
  extent->leased = false;
  extent->identity = {};
  return Status::Ok();
}

}  // namespace pih
