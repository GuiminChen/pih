#include "pih/model/engine_owned_resource_census_digest.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

constexpr std::size_t kResourceKindCount = 6;
constexpr std::size_t kMaximumRecordBytes = 64U * 1024U;
constexpr std::size_t kMaximumRecordCount = 1024U * 1024U;

bool less_bytes(std::span<const std::byte> left,
                std::span<const std::byte> right) {
  return std::lexicographical_compare(left.begin(), left.end(),
                                      right.begin(), right.end());
}

bool equal_bytes(std::span<const std::byte> left,
                 std::span<const std::byte> right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin());
}

Result<Sha256Digest> hash_record(std::span<const std::byte> identity) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:engine-owned-resource-census-record:v1", 1);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_bytes(1, identity);
  if (!status.ok()) return status;
  return builder->finalize();
}

}  // namespace

Result<Sha256Digest> engine_owned_resource_census_digest(
    EngineOwnedResourceKind kind,
    std::span<const EngineOwnedResourceCensusRecord> records) {
  const auto kind_index = static_cast<std::size_t>(kind);
  if (kind_index >= kResourceKindCount) {
    return Status::InvalidArgument("invalid engine owned-resource census kind");
  }
  if (records.size() > kMaximumRecordCount) {
    return Status::ResourceExhausted(
        "engine owned-resource census has too many records");
  }

  std::vector<std::span<const std::byte>> identities;
  identities.reserve(records.size());
  for (const auto& record : records) {
    if (record.identity.empty()) {
      return Status::InvalidArgument(
          "engine owned-resource census identity is empty");
    }
    if (record.identity.size() > kMaximumRecordBytes) {
      return Status::ResourceExhausted(
          "engine owned-resource census identity is too large");
    }
    identities.push_back(record.identity);
  }
  std::sort(identities.begin(), identities.end(), less_bytes);
  for (std::size_t index = 1; index < identities.size(); ++index) {
    if (equal_bytes(identities[index - 1], identities[index])) {
      return Status::InvalidArgument(
          "engine owned-resource census identity is duplicated");
    }
  }

  std::vector<std::byte> ordered_record_hashes;
  ordered_record_hashes.reserve(identities.size() * Sha256Digest{}.bytes.size());
  for (const auto identity : identities) {
    auto digest = hash_record(identity);
    if (!digest.ok()) return digest.status();
    ordered_record_hashes.insert(ordered_record_hashes.end(),
                                 digest->bytes.begin(), digest->bytes.end());
  }

  auto collection = CanonicalHashBuilder::Create(
      "pih:engine-owned-resource-census:v1", 3);
  if (!collection.ok()) return collection.status();
  auto status = collection->add_u32(1, static_cast<std::uint32_t>(kind_index));
  if (status.ok()) {
    status = collection->add_u64(2, static_cast<std::uint64_t>(records.size()));
  }
  if (status.ok()) status = collection->add_bytes(3, ordered_record_hashes);
  if (!status.ok()) return status;
  return collection->finalize();
}

}  // namespace pih
