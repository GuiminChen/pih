#include "pih/model/engine_owned_resource_manifest_record.h"

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

bool is_zero(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return false;
  }
  return true;
}

}  // namespace

Result<std::vector<std::byte>> encode_engine_owned_resource_manifest_record(
    const EngineOwnedResourceManifestRecord& record) {
  const auto kind = static_cast<std::uint32_t>(record.kind);
  const auto state = static_cast<std::uint32_t>(record.state);
  if (kind >= 6 || state >= 3 || is_zero(record.owner_identity) ||
      is_zero(record.resource_identity) || record.object_count == 0) {
    return Status::InvalidArgument(
        "engine owned-resource manifest record is invalid");
  }

  auto builder = CanonicalHashBuilder::Create(
      "pih:engine-owned-resource-manifest-record:v1", 6);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, kind);
  if (status.ok()) status = builder->add_hash(2, record.owner_identity);
  if (status.ok()) status = builder->add_hash(3, record.resource_identity);
  if (status.ok()) status = builder->add_u64(4, record.backing_bytes);
  if (status.ok()) status = builder->add_u64(5, record.object_count);
  if (status.ok()) status = builder->add_u32(6, state);
  if (!status.ok()) return status;
  auto digest = builder->finalize();
  if (!digest.ok()) return digest.status();
  return std::vector<std::byte>(digest->bytes.begin(), digest->bytes.end());
}

}  // namespace pih
