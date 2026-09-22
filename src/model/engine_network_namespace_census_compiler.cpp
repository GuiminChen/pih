#include "pih/model/engine_network_namespace_census_compiler.h"

#include <limits>
#include <map>

namespace pih {
namespace {

constexpr std::size_t kMaximumNetworkCensusRows = 1024U * 1024U;

bool zero_network_row_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return false;
  }
  return true;
}

}  // namespace

Result<EngineNetworkNamespaceCensusCompiler>
EngineNetworkNamespaceCensusCompiler::Create(
    std::uint64_t namespace_identity,
    EngineNetworkNamespaceCensusBackend& backend) {
  if (namespace_identity == 0) {
    return Status::InvalidArgument(
        "network namespace census identity is zero");
  }
  return EngineNetworkNamespaceCensusCompiler(namespace_identity, backend);
}

Result<EngineNetworkNamespaceSnapshot>
EngineNetworkNamespaceCensusCompiler::capture(
    std::uint64_t namespace_identity) {
  if (namespace_identity != namespace_identity_) {
    return Status::FailedPrecondition(
        "network namespace census request identity drifted");
  }
  auto raw = backend_->capture(namespace_identity);
  if (!raw.ok()) return raw.status();
  if (raw->namespace_identity != namespace_identity_ ||
      raw->namespace_handle_present == raw->namespace_destroyed) {
    return Status::FailedPrecondition(
        "network namespace raw snapshot state is ambiguous");
  }
  if (raw->namespace_destroyed) {
    if (!raw->rows.empty()) {
      return Status::FailedPrecondition(
          "destroyed network namespace has raw socket rows");
    }
    return EngineNetworkNamespaceSnapshot{
        namespace_identity_, false, true, {}};
  }
  if (raw->rows.size() > kMaximumNetworkCensusRows) {
    return Status::ResourceExhausted(
        "network namespace census has too many socket rows");
  }

  std::map<std::string, EngineNetworkOwnerObservation> aggregated;
  for (const auto& row : raw->rows) {
    if (!row.ownership_complete) {
      return Status::Unavailable(
          "network socket owner classification is unavailable");
    }
    if (!row.kernel_backing_complete) {
      return Status::Unavailable(
          "network socket kernel backing is unavailable");
    }
    const auto lifecycle = static_cast<std::uint32_t>(row.lifecycle);
    if (zero_network_row_digest(row.owner_identity) ||
        zero_network_row_digest(row.resource_identity) || lifecycle >= 2) {
      return Status::FailedPrecondition(
          "network socket census row is invalid");
    }
    const auto key = row.owner_identity.hex() + row.resource_identity.hex() +
                     std::to_string(lifecycle);
    auto entry = aggregated
                     .try_emplace(
                         key, EngineNetworkOwnerObservation{
                                  row.owner_identity, row.resource_identity,
                                  0, 0,
                                  lifecycle == 0
                                      ? EngineOwnedResourceLifecycleState::kActive
                                      : EngineOwnedResourceLifecycleState::kRetired})
                     .first;
    auto& owner = entry->second;
    if (owner.object_count == std::numeric_limits<std::uint64_t>::max() ||
        row.kernel_backing_bytes >
            std::numeric_limits<std::uint64_t>::max() - owner.backing_bytes) {
      return Status::ResourceExhausted(
          "network socket census aggregation overflowed");
    }
    ++owner.object_count;
    owner.backing_bytes += row.kernel_backing_bytes;
  }

  std::vector<EngineNetworkOwnerObservation> owners;
  owners.reserve(aggregated.size());
  for (auto& [key, owner] : aggregated) owners.push_back(std::move(owner));
  return EngineNetworkNamespaceSnapshot{
      namespace_identity_, true, false, std::move(owners)};
}

}  // namespace pih
