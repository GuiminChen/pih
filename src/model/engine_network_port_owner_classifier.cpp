#include "pih/model/engine_network_port_owner_classifier.h"

#include <algorithm>

namespace pih {
namespace {

bool zero_port_owner_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return false;
  }
  return true;
}

}  // namespace

Result<EngineNetworkPortOwnerClassifier>
EngineNetworkPortOwnerClassifier::Create(
    std::uint64_t namespace_identity,
    std::span<const EngineNetworkPortOwnerBinding> bindings,
    EngineNetworkPortCensusBackend& backend) {
  if (namespace_identity == 0 || bindings.empty()) {
    return Status::InvalidArgument(
        "network port owner manifest is empty or unbound");
  }
  std::vector<EngineNetworkPortOwnerBinding> ordered(bindings.begin(),
                                                      bindings.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& left,
                                                const auto& right) {
    return left.first_port < right.first_port;
  });
  for (std::size_t index = 0; index < ordered.size(); ++index) {
    const auto& binding = ordered[index];
    if (binding.first_port == 0 || binding.first_port > binding.last_port ||
        zero_port_owner_digest(binding.owner_identity) ||
        zero_port_owner_digest(binding.resource_identity) ||
        (index != 0 && ordered[index - 1].last_port >= binding.first_port)) {
      return Status::InvalidArgument(
          "network port owner manifest is invalid");
    }
  }
  return EngineNetworkPortOwnerClassifier(namespace_identity,
                                           std::move(ordered), backend);
}

Result<EngineNetworkNamespaceRawSnapshot>
EngineNetworkPortOwnerClassifier::capture(std::uint64_t namespace_identity) {
  if (namespace_identity != namespace_identity_) {
    return Status::FailedPrecondition(
        "network port owner request identity drifted");
  }
  auto raw = backend_->capture(namespace_identity);
  if (!raw.ok()) return raw.status();
  if (raw->namespace_identity != namespace_identity_ ||
      raw->namespace_handle_present == raw->namespace_destroyed) {
    return Status::FailedPrecondition(
        "network port raw snapshot state is ambiguous");
  }
  if (raw->namespace_destroyed) {
    if (!raw->rows.empty()) {
      return Status::FailedPrecondition(
          "destroyed network namespace has port rows");
    }
    return EngineNetworkNamespaceRawSnapshot{
        namespace_identity_, false, true, {}};
  }

  std::vector<EngineNetworkSocketCensusRow> rows;
  rows.reserve(raw->rows.size());
  for (const auto& row : raw->rows) {
    if (row.local_port == 0 ||
        static_cast<std::uint32_t>(row.lifecycle) >= 2) {
      return Status::FailedPrecondition(
          "network port census row is invalid");
    }
    const auto found = std::find_if(
        bindings_.begin(), bindings_.end(), [&](const auto& binding) {
          return row.local_port >= binding.first_port &&
                 row.local_port <= binding.last_port;
        });
    if (found == bindings_.end()) {
      rows.push_back({false, {}, {}, row.lifecycle,
                      row.kernel_backing_complete,
                      row.kernel_backing_bytes});
    } else {
      rows.push_back({true, found->owner_identity, found->resource_identity,
                      row.lifecycle, row.kernel_backing_complete,
                      row.kernel_backing_bytes});
    }
  }
  return EngineNetworkNamespaceRawSnapshot{
      namespace_identity_, true, false, std::move(rows)};
}

}  // namespace pih
