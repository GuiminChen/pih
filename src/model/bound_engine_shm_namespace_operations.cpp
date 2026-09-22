#include "pih/model/bound_engine_shm_namespace_operations.h"

#include <algorithm>
#include <set>

namespace pih {
namespace {

bool safe_basename(std::string_view value) {
  if (value.empty() || value.size() > 255 || value == "." || value == "..")
    return false;
  for (const unsigned char byte : value) {
    if (byte < 0x21 || byte > 0x7e || byte == '/' || byte == '\\')
      return false;
  }
  return true;
}

}  // namespace

Result<BoundEngineShmNamespaceOperations>
BoundEngineShmNamespaceOperations::Create(
    std::span<const EngineShmNamespaceHandleBinding> bindings,
    EngineShmNamespaceStatBackend& backend) {
  if (bindings.empty()) {
    return Status::InvalidArgument(
        "bound SHM namespace operations manifest is empty");
  }
  std::set<std::uint64_t> identities;
  std::set<std::string> namespace_entries;
  for (const auto& binding : bindings) {
    const auto entry = std::to_string(binding.directory_descriptor) + ":" +
                       binding.basename;
    if (binding.object_identity == 0 || binding.directory_descriptor < 0 ||
        !safe_basename(binding.basename) ||
        !identities.insert(binding.object_identity).second ||
        !namespace_entries.insert(entry).second) {
      return Status::InvalidArgument(
          "bound SHM namespace operations manifest is invalid");
    }
  }
  return BoundEngineShmNamespaceOperations(
      std::vector<EngineShmNamespaceHandleBinding>(bindings.begin(),
                                                   bindings.end()),
      backend);
}

Result<EngineShmObjectObservation>
BoundEngineShmNamespaceOperations::observe(std::uint64_t object_identity) {
  const auto found = std::find_if(
      bindings_.begin(), bindings_.end(), [&](const auto& binding) {
        return binding.object_identity == object_identity;
      });
  if (found == bindings_.end()) {
    return Status::FailedPrecondition("SHM object identity is not bound");
  }
  auto result = backend_->stat_at(found->directory_descriptor,
                                  found->basename);
  if (!result.ok()) return result.status();
  return EngineShmObjectObservation{
      object_identity, result->present, result->regular_file,
      result->device_identity, result->inode_identity, result->size_bytes};
}

}  // namespace pih
