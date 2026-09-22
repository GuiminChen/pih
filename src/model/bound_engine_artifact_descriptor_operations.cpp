#include "pih/model/bound_engine_artifact_descriptor_operations.h"

#include <algorithm>
#include <set>

namespace pih {

Result<BoundEngineArtifactDescriptorOperations>
BoundEngineArtifactDescriptorOperations::Create(
    std::span<const EngineArtifactDescriptorHandleBinding> bindings,
    EngineArtifactDescriptorStatBackend& backend) {
  if (bindings.empty()) {
    return Status::InvalidArgument(
        "bound artifact descriptor operations manifest is empty");
  }
  std::set<std::uint64_t> identities;
  std::set<std::int32_t> descriptors;
  for (const auto& binding : bindings) {
    if (binding.descriptor_identity == 0 || binding.descriptor < 0 ||
        !identities.insert(binding.descriptor_identity).second ||
        !descriptors.insert(binding.descriptor).second) {
      return Status::InvalidArgument(
          "bound artifact descriptor operations manifest is invalid");
    }
  }
  return BoundEngineArtifactDescriptorOperations(
      std::vector<EngineArtifactDescriptorHandleBinding>(bindings.begin(),
                                                         bindings.end()),
      backend);
}

Result<EngineArtifactDescriptorObservation>
BoundEngineArtifactDescriptorOperations::observe(
    std::uint64_t descriptor_identity) {
  const auto found = std::find_if(
      bindings_.begin(), bindings_.end(), [&](const auto& binding) {
        return binding.descriptor_identity == descriptor_identity;
      });
  if (found == bindings_.end()) {
    return Status::FailedPrecondition(
        "artifact descriptor identity is not bound");
  }
  auto result = backend_->stat(found->descriptor);
  if (!result.ok()) return result.status();
  return EngineArtifactDescriptorObservation{
      descriptor_identity, result->open, result->regular_file,
      result->device_identity,
      result->inode_identity, result->size_bytes};
}

}  // namespace pih
