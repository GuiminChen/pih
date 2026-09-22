#include "pih/model/bound_engine_listener_operations.h"

#include <algorithm>
#include <set>

namespace pih {

Result<BoundEngineListenerOperations> BoundEngineListenerOperations::Create(
    std::span<const EngineListenerHandleBinding> bindings,
    EngineListenerKernelBackend& kernel,
    EngineListenerAcceptAuthority& authority) {
  if (bindings.empty()) {
    return Status::InvalidArgument("bound listener manifest is empty");
  }
  std::set<std::uint64_t> identities;
  std::set<std::int32_t> descriptors;
  for (const auto& binding : bindings) {
    if (binding.listener_identity == 0 || binding.descriptor < 0 ||
        !identities.insert(binding.listener_identity).second ||
        !descriptors.insert(binding.descriptor).second) {
      return Status::InvalidArgument("bound listener manifest is invalid");
    }
  }
  return BoundEngineListenerOperations(
      std::vector<EngineListenerHandleBinding>(bindings.begin(),
                                               bindings.end()),
      kernel, authority);
}

Result<EngineListenerObservation> BoundEngineListenerOperations::observe(
    std::uint64_t listener_identity) {
  const auto found = std::find_if(
      bindings_.begin(), bindings_.end(), [&](const auto& binding) {
        return binding.listener_identity == listener_identity;
      });
  if (found == bindings_.end()) {
    return Status::FailedPrecondition("listener identity is not bound");
  }
  auto kernel = kernel_->inspect(found->descriptor);
  if (!kernel.ok()) return kernel.status();
  if (!kernel->open) {
    return EngineListenerObservation{listener_identity, false, false, false,
                                     false, 0, 0};
  }
  auto authority = authority_->enabled(listener_identity);
  if (!authority.ok()) return authority.status();
  return EngineListenerObservation{
      listener_identity, true, kernel->socket, kernel->kernel_listening,
      *authority, kernel->device_identity, kernel->inode_identity};
}

}  // namespace pih
