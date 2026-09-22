#include "pih/model/engine_inet_diag_namespace_handle_gate.h"

namespace pih {

Result<EngineInetDiagNamespaceHandleGate>
EngineInetDiagNamespaceHandleGate::Create(
    std::int32_t namespace_descriptor,
    std::int32_t netlink_descriptor,
    EngineInetDiagNamespaceHandleProbe& probe) {
  if (namespace_descriptor < 0 || netlink_descriptor < 0 ||
      namespace_descriptor == netlink_descriptor) {
    return Status::InvalidArgument(
        "inet_diag namespace handle descriptors are invalid");
  }
  return EngineInetDiagNamespaceHandleGate(
      namespace_descriptor, netlink_descriptor, probe);
}

Result<EngineInetDiagNamespaceHandleState>
EngineInetDiagNamespaceHandleGate::inspect() {
  auto observation = probe_->inspect(namespace_descriptor_,
                                     netlink_descriptor_);
  if (!observation.ok()) return observation.status();
  if (!observation->namespace_descriptor_open &&
      !observation->netlink_descriptor_open) {
    if (observation->namespace_is_network_namespace ||
        observation->netlink_is_sock_diag) {
      return Status::FailedPrecondition(
          "closed inet_diag handles retained type evidence");
    }
    return EngineInetDiagNamespaceHandleState::kDestroyed;
  }
  if (!observation->namespace_descriptor_open ||
      !observation->netlink_descriptor_open) {
    return Status::FailedPrecondition(
        "inet_diag namespace handles are partially torn down");
  }
  if (!observation->namespace_is_network_namespace ||
      !observation->netlink_is_sock_diag) {
    return Status::FailedPrecondition(
        "inet_diag namespace handle type drifted");
  }
  return EngineInetDiagNamespaceHandleState::kPresent;
}

}  // namespace pih
