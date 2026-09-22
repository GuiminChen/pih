#include "pih/model/bound_engine_inet_diag_transport.h"

namespace pih {

Result<BoundEngineInetDiagTransport> BoundEngineInetDiagTransport::Create(
    EngineInetDiagTransportBinding binding,
    EngineInetDiagDescriptorBackend& backend) {
  if (binding.namespace_identity == 0 ||
      binding.namespace_descriptor < 0 || binding.netlink_descriptor < 0 ||
      binding.namespace_descriptor == binding.netlink_descriptor) {
    return Status::InvalidArgument(
        "bound inet_diag transport binding is invalid");
  }
  return BoundEngineInetDiagTransport(binding, backend);
}

Result<EngineInetDiagNamespaceHandleState>
BoundEngineInetDiagTransport::namespace_state(
    std::uint64_t namespace_identity) {
  if (namespace_identity != binding_.namespace_identity) {
    return Status::FailedPrecondition(
        "inet_diag namespace identity is not bound");
  }
  return backend_->namespace_state(binding_.namespace_descriptor,
                                   binding_.netlink_descriptor);
}

Result<EngineInetDiagDumpTicket> BoundEngineInetDiagTransport::begin_dump(
    std::uint64_t namespace_identity,
    std::uint64_t maximum_duration_ms) {
  if (namespace_identity != binding_.namespace_identity) {
    return Status::FailedPrecondition(
        "inet_diag namespace identity is not bound");
  }
  return backend_->begin_dump(binding_.netlink_descriptor,
                              maximum_duration_ms);
}

Result<std::optional<EngineInetDiagMultipartMessage>>
BoundEngineInetDiagTransport::receive(
    const EngineInetDiagDumpTicket& ticket) {
  return backend_->receive(binding_.netlink_descriptor, ticket);
}

}  // namespace pih
