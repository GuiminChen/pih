#pragma once

#include "pih/model/engine_inet_diag_transport_adapter.h"

namespace pih {

struct EngineInetDiagTransportBinding final {
  std::uint64_t namespace_identity = 0;
  std::int32_t namespace_descriptor = -1;
  std::int32_t netlink_descriptor = -1;
};

class EngineInetDiagDescriptorBackend {
 public:
  virtual ~EngineInetDiagDescriptorBackend() = default;
  virtual Result<EngineInetDiagNamespaceHandleState> namespace_state(
      std::int32_t namespace_descriptor,
      std::int32_t netlink_descriptor) = 0;
  virtual Result<EngineInetDiagDumpTicket> begin_dump(
      std::int32_t netlink_descriptor,
      std::uint64_t maximum_duration_ms) = 0;
  virtual Result<std::optional<EngineInetDiagMultipartMessage>> receive(
      std::int32_t netlink_descriptor,
      const EngineInetDiagDumpTicket& ticket) = 0;
};

class BoundEngineInetDiagTransport final : public EngineInetDiagTransport {
 public:
  static Result<BoundEngineInetDiagTransport> Create(
      EngineInetDiagTransportBinding binding,
      EngineInetDiagDescriptorBackend& backend);

  Result<EngineInetDiagNamespaceHandleState> namespace_state(
      std::uint64_t namespace_identity) override;
  Result<EngineInetDiagDumpTicket> begin_dump(
      std::uint64_t namespace_identity,
      std::uint64_t maximum_duration_ms) override;
  Result<std::optional<EngineInetDiagMultipartMessage>> receive(
      const EngineInetDiagDumpTicket& ticket) override;

 private:
  BoundEngineInetDiagTransport(
      EngineInetDiagTransportBinding binding,
      EngineInetDiagDescriptorBackend& backend) noexcept
      : binding_(binding), backend_(&backend) {}

  EngineInetDiagTransportBinding binding_{};
  EngineInetDiagDescriptorBackend* backend_ = nullptr;
};

}  // namespace pih
