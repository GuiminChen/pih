#pragma once

#include "pih/model/engine_inet_diag_transport_adapter.h"

namespace pih {

struct EngineInetDiagNamespaceHandleObservation final {
  bool namespace_descriptor_open = false;
  bool namespace_is_network_namespace = false;
  bool netlink_descriptor_open = false;
  bool netlink_is_sock_diag = false;
};

class EngineInetDiagNamespaceHandleProbe {
 public:
  virtual ~EngineInetDiagNamespaceHandleProbe() = default;
  virtual Result<EngineInetDiagNamespaceHandleObservation> inspect(
      std::int32_t namespace_descriptor,
      std::int32_t netlink_descriptor) = 0;
};

class EngineInetDiagNamespaceHandleGate final {
 public:
  static Result<EngineInetDiagNamespaceHandleGate> Create(
      std::int32_t namespace_descriptor,
      std::int32_t netlink_descriptor,
      EngineInetDiagNamespaceHandleProbe& probe);

  Result<EngineInetDiagNamespaceHandleState> inspect();

 private:
  EngineInetDiagNamespaceHandleGate(
      std::int32_t namespace_descriptor,
      std::int32_t netlink_descriptor,
      EngineInetDiagNamespaceHandleProbe& probe) noexcept
      : namespace_descriptor_(namespace_descriptor),
        netlink_descriptor_(netlink_descriptor), probe_(&probe) {}

  std::int32_t namespace_descriptor_ = -1;
  std::int32_t netlink_descriptor_ = -1;
  EngineInetDiagNamespaceHandleProbe* probe_ = nullptr;
};

}  // namespace pih
