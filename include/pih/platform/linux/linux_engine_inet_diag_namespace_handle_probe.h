#pragma once

#include "pih/model/engine_inet_diag_namespace_handle_gate.h"

namespace pih {

class LinuxEngineInetDiagNamespaceHandleProbe final
    : public EngineInetDiagNamespaceHandleProbe {
 public:
  Result<EngineInetDiagNamespaceHandleObservation> inspect(
      std::int32_t namespace_descriptor,
      std::int32_t netlink_descriptor) override;
};

}  // namespace pih
