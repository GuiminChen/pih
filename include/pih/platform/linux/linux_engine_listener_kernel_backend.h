#pragma once

#include "pih/model/bound_engine_listener_operations.h"

namespace pih {

class LinuxEngineListenerKernelBackend final
    : public EngineListenerKernelBackend {
 public:
  Result<EngineListenerKernelState> inspect(
      std::int32_t descriptor) override;
};

}  // namespace pih
