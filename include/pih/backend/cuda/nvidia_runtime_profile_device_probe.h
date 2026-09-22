#pragma once

#include "pih/model/runtime_profile_device_gate.h"

namespace pih {

class NvidiaRuntimeProfileDeviceProbe final : public RuntimeProfileDeviceProbe {
 public:
  Result<RuntimeProfileDeviceObservation> observe(
      std::int32_t ordinal) override;
};

}  // namespace pih
