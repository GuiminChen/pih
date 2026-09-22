#pragma once

#include "pih/model/engine_allocation_lease.h"

namespace pih {

class LinuxEngineAllocationLeaseProbe final {
 public:
  static Result<EngineAllocationLeaseObservation> Observe(
      std::int32_t inherited_lease_fd);
};

}  // namespace pih
