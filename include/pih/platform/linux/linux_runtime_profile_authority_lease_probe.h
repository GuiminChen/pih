#pragma once

#include <array>
#include <cstdint>

#include "pih/model/runtime_profile_authority_lease.h"

namespace pih {

class LinuxRuntimeProfileAuthorityLeaseProbe final
    : public RuntimeProfileAuthorityLeaseProbe {
 public:
  static Result<LinuxRuntimeProfileAuthorityLeaseProbe> Create(
      std::array<std::int32_t, 5> inherited_fds);
  Result<RuntimeProfileAuthorityLeaseObservation> observe(
      const RuntimeProfileAuthorityDescriptor& descriptor) override;

 private:
  explicit LinuxRuntimeProfileAuthorityLeaseProbe(
      std::array<std::int32_t, 5> inherited_fds) noexcept;
  std::array<std::int32_t, 5> inherited_fds_{};
};

}  // namespace pih
