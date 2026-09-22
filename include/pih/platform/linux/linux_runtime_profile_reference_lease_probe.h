#pragma once

#include <array>
#include <cstdint>

#include "pih/model/runtime_profile_reference_lease.h"

namespace pih {

class LinuxRuntimeProfileReferenceLeaseProbe final
    : public RuntimeProfileReferenceLeaseProbe {
 public:
  static Result<LinuxRuntimeProfileReferenceLeaseProbe> Create(
      std::array<std::int32_t, 7> inherited_fds);
  Result<RuntimeProfileReferenceLeaseObservation> observe(
      const RuntimeProfileReferenceDescriptor& descriptor) override;
  Result<std::vector<std::byte>> read_bounded_object(
      const RuntimeProfileReferenceDescriptor& descriptor,
      std::uint64_t maximum_bytes) const;

 private:
  explicit LinuxRuntimeProfileReferenceLeaseProbe(
      std::array<std::int32_t, 7> inherited_fds) noexcept;
  std::array<std::int32_t, 7> inherited_fds_{};
};

}  // namespace pih
