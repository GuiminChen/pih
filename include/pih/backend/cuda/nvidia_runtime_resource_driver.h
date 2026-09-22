#pragma once

#include "pih/backend/cuda/cuda_runtime_resources.h"

namespace pih {

class NvidiaRuntimeResourceDriver final : public CudaRuntimeResourceDriver {
 public:
  Result<std::uintptr_t> retain_primary_context(
      std::int32_t device_ordinal, std::uint32_t context_flags) override;
  Status bind_runtime(std::int32_t device_ordinal,
                      std::uintptr_t context) override;
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t context) override;
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t context) override;
  void destroy_event(DriverEventHandle event) noexcept override;
  void destroy_stream(DriverStreamHandle stream) noexcept override;
  void release_primary_context(std::int32_t device_ordinal,
                               std::uintptr_t context) noexcept override;

 private:
  Status require_current(std::uintptr_t context) const;
};

}  // namespace pih
