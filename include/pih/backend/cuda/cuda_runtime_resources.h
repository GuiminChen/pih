#pragma once

#include <cstdint>

#include "pih/backend/cuda/completion_event_slot.h"

namespace pih {

struct CudaRuntimeResourceIdentity final {
  std::int32_t device_ordinal;
  std::uint32_t rank;
  std::uint64_t worker_generation;
  std::uint32_t context_flags;
  std::uintptr_t context;
  DriverStreamHandle stream;
  DriverEventHandle event;
  DriverStreamHandle scrub_stream;
  DriverEventHandle scrub_event;
  DriverStreamHandle diagnostic_stream;
  DriverEventHandle diagnostic_event;
  DriverEventHandle deepseek_expert_event;
  DriverStreamHandle deepseek_paging_stream;
};

class CudaRuntimeResourceDriver {
 public:
  virtual ~CudaRuntimeResourceDriver() = default;
  virtual Result<std::uintptr_t> retain_primary_context(
      std::int32_t device_ordinal, std::uint32_t context_flags) = 0;
  virtual Status bind_runtime(std::int32_t device_ordinal,
                              std::uintptr_t context) = 0;
  virtual Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t context) = 0;
  virtual Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t context) = 0;
  virtual void destroy_event(DriverEventHandle event) noexcept = 0;
  virtual void destroy_stream(DriverStreamHandle stream) noexcept = 0;
  virtual void release_primary_context(std::int32_t device_ordinal,
                                       std::uintptr_t context) noexcept = 0;
};

class CudaRuntimeResources final {
 public:
  static Result<CudaRuntimeResources> Create(
      std::int32_t device_ordinal, std::uint32_t rank,
      std::uint64_t worker_generation, std::uint32_t context_flags,
      CudaRuntimeResourceDriver& driver);

  ~CudaRuntimeResources();
  CudaRuntimeResources(const CudaRuntimeResources&) = delete;
  CudaRuntimeResources& operator=(const CudaRuntimeResources&) = delete;
  CudaRuntimeResources(CudaRuntimeResources&& other) noexcept;
  CudaRuntimeResources& operator=(CudaRuntimeResources&& other) noexcept;

  [[nodiscard]] const CudaRuntimeResourceIdentity& identity() const noexcept {
    return identity_;
  }

 private:
  CudaRuntimeResources(CudaRuntimeResourceIdentity identity,
                       CudaRuntimeResourceDriver& driver)
      : identity_(identity), driver_(&driver) {}
  void reset() noexcept;

  CudaRuntimeResourceIdentity identity_{};
  CudaRuntimeResourceDriver* driver_ = nullptr;
};

}  // namespace pih
