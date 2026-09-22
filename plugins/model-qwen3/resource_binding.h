#pragma once

#include <cstdlib>

#include "memory_binding.h"
#include "runtime_resources.h"
#include "pih/contracts/nvidia_cuda_resources_v1.h"

namespace pih::qwen_plugin {
inline bool ValidResourceApi(const pih_nvidia_cuda_resources_api_v1& api) noexcept {
  return api.struct_size == sizeof(api) &&
      api.contract_version == PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1 && api.context &&
      api.retain_primary_context && api.bind_runtime && api.create_nonblocking_stream &&
      api.create_disable_timing_event && api.destroy_event && api.destroy_stream &&
      api.release_primary_context;
}

// Holds only a borrowed capability table; the backend owns all native handles.
// The driver must outlive RuntimeResources and the provider must outlive both.
class CapabilityResourceDriver final : public CudaRuntimeResourceDriver {
 public:
  explicit CapabilityResourceDriver(const pih_nvidia_cuda_resources_api_v1& api)
      : api_(&api) {}
  Result<std::uintptr_t> retain_primary_context(std::int32_t ordinal,
                                               std::uint32_t flags) override {
    if (!ValidResourceApi(*api_) || ordinal < 0 || flags != PIH_CUDA_CONTEXT_SCHED_YIELD_V1)
      return Status::InvalidArgument("Qwen context capability request invalid");
    std::uintptr_t handle{};
    const auto status = MemoryStatus(api_->retain_primary_context(api_->context, ordinal, flags, &handle));
    if (!status.ok()) return status;
    if (!handle) return Status::FailedPrecondition("Qwen context provider returned a null handle");
    return handle;
  }
  Status bind_runtime(std::int32_t ordinal, std::uintptr_t context) override {
    if (!ValidResourceApi(*api_) || ordinal < 0 || !context)
      return Status::InvalidArgument("Qwen runtime capability request invalid");
    return MemoryStatus(api_->bind_runtime(api_->context, ordinal, context));
  }
  Result<DriverStreamHandle> create_nonblocking_stream(std::uintptr_t context) override {
    return Create(context, true);
  }
  Result<DriverEventHandle> create_disable_timing_event(std::uintptr_t context) override {
    return Create(context, false);
  }
  void destroy_event(DriverEventHandle event) noexcept override {
    Retire([&] { return api_->destroy_event(api_->context, event); });
  }
  void destroy_stream(DriverStreamHandle stream) noexcept override {
    Retire([&] { return api_->destroy_stream(api_->context, stream); });
  }
  void release_primary_context(std::int32_t ordinal, std::uintptr_t context) noexcept override {
    Retire([&] { return api_->release_primary_context(api_->context, ordinal, context); });
  }
 private:
  Result<std::uintptr_t> Create(std::uintptr_t context, bool stream) {
    if (!ValidResourceApi(*api_) || !context)
      return Status::InvalidArgument("Qwen resource capability request invalid");
    std::uintptr_t handle{};
    const auto create = stream ? api_->create_nonblocking_stream : api_->create_disable_timing_event;
    const auto status = MemoryStatus(create(api_->context, context, &handle));
    if (!status.ok()) return status;
    if (!handle) return Status::FailedPrecondition("Qwen resource provider returned a null handle");
    return handle;
  }
  template<class Operation> void Retire(Operation operation) noexcept {
    try {
      if (!ValidResourceApi(*api_)) std::abort();
      const auto status = operation();
      if (!pih_status_is_ok_v1(&status)) std::abort();
    } catch (...) { std::abort(); }
  }
  const pih_nvidia_cuda_resources_api_v1* api_;
};
}  // namespace pih::qwen_plugin
