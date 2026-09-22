#pragma once

#include <limits>
#include <cstdlib>

#include "memory_binding.h"
#include "pih/backend/cuda/atomic_completion_evidence_provider.h"
#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/backend/cuda/cuda_runtime_resources.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/model/qwen3_bf16_execution_prelude.h"

namespace pih::qwen_plugin {
// Called before any state member (weights, arenas, modules) is destroyed,
// including partial-load and exception unwinding. An unretired GPU borrower
// must never observe released memory; failed synchronization is fail-stop.
inline void RetireStreamsOrTerminate(const pih_nvidia_cuda_async_api_v1& api,
                                    const CudaRuntimeResourceIdentity& identity) noexcept {
  try {
    for (auto stream : {identity.stream, identity.scrub_stream, identity.diagnostic_stream}) {
      if (!stream) continue;
      const auto status = api.synchronize_stream(api.context, identity.context, stream);
      if (!pih_status_is_valid_v1(&status) || !pih_status_is_ok_v1(&status)) std::abort();
    }
  } catch (...) { std::abort(); }
}
inline bool ValidAsyncApi(const pih_nvidia_cuda_async_api_v1& api) noexcept {
  return api.struct_size == sizeof(api) &&
      api.contract_version == PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 && api.context &&
      api.activate_context && api.validate_pinned_host && api.copy_async &&
      api.memset_async && api.record_event && api.query_event &&
      api.synchronize_stream && api.require_clean_last_error;
}

class CapabilityErrorClearDriver final : public QwenBf16DeviceErrorClearDriver {
 public:
  CapabilityErrorClearDriver(const pih_nvidia_cuda_async_api_v1& api,
                              std::uintptr_t context, std::int32_t rank)
      : api_(&api), context_(context), rank_(rank) {}
  Status clear_u32_async(const TensorView& target, std::int32_t rank,
                         DriverStreamHandle stream) override {
    const auto address = reinterpret_cast<std::uintptr_t>(target.data());
    if (!ValidAsyncApi(*api_) || !context_ || rank_ < 0 || rank != rank_ || !stream ||
        !address || address % alignof(std::uint32_t) || address > UINTPTR_MAX - 3 ||
        target.dtype() != DType::kUInt8 || target.rank() != 1 || target.dim(0) != 4 ||
        target.stride(0) != 1 || target.device().type() != DeviceType::kCuda ||
        target.device().index() != rank_)
      return Status::InvalidArgument("Qwen error-clear capability target invalid");
    const auto before = MemoryStatus(api_->require_clean_last_error(api_->context, context_));
    if (!before.ok()) return before;
    const auto cleared = MemoryStatus(api_->memset_async(api_->context, context_, address, 0, 4, stream));
    const auto after = MemoryStatus(api_->require_clean_last_error(api_->context, context_));
    return cleared.ok() ? after : cleared;
  }
 private:
  const pih_nvidia_cuda_async_api_v1* api_;
  std::uintptr_t context_;
  std::int32_t rank_;
};

class CapabilityTypedCopyDriver final : public TypedCopyDriver {
 public:
  CapabilityTypedCopyDriver(const pih_nvidia_cuda_async_api_v1& api, std::uintptr_t context)
      : api_(&api), context_(context) {}
  std::uintptr_t context_identity() const noexcept override { return context_; }
  Status copy(CudaCopyKind kind, std::uintptr_t destination, std::uintptr_t source,
              std::uint64_t bytes, DriverStreamHandle stream) override {
    if (!ValidAsyncApi(*api_) || !context_ || !destination || !source || !bytes || !stream)
      return Status::InvalidArgument("Qwen typed copy capability request invalid");
    if (bytes > std::numeric_limits<std::size_t>::max() ||
        bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - destination ||
        bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - source)
      return Status::ResourceExhausted("Qwen typed copy exceeds address space");
    std::uint32_t direction{};
    switch (kind) {
      case CudaCopyKind::kHostToDevice: direction = PIH_CUDA_COPY_H2D_V1; break;
      case CudaCopyKind::kDeviceToHost: direction = PIH_CUDA_COPY_D2H_V1; break;
      case CudaCopyKind::kDeviceToDevice: direction = PIH_CUDA_COPY_D2D_V1; break;
      default: return Status::InvalidArgument("Qwen typed copy kind invalid");
    }
    // Do not switch contexts to hide a caller-thread ownership mismatch. The
    // provider checks the current context and peeks without clearing errors.
    const auto before = MemoryStatus(api_->require_clean_last_error(api_->context, context_));
    if (!before.ok()) return before;
    const auto copied = MemoryStatus(api_->copy_async(
        api_->context, context_, destination, source, bytes, direction, stream));
    const auto after = MemoryStatus(api_->require_clean_last_error(api_->context, context_));
    return copied.ok() ? after : copied;
  }
 private:
  const pih_nvidia_cuda_async_api_v1* api_;
  std::uintptr_t context_;
};

class CapabilityEventDriver final : public CompletionEventDriver,
                                    public CompletionLastErrorProbe {
 public:
  CapabilityEventDriver(const pih_nvidia_cuda_async_api_v1& api, std::uintptr_t context)
      : api_(&api), context_(context) {}
  Status record(DriverEventHandle event, DriverStreamHandle stream) override {
    if (!ValidAsyncApi(*api_) || !context_ || !event || !stream)
      return Status::InvalidArgument("Qwen event record capability request invalid");
    return MemoryStatus(api_->record_event(api_->context, context_, event, stream));
  }
  Result<CudaEventQueryResult> query(DriverEventHandle event) override {
    if (!ValidAsyncApi(*api_) || !context_ || !event)
      return Status::InvalidArgument("Qwen event query capability request invalid");
    std::uint32_t result{};
    const auto status = MemoryStatus(api_->query_event(api_->context, context_, event, &result));
    if (!status.ok()) return status;
    if (result == PIH_CUDA_EVENT_COMPLETE_V1) return CudaEventQueryResult::kSuccess;
    if (result == PIH_CUDA_EVENT_PENDING_V1) return CudaEventQueryResult::kNotReady;
    return Status::FailedPrecondition("Qwen event provider returned an invalid result");
  }
  Status require_clean_last_error() override {
    if (!ValidAsyncApi(*api_) || !context_)
      return Status::InvalidArgument("Qwen last-error capability binding invalid");
    return MemoryStatus(api_->require_clean_last_error(api_->context, context_));
  }
 private:
  const pih_nvidia_cuda_async_api_v1* api_;
  std::uintptr_t context_;
};
}  // namespace pih::qwen_plugin
