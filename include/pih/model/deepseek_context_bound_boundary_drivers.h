#pragma once

#include "pih/backend/cuda/atomic_completion_evidence_provider.h"
#include "pih/model/deepseek_context_bound_nccl_api.h"
#include "pih/model/deepseek_nccl_endpoint_warmup_runner.h"

namespace pih {

class DeepSeekContextBoundCompletionDriver final
    : public CompletionEventDriver, public CompletionLastErrorProbe {
 public:
  static Result<DeepSeekContextBoundCompletionDriver> Create(
      std::uintptr_t context_identity, CompletionEventDriver& events,
      CompletionLastErrorProbe& last_error,
      DeepSeekNcclContextActivator& activator);

  Status record(DriverEventHandle event,
                DriverStreamHandle stream) override;
  Result<CudaEventQueryResult> query(DriverEventHandle event) override;
  Status require_clean_last_error() override;

 private:
  DeepSeekContextBoundCompletionDriver(
      std::uintptr_t context_identity, CompletionEventDriver& events,
      CompletionLastErrorProbe& last_error,
      DeepSeekNcclContextActivator& activator) noexcept
      : context_identity_(context_identity), events_(&events),
        last_error_(&last_error), activator_(&activator) {}
  Status activate();

  std::uintptr_t context_identity_ = 0;
  CompletionEventDriver* events_ = nullptr;
  CompletionLastErrorProbe* last_error_ = nullptr;
  DeepSeekNcclContextActivator* activator_ = nullptr;
};

class DeepSeekContextBoundWarmupPayloadOperations final
    : public DeepSeekNcclWarmupPayloadOperations {
 public:
  static Result<DeepSeekContextBoundWarmupPayloadOperations> Create(
      std::uintptr_t context_identity,
      DeepSeekNcclWarmupPayloadOperations& payload,
      DeepSeekNcclContextActivator& activator);

  Status prepare(DeepSeekNcclRole role, void* device_buffer,
                 std::uint64_t bytes, DriverStreamHandle stream,
                 std::uint64_t pattern_identity) override;
  Result<Sha256Digest> digest(const void* device_buffer,
                              std::uint64_t bytes) override;

 private:
  DeepSeekContextBoundWarmupPayloadOperations(
      std::uintptr_t context_identity,
      DeepSeekNcclWarmupPayloadOperations& payload,
      DeepSeekNcclContextActivator& activator) noexcept
      : context_identity_(context_identity), payload_(&payload),
        activator_(&activator) {}
  Status activate();

  std::uintptr_t context_identity_ = 0;
  DeepSeekNcclWarmupPayloadOperations* payload_ = nullptr;
  DeepSeekNcclContextActivator* activator_ = nullptr;
};

}  // namespace pih
