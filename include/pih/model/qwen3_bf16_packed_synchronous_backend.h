#pragma once

#include <vector>

#include "pih/model/qwen3_bf16_packed_batch_transaction.h"
#include "pih/model/qwen3_bf16_packed_step_builder.h"
#include "pih/model/qwen3_bf16_packed_step_transaction.h"
#include "pih/model/qwen3_bf16_synchronous_backend.h"

namespace pih {

using QwenBf16PackedBackendDrivers = QwenBf16SynchronousBackendDrivers;

struct QwenBf16PackedBackendArenas final {
  QwenBf16StepDeviceOwners device;
  CudaCopyEndpoint pinned_staging, device_staging, sampled_tokens,
      device_error, pinned_result;
  std::span<std::byte> pinned_staging_backing;
  std::span<std::byte> pinned_result_backing;
};

struct QwenBf16PackedBackendIdentity final {
  std::uint64_t epoch, first_request_generation, first_event_generation,
      first_plan_id, timeout_ns;
  std::uintptr_t context_identity;
  DriverStreamHandle stream;
  DriverEventHandle event;
  std::int32_t owning_rank;
  std::uint32_t slot_count, sample_capacity;
  float rms_epsilon, attention_scale;
};

enum class QwenBf16PackedBackendState : std::uint8_t {
  kReady = 1, kRunning = 2, kPoisoned = 3,
};

class QwenBf16PackedSynchronousBackend final
    : public QwenBf16PackedBatchBackend {
 public:
  static Result<QwenBf16PackedSynchronousBackend> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> legacy_functions,
      std::span<const ResolvedKernelFunction> packed_functions,
      const QwenBf16WeightResourceSet& weights,
      QwenBf16PackedBackendArenas arenas,
      QwenBf16PackedBackendIdentity identity,
      QwenBf16PackedBackendDrivers drivers);

  QwenBf16PackedSynchronousBackend(
      const QwenBf16PackedSynchronousBackend&) = delete;
  QwenBf16PackedSynchronousBackend(QwenBf16PackedSynchronousBackend&&) noexcept = default;

  Result<QwenBf16PackedBatchExecutionView> execute_packed(
      const PackedTokenPlan& plan, PackedTokenMetadataView metadata,
      QwenBf16PackedKvMetadataView kv_metadata,
      std::span<const QwenBf16PackedKvBinding> bindings,
      std::span<const QwenKvAppendPlan> append_plans,
      std::span<const Qwen3SamplingDescriptor> sampling) override;
  [[nodiscard]] QwenBf16PackedBackendState state() const noexcept { return state_; }

 private:
  QwenBf16PackedSynchronousBackend(
      const QwenBf16CommandBuffer& c,
      std::span<const ResolvedKernelFunction> l,
      std::span<const ResolvedKernelFunction> p,
      const QwenBf16WeightResourceSet& w, QwenBf16PackedBackendArenas a,
      QwenBf16PackedBackendIdentity i, QwenBf16PackedBackendDrivers d)
      : commands_(&c), legacy_(l), packed_(p), weights_(&w), arenas_(a),
        identity_(i), drivers_(d), request_generation_(i.first_request_generation),
        event_generation_(i.first_event_generation), plan_id_(i.first_plan_id) {}

  const QwenBf16CommandBuffer* commands_;
  std::span<const ResolvedKernelFunction> legacy_, packed_;
  const QwenBf16WeightResourceSet* weights_;
  QwenBf16PackedBackendArenas arenas_;
  QwenBf16PackedBackendIdentity identity_;
  QwenBf16PackedBackendDrivers drivers_;
  std::uint64_t request_generation_, event_generation_, plan_id_;
  std::vector<std::uint32_t> sampled_tokens_;
  std::vector<QwenBf16PackedSampleReceipt> sampling_receipts_;
  QwenBf16PackedBackendState state_ = QwenBf16PackedBackendState::kReady;
};

}  // namespace pih
