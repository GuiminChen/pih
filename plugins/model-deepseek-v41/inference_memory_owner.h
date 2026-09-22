#pragma once
#include "inference_memory.h"
#include "inference_request.h"

namespace pih::deepseek_v41 {
enum class InferenceMemoryState { kEmpty, kInitializing, kReady, kRunning, kRetiring, kReleased, kQuarantined };
// Explicit-release ledger. Destruction never synchronizes or frees resources.
// Capabilities, plan, retained context, stream and exclusive event are borrowed.
class InferenceMemoryOwner final {
 public:
  InferenceMemoryOwner(const InferenceMemoryPlan& plan, const pih_nvidia_cuda_memory_api_v1& memory,
      const pih_nvidia_cuda_async_api_v1& async, std::int32_t device,
      std::uintptr_t context, std::uintptr_t stream, std::uintptr_t retirement_event);
  InferenceMemoryOwner(const InferenceMemoryOwner&) = delete;
  InferenceMemoryOwner& operator=(const InferenceMemoryOwner&) = delete;
  Status Allocate();
  // Completes initialization/step retirement, or advances the active step.
  Result<bool> Advance();
  Status StartStep(const FlashConfig& config, BlockSequence& sequence, EngramHashState& hashes,
      std::span<const std::uint32_t> tokens, const BackboneWeightUpload& weights,
      ExpertWorkspaceOwner& workspace, std::uintptr_t communicator, std::uintptr_t completion_event,
      const SamplingParameters& parameters, const SamplingIdentity& identity,
      InferenceOperation::Clock::time_point deadline);
  Status StartRequest(const FlashConfig& config, BlockSequence& sequence, EngramHashState& hashes,
      const InferenceRequest& request, const BackboneWeightUpload& weights, ExpertWorkspaceOwner& workspace,
      std::uintptr_t communicator, std::uintptr_t completion_event, InferenceOperation::Clock::time_point deadline);
  Result<EngramDeviceRegion> Logits() const;
  Result<SamplingObservation> Candidate() const;
  Result<RankStepReceipt> Receipt() const;
  Status Release();
  void Quarantine() noexcept { state_ = InferenceMemoryState::kQuarantined; operation_.reset(); }
  InferenceMemoryState state() const noexcept { return state_; }
  const pih_cuda_allocation_v1& device_record() const noexcept { return device_allocation_; }
  const pih_cuda_allocation_v1& host_record() const noexcept { return host_allocation_; }
  bool device_released() const noexcept { return device_released_; }
  bool host_released() const noexcept { return host_released_; }
 private:
  Result<bool> AdvanceImpl();
  const InferenceMemoryPlan& plan_;
  const pih_nvidia_cuda_memory_api_v1& memory_;
  const pih_nvidia_cuda_async_api_v1& async_;
  std::int32_t device_;
  std::uintptr_t context_, stream_, event_;
  pih_cuda_allocation_v1 device_allocation_{sizeof(pih_cuda_allocation_v1), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  pih_cuda_allocation_v1 host_allocation_{sizeof(pih_cuda_allocation_v1), PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1};
  bool device_released_ = false, host_released_ = false;
  std::unique_ptr<InferenceOperation> operation_;
  InferenceMemoryState state_ = InferenceMemoryState::kEmpty;
};
}  // namespace pih::deepseek_v41
