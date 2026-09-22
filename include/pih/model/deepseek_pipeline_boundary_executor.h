#pragma once

#include "pih/model/deepseek_nccl_boundary_lease.h"
#include "pih/model/deepseek_pipeline_transaction.h"

namespace pih {

enum class DeepSeekBoundaryExecutorState : std::uint8_t {
  kReady,
  kIssuePending,
  kDeviceInFlight,
  kCompleteVerified,
  kPoisoned,
};

class DeepSeekBoundaryOperation {
 public:
  virtual ~DeepSeekBoundaryOperation() = default;
  virtual Status submit(DeepSeekNcclP2pBindingTarget& binding,
                        DeepSeekNcclP2pDriver& nccl,
                        CompletionEventDriver& event_driver) = 0;
  virtual Status poll_issue(DeepSeekNcclP2pDriver& nccl,
                            CompletionEventDriver& event_driver) = 0;
  virtual Status poll_completion(DeepSeekNcclP2pDriver& nccl,
                                 CompletionEventDriver& event_driver,
                                 CompletionEvidenceProvider& evidence) = 0;
  virtual Status expire(std::uint64_t now_ns) {
    return Status::FailedPrecondition(
        "DeepSeek boundary operation has no deadline frontier");
  }
  [[nodiscard]] virtual DeepSeekBoundaryExecutorState state() const noexcept = 0;
  [[nodiscard]] virtual std::uintptr_t leased_buffer_address()
      const noexcept { return 0; }
  [[nodiscard]] virtual std::uint64_t leased_buffer_bytes()
      const noexcept { return 0; }
};

class DeepSeekPipelineBoundaryExecutor final : public DeepSeekBoundaryOperation {
 public:
  static Result<DeepSeekPipelineBoundaryExecutor> Create(
      const DeepSeekPipelineTransaction& transaction,
      DeepSeekNcclP2pPlan& operation,
      const DeepSeekNcclBoundaryLease& boundary,
      DeepSeekNcclOperationSequencer& sequencer,
      CompletionEventSlot& completion_event,
      CudaCompletionFrontier& completion_frontier);

  Status submit(DeepSeekNcclP2pBindingTarget& binding,
                DeepSeekNcclP2pDriver& nccl,
                CompletionEventDriver& event_driver) override;
  Status poll_issue(DeepSeekNcclP2pDriver& nccl,
                    CompletionEventDriver& event_driver) override;
  Status poll_completion(DeepSeekNcclP2pDriver& nccl,
                         CompletionEventDriver& event_driver,
                         CompletionEvidenceProvider& evidence) override;
  Status expire(std::uint64_t now_ns) override;
  [[nodiscard]] DeepSeekBoundaryExecutorState state() const noexcept override {
    return state_;
  }
  [[nodiscard]] std::uintptr_t leased_buffer_address()
      const noexcept override {
    return boundary_ == nullptr
               ? 0
               : reinterpret_cast<std::uintptr_t>(boundary_->data());
  }
  [[nodiscard]] std::uint64_t leased_buffer_bytes()
      const noexcept override {
    return boundary_ == nullptr ? 0 : boundary_->bytes();
  }

 private:
  DeepSeekPipelineBoundaryExecutor(
      const DeepSeekPipelineTransaction& transaction,
      DeepSeekNcclP2pPlan& operation,
      const DeepSeekNcclBoundaryLease& boundary,
      DeepSeekNcclOperationSequencer& sequencer,
      CompletionEventSlot& completion_event,
      CudaCompletionFrontier& completion_frontier) noexcept
      : transaction_(&transaction), operation_(&operation), boundary_(&boundary),
        sequencer_(&sequencer), completion_event_(&completion_event),
        completion_frontier_(&completion_frontier) {}
  Status record_event(CompletionEventDriver& event_driver);
  Status poison(Status status);

  const DeepSeekPipelineTransaction* transaction_ = nullptr;
  DeepSeekNcclP2pPlan* operation_ = nullptr;
  const DeepSeekNcclBoundaryLease* boundary_ = nullptr;
  DeepSeekNcclOperationSequencer* sequencer_ = nullptr;
  CompletionEventSlot* completion_event_ = nullptr;
  CudaCompletionFrontier* completion_frontier_ = nullptr;
  DeepSeekBoundaryExecutorState state_ = DeepSeekBoundaryExecutorState::kReady;
};

}  // namespace pih
