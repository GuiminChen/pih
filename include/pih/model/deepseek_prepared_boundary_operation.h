#pragma once

#include <memory>

#include "pih/model/deepseek_pipeline_boundary_executor.h"

namespace pih {

class DeepSeekPreparedBoundaryOperation final
    : public DeepSeekBoundaryOperation {
 public:
  static Result<DeepSeekPreparedBoundaryOperation> Create(
      DeepSeekPipelineTransaction& transaction,
      DeepSeekNcclP2pManifest manifest, Buffer& buffer,
      std::uint64_t buffer_owner_id, std::uintptr_t context_identity,
      std::uintptr_t boundary_stream, DriverEventHandle completion_event,
      DeepSeekNcclOperationSequencer& sequencer,
      std::uint64_t submit_ns, std::uint64_t deadline_ns);

  Status submit(DeepSeekNcclP2pBindingTarget& binding,
                DeepSeekNcclP2pDriver& nccl,
                CompletionEventDriver& event_driver) override;
  Status poll_issue(DeepSeekNcclP2pDriver& nccl,
                    CompletionEventDriver& event_driver) override;
  Status poll_completion(DeepSeekNcclP2pDriver& nccl,
                         CompletionEventDriver& event_driver,
                         CompletionEvidenceProvider& evidence) override;
  Status expire(std::uint64_t now_ns) override;
  [[nodiscard]] DeepSeekBoundaryExecutorState state() const noexcept override;
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
  [[nodiscard]] const DeepSeekNcclP2pManifest& manifest() const noexcept {
    return operation_->manifest();
  }

 private:
  DeepSeekPreparedBoundaryOperation(
      DeepSeekPipelineTransaction& transaction,
      std::unique_ptr<DeepSeekNcclP2pPlan> operation,
      std::unique_ptr<DeepSeekNcclBoundaryLease> boundary,
      DeepSeekNcclOperationSequencer& sequencer,
      std::unique_ptr<CompletionEventSlot> completion_event,
      std::unique_ptr<CudaCompletionFrontier> completion_frontier) noexcept
      : transaction_(&transaction), operation_(std::move(operation)),
        boundary_(std::move(boundary)), sequencer_(&sequencer),
        completion_event_(std::move(completion_event)),
        completion_frontier_(std::move(completion_frontier)) {}

  Status ensure_executor();

  DeepSeekPipelineTransaction* transaction_;
  std::unique_ptr<DeepSeekNcclP2pPlan> operation_;
  std::unique_ptr<DeepSeekNcclBoundaryLease> boundary_;
  DeepSeekNcclOperationSequencer* sequencer_;
  std::unique_ptr<CompletionEventSlot> completion_event_;
  std::unique_ptr<CudaCompletionFrontier> completion_frontier_;
  std::unique_ptr<DeepSeekPipelineBoundaryExecutor> executor_;
  DeepSeekBoundaryExecutorState state_ =
      DeepSeekBoundaryExecutorState::kReady;
};

}  // namespace pih
