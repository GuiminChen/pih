#pragma once

#include <memory>

#include "pih/model/deepseek_boundary_credit_tracker.h"
#include "pih/model/deepseek_prepared_boundary_operation.h"

namespace pih {

class DeepSeekTrackedBoundaryOperation final
    : public DeepSeekBoundaryOperation {
 public:
  static Result<std::unique_ptr<DeepSeekTrackedBoundaryOperation>> Create(
      std::unique_ptr<DeepSeekPreparedBoundaryOperation> operation,
      DeepSeekBoundaryCreditTracker& tracker,
      DeepSeekBoundaryCreditHandle handle);

  DeepSeekTrackedBoundaryOperation(
      const DeepSeekTrackedBoundaryOperation&) = delete;
  DeepSeekTrackedBoundaryOperation& operator=(
      const DeepSeekTrackedBoundaryOperation&) = delete;
  ~DeepSeekTrackedBoundaryOperation() override;

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
    return operation_->state();
  }
  [[nodiscard]] std::uintptr_t leased_buffer_address()
      const noexcept override {
    return operation_ == nullptr ? 0 : operation_->leased_buffer_address();
  }
  [[nodiscard]] std::uint64_t leased_buffer_bytes()
      const noexcept override {
    return operation_ == nullptr ? 0 : operation_->leased_buffer_bytes();
  }
  [[nodiscard]] const DeepSeekNcclP2pManifest& manifest() const noexcept {
    return operation_->manifest();
  }

 private:
  DeepSeekTrackedBoundaryOperation(
      std::unique_ptr<DeepSeekPreparedBoundaryOperation> operation,
      DeepSeekBoundaryCreditTracker& tracker,
      DeepSeekBoundaryCreditHandle handle) noexcept
      : operation_(std::move(operation)), tracker_(&tracker), handle_(handle) {}
  Status quarantine(Status cause) noexcept;

  std::unique_ptr<DeepSeekPreparedBoundaryOperation> operation_;
  DeepSeekBoundaryCreditTracker* tracker_;
  DeepSeekBoundaryCreditHandle handle_;
  bool committed_ = false;
  bool complete_verified_ = false;
};

}  // namespace pih
