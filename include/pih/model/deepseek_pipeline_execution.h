#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pih/model/deepseek_pipeline_coordinator.h"
#include "pih/model/deepseek_request_registry.h"

namespace pih {

struct DeepSeekRequestIdentity final {
  std::uint64_t request_id = 0;
  std::uint64_t request_generation = 0;
  bool terminal_on_complete = true;
};

class DeepSeekPipelineExecution final {
 public:
  static Result<DeepSeekPipelineExecution> Create(
      DeepSeekRequestRegistry& requests,
      DeepSeekPipelineResourceSet& resources,
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRequestIdentity> request_identities);

  DeepSeekPipelineExecution(const DeepSeekPipelineExecution&) = delete;
  DeepSeekPipelineExecution& operator=(const DeepSeekPipelineExecution&) = delete;
  DeepSeekPipelineExecution(DeepSeekPipelineExecution&&) noexcept = default;
  DeepSeekPipelineExecution& operator=(
      DeepSeekPipelineExecution&& other) noexcept;

  Status stage_ready(std::uint32_t rank);
  [[nodiscard]] Status validate_stage_reject(std::uint32_t rank) const;
  Status stage_reject(std::uint32_t rank, Status reason);
  [[nodiscard]] Status validate_commit() const;
  Status commit();
  [[nodiscard]] Status validate_cancel_request(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Status cancel_request(std::uint64_t request_id,
                        std::uint64_t request_generation);
  [[nodiscard]] Status validate_stage_complete(std::uint32_t rank) const;
  Status stage_complete(std::uint32_t rank);
  [[nodiscard]] Status validate_stage_failed(std::uint32_t rank) const;
  Status stage_failed(std::uint32_t rank, Status reason);

  [[nodiscard]] DeepSeekPipelineCoordinatorState state() const noexcept {
    return coordinator_->state();
  }
  [[nodiscard]] bool contains_request(
      std::uint64_t request_id, std::uint64_t request_generation) const noexcept;
  [[nodiscard]] const std::vector<DeepSeekRequestIdentity>& request_identities()
      const noexcept { return request_identities_; }
  [[nodiscard]] std::uint64_t plan_sequence() const noexcept {
    return transaction_->descriptor().plan_sequence;
  }

 private:
  DeepSeekPipelineExecution(
      DeepSeekRequestRegistry& requests,
      std::vector<DeepSeekRequestIdentity> request_identities,
      std::unique_ptr<DeepSeekPipelineTransaction> transaction,
      std::unique_ptr<DeepSeekPipelineCoordinator> coordinator)
      : requests_(&requests), request_identities_(std::move(request_identities)),
        transaction_(std::move(transaction)), coordinator_(std::move(coordinator)) {}
  void rollback_prepared_requests() noexcept;

  DeepSeekRequestRegistry* requests_ = nullptr;
  std::vector<DeepSeekRequestIdentity> request_identities_;
  std::unique_ptr<DeepSeekPipelineTransaction> transaction_;
  std::unique_ptr<DeepSeekPipelineCoordinator> coordinator_;
};

}  // namespace pih
