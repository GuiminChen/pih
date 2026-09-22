#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <optional>
#include <vector>

#include "pih/model/deepseek_pipeline_execution.h"
#include "pih/model/deepseek_sampler.h"
#include "pih/scheduler/output_burst_credit_pool.h"

namespace pih {

enum class DeepSeekFinishReason : std::uint8_t {
  kNone = 0,
  kStop,
  kLength,
  kToolCalls,
};

struct DeepSeekAcceptedTokenBatch final {
  std::vector<std::uint32_t> token_ids;
  DeepSeekFinishReason finish_reason = DeepSeekFinishReason::kNone;
  struct SamplingReceipt final {
    std::uint64_t config_id = 0;
    std::uint64_t sample_ordinal = 0;
    float selected_logprob = 0.0F;
    std::uint32_t rng_word = 0;
  };
  std::vector<SamplingReceipt> sampling_receipts;
  struct SelectedLogprobReceipt final {
    struct TopLogprob final {
      std::uint32_t token_id = 0;
      float logprob = 0.0F;
      std::uint32_t rank = 0;
    };
    std::uint64_t config_id = 0;
    float selected_logprob = 0.0F;
    std::vector<TopLogprob> top_logprobs;
  };
  std::vector<SelectedLogprobReceipt> selected_logprobs;
};

struct DeepSeekAcceptedTokenSnapshot final {
  std::vector<std::uint32_t> token_ids;
  std::uint64_t accepted_completion_count = 0;
  std::uint64_t model_processed_length = 0;
  std::optional<std::uint32_t> pending_input_token;
  DeepSeekFinishReason finish_reason = DeepSeekFinishReason::kNone;
  std::uint64_t sample_ordinal = 0;
  std::vector<float> selected_logprobs;
  std::vector<std::vector<DeepSeekAcceptedTokenBatch::SelectedLogprobReceipt::TopLogprob>>
      top_logprobs;
  std::uint32_t minimum_completion_tokens = 0;
};

class DeepSeekControlPlane final {
 public:
  static Result<DeepSeekControlPlane> Create(
      std::uint64_t engine_epoch, DeepSeekPipelineCapacity capacity);

  DeepSeekControlPlane(const DeepSeekControlPlane&) = delete;
  DeepSeekControlPlane& operator=(const DeepSeekControlPlane&) = delete;
  DeepSeekControlPlane(DeepSeekControlPlane&&) noexcept = default;
  DeepSeekControlPlane& operator=(DeepSeekControlPlane&&) noexcept = default;

  Status submit(std::uint64_t request_id, std::uint64_t request_generation);
  Status cancel(std::uint64_t request_id, std::uint64_t request_generation);
  [[nodiscard]] Status validate_retire(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Status retire(std::uint64_t request_id, std::uint64_t request_generation);
  Status finish_request(std::uint64_t request_id,
                        std::uint64_t request_generation);
  Status configure_ledger(std::uint64_t request_id,
                          std::uint64_t request_generation,
                          std::uint64_t prompt_token_count,
                          std::uint32_t maximum_completion_tokens,
                          std::uint32_t minimum_completion_tokens);
  Status configure_sampling(std::uint64_t request_id,
                            std::uint64_t request_generation,
                            DeepSeekRequestSamplingConfig config);
  Result<DeepSeekRequestSamplingConfig> sampling_config(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Status stage_accepted_tokens(
      std::uint64_t plan_sequence,
      std::vector<DeepSeekAcceptedTokenBatch> batches);
  Status prepare_plan(DeepSeekPipelinePlanDescriptor descriptor,
                      std::vector<DeepSeekRequestIdentity> identities);
  Status stage_ready(std::uint32_t rank);
  Status stage_reject(std::uint32_t rank, Status reason);
  [[nodiscard]] Status validate_commit_plan() const;
  Status commit_plan();
  Status prepare_stage_complete(std::uint32_t rank);
  Status stage_complete(std::uint32_t rank);
  Status acknowledge_output_plan(std::uint64_t plan_sequence);
  Status stage_failed(std::uint32_t rank, Status reason);
  void fail_all() noexcept;
  void cancel_all() noexcept;

  Result<DeepSeekRequestState> request_state(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Result<DeepSeekPipelineCoordinatorState> execution_state() const;
  Result<DeepSeekAcceptedTokenSnapshot> accepted_token_snapshot(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  [[nodiscard]] bool execution_live() const noexcept {
    return execution_.has_value();
  }
  [[nodiscard]] std::size_t active_request_count() const noexcept {
    return requests_->active_count();
  }
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }

 private:
  DeepSeekControlPlane(
      std::uint64_t engine_epoch,
      std::unique_ptr<DeepSeekRequestRegistry> requests,
      std::unique_ptr<DeepSeekPipelineResourceSet> resources,
      OutputBurstCreditPool output_credits)
      : engine_epoch_(engine_epoch), requests_(std::move(requests)),
        resources_(std::move(resources)),
        output_credits_(std::move(output_credits)),
        pending_output_bursts_(DeepSeekPipelineCapacity::kBoundaryCredits),
        pending_output_owners_(DeepSeekPipelineCapacity::kBoundaryCredits) {}

  std::uint64_t engine_epoch_ = 0;
  std::unique_ptr<DeepSeekRequestRegistry> requests_;
  std::unique_ptr<DeepSeekPipelineResourceSet> resources_;
  std::optional<DeepSeekPipelineExecution> execution_;
  OutputBurstCreditPool output_credits_;
  std::optional<OutputBurstLease> current_output_burst_;
  std::vector<DeepSeekRequestIdentity> current_output_owners_;
  std::vector<std::optional<OutputBurstLease>> pending_output_bursts_;
  std::vector<std::optional<std::vector<DeepSeekRequestIdentity>>>
      pending_output_owners_;
  struct Ledger final {
    std::uint64_t prompt_token_count = 0;
    std::uint32_t maximum_completion_tokens = 0;
    std::uint32_t minimum_completion_tokens = 0;
    std::vector<std::uint32_t> token_ids;
    DeepSeekFinishReason finish_reason = DeepSeekFinishReason::kNone;
    std::optional<DeepSeekRequestSamplingConfig> sampling;
    std::uint64_t sample_ordinal = 0;
    std::vector<float> selected_logprobs;
    std::vector<std::vector<DeepSeekAcceptedTokenBatch::SelectedLogprobReceipt::TopLogprob>>
        top_logprobs;
  };
  struct TentativeAccepted final {
    std::uint64_t plan_sequence = 0;
    std::vector<DeepSeekAcceptedTokenBatch> batches;
  };
  using LedgerKey = std::pair<std::uint64_t, std::uint64_t>;
  Status validate_batch(const Ledger& ledger,
                        const DeepSeekAcceptedTokenBatch& batch) const;
  std::map<LedgerKey, Ledger> ledgers_;
  std::optional<TentativeAccepted> tentative_accepted_;
};

}  // namespace pih
