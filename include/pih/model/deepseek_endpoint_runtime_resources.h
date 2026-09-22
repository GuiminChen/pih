#pragma once

#include <memory>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/buffer.h"
#include "pih/model/deepseek_endpoint_sequence_executor.h"
#include "pih/model/deepseek_sampler.h"
#include "pih/model/deepseek_stage_mapping_plan.h"

namespace pih {

class DeepSeekEndpointRuntimeResources final {
 public:
  static Result<DeepSeekEndpointRuntimeResources> Allocate(
      DeepSeekStagePlan stage, DeepSeekEndpointSequenceOperations* operations,
      RegisteredPinnedAllocator& allocator);
  static Result<DeepSeekEndpointRuntimeResources> Allocate(
      DeepSeekStagePlan stage, DeepSeekEndpointSequenceOperations& operations,
      RegisteredPinnedAllocator& allocator) {
    return Allocate(stage, &operations, allocator);
  }

  [[nodiscard]] DeepSeekEndpointSequenceExecutor* executor() noexcept {
    return executor_.get();
  }
  [[nodiscard]] bool owns_embedding() const noexcept {
    return stage_.owns_embedding;
  }
  [[nodiscard]] bool owns_lm_head() const noexcept {
    return stage_.owns_lm_head;
  }
  Result<std::uint32_t> sampled_token() const;
  Result<DeepSeekSamplingResult> sampling_result(
      std::uint32_t top_logprobs_count = 0) const;

 private:
  DeepSeekEndpointRuntimeResources(
      DeepSeekStagePlan stage, std::unique_ptr<Buffer> host_error,
      std::unique_ptr<Buffer> host_sampled_token,
      std::unique_ptr<Buffer> host_selected_logprob,
      std::unique_ptr<Buffer> host_rng_word,
      std::unique_ptr<Buffer> host_top_ids,
      std::unique_ptr<Buffer> host_top_logprobs,
      std::unique_ptr<DeepSeekEndpointSequenceExecutor> executor) noexcept
      : stage_(stage), host_error_(std::move(host_error)),
        host_sampled_token_(std::move(host_sampled_token)),
        host_selected_logprob_(std::move(host_selected_logprob)),
        host_rng_word_(std::move(host_rng_word)),
        host_top_ids_(std::move(host_top_ids)),
        host_top_logprobs_(std::move(host_top_logprobs)),
        executor_(std::move(executor)) {}

  DeepSeekStagePlan stage_;
  std::unique_ptr<Buffer> host_error_;
  std::unique_ptr<Buffer> host_sampled_token_;
  std::unique_ptr<Buffer> host_selected_logprob_;
  std::unique_ptr<Buffer> host_rng_word_;
  std::unique_ptr<Buffer> host_top_ids_;
  std::unique_ptr<Buffer> host_top_logprobs_;
  std::unique_ptr<DeepSeekEndpointSequenceExecutor> executor_;
};

}  // namespace pih
