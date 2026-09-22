#pragma once

#include "pih/model/qwen3_bf16_packed_result_layout.h"
#include "pih/model/qwen3_bf16_packed_staging_resources.h"
#include "pih/model/qwen3_bf16_resource_set.h"

namespace pih {

class QwenBf16PackedResourceSet final {
 public:
  [[nodiscard]] const QwenBf16ResourceSet& activations() const noexcept {
    return activations_;
  }
  [[nodiscard]] const QwenBf16PackedStagingResources& metadata() const noexcept {
    return metadata_;
  }
  [[nodiscard]] std::uint32_t execution_bucket_tokens() const noexcept {
    return execution_bucket_tokens_;
  }
  [[nodiscard]] std::uint32_t sample_count() const noexcept {
    return sample_count_;
  }
  [[nodiscard]] std::uint32_t sequence_count() const noexcept {
    return sequence_count_;
  }
  [[nodiscard]] const TensorView& sampler_workspace_ids() const noexcept {
    return sampler_workspace_ids_;
  }
  [[nodiscard]] const TensorView& selected_logprobs() const noexcept {
    return selected_logprobs_;
  }
  [[nodiscard]] const TensorView& rng_words() const noexcept {
    return rng_words_;
  }
  [[nodiscard]] const TensorView& top_token_ids() const noexcept {
    return top_token_ids_;
  }
  [[nodiscard]] const TensorView& top_logprobs() const noexcept {
    return top_logprobs_;
  }
  [[nodiscard]] const TensorView& top_counts() const noexcept {
    return top_counts_;
  }

 private:
  friend class QwenBf16PackedResourceFactory;
  QwenBf16PackedResourceSet(QwenBf16ResourceSet activations,
                            QwenBf16PackedStagingResources metadata,
                            TensorView sampler_workspace_ids,
                            TensorView selected_logprobs, TensorView rng_words,
                            TensorView top_token_ids, TensorView top_logprobs,
                            TensorView top_counts,
                            std::uint32_t execution_bucket_tokens,
                            std::uint32_t sample_count,
                            std::uint32_t sequence_count)
      : activations_(std::move(activations)), metadata_(std::move(metadata)),
        sampler_workspace_ids_(std::move(sampler_workspace_ids)),
        selected_logprobs_(std::move(selected_logprobs)),
        rng_words_(std::move(rng_words)),
        top_token_ids_(std::move(top_token_ids)),
        top_logprobs_(std::move(top_logprobs)),
        top_counts_(std::move(top_counts)),
        execution_bucket_tokens_(execution_bucket_tokens),
        sample_count_(sample_count), sequence_count_(sequence_count) {}

  QwenBf16ResourceSet activations_;
  QwenBf16PackedStagingResources metadata_;
  TensorView sampler_workspace_ids_;
  TensorView selected_logprobs_;
  TensorView rng_words_;
  TensorView top_token_ids_;
  TensorView top_logprobs_;
  TensorView top_counts_;
  std::uint32_t execution_bucket_tokens_ = 0;
  std::uint32_t sample_count_ = 0;
  std::uint32_t sequence_count_ = 0;
};

class QwenBf16PackedResourceFactory final {
 public:
  static Result<QwenBf16PackedResourceSet> Create(
      std::uint64_t request_generation, std::int32_t owning_rank,
      std::uint32_t slot_count, std::uint32_t sample_count,
      const QwenBf16ExecutionArenaLayout& execution_layout,
      const QwenBf16PackedStepStagingLayout& staging_layout,
      const QwenBf16StepDeviceOwners& owners);
};

}  // namespace pih
