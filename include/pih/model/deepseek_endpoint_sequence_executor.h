#pragma once

#include <optional>

#include "pih/backend/cuda/deepseek_endpoint.h"
#include "pih/backend/cuda/deepseek_rms_norm.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {

struct DeepSeekHeadSequenceSubmission final {
  DeepSeekHcHeadLaunch hc;
  DeepSeekRmsNormLaunch rms;
  DeepSeekLmHeadLaunch lm;
  DeepSeekArgmaxLaunch sample;
  std::optional<DeepSeekStochasticSampleLaunch> stochastic_sample;
};

class DeepSeekEndpointSequenceOperations {
 public:
  virtual ~DeepSeekEndpointSequenceOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status embedding(DeepSeekEmbeddingLaunch launch) = 0;
  virtual Status hc_head(DeepSeekHcHeadLaunch launch) = 0;
  virtual Status rms_norm(DeepSeekRmsNormLaunch launch) = 0;
  virtual Status lm_head(DeepSeekLmHeadLaunch launch) = 0;
  virtual Status sample(DeepSeekArgmaxLaunch launch) = 0;
  virtual Status stochastic_sample(DeepSeekStochasticSampleLaunch) {
    return Status::FailedPrecondition(
        "DeepSeek stochastic sampling operations are unavailable");
  }
  virtual Status copy_token_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
  virtual Status copy_logprob_d2h_async(float*, std::uintptr_t,
                                         std::uintptr_t) {
    return Status::FailedPrecondition(
        "DeepSeek logprob copy operation is unavailable");
  }
  virtual Status copy_rng_d2h_async(std::uint32_t*, std::uintptr_t,
                                    std::uintptr_t) {
    return Status::FailedPrecondition(
        "DeepSeek RNG copy operation is unavailable");
  }
  virtual Status copy_top_ids_d2h_async(std::uint32_t*, std::uintptr_t,
                                        std::uint32_t, std::uintptr_t) {
    return Status::FailedPrecondition(
        "DeepSeek top-token copy operation is unavailable");
  }
  virtual Status copy_top_logprobs_d2h_async(float*, std::uintptr_t,
                                             std::uint32_t,
                                             std::uintptr_t) {
    return Status::FailedPrecondition(
        "DeepSeek top-logprob copy operation is unavailable");
  }
};

class DeepSeekEndpointSequenceExecutor final {
 public:
  static Result<DeepSeekEndpointSequenceExecutor> Create(
      DeepSeekEndpointSequenceOperations& operations,
      std::uint32_t* host_error_flag, std::uint32_t* host_sampled_token,
      float* host_selected_logprob = nullptr,
      std::uint32_t* host_rng_word = nullptr,
      std::uint32_t* host_top_ids = nullptr,
      float* host_top_logprobs = nullptr);
  Status launch_embedding(
      DeepSeekEmbeddingLaunch launch,
      DeepSeekAttentionSequenceTransaction& transaction);
  Status launch_head(
      const DeepSeekHeadSequenceSubmission& submission,
      DeepSeekAttentionSequenceTransaction& transaction);

 private:
  Result<bool> claim(std::uintptr_t device_error, std::uintptr_t stream,
                     DeepSeekAttentionSequenceTransaction& transaction);
  Status run(Status status);
  DeepSeekEndpointSequenceOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  std::uint32_t* host_sampled_token_ = nullptr;
  float* host_selected_logprob_ = nullptr;
  std::uint32_t* host_rng_word_ = nullptr;
  std::uint32_t* host_top_ids_ = nullptr;
  float* host_top_logprobs_ = nullptr;
  bool poisoned_ = false;
};

}  // namespace pih
