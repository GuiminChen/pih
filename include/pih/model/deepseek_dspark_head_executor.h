#pragma once

#include <array>

#include "pih/backend/cuda/deepseek_endpoint.h"
#include "pih/backend/cuda/deepseek_rms_norm.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {

struct DeepSeekDsparkHeadSubmission final {
  static constexpr std::size_t kBlockSize = 5;
  DeepSeekHcHeadLaunch hc;
  DeepSeekRmsNormLaunch rms;
  DeepSeekLmHeadLaunch lm;
  std::array<DeepSeekDsparkMarkovLaunch, kBlockSize> markov;
  std::array<DeepSeekArgmaxLaunch, kBlockSize> argmax;
  DeepSeekDsparkConfidenceLaunch confidence;
};

class DeepSeekDsparkHeadOperations {
 public:
  virtual ~DeepSeekDsparkHeadOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status hc_head(DeepSeekHcHeadLaunch launch) = 0;
  virtual Status rms_norm(DeepSeekRmsNormLaunch launch) = 0;
  virtual Status lm_head(DeepSeekLmHeadLaunch launch) = 0;
  virtual Status markov(DeepSeekDsparkMarkovLaunch launch) = 0;
  virtual Status argmax(DeepSeekArgmaxLaunch launch) = 0;
  virtual Status confidence(DeepSeekDsparkConfidenceLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
};

class DeepSeekDsparkHeadExecutor final {
 public:
  static Result<DeepSeekDsparkHeadExecutor> Create(
      DeepSeekDsparkHeadOperations& operations,
      std::uint32_t* host_error_flag);
  Status launch(const DeepSeekDsparkHeadSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);

 private:
  Status run(Status status);
  DeepSeekDsparkHeadOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  bool poisoned_ = false;
};

}  // namespace pih
