#pragma once

#include "pih/backend/cuda/deepseek_attention_dense_primitives.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_fp8_gemm.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {
struct DeepSeekAttentionOutputProjectionSubmission final {
  DeepSeekRotaryLaunch inverse_rope;
  DeepSeekGroupedFp8GemmLaunch wo_a;
  DeepSeekFp8ActivationQuantLaunch quant;
  DeepSeekFp8GemmLaunch wo_b;
};
Status validate_deepseek_attention_output_projection_submission(
    const DeepSeekAttentionOutputProjectionSubmission& submission);
class DeepSeekAttentionOutputProjectionOperations {
 public:
  virtual ~DeepSeekAttentionOutputProjectionOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status rotary(DeepSeekRotaryLaunch launch) = 0;
  virtual Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch launch) = 0;
  virtual Status quant(DeepSeekFp8ActivationQuantLaunch launch) = 0;
  virtual Status gemm(DeepSeekFp8GemmLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
};
class DeepSeekAttentionOutputProjectionCoordinator final {
 public:
  static Result<DeepSeekAttentionOutputProjectionCoordinator> Create(
      DeepSeekAttentionOutputProjectionOperations& operations,
      std::uint32_t* host_error_flag);
  Status launch(const DeepSeekAttentionOutputProjectionSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);
 private:
  Status run(Status status);
  DeepSeekAttentionOutputProjectionOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  bool poisoned_ = false;
};
}  // namespace pih
