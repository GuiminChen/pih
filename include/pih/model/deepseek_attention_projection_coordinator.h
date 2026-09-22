#pragma once

#include "pih/backend/cuda/deepseek_attention_dense_primitives.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_fp8_gemm.h"
#include "pih/backend/cuda/deepseek_rms_norm.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {

struct DeepSeekAttentionProjectionSubmission final {
  DeepSeekFp8ActivationQuantLaunch input_quant;
  DeepSeekFp8GemmLaunch wq_a;
  DeepSeekRmsNormLaunch q_norm;
  DeepSeekFp8ActivationQuantLaunch q_quant;
  DeepSeekFp8GemmLaunch wq_b;
  DeepSeekHeadRmsLaunch q_head_rms;
  DeepSeekRotaryLaunch q_rope;
  DeepSeekFp8GemmLaunch wkv;
  DeepSeekRmsNormLaunch kv_norm;
  DeepSeekRotaryLaunch kv_rope;
  DeepSeekKvFp8SimulateLaunch kv_simulate;
};

Status validate_deepseek_attention_projection_submission(
    const DeepSeekAttentionProjectionSubmission& submission);

class DeepSeekAttentionProjectionOperations {
 public:
  virtual ~DeepSeekAttentionProjectionOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status quant(DeepSeekFp8ActivationQuantLaunch launch) = 0;
  virtual Status gemm(DeepSeekFp8GemmLaunch launch) = 0;
  virtual Status rms(DeepSeekRmsNormLaunch launch) = 0;
  virtual Status head_rms(DeepSeekHeadRmsLaunch launch) = 0;
  virtual Status rotary(DeepSeekRotaryLaunch launch) = 0;
  virtual Status kv_simulate(DeepSeekKvFp8SimulateLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
};

class DeepSeekAttentionProjectionCoordinator final {
 public:
  static Result<DeepSeekAttentionProjectionCoordinator> Create(
      DeepSeekAttentionProjectionOperations& operations,
      std::uint32_t* host_error_flag);
  Status launch(const DeepSeekAttentionProjectionSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);

 private:
  Status run(Status status);
  DeepSeekAttentionProjectionOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  bool poisoned_ = false;
};

}  // namespace pih
