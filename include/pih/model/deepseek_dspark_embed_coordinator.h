#pragma once

#include "pih/backend/cuda/deepseek_endpoint.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_fp8_gemm.h"
#include "pih/backend/cuda/deepseek_rms_norm.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {
struct DeepSeekDsparkEmbedSubmission final {
  DeepSeekFp8ActivationQuantLaunch main_quant;
  DeepSeekFp8GemmLaunch main_proj;
  DeepSeekRmsNormLaunch main_norm;
  DeepSeekDsparkDraftInitLaunch draft_init;
};
class DeepSeekDsparkEmbedOperations {
 public:
  virtual ~DeepSeekDsparkEmbedOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status quant(DeepSeekFp8ActivationQuantLaunch launch) = 0;
  virtual Status gemm(DeepSeekFp8GemmLaunch launch) = 0;
  virtual Status rms(DeepSeekRmsNormLaunch launch) = 0;
  virtual Status draft_init(DeepSeekDsparkDraftInitLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
};
class DeepSeekDsparkEmbedCoordinator final {
 public:
  static Result<DeepSeekDsparkEmbedCoordinator> Create(
      DeepSeekDsparkEmbedOperations& operations,
      std::uint32_t* host_error_flag);
  Status launch(const DeepSeekDsparkEmbedSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);
  // Prefill needs only the target-layer projection and normalization.  The
  // draft-token embedding is decode proposal work and is deliberately not
  // submitted by this entry point.
  Status launch_prefill_state(
      const DeepSeekDsparkEmbedSubmission& submission,
      DeepSeekAttentionSequenceTransaction& transaction);
 private:
  Status run(Status status);
  Status launch_internal(
      const DeepSeekDsparkEmbedSubmission& submission,
      DeepSeekAttentionSequenceTransaction& transaction,
      bool initialize_draft);
  DeepSeekDsparkEmbedOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  bool poisoned_ = false;
};
}  // namespace pih
