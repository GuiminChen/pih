#pragma once

#include <cstddef>
#include <cstdint>

#include "pih/backend/cuda/deepseek_compressor_pooling.h"
#include "pih/backend/cuda/deepseek_compressor_bf16_projection.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"
#include "pih/model/deepseek_fixed_state_layout.h"

namespace pih {

struct DeepSeekCompressorStateSubmission final {
  std::uint32_t layer_id = 0;
  bool indexer = false;
  std::uintptr_t kv_projection_f32 = 0;
  std::uintptr_t gate_projection_f32 = 0;
  std::uintptr_t ape_row_f32 = 0;
  std::uintptr_t output_f32 = 0;
  std::uintptr_t device_error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t batch_count = 0;
  std::uint32_t absolute_position = 0;
  DeepSeekCompressorBf16ProjectionLaunch projection;
};

class DeepSeekCompressorStateOperations {
 public:
  virtual ~DeepSeekCompressorStateOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status pooling(DeepSeekCompressorPoolingLaunch launch) = 0;
  virtual Status projection(
      DeepSeekCompressorBf16ProjectionLaunch) {
    return Status::FailedPrecondition(
        "DeepSeek compressor projection operation is unavailable");
  }
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
};

class DeepSeekCompressorStateWriter final {
 public:
  static Result<DeepSeekCompressorStateWriter> Create(
      DeepSeekFixedStateLayout layout,
      DeepSeekCompressorStateOperations& operations,
      std::uint32_t* host_error_flag);

  Status launch(const DeepSeekCompressorStateSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);

 private:
  DeepSeekFixedStateLayout layout_;
  DeepSeekCompressorStateOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  bool poisoned_ = false;
};

}  // namespace pih
