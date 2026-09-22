#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pih/backend/cuda/deepseek_compressor_bf16_store.h"
#include "pih/model/deepseek_attention_page_arena.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {

class DeepSeekCompressedPageOperations {
 public:
  virtual ~DeepSeekCompressedPageOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status copy_d2d_async(std::uintptr_t destination,
                                std::uintptr_t source,
                                std::size_t bytes,
                                std::uintptr_t stream) = 0;
  virtual Status store(DeepSeekCompressorBf16StoreLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
};

struct DeepSeekRatio4SlotSubmission final {
  DeepSeekRatio4PagePair target;
  DeepSeekRatio4PagePair committed;
  bool tail_cow = false;
  std::uintptr_t main_compressed_f32 = 0;
  std::uintptr_t index_compressed_f32 = 0;
  std::uintptr_t main_rms_weight_bf16 = 0;
  std::uintptr_t index_rms_weight_bf16 = 0;
  std::uintptr_t main_cos_sin_cache_f32 = 0;
  std::uintptr_t index_cos_sin_cache_f32 = 0;
  std::uintptr_t device_error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t absolute_position = 0;
  float rms_epsilon = 0.0F;
};

struct DeepSeekRatio128SlotSubmission final {
  DeepSeekRatio128Page target;
  DeepSeekRatio128Page committed;
  bool tail_cow = false;
  std::uintptr_t compressed_f32 = 0;
  std::uintptr_t rms_weight_bf16 = 0;
  std::uintptr_t cos_sin_cache_f32 = 0;
  std::uintptr_t device_error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t absolute_position = 0;
  float rms_epsilon = 0.0F;
};

class DeepSeekCompressedPageWriter final {
 public:
  static Result<DeepSeekCompressedPageWriter> Create(
      DeepSeekAttentionPageArena arena,
      DeepSeekCompressedPageOperations& operations,
      std::uint32_t* host_error_flag);

  Status write_ratio4(const DeepSeekRatio4SlotSubmission& submission,
                      DeepSeekAttentionSequenceTransaction& transaction);
  Status write_ratio128(const DeepSeekRatio128SlotSubmission& submission,
                        DeepSeekAttentionSequenceTransaction& transaction);

 private:
  DeepSeekAttentionPageArena arena_;
  DeepSeekCompressedPageOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  std::uint64_t copied_epoch_ = 0;
  std::vector<std::uint64_t> copied_tail_targets_;
  bool poisoned_ = false;
};

}  // namespace pih
