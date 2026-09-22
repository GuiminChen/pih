#pragma once

#include <cstdint>

#include "pih/model/deepseek_compressed_page_writer.h"
#include "pih/model/deepseek_compressor_state_writer.h"

namespace pih {

struct DeepSeekDeferredCompressedPageResources final {
  std::uintptr_t main_rms_weight_bf16 = 0;
  std::uintptr_t index_rms_weight_bf16 = 0;
  std::uintptr_t main_cos_sin_cache_f32 = 0;
  std::uintptr_t index_cos_sin_cache_f32 = 0;
  float rms_epsilon = 0.0F;
};

struct DeepSeekCompressedLayerUpdateSubmission final {
  std::uint32_t ratio = 0;
  DeepSeekCompressorStateSubmission main_state;
  DeepSeekCompressorStateSubmission index_state;
  bool has_completed_slot = false;
  DeepSeekRatio4SlotSubmission ratio4_slot;
  DeepSeekRatio128SlotSubmission ratio128_slot;
  bool defer_page_mutation = false;
  DeepSeekDeferredCompressedPageResources deferred_page;
};

class DeepSeekCompressedLayerUpdateCoordinator final {
 public:
  static Result<DeepSeekCompressedLayerUpdateCoordinator> Create(
      DeepSeekCompressorStateWriter& state_writer,
      DeepSeekCompressedPageWriter& page_writer);

  Status validate(
      const DeepSeekCompressedLayerUpdateSubmission& submission,
      const DeepSeekAttentionSequenceTransaction& transaction) const;

  Status launch(const DeepSeekCompressedLayerUpdateSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);

 private:
  DeepSeekCompressorStateWriter* state_writer_ = nullptr;
  DeepSeekCompressedPageWriter* page_writer_ = nullptr;
  bool poisoned_ = false;
};

}  // namespace pih
