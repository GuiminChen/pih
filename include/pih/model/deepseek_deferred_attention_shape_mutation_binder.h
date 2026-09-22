#pragma once

#include "pih/model/deepseek_attention_plan_input_shape_assembler.h"
#include "pih/model/deepseek_compressed_page_mutation_assembler.h"

namespace pih {

class DeepSeekDeferredAttentionShapeMutationBinder final {
 public:
  static Status Bind(
      std::uint32_t sequence,
      std::span<DeepSeekAttentionLayerPlanShapeSeed> shapes,
      DeepSeekAttentionSequenceTransaction& transaction,
      DeepSeekRatio4PagePool& ratio4_pool,
      DeepSeekRatio128PagePool& ratio128_pool);
};

}  // namespace pih
