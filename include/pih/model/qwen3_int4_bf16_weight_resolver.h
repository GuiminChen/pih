#pragma once

#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_int4_weight_resource_set.h"

namespace pih {

class QwenInt4Bf16WeightResolver final {
 public:
  static Result<TensorView> Resolve(
      const QwenBf16PreparedCommand& command,
      const QwenInt4WeightResourceSet& weights);
};

}  // namespace pih
