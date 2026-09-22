#pragma once

#include "pih/model/qwen3_bf16_packed_resource_factory.h"
#include "pih/model/qwen3_int4_lm_head_materializer.h"

namespace pih {

class QwenInt4PackedLmHeadMaterializer final {
 public:
  static Result<QwenInt4LmHeadBinding> Create(
      const QwenBf16PreparedCommand& command,
      const QwenBf16PackedResourceSet& resources,
      const QwenInt4WeightResourceSet& weights,
      std::uint64_t request_generation);
};

}  // namespace pih
