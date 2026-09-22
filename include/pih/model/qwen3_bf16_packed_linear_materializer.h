#pragma once

#include "pih/model/qwen3_bf16_linear_materializer.h"
#include "pih/model/qwen3_bf16_packed_resource_factory.h"

namespace pih {

class QwenBf16PackedLinearMaterializer final {
 public:
  static Result<QwenBf16LinearBinding> Create(
      const QwenBf16PreparedCommand& command,
      const QwenBf16PackedResourceSet& resources,
      const QwenBf16WeightResourceSet& weights,
      std::uint64_t request_generation);
};

}  // namespace pih
