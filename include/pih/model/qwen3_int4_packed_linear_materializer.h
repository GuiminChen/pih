#pragma once

#include "pih/model/qwen3_bf16_packed_resource_factory.h"
#include "pih/model/qwen3_int4_linear_materializer.h"

namespace pih {

class QwenInt4PackedLinearMaterializer final {
 public:
  static Result<QwenInt4DispatchPlan> Create(
      const QwenBf16PreparedCommand& command,
      const ResolvedKernelFunction& function,
      const QwenBf16PackedResourceSet& resources,
      const QwenInt4WeightResourceSet& weights,
      std::uint64_t request_generation);
};

}  // namespace pih
