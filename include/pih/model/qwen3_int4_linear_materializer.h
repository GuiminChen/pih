#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_bf16_resource_set.h"
#include "pih/model/qwen3_int4_dispatch_plan.h"
#include "pih/model/qwen3_int4_weight_resource_set.h"

namespace pih {

class QwenInt4LinearMaterializer final {
 public:
  static Result<QwenInt4DispatchPlan> Create(
      const QwenBf16PreparedCommand& command,
      const ResolvedKernelFunction& function,
      const QwenBf16ResourceSet& activations,
      const QwenInt4WeightResourceSet& weights,
      std::uint64_t request_generation);
};

}  // namespace pih
