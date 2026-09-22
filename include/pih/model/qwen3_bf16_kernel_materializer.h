#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_bf16_dispatch_plan.h"
#include "pih/model/qwen3_bf16_resource_set.h"
#include "pih/model/qwen3_int4_weight_resource_set.h"

namespace pih {

struct QwenBf16KernelContext final {
  std::uint64_t request_generation;
  std::uint32_t owner_sequence_index;
  std::uint64_t query_start_position;
  std::uint32_t key_token_count;
  std::uint32_t slot_count;
  float rms_epsilon;
  float attention_scale;
};

class QwenBf16KernelMaterializer final {
 public:
  static Result<QwenBf16DispatchPlan> Create(
      const QwenBf16PreparedCommand& command,
      const ResolvedKernelFunction& function,
      const QwenBf16ResourceSet& resources,
      const QwenBf16WeightResourceSet& weights,
      const QwenBf16KernelContext& context);
  static Result<QwenBf16DispatchPlan> Create(
      const QwenBf16PreparedCommand& command,
      const ResolvedKernelFunction& function,
      const QwenBf16ResourceSet& resources,
      const QwenInt4WeightResourceSet& weights,
      const QwenBf16KernelContext& context);
};

}  // namespace pih
