#pragma once

#include "pih/model/qwen3_bf16_kernel_materializer.h"
#include "pih/model/qwen3_bf16_packed_dispatch_plan.h"
#include "pih/model/qwen3_bf16_packed_resource_factory.h"

namespace pih {

struct QwenBf16PackedKernelContext final {
  std::uint64_t request_generation;
  std::uint32_t real_token_count;
  std::uint32_t sequence_count;
  std::uint32_t slot_count;
  float rms_epsilon;
  float attention_scale;
};

class QwenBf16PackedKernelMaterializer final {
 public:
  static bool uses_packed_primitive(
      const QwenBf16PreparedCommand& command) noexcept;
  static Result<QwenBf16PackedDispatchPlan> CreatePacked(
      const QwenBf16PreparedCommand& command,
      const ResolvedKernelFunction& function,
      const QwenBf16PackedResourceSet& resources,
      const QwenBf16WeightResourceSet& weights,
      const QwenBf16PackedKernelContext& context);
  static Result<QwenBf16PackedDispatchPlan> CreatePacked(
      const QwenBf16PreparedCommand& command,
      const ResolvedKernelFunction& function,
      const QwenBf16PackedResourceSet& resources,
      const QwenInt4WeightResourceSet& weights,
      const QwenBf16PackedKernelContext& context);
  static Result<QwenBf16DispatchPlan> CreateLegacy(
      const QwenBf16PreparedCommand& command,
      const ResolvedKernelFunction& function,
      const QwenBf16PackedResourceSet& resources,
      const QwenBf16WeightResourceSet& weights,
      const QwenBf16PackedKernelContext& context);
  static Result<QwenBf16DispatchPlan> CreateLegacy(
      const QwenBf16PreparedCommand& command,
      const ResolvedKernelFunction& function,
      const QwenBf16PackedResourceSet& resources,
      const QwenInt4WeightResourceSet& weights,
      const QwenBf16PackedKernelContext& context);
  static Result<QwenBf16PackedDispatchPlan> CreateSampleHidden(
      const ResolvedKernelFunction& function,
      const QwenBf16PackedResourceSet& resources,
      const QwenBf16PackedKernelContext& context);
};

}  // namespace pih
