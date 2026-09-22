#pragma once

#include <cstdint>
#include <span>

#include "pih/model/qwen3_bf16_linear_plan.h"
#include "pih/model/qwen3_bf16_linear_shape_selector.h"
#include "pih/model/qwen3_bf16_prepared_execution.h"

namespace pih {

class NvidiaQwenBf16ExecutionDriver final
    : public QwenBf16LinearExecutionDriver {
 public:
  static Result<NvidiaQwenBf16ExecutionDriver> Create(
      const QwenBf16LinearPlanSet& plans, void* workspace,
      std::uint64_t workspace_bytes, std::int32_t owning_rank);
  static Result<NvidiaQwenBf16ExecutionDriver> Create(
      const QwenBf16LinearPlanSet& prefill_plans,
      const QwenBf16LinearPlanSet& decode_plans, void* workspace,
      std::uint64_t workspace_bytes, std::int32_t owning_rank);
  static Result<NvidiaQwenBf16ExecutionDriver> Create(
      const QwenBf16LinearPlanSet& prefill_plans,
      const QwenBf16LinearPlanSet& decode_plans,
      std::span<const QwenBf16LinearPlanSet* const> packed_lm_head_plans,
      void* workspace, std::uint64_t workspace_bytes,
      std::int32_t owning_rank);
  static Result<NvidiaQwenBf16ExecutionDriver> Create(
      const QwenBf16LinearPlanSet& prefill_plans,
      const QwenBf16LinearPlanSet& tail_prefill_plans,
      const QwenBf16LinearPlanSet& decode_plans, void* workspace,
      std::uint64_t workspace_bytes, std::int32_t owning_rank);
  static Result<NvidiaQwenBf16ExecutionDriver> Create(
      const QwenBf16LinearPlanSet& prefill_plans,
      const QwenBf16LinearPlanSet& tail_prefill_plans,
      const QwenBf16LinearPlanSet& decode_plans,
      std::span<const QwenBf16LinearPlanSet* const> packed_lm_head_plans,
      void* workspace, std::uint64_t workspace_bytes,
      std::int32_t owning_rank);

  Status execute(const QwenBf16LinearBinding& binding,
                 DriverStreamHandle stream) override;

  [[nodiscard]] std::uintptr_t context_identity() const noexcept {
    return context_identity_;
  }
  [[nodiscard]] std::int32_t owning_rank() const noexcept {
    return owning_rank_;
  }

 private:
  NvidiaQwenBf16ExecutionDriver(
      const QwenBf16LinearPlanSet& prefill_plans,
      const QwenBf16LinearPlanSet& tail_prefill_plans,
      const QwenBf16LinearPlanSet& decode_plans, void* workspace,
      std::uint64_t workspace_bytes, std::int32_t owning_rank,
      std::uintptr_t context_identity,
      QwenBf16LinearShapeSelector selector,
      std::span<const QwenBf16LinearPlanSet* const> packed_lm_head_plans)
      : prefill_plans_(&prefill_plans),
        tail_prefill_plans_(&tail_prefill_plans),
        decode_plans_(&decode_plans),
        workspace_(workspace),
        workspace_bytes_(workspace_bytes),
        owning_rank_(owning_rank),
        context_identity_(context_identity),
        selector_(selector),
        packed_lm_head_plans_(packed_lm_head_plans) {}

  Status require_current_owner() const;

  const QwenBf16LinearPlanSet* prefill_plans_;
  const QwenBf16LinearPlanSet* tail_prefill_plans_;
  const QwenBf16LinearPlanSet* decode_plans_;
  void* workspace_;
  std::uint64_t workspace_bytes_;
  std::int32_t owning_rank_;
  std::uintptr_t context_identity_;
  QwenBf16LinearShapeSelector selector_;
  std::span<const QwenBf16LinearPlanSet* const> packed_lm_head_plans_;
};

}  // namespace pih
