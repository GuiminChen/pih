#include "pih/model/qwen3_int4_step_builder.h"

#include <cmath>
#include <utility>

namespace pih {

Result<QwenInt4BuiltStep> QwenInt4StepBuilder::Create(
    std::span<const std::int64_t> tokens, std::uint64_t first_position,
    const QwenKvBlockTable& block_table,
    const QwenKvAppendPlan& append_plan,
    std::uint64_t request_generation, std::int32_t owning_rank,
    std::uint32_t slot_count, float rms_epsilon, float attention_scale,
    const QwenBf16CommandBuffer& commands,
    std::span<const ResolvedKernelFunction> functions,
    const QwenInt4WeightResourceSet& weights,
    const QwenBf16StepDeviceOwners& owners,
    std::span<std::byte> pinned_staging_backing) {
  if (request_generation == 0 || owning_rank < 0 ||
      weights.owning_rank() != owning_rank || !std::isfinite(rms_epsilon) ||
      rms_epsilon <= 0.0F || !std::isfinite(attention_scale) ||
      attention_scale <= 0.0F) {
    return Status::InvalidArgument("Qwen INT4 step identity is invalid");
  }
  auto input=QwenBf16StepInputPlan::Create(
      tokens,first_position,block_table,append_plan);
  if(!input.ok()) return input.status();
  auto staging=QwenBf16StepStagingLayout::Create(*input);
  if(!staging.ok()) return staging.status();
  const Status staged=staging->materialize(*input,pinned_staging_backing);
  if(!staged.ok()) return staged;
  auto arena=QwenBf16ExecutionArenaLayout::Create(tokens.size(),1);
  if(!arena.ok()) return arena.status();
  auto resources=QwenBf16StepResourceFactory::Create(
      request_generation,owning_rank,slot_count,*arena,*staging,owners);
  if(!resources.ok()) return resources.status();
  const QwenBf16KernelContext context{
      request_generation,block_table.descriptor().owner_sequence_index,
      first_position,append_plan.target_committed_tokens,slot_count,
      rms_epsilon,attention_scale};
  auto execution=QwenInt4PreparedExecution::Create(
      commands,functions,*resources,weights,context);
  if(!execution.ok()) return execution.status();
  return QwenInt4BuiltStep(std::move(*staging),std::move(*arena),
                           std::move(*resources),std::move(*execution));
}

}  // namespace pih
