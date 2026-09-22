#include "pih/model/qwen3_bf16_engine_arenas.h"

#include <array>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Status validate(const Buffer& buffer, std::uint64_t expected_bytes,
                std::int32_t owning_rank) {
  if (buffer.size_bytes() != expected_bytes || buffer.generation() == 0 ||
      buffer.device().type() != DeviceType::kCuda ||
      buffer.device().index() != owning_rank ||
      (expected_bytes != 0 && buffer.data() == nullptr)) {
    return Status::FailedPrecondition(
        "Qwen engine allocator returned an invalid device allocation");
  }
  return Status::Ok();
}

QwenBf16DeviceArenaOwner owner(const Buffer& buffer) {
  return {reinterpret_cast<std::uintptr_t>(buffer.data()),
          buffer.size_bytes(), buffer.generation()};
}

}  // namespace

Result<QwenBf16EngineDeviceArenas> QwenBf16EngineDeviceArenas::Allocate(
    const QwenBf16EngineResourcePlan& plan, Allocator& allocator,
    std::int32_t owning_rank) {
  if (owning_rank < 0) {
    return Status::InvalidArgument("Qwen engine owning rank is invalid");
  }
  auto step_staging = Buffer::Allocate(
      allocator, plan.step_staging_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!step_staging.ok()) return step_staging.status();
  auto activations = Buffer::Allocate(
      allocator, plan.activation_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!activations.ok()) return activations.status();
  auto mlp = Buffer::Allocate(
      allocator, plan.mlp_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!mlp.ok()) return mlp.status();
  auto rope = Buffer::Allocate(
      allocator, plan.rope_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!rope.ok()) return rope.status();
  auto logits = Buffer::Allocate(
      allocator, plan.logits_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!logits.ok()) return logits.status();
  auto sampled = Buffer::Allocate(
      allocator, plan.sampled_token_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!sampled.ok()) return sampled.status();
  auto error = Buffer::Allocate(
      allocator, plan.device_error_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!error.ok()) return error.status();
  auto kv = Buffer::Allocate(
      allocator, plan.kv_backing_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!kv.ok()) return kv.status();
  auto metadata = Buffer::Allocate(
      allocator, plan.kv_metadata_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!metadata.ok()) return metadata.status();
  auto workspace = Buffer::Allocate(
      allocator, plan.linear_workspace_bytes(), QwenBf16ExecutionArenaLayout::kAlignment);
  if (!workspace.ok()) return workspace.status();
  const std::array<std::pair<const Buffer*, std::uint64_t>, 10> buffers{{
      {&*step_staging, plan.step_staging_bytes()},
      {&*activations, plan.activation_bytes()}, {&*mlp, plan.mlp_bytes()},
      {&*rope, plan.rope_bytes()}, {&*logits, plan.logits_bytes()},
      {&*sampled, plan.sampled_token_bytes()},
      {&*error, plan.device_error_bytes()}, {&*kv, plan.kv_backing_bytes()},
      {&*metadata, plan.kv_metadata_bytes()},
      {&*workspace, plan.linear_workspace_bytes()}}};
  std::uint64_t total = 0;
  for (const auto& [buffer, expected] : buffers) {
    const Status valid = validate(*buffer, expected, owning_rank);
    if (!valid.ok()) return valid;
    auto next = checked_add_u64(total, expected);
    if (!next.ok()) return next.status();
    total = *next;
  }
  return QwenBf16EngineDeviceArenas(
      std::move(*step_staging), std::move(*activations), std::move(*mlp),
      std::move(*rope), std::move(*logits), std::move(*sampled),
      std::move(*error), std::move(*kv), std::move(*metadata),
      std::move(*workspace), total);
}

QwenBf16StepDeviceOwners QwenBf16EngineDeviceArenas::step_owners()
    const noexcept {
  return {owner(step_staging_), owner(activations_), owner(mlp_), owner(rope_),
          owner(logits_), owner(sampled_token_), owner(device_error_),
          owner(kv_backing_), owner(kv_metadata_)};
}

}  // namespace pih
