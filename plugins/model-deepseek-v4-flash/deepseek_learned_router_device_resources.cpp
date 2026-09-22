#include "pih/model/deepseek_learned_router_device_resources.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekLearnedRouterDeviceResources>
DeepSeekLearnedRouterDeviceResources::Allocate(
    std::uint32_t maximum_tokens, DeepSeekExpertComputeArena expert_arena,
    Allocator& allocator, std::uint64_t context_identity,
    std::int32_t device_ordinal) {
  if (maximum_tokens == 0 || maximum_tokens > 4096 ||
      context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument(
        "DeepSeek learned router device identity is invalid");
  }
  auto score_elements = checked_mul_u64(
      maximum_tokens, DeepSeekLearnedRouterCoordinator::kExpertCount);
  if (!score_elements.ok()) return score_elements.status();
  auto score_bytes = checked_mul_u64(*score_elements, sizeof(float));
  if (!score_bytes.ok()) return score_bytes.status();
  if (expert_arena.error_flag_u32.address == 0 ||
      expert_arena.error_flag_u32.bytes < sizeof(std::uint32_t)) {
    return Status::InvalidArgument(
        "DeepSeek expert arena cannot back learned router scratch");
  }
  auto scores = Buffer::Allocate(allocator, *score_bytes, 256);
  if (!scores.ok()) return scores.status();
  if (scores->data() == nullptr || scores->size_bytes() != *score_bytes ||
      scores->generation() == 0 ||
      scores->device().type() != DeviceType::kCuda ||
      scores->device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek learned router allocator returned invalid scores");
  }
  const DeepSeekLearnedRouterDeviceView view{
      reinterpret_cast<std::uintptr_t>(scores->data()),
      expert_arena.error_flag_u32.address};
  return DeepSeekLearnedRouterDeviceResources(
      std::move(*scores), view, maximum_tokens);
}

}  // namespace pih
