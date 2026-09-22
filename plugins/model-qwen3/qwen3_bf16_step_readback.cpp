#include "pih/model/qwen3_bf16_step_readback.h"

#include <array>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenBf16StepReadback> QwenBf16StepReadback::Create(
    CudaCopyEndpoint sampled_token_device,
    CudaCopyEndpoint device_error_device,
    CudaCopyEndpoint pinned_host_result_arena,
    std::uintptr_t primary_context_identity, DriverStreamHandle stream,
    std::uint64_t completion_event_generation, std::uint64_t first_plan_id) {
  if (first_plan_id == 0 || first_plan_id == UINT64_MAX) {
    return Status::InvalidArgument(
        "Qwen step readback plan identity is invalid");
  }
  auto host_end = checked_add_u64(pinned_host_result_arena.offset,
                                  QwenBf16StepResultLayout::kTotalBytes);
  if (!host_end.ok()) return host_end.status();
  if (*host_end > pinned_host_result_arena.allocation_bytes) {
    return Status::InvalidArgument(
        "Qwen step readback result arena exceeds its owner");
  }
  const std::array<CudaCopyEndpoint, kCopyCount> sources{
      sampled_token_device, device_error_device};
  const std::array<QwenBf16ArenaSpan, kCopyCount> spans{
      QwenBf16StepResultLayout::sampled_token(),
      QwenBf16StepResultLayout::device_error()};
  const std::array<std::uint64_t, kCopyCount> alignments{8, 4};
  std::vector<CudaTypedCopyPlan> copies;
  copies.reserve(kCopyCount);
  for (std::size_t index = 0; index < kCopyCount; ++index) {
    auto host_offset = checked_add_u64(pinned_host_result_arena.offset,
                                       spans[index].offset_bytes);
    if (!host_offset.ok()) return host_offset.status();
    auto destination = pinned_host_result_arena;
    destination.offset = *host_offset;
    auto copy = CudaTypedCopyPlan::Create(
        first_plan_id + index, CudaCopyPurpose::kResult,
        CudaCopyKind::kDeviceToHost, sources[index], destination,
        spans[index].size_bytes, alignments[index], primary_context_identity,
        stream, completion_event_generation);
    if (!copy.ok()) return copy.status();
    copies.push_back(std::move(*copy));
  }
  return QwenBf16StepReadback(std::move(copies));
}

Status QwenBf16StepReadback::submit(TypedCopyDriver& driver) {
  if (state_ != QwenBf16StepReadbackState::kPrepared) {
    return Status::FailedPrecondition("Qwen step readback is not submit-ready");
  }
  for (; submitted_copies_ < copies_.size(); ++submitted_copies_) {
    const Status status = copies_[submitted_copies_].submit(driver);
    if (!status.ok()) {
      state_ = QwenBf16StepReadbackState::kPoisoned;
      return status;
    }
  }
  state_ = QwenBf16StepReadbackState::kSubmitted;
  return Status::Ok();
}

}  // namespace pih
