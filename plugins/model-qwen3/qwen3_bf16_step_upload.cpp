#include "pih/model/qwen3_bf16_step_upload.h"

#include <array>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenBf16StepUpload> QwenBf16StepUpload::Create(
    const QwenBf16StepStagingLayout& layout,
    CudaCopyEndpoint pinned_host_arena, CudaCopyEndpoint device_arena,
    std::uintptr_t primary_context_identity, DriverStreamHandle stream,
    std::uint64_t completion_event_generation, std::uint64_t first_plan_id) {
  if (first_plan_id == 0 ||
      first_plan_id > UINT64_MAX - (kCopyCount - 1)) {
    return Status::InvalidArgument("Qwen step upload plan identity is invalid");
  }
  auto host_end =
      checked_add_u64(pinned_host_arena.offset, layout.total_bytes());
  auto device_end = checked_add_u64(device_arena.offset, layout.total_bytes());
  if (!host_end.ok()) return host_end.status();
  if (!device_end.ok()) return device_end.status();
  if (*host_end > pinned_host_arena.allocation_bytes ||
      *device_end > device_arena.allocation_bytes) {
    return Status::InvalidArgument(
        "Qwen step upload arena range exceeds its owner");
  }
  const std::array<QwenBf16ArenaSpan, kCopyCount> spans{
      layout.token_ids(), layout.positions(), layout.append_handles(),
      layout.token_offsets(), layout.visible_handles()};
  const std::array<std::uint64_t, kCopyCount> alignments{8, 8, 4, 2, 4};
  std::vector<CudaTypedCopyPlan> copies;
  copies.reserve(kCopyCount);
  for (std::size_t index = 0; index < spans.size(); ++index) {
    auto host_offset =
        checked_add_u64(pinned_host_arena.offset, spans[index].offset_bytes);
    auto device_offset =
        checked_add_u64(device_arena.offset, spans[index].offset_bytes);
    if (!host_offset.ok()) return host_offset.status();
    if (!device_offset.ok()) return device_offset.status();
    auto source = pinned_host_arena;
    auto destination = device_arena;
    source.offset = *host_offset;
    destination.offset = *device_offset;
    auto copy = CudaTypedCopyPlan::Create(
        first_plan_id + index, CudaCopyPurpose::kInput,
        CudaCopyKind::kHostToDevice, source, destination,
        spans[index].size_bytes, alignments[index], primary_context_identity,
        stream, completion_event_generation);
    if (!copy.ok()) return copy.status();
    copies.push_back(std::move(*copy));
  }
  return QwenBf16StepUpload(std::move(copies));
}

Status QwenBf16StepUpload::submit(TypedCopyDriver& driver) {
  if (state_ != QwenBf16StepUploadState::kPrepared) {
    return Status::FailedPrecondition("Qwen step upload is not submit-ready");
  }
  for (; submitted_copies_ < copies_.size(); ++submitted_copies_) {
    const Status status = copies_[submitted_copies_].submit(driver);
    if (!status.ok()) {
      state_ = QwenBf16StepUploadState::kPoisoned;
      return status;
    }
  }
  state_ = QwenBf16StepUploadState::kSubmitted;
  return Status::Ok();
}

}  // namespace pih
