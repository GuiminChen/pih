#include "pih/model/deepseek_index_selection_driver.h"

#include <algorithm>
#include <cstring>

namespace pih {

Result<DeepSeekIndexSelectionDriver> DeepSeekIndexSelectionDriver::Create(
    DeepSeekIndexSelectionOperations& operations,
    DeepSeekIndexSelectionHostStaging staging,
    std::uintptr_t completion_event) {
  if (staging.scores == nullptr || staging.error_flag == nullptr ||
      staging.score_capacity == 0 || completion_event == 0) {
    return Status::InvalidArgument(
        "DeepSeek index selection resources are invalid");
  }
  auto status = operations.validate_host_staging(staging);
  if (!status.ok()) return status;
  DeepSeekIndexSelectionDriver driver;
  driver.operations_ = &operations;
  driver.staging_ = staging;
  driver.completion_event_ = completion_event;
  return driver;
}

Status DeepSeekIndexSelectionDriver::begin(
    const DeepSeekIndexSelectionSubmission& submission) {
  auto status = validate(submission);
  if (!status.ok()) return status;
  submission_ = submission;
  visible_slot_counts_.assign(submission.visible_slot_counts.begin(),
                              submission.visible_slot_counts.end());
  submission_.visible_slot_counts = visible_slot_counts_;
  selectors_.assign(submission.query_count, DeepSeekOnlineTopK{});
  next_slot_ = 0;
  inflight_slot_count_ = 0;
  logical_page_count_ = 0;
  physical_page_count_ = 0;
  paged_ = false;
  *staging_.error_flag = 0;
  state_ = State::kReady;
  return Status::Ok();
}

Status DeepSeekIndexSelectionDriver::begin_paged(
    const DeepSeekIndexSelectionSubmission& submission,
    std::span<const std::uint32_t> page_slots,
    std::uint32_t physical_page_count) {
  auto status = validate(submission);
  const auto required_pages =
      (static_cast<std::uint64_t>(submission.total_slot_count) + 63U) / 64U;
  if (!status.ok()) return status;
  if (page_slots.size() != required_pages || physical_page_count == 0 ||
      staging_.page_slots == nullptr ||
      page_slots.size() > staging_.page_slot_capacity ||
      submission.arena.page_slots_u32 == 0 ||
      page_slots.size() > submission.arena.page_slot_capacity ||
      std::ranges::any_of(page_slots, [physical_page_count](auto slot) {
        return slot >= physical_page_count;
      })) {
    return Status::InvalidArgument(
        "DeepSeek paged index selection table is invalid");
  }
  std::memcpy(staging_.page_slots, page_slots.data(),
              page_slots.size_bytes());
  status = operations_->copy_h2d_async(
      submission.arena.page_slots_u32, staging_.page_slots,
      page_slots.size_bytes(), submission.stream);
  if (!status.ok()) {
    state_ = State::kPoisoned;
    return status;
  }
  submission_ = submission;
  visible_slot_counts_.assign(submission.visible_slot_counts.begin(),
                              submission.visible_slot_counts.end());
  submission_.visible_slot_counts = visible_slot_counts_;
  selectors_.assign(submission.query_count, DeepSeekOnlineTopK{});
  next_slot_ = 0;
  inflight_slot_count_ = 0;
  logical_page_count_ = static_cast<std::uint32_t>(page_slots.size());
  physical_page_count_ = physical_page_count;
  paged_ = true;
  *staging_.error_flag = 0;
  state_ = State::kReady;
  return Status::Ok();
}

Status DeepSeekIndexSelectionDriver::validate(
    const DeepSeekIndexSelectionSubmission& submission) const {
  if (state_ == State::kInflight || state_ == State::kPoisoned) {
    return Status::FailedPrecondition(
        "DeepSeek index selection driver is not reusable");
  }
  const auto required_capacity =
      static_cast<std::uint64_t>(submission.query_count) *
      std::min(submission.total_slot_count,
               DeepSeekIndexScoreLaunch::kMaximumSlotTile);
  if (submission.query_bf16 == 0 || submission.index_kv_bf16 == 0 ||
      submission.head_weight_f32 == 0 || submission.arena.score_f32 == 0 ||
      submission.arena.error_flag_u32 == 0 || submission.stream == 0 ||
      submission.query_count == 0 || submission.query_count > 4096 ||
      submission.head_count != 64 ||
      submission.total_slot_count == 0 ||
      submission.total_slot_count > 262144 ||
      submission.visible_slot_counts.size() != submission.query_count ||
      required_capacity > staging_.score_capacity) {
    return Status::InvalidArgument(
        "DeepSeek index selection submission is invalid");
  }
  for (const auto visible : submission.visible_slot_counts) {
    if (visible > submission.total_slot_count) {
      return Status::InvalidArgument(
          "DeepSeek index selection visibility is invalid");
    }
  }
  return Status::Ok();
}

Status DeepSeekIndexSelectionDriver::launch_next_tile() {
  if (state_ != State::kReady) {
    return Status::FailedPrecondition(
        "DeepSeek index selection tile is not launchable");
  }
  inflight_slot_count_ = std::min<std::uint32_t>(
      DeepSeekIndexScoreLaunch::kMaximumSlotTile,
      submission_.total_slot_count - next_slot_);
  auto fail = [this](Status status) {
    if (!status.ok()) state_ = State::kPoisoned;
    return status;
  };
  auto status = fail(operations_->zero_u32_async(
      submission_.arena.error_flag_u32, submission_.stream));
  if (!status.ok()) return status;
  const auto kv_tile = paged_
      ? submission_.index_kv_bf16
      : submission_.index_kv_bf16 +
            static_cast<std::uint64_t>(next_slot_) * 128U *
                sizeof(std::uint16_t);
  status = fail(operations_->score(
      {submission_.query_bf16, kv_tile, submission_.head_weight_f32,
       submission_.arena.score_f32, submission_.arena.error_flag_u32,
       submission_.stream, submission_.query_count, submission_.head_count,
       inflight_slot_count_,
       paged_ ? submission_.arena.page_slots_u32 : 0,
       paged_ ? next_slot_ : 0,
       paged_ ? logical_page_count_ : 0,
       paged_ ? physical_page_count_ : 0}));
  if (!status.ok()) return status;
  const auto elements = submission_.query_count * inflight_slot_count_;
  status = fail(operations_->copy_d2h_async(
      staging_.scores, submission_.arena.score_f32,
      static_cast<std::size_t>(elements) * sizeof(float),
      submission_.stream));
  if (!status.ok()) return status;
  status = fail(operations_->copy_d2h_async(
      staging_.error_flag, submission_.arena.error_flag_u32,
      sizeof(std::uint32_t), submission_.stream));
  if (!status.ok()) return status;
  status = fail(operations_->record_event(completion_event_, submission_.stream));
  if (!status.ok()) return status;
  state_ = State::kInflight;
  return Status::Ok();
}

Result<DeepSeekExpertAsyncStatus> DeepSeekIndexSelectionDriver::poll_tile() {
  if (state_ == State::kPoisoned) return DeepSeekExpertAsyncStatus::kError;
  if (state_ != State::kInflight) {
    return Status::FailedPrecondition(
        "DeepSeek index selection has no inflight tile");
  }
  auto event = operations_->query_event(completion_event_);
  if (!event.ok()) {
    state_ = State::kPoisoned;
    return event.status();
  }
  if (*event == DeepSeekExpertAsyncStatus::kInProgress) return *event;
  if (*event == DeepSeekExpertAsyncStatus::kError || *staging_.error_flag != 0) {
    state_ = State::kPoisoned;
    return DeepSeekExpertAsyncStatus::kError;
  }
  if (*event != DeepSeekExpertAsyncStatus::kSuccess) {
    state_ = State::kPoisoned;
    return Status::Internal(
        "DeepSeek index selection received invalid event state");
  }
  for (std::uint32_t query = 0; query < submission_.query_count; ++query) {
    const auto visible = visible_slot_counts_[query];
    if (next_slot_ >= visible) continue;
    const auto count = std::min(inflight_slot_count_, visible - next_slot_);
    const auto row = std::span<const float>(
        staging_.scores + static_cast<std::size_t>(query) *
                              inflight_slot_count_,
        count);
    auto status = selectors_[query].consume(next_slot_, row);
    if (!status.ok()) {
      state_ = State::kPoisoned;
      return status;
    }
  }
  next_slot_ += inflight_slot_count_;
  inflight_slot_count_ = 0;
  state_ = next_slot_ == submission_.total_slot_count ? State::kComplete
                                                       : State::kReady;
  return DeepSeekExpertAsyncStatus::kSuccess;
}

bool DeepSeekIndexSelectionDriver::complete() const {
  return state_ == State::kComplete;
}

Result<std::vector<std::vector<std::uint32_t>>>
DeepSeekIndexSelectionDriver::finish() const {
  if (state_ != State::kComplete) {
    return Status::FailedPrecondition(
        "DeepSeek index selection is not complete");
  }
  std::vector<std::vector<std::uint32_t>> result;
  result.reserve(selectors_.size());
  for (std::size_t query = 0; query < selectors_.size(); ++query) {
    if (visible_slot_counts_[query] == 0) {
      result.emplace_back();
      continue;
    }
    const auto& selector = selectors_[query];
    auto indices = selector.finish();
    if (!indices.ok()) return indices.status();
    result.push_back(std::move(*indices));
  }
  return result;
}

}  // namespace pih
