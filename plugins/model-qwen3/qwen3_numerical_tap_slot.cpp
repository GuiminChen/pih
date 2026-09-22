#include "pih/model/qwen3_numerical_tap_slot.h"

namespace pih {

Result<QwenNumericalTapSlot> QwenNumericalTapSlot::Create(
    std::uint64_t owner_id, std::uint64_t allocation_generation,
    std::uint64_t capacity_bytes) {
  if (owner_id == 0 || allocation_generation == 0 || capacity_bytes == 0) {
    return Status::InvalidArgument("Qwen numerical tap slot identity is invalid");
  }
  return QwenNumericalTapSlot(owner_id, allocation_generation, capacity_bytes);
}

Status QwenNumericalTapSlot::poison(const char* message) {
  state_ = QwenNumericalTapSlotState::kPoisoned;
  return Status::FailedPrecondition(message);
}

Status QwenNumericalTapSlot::begin(
    std::uint64_t capture_generation, std::size_t capture_index,
    std::uint64_t source_owner_id, std::uint64_t source_generation,
    std::uint64_t bytes, std::uint64_t producer_plan_generation) {
  if (state_ != QwenNumericalTapSlotState::kFree) {
    return Status::FailedPrecondition("Qwen numerical tap slot is not free");
  }
  if (capture_generation == 0 || capture_generation <= last_generation_ ||
      source_owner_id == 0 || source_generation == 0 || bytes == 0 ||
      bytes > capacity_bytes_ || producer_plan_generation == 0) {
    return Status::InvalidArgument("Qwen numerical tap capture identity is invalid");
  }
  generation_ = capture_generation;
  capture_index_ = capture_index;
  source_owner_id_ = source_owner_id;
  source_generation_ = source_generation;
  bytes_ = bytes;
  producer_plan_generation_ = producer_plan_generation;
  copy_plan_id_ = 0;
  copy_event_generation_ = 0;
  state_ = QwenNumericalTapSlotState::kProducerPending;
  return Status::Ok();
}

Status QwenNumericalTapSlot::complete_producer(
    const CudaCompletionFrontier& frontier) {
  if (state_ != QwenNumericalTapSlotState::kProducerPending ||
      !frontier.publication_authorized() ||
      frontier.key().plan_generation != producer_plan_generation_) {
    return poison("Qwen numerical tap producer frontier is invalid");
  }
  state_ = QwenNumericalTapSlotState::kCopyReady;
  return Status::Ok();
}

Status QwenNumericalTapSlot::submit_copy(CudaTypedCopyPlan& copy,
                                         TypedCopyDriver& driver) {
  if (state_ != QwenNumericalTapSlotState::kCopyReady ||
      copy.source_owner_id() != source_owner_id_ ||
      copy.source_generation() != source_generation_ ||
      copy.destination_owner_id() != owner_id_ ||
      copy.destination_generation() != allocation_generation_ ||
      copy.bytes() != bytes_ ||
      copy.completion_event_generation() != generation_) {
    return poison("Qwen numerical tap copy identity drifted");
  }
  const Status submitted = copy.submit(driver);
  if (!submitted.ok()) {
    state_ = QwenNumericalTapSlotState::kPoisoned;
    return submitted;
  }
  copy_plan_id_ = copy.plan_id();
  copy_event_generation_ = copy.completion_event_generation();
  state_ = QwenNumericalTapSlotState::kCopyInFlight;
  return Status::Ok();
}

Status QwenNumericalTapSlot::complete_copy(
    const CudaCompletionFrontier& frontier) {
  if (state_ != QwenNumericalTapSlotState::kCopyInFlight ||
      !frontier.publication_authorized() ||
      frontier.key().plan_generation != copy_plan_id_ ||
      frontier.event_generation() != copy_event_generation_) {
    return poison("Qwen numerical tap copy frontier is invalid");
  }
  state_ = QwenNumericalTapSlotState::kCopyComplete;
  return Status::Ok();
}

Result<QwenNumericalTapReceipt> QwenNumericalTapSlot::seal(
    std::uint64_t capture_generation,
    std::span<const std::byte> observed_bytes) {
  if (state_ != QwenNumericalTapSlotState::kCopyComplete ||
      capture_generation != generation_ || observed_bytes.size() != bytes_) {
    return poison("Qwen numerical tap writer bytes are incomplete");
  }
  auto digest = sha256(observed_bytes);
  if (!digest.ok()) {
    state_ = QwenNumericalTapSlotState::kPoisoned;
    return digest.status();
  }
  state_ = QwenNumericalTapSlotState::kSealed;
  return QwenNumericalTapReceipt{capture_index_, generation_, bytes_, *digest};
}

Status QwenNumericalTapSlot::release(std::uint64_t capture_generation) {
  if (state_ != QwenNumericalTapSlotState::kSealed ||
      capture_generation != generation_) {
    return Status::FailedPrecondition(
        "Qwen numerical tap slot cannot be released before sealing");
  }
  last_generation_ = generation_;
  state_ = QwenNumericalTapSlotState::kFree;
  return Status::Ok();
}

}  // namespace pih
