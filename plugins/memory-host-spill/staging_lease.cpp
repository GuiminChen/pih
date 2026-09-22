#include "pih/backend/cuda/staging_lease.h"

namespace pih {

Result<CudaStagingLease> CudaStagingLease::Create(
    std::uint64_t owner_id, std::uint64_t capacity_bytes) {
  if (owner_id == 0 || capacity_bytes == 0) {
    return Status::InvalidArgument("CUDA staging slot identity is invalid");
  }
  return CudaStagingLease(owner_id, capacity_bytes);
}

bool CudaStagingLease::generation_matches(
    std::uint64_t generation) const noexcept {
  return generation != 0 && generation == generation_;
}

Status CudaStagingLease::begin_fill(CudaStagingState filling_state,
                                    std::uint64_t generation,
                                    std::uint64_t bytes) {
  if (state_ != CudaStagingState::kFree) {
    return Status::FailedPrecondition("CUDA staging slot is not free");
  }
  if (generation == 0 || generation <= last_generation_ || bytes == 0 ||
      bytes > capacity_bytes_) {
    return Status::InvalidArgument("CUDA staging fill identity is invalid");
  }
  generation_ = generation;
  bytes_ = bytes;
  copy_plan_id_ = 0;
  producer_plan_id_ = 0;
  state_ = filling_state;
  return Status::Ok();
}

Status CudaStagingLease::begin_cpu_fill(std::uint64_t generation,
                                        std::uint64_t bytes) {
  return begin_fill(CudaStagingState::kCpuFilling, generation, bytes);
}

Status CudaStagingLease::begin_gpu_fill(std::uint64_t generation,
                                        std::uint64_t bytes,
                                        std::uint64_t producer_plan_id) {
  if (producer_plan_id == 0) {
    return Status::InvalidArgument("CUDA staging producer identity is invalid");
  }
  const Status status =
      begin_fill(CudaStagingState::kGpuFilling, generation, bytes);
  if (!status.ok()) return status;
  producer_plan_id_ = producer_plan_id;
  return Status::Ok();
}

Status CudaStagingLease::mark_ready(std::uint64_t generation) {
  if (state_ != CudaStagingState::kCpuFilling ||
      !generation_matches(generation)) {
    return Status::FailedPrecondition("CUDA staging fill is not active");
  }
  state_ = CudaStagingState::kReadyToSubmit;
  return Status::Ok();
}

Status CudaStagingLease::complete_gpu_fill(
    const CudaCompletionFrontier& frontier) {
  if (state_ != CudaStagingState::kGpuFilling ||
      frontier.key().plan_generation != producer_plan_id_ ||
      !frontier.publication_authorized()) {
    return Status::FailedPrecondition(
        "CUDA staging GPU fill completion is not publishable");
  }
  state_ = CudaStagingState::kReadyToSubmit;
  return Status::Ok();
}

Status CudaStagingLease::submit_copy(CudaTypedCopyPlan& plan,
                                     TypedCopyDriver& driver) {
  if (state_ != CudaStagingState::kReadyToSubmit || plan.no_op() ||
      plan.bytes() != bytes_) {
    return Status::FailedPrecondition("CUDA staging copy is not ready");
  }
  const bool is_source = plan.source_owner_id() == owner_id_ &&
                         plan.source_generation() == generation_;
  const bool is_destination = plan.destination_owner_id() == owner_id_ &&
                              plan.destination_generation() == generation_;
  if (is_source == is_destination) {
    return Status::InvalidArgument(
        "CUDA staging copy endpoint identity mismatched");
  }
  const Status submitted = plan.submit(driver);
  if (!submitted.ok()) {
    state_ = CudaStagingState::kSuspect;
    return submitted;
  }
  copy_plan_id_ = plan.plan_id();
  state_ = CudaStagingState::kInFlight;
  return Status::Ok();
}

Status CudaStagingLease::complete_copy(
    const CudaCompletionFrontier& frontier) {
  if (state_ != CudaStagingState::kInFlight ||
      frontier.key().phase != CudaCompletionPhase::kCopy ||
      frontier.key().plan_generation != copy_plan_id_ ||
      !frontier.publication_authorized()) {
    return Status::FailedPrecondition(
        "CUDA staging copy completion is not publishable");
  }
  state_ = CudaStagingState::kCompleteVerified;
  return Status::Ok();
}

Status CudaStagingLease::consume(std::uint64_t generation) {
  if (state_ != CudaStagingState::kCompleteVerified ||
      !generation_matches(generation)) {
    return Status::FailedPrecondition("CUDA staging result is not consumable");
  }
  state_ = CudaStagingState::kConsumed;
  return Status::Ok();
}

Status CudaStagingLease::release(std::uint64_t generation) {
  if (state_ != CudaStagingState::kConsumed ||
      !generation_matches(generation)) {
    return Status::FailedPrecondition("CUDA staging lease is not releasable");
  }
  last_generation_ = generation_;
  generation_ = 0;
  bytes_ = 0;
  copy_plan_id_ = 0;
  producer_plan_id_ = 0;
  state_ = CudaStagingState::kFree;
  return Status::Ok();
}

Status CudaStagingLease::fail(std::uint64_t generation) {
  if (state_ == CudaStagingState::kFree ||
      state_ == CudaStagingState::kSuspect ||
      !generation_matches(generation)) {
    return Status::FailedPrecondition("CUDA staging failure identity is invalid");
  }
  state_ = CudaStagingState::kSuspect;
  return Status::Internal("CUDA staging slot became suspect");
}

}  // namespace pih
