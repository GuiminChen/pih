#include "weight_memory_owner.h"
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
namespace {
Status Provider(pih_status_v1 status) {
  if (!pih_status_is_valid_v1(&status)) return Status::Internal("Malformed weight memory provider status; retain ledger");
  if (!pih_status_is_ok_v1(&status)) return Status::FailedPrecondition("Weight memory provider failed; retain ledger");
  return Status::Ok();
}
Status Allocation(const pih_cuda_allocation_v1& a, unsigned kind, int device, std::uint64_t bytes) {
  if (a.struct_size != sizeof(a) || a.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      a.memory_kind != kind || a.device_ordinal != device || !a.generation || a.bytes != bytes ||
      a.alignment < 256 || (a.alignment & (a.alignment - 1)) || !a.address || a.address % a.alignment ||
      a.bytes > std::numeric_limits<std::uintptr_t>::max() - a.address)
    return Status::FailedPrecondition("Weight allocation identity or extent invalid; retain ledger");
  return Status::Ok();
}
}
Status WeightMemoryOwner::Fail(Status status) {
  state_ = WeightMemoryState::kQuarantined;
  return status;
}
Status WeightMemoryOwner::Start(std::uint64_t budget, std::uint64_t staging, Clock::time_point deadline) {
  if (state_ != WeightMemoryState::kEmpty) return Status::FailedPrecondition("Weight owner is single-use");
  const auto bytes = files_.catalog().device_bytes();
  if (!bytes || bytes > budget || !staging || staging > (1U << 20) || staging % 256 ||
      device_ < 0 || !context_ || !stream_ || !event_ || deadline <= Clock::now())
    return Status::InvalidArgument("Weight memory budget, staging, handles or deadline invalid");
  if (memory_.struct_size != sizeof(memory_) || memory_.contract_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      !memory_.context || !memory_.allocate_device || !memory_.deallocate_device ||
      !memory_.allocate_pinned_host || !memory_.deallocate_pinned_host ||
      async_.struct_size != sizeof(async_) || async_.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      !async_.context || !async_.activate_context || !async_.record_event || !async_.query_event)
    return Status::InvalidArgument("Weight memory capabilities invalid");
  state_ = WeightMemoryState::kQuarantined;
  deadline_ = deadline;
  auto status = Provider(async_.activate_context(async_.context, context_)); if (!status.ok()) return status;
  std::uint32_t idle = 0;
  status = Provider(async_.query_event(async_.context, context_, event_, &idle)); if (!status.ok()) return status;
  if (idle != PIH_CUDA_EVENT_COMPLETE_V1) return Status::FailedPrecondition("Weight upload event is not idle");
  status = Provider(memory_.allocate_device(memory_.context, device_, bytes, 256, &device_allocation_));
  if (!status.ok()) return status;
  status = Allocation(device_allocation_, PIH_CUDA_MEMORY_DEVICE_V1, device_, bytes); if (!status.ok()) return status;
  status = Provider(memory_.allocate_pinned_host(memory_.context, -1, staging, 256, &host_allocation_));
  if (!status.ok()) return status;
  status = Allocation(host_allocation_, PIH_CUDA_MEMORY_PINNED_HOST_V1, -1, staging); if (!status.ok()) return status;
  if (device_allocation_.address < host_allocation_.address + host_allocation_.bytes &&
      host_allocation_.address < device_allocation_.address + device_allocation_.bytes)
    return Status::FailedPrecondition("Weight device and staging allocations overlap");
  auto upload = BackboneWeightUpload::Start(files_, {{device_allocation_.address, bytes},
      {host_allocation_.address, staging}, stream_, event_}, deadline);
  if (!upload.ok()) return upload.status();
  upload_ = std::move(*upload);
  state_ = WeightMemoryState::kUploading;
  return Status::Ok();
}
Result<bool> WeightMemoryOwner::Advance() {
  if (state_ == WeightMemoryState::kReady || state_ == WeightMemoryState::kRetired) return true;
  if (state_ != WeightMemoryState::kUploading && state_ != WeightMemoryState::kRetiring)
    return Status::FailedPrecondition("Weight owner has no pending work");
  if (Clock::now() >= deadline_) return Fail(Status::DeadlineExceeded("Weight memory transition expired"));
  const auto active = Provider(async_.activate_context(async_.context, context_)); if (!active.ok()) return Fail(active);
  try {
    if (state_ == WeightMemoryState::kUploading) {
      auto result = upload_->Advance(); if (!result.ok()) return Fail(result.status());
      if (*result != WeightUploadState::kComplete) return false;
      state_ = WeightMemoryState::kReady;
    } else {
      std::uint32_t ready = 0;
      const auto status = Provider(async_.query_event(async_.context, context_, event_, &ready));
      if (!status.ok()) return Fail(status);
      if (ready == PIH_CUDA_EVENT_PENDING_V1) return false;
      if (ready != PIH_CUDA_EVENT_COMPLETE_V1) return Fail(Status::Internal("Invalid weight retirement event status"));
      state_ = WeightMemoryState::kRetired;
    }
    if (Clock::now() >= deadline_) return Fail(Status::DeadlineExceeded("Weight memory transition expired"));
    return true;
  } catch (const std::bad_alloc&) {
    return Fail(Status::ResourceExhausted("Weight upload continuation allocation failed"));
  }
}
Result<const BackboneWeightUpload*> WeightMemoryOwner::Upload() const {
  if (state_ != WeightMemoryState::kReady) return Status::FailedPrecondition("Weights are not admitted ready");
  return upload_.get();
}
Status WeightMemoryOwner::BeginRetirement(Clock::time_point deadline) {
  if (state_ != WeightMemoryState::kReady || deadline <= Clock::now())
    return Status::FailedPrecondition("Weights are not ready for retirement");
  state_ = WeightMemoryState::kQuarantined;
  auto status = Provider(async_.activate_context(async_.context, context_)); if (!status.ok()) return status;
  status = Provider(async_.record_event(async_.context, context_, event_, stream_)); if (!status.ok()) return status;
  deadline_ = deadline;
  state_ = WeightMemoryState::kRetiring;
  return Status::Ok();
}
Status WeightMemoryOwner::Release() {
  if (state_ != WeightMemoryState::kRetired) return Status::FailedPrecondition("Weight consumers must retire before release");
  state_ = WeightMemoryState::kQuarantined;
  upload_.reset();
  auto status = Provider(async_.activate_context(async_.context, context_)); if (!status.ok()) return status;
  status = Provider(memory_.deallocate_pinned_host(memory_.context, &host_allocation_)); if (!status.ok()) return status;
  host_released_ = true;
  status = Provider(memory_.deallocate_device(memory_.context, &device_allocation_)); if (!status.ok()) return status;
  device_released_ = true;
  state_ = WeightMemoryState::kReleased;
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
