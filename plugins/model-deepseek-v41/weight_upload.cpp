#include "weight_upload.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
namespace {
Status Cuda(cudaError_t error) {
  return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error));
}
bool Valid(EngramDeviceRegion r) {
  return r.address && r.bytes && r.address % 256 == 0 &&
      r.bytes <= std::numeric_limits<std::uintptr_t>::max() - r.address;
}
Status Payload(const LocatedWeight& located, std::uint64_t offset, std::span<const std::byte> bytes) {
  const auto& t = located.weight.tensor;
  const auto storage = t.storage;
  const unsigned width = storage == WeightStorage::kF32 ? 4 : storage == WeightStorage::kBF16 ? 2 : 1;
  if (offset % width || bytes.size() % width) return Status::InvalidArgument("Misaligned weight payload chunk");
  const bool table = t.name.find(".engram.embed.") != std::string::npos;
  const auto padding_start = located.weight.valid * t.columns * width;
  for (std::size_t i = 0; i < bytes.size(); i += width) {
    std::uint32_t bits = 0;
    for (unsigned j = 0; j < width; ++j)
      bits |= std::uint32_t(std::to_integer<unsigned char>(bytes[i + j])) << (8 * j);
    const bool invalid = storage == WeightStorage::kF32 ? (bits & 0x7f800000U) == 0x7f800000U :
        storage == WeightStorage::kBF16 ? (bits & 0x7f80U) == 0x7f80U :
        storage == WeightStorage::kE4M3FN ? (bits & 0x7fU) == 0x7fU :
        storage == WeightStorage::kE8M0 ? bits == 255 : false;
    if (invalid) return Status::InvalidArgument("Non-finite V4.1 weight or scale payload");
    if (table && located.weight.padding && offset + i >= padding_start &&
        bits != (storage == WeightStorage::kE8M0 ? 127U : 0U))
      return Status::InvalidArgument("Noncanonical V4.1 Engram table padding");
  }
  return Status::Ok();
}
}
Result<std::unique_ptr<BackboneWeightUpload>> BackboneWeightUpload::Start(
    const BackboneWeightFiles& files, WeightUploadResources r, Clock::time_point deadline) {
  if (!r.stream || !r.event || !Valid(r.device) || !Valid(r.staging) ||
      r.device.bytes != files.catalog().device_bytes() || r.staging.bytes > (1U << 20) ||
      r.staging.bytes % 256 || (r.device.address < r.staging.address + r.staging.bytes &&
      r.staging.address < r.device.address + r.device.bytes))
    return Status::InvalidArgument("V4.1 upload requires exact device arena and disjoint aligned pinned staging");
  if (Clock::now() >= deadline) return Status::DeadlineExceeded("Weight upload deadline expired");
  int device = -1;
  auto status = Cuda(cudaGetDevice(&device)); if (!status.ok()) return status;
  for (const auto region : {r.device, r.staging}) {
    cudaPointerAttributes a{};
    status = Cuda(cudaPointerGetAttributes(&a, reinterpret_cast<void*>(region.address)));
    if (!status.ok()) return status;
    if (region.address == r.device.address ? (a.type != cudaMemoryTypeDevice || a.device != device) : a.type != cudaMemoryTypeHost)
      return Status::FailedPrecondition("Weight upload memory domain invalid");
  }
  const auto event = cudaEventQuery(reinterpret_cast<cudaEvent_t>(r.event));
  if (event == cudaErrorNotReady) return Status::FailedPrecondition("Weight upload event is in use");
  status = Cuda(event); if (!status.ok()) return status;
  status = files.Revalidate(); if (!status.ok()) return status;
  try {
    auto result = std::unique_ptr<BackboneWeightUpload>(new BackboneWeightUpload);
    result->files_ = &files; result->resources_ = r; result->deadline_ = deadline;
    result->device_ = device; result->state_ = WeightUploadState::kReady;
    return result;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Weight upload state allocation failed"); }
}
Result<WeightUploadState> BackboneWeightUpload::Advance() {
  if (state_ == WeightUploadState::kFailed) return Status::FailedPrecondition("Weight upload failed; retire resources");
  if (state_ == WeightUploadState::kComplete) return state_;
  const auto previous = state_; state_ = WeightUploadState::kFailed;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Weight upload deadline expired; retire resources");
  int device = -1;
  auto status = Cuda(cudaGetDevice(&device)); if (!status.ok()) return status;
  if (device != device_) return Status::FailedPrecondition("Weight upload CUDA device changed");
  const auto event = reinterpret_cast<cudaEvent_t>(resources_.event);
  if (previous == WeightUploadState::kWaitingCopy) {
    const auto ready = cudaEventQuery(event);
    if (ready == cudaErrorNotReady) { state_ = previous; return state_; }
    status = Cuda(ready); if (!status.ok()) return status;
    offset_ += pending_; pending_ = 0;
    if (offset_ == files_->catalog().weights()[tensor_].weight.tensor.bytes) { ++tensor_; offset_ = 0; }
    if (tensor_ == files_->catalog().weights().size()) {
      status = files_->Revalidate(); if (!status.ok()) return status;
      if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Weight upload expired at admission");
      state_ = WeightUploadState::kComplete; return state_;
    }
    state_ = WeightUploadState::kReady; return state_;
  }
  const auto& weight = files_->catalog().weights()[tensor_];
  const auto count = std::min(resources_.staging.bytes, weight.weight.tensor.bytes - offset_);
  const auto staging = std::span(reinterpret_cast<std::byte*>(resources_.staging.address), static_cast<std::size_t>(count));
  status = files_->Read(weight.weight.tensor.name, offset_, staging); if (!status.ok()) return status;
  status = Payload(weight, offset_, staging); if (!status.ok()) return status;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Weight upload expired before copy");
  const auto stream = reinterpret_cast<cudaStream_t>(resources_.stream);
  status = Cuda(cudaMemcpyAsync(reinterpret_cast<void*>(resources_.device.address + weight.device_offset + offset_),
      staging.data(), count, cudaMemcpyHostToDevice, stream));
  if (!status.ok()) return status;
  status = Cuda(cudaEventRecord(event, stream)); if (!status.ok()) return status;
  pending_ = count; state_ = WeightUploadState::kWaitingCopy; return state_;
}
Result<EngramDeviceRegion> BackboneWeightUpload::Find(std::string_view name) const {
  if (state_ != WeightUploadState::kComplete) return Status::FailedPrecondition("Weight upload is not admitted complete");
  const auto* weight = files_->catalog().Find(name);
  if (!weight) return Status::InvalidArgument("Unknown uploaded weight name");
  return EngramDeviceRegion{resources_.device.address + weight->device_offset, weight->weight.tensor.bytes};
}
Status BackboneWeightUpload::ValidateScratch(std::span<const EngramDeviceRegion> writes) const {
  if (state_ != WeightUploadState::kComplete) return Status::FailedPrecondition("Weight upload is not admitted complete");
  const auto arena = resources_.device;
  for (const auto r : writes) {
    if (!r.address && !r.bytes) continue;
    if (!r.address || !r.bytes || r.bytes > std::numeric_limits<std::uintptr_t>::max() - r.address)
      return Status::InvalidArgument("Scratch extent invalid");
    if (r.address < arena.address + arena.bytes && arena.address < r.address + r.bytes)
      return Status::InvalidArgument("Scratch overwrites immutable uploaded weights");
  }
  return Status::Ok();
}
Result<EngramDeviceRegion> BackboneWeightUpload::Arena() const {
  if (state_ != WeightUploadState::kComplete) return Status::FailedPrecondition("Weight upload is not admitted complete");
  return resources_.device;
}
}  // namespace pih::deepseek_v41
