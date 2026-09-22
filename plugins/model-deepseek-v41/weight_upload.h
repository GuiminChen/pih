#pragma once
#include "weight_files.h"
#include "engram_launch.h"
#include <chrono>

namespace pih::deepseek_v41 {
enum class WeightUploadState { kReady, kWaitingCopy, kComplete, kFailed };
struct WeightUploadResources final {
  // Borrowed exclusive regions/stream/event, allocated through plugin contracts.
  EngramDeviceRegion device, staging;
  std::uintptr_t stream = 0, event = 0;
};
class BackboneWeightUpload final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<std::unique_ptr<BackboneWeightUpload>> Start(const BackboneWeightFiles& files,
      WeightUploadResources resources, Clock::time_point deadline);
  BackboneWeightUpload(const BackboneWeightUpload&) = delete;
  BackboneWeightUpload& operator=(const BackboneWeightUpload&) = delete;
  // At most one bounded read/copy per call. No implicit synchronization or retry.
  Result<WeightUploadState> Advance();
  Result<EngramDeviceRegion> Find(std::string_view tensor) const;
  Result<EngramDeviceRegion> Arena() const;
  const BackboneWeightCatalog& catalog() const noexcept { return files_->catalog(); }
  // Reject writes anywhere in the full immutable weight arena, not just the
  // particular layer's currently bound tensors. Requires completed upload.
  Status ValidateScratch(std::span<const EngramDeviceRegion> writes) const;
  WeightUploadState state() const noexcept { return state_; }
  // Destruction does not retire work or free resources. On failure/early
  // destruction the owner must retain and safely retire all borrowed resources.
 private:
  BackboneWeightUpload() = default;
  const BackboneWeightFiles* files_ = nullptr;
  WeightUploadResources resources_{};
  Clock::time_point deadline_{};
  int device_ = -1;
  std::size_t tensor_ = 0;
  std::uint64_t offset_ = 0, pending_ = 0;
  WeightUploadState state_ = WeightUploadState::kFailed;
};
}  // namespace pih::deepseek_v41
