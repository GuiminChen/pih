#pragma once
#include "head_operation.h"
#include "sampling.h"

namespace pih::deepseek_v41 {
class SamplingOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  SamplingOperation(const SamplingOperation&) = delete;
  SamplingOperation& operator=(const SamplingOperation&) = delete;
  SamplingOperation(SamplingOperation&& other) noexcept;
  SamplingOperation& operator=(SamplingOperation&&) = delete;
  ~SamplingOperation();
  static Result<SamplingOperation> Start(const HeadOperation& head, SamplingLaunch launch,
      const SamplingIdentity& identity, EngramDeviceRegion host_candidate,
      const EngramCompletionResources& resources, Clock::time_point deadline);
  Result<bool> Poll();
  Result<SamplingObservation> Candidate() const;
 private:
  SamplingOperation() = default;
  BlockSequence* sequence_ = nullptr;
  EngramDeviceRegion host_candidate_{};
  SamplingParameters parameters_{};
  SamplingObservation observation_{};
  std::optional<EngramCompletion> completion_;
  Clock::time_point deadline_{};
  bool failed_ = true, complete_ = false;
};
}  // namespace pih::deepseek_v41
