#pragma once
#include "expert_dispatch.h"
#include "engram_completion.h"
#include <chrono>
#include <optional>

namespace pih::deepseek_v41 {
class ExpertCounts final {
 public:
  using Clock = std::chrono::steady_clock;
  ExpertCounts(const ExpertCounts&) = delete;
  ExpertCounts& operator=(const ExpertCounts&) = delete;
  ExpertCounts(ExpertCounts&& other) noexcept;
  ExpertCounts& operator=(ExpertCounts&&) = delete;
  // Borrows exact local-expert-sized pinned host storage, disjoint from the
  // pinned error slot, and an exclusive event. Retain all resources until safe
  // completion/retirement, including when Start or Poll fails after submission.
  static Result<ExpertCounts> Start(const ExpertDispatchLaunch& dispatch,
      EngramDeviceRegion host_counts, const EngramCompletionResources& resources,
      Clock::time_point deadline);
  Result<bool> Poll();
  // Accessible only after successful Poll; returns the internally copied count.
  Result<std::uint32_t> Rows(std::uint32_t expert) const;
  // Descriptor identity only; owner must keep device contents immutable.
  Status ValidatePlan(const ExpertDispatchLaunch& dispatch) const;
 private:
  ExpertCounts() = default;
  std::optional<EngramCompletion> completion_;
  EngramDeviceRegion host_counts_{};
  ExpertDispatchLaunch dispatch_{};
  std::array<std::uint32_t, 384> counts_{};
  std::uint32_t first_ = 0, local_ = 0, tokens_ = 0, picks_ = 0;
  Clock::time_point deadline_{};
  bool failed_ = true, complete_ = false;
};
}  // namespace pih::deepseek_v41
