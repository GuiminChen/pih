#pragma once
#include "config.h"
#include "rope_launch.h"
#include <optional>

namespace pih::deepseek_v41 {
struct StepPhasesLaunch final {
  RopeSequenceLaunch query;
  // Only compressed-KV owner layers with newly completed groups need this.
  std::optional<RopeSequenceLaunch> compressed;
  std::uint32_t start = 0;
};
Status ValidateStepPhases(const FlashConfig& config, const StepPhasesLaunch& launch);
Status LaunchStepPhases(const FlashConfig& config, const StepPhasesLaunch& launch);

// Four disjoint, aligned segments for query and compressed positions/tables.
// No allocation, initialization or CUDA submission occurs during planning.
class StepPhasesPlan final {
 public:
  static Result<StepPhasesPlan> Create(const FlashConfig& config,
      std::uint32_t token_capacity, std::uint64_t byte_budget);
  std::uint64_t bytes() const noexcept { return bytes_; }
  Result<StepPhasesLaunch> Bind(const FlashConfig& config, EngramDeviceRegion arena,
      std::uint32_t layer, std::uint32_t start, std::uint32_t tokens,
      EngramDeviceRegion error, std::uintptr_t stream) const;
 private:
  StepPhasesPlan() = default;
  std::array<std::uint64_t, 4> offsets_{};
  Sha256Digest config_sha256_;
  std::uint64_t bytes_ = 0;
  std::uint32_t tokens_ = 0;
};
}  // namespace pih::deepseek_v41
