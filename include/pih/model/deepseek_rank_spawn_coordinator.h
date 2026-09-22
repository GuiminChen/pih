#pragma once

#include <optional>
#include <span>
#include <vector>

#include "pih/model/deepseek_rank_capacity_plan_instance.h"
#include "pih/model/deepseek_rank_process_supervisor.h"

namespace pih {

class DeepSeekRankSpawnResourceCollector {
 public:
  virtual ~DeepSeekRankSpawnResourceCollector() = default;
  virtual Result<DeepSeekRankSpawnResourceObservation> collect() = 0;
};

class DeepSeekRankSpawnCoordinator final {
 public:
  static Result<DeepSeekRankSpawnCoordinator> Create(
      std::span<const DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan plan,
      DeepSeekRankCapacityPlanInstance capacity_plan_instance,
      DeepSeekRankSpawnResourceCollector& collector,
      DeepSeekRankProcessDriver& driver);

  DeepSeekRankSpawnCoordinator(const DeepSeekRankSpawnCoordinator&) = delete;
  DeepSeekRankSpawnCoordinator& operator=(
      const DeepSeekRankSpawnCoordinator&) = delete;
  DeepSeekRankSpawnCoordinator(DeepSeekRankSpawnCoordinator&&) noexcept =
      default;
  DeepSeekRankSpawnCoordinator& operator=(
      DeepSeekRankSpawnCoordinator&&) noexcept = default;

  Status launch();
  [[nodiscard]] bool launch_attempted() const noexcept {
    return launch_attempted_;
  }
  [[nodiscard]] DeepSeekRankProcessSupervisor* supervisor() noexcept {
    return supervisor_ ? &*supervisor_ : nullptr;
  }
  [[nodiscard]] const DeepSeekRankProcessSupervisor* supervisor()
      const noexcept {
    return supervisor_ ? &*supervisor_ : nullptr;
  }
  [[nodiscard]] const Sha256Digest& preflight_receipt_root() const noexcept {
    return preflight_receipt_root_;
  }
  [[nodiscard]] const Sha256Digest& spawn_authorization_root() const noexcept {
    return spawn_authorization_root_;
  }

 private:
  DeepSeekRankSpawnCoordinator(
      std::vector<DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan plan,
      DeepSeekRankCapacityPlanInstance capacity_plan_instance,
      DeepSeekRankSpawnResourceCollector& collector,
      DeepSeekRankProcessDriver& driver) noexcept;

  std::vector<DeepSeekRankProcessManifest> manifests_;
  DeepSeekRankSpawnResourcePlan plan_;
  std::optional<DeepSeekRankCapacityPlanInstance> capacity_plan_instance_;
  DeepSeekRankSpawnResourceCollector* collector_ = nullptr;
  DeepSeekRankProcessDriver* driver_ = nullptr;
  std::optional<DeepSeekRankProcessSupervisor> supervisor_;
  Sha256Digest preflight_receipt_root_{};
  Sha256Digest spawn_authorization_root_{};
  bool launch_attempted_ = false;
};

}  // namespace pih
