#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_post_exec_resource_exchange.h"
#include "pih/model/deepseek_rank_spawn_coordinator.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankStartupBarrierAbi =
    "pih_deepseek_rank_startup_barrier_v1";

class DeepSeekRankStartupBarrier final {
 public:
  static Result<DeepSeekRankStartupBarrier> Create(
      DeepSeekRankSpawnCoordinator& spawn_coordinator,
      std::span<const DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan spawn_plan,
      std::span<const DeepSeekRankPostExecResourcePlan> post_exec_plans,
      std::uint64_t post_exec_deadline_ns,
      DeepSeekRankPostExecResourceChannel& channel);

  Status advance(std::uint64_t now_ns);
  [[nodiscard]] bool exec_ready() const noexcept;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] const DeepSeekRankPostExecResourceSeal* resource_seal()
      const noexcept;

 private:
  DeepSeekRankStartupBarrier(
      DeepSeekRankSpawnCoordinator& spawn_coordinator,
      std::vector<DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan spawn_plan,
      std::vector<DeepSeekRankPostExecResourcePlan> post_exec_plans,
      std::uint64_t post_exec_deadline_ns,
      DeepSeekRankPostExecResourceChannel& channel) noexcept;
  Status fail(Status cause) noexcept;

  DeepSeekRankSpawnCoordinator* spawn_coordinator_ = nullptr;
  std::vector<DeepSeekRankProcessManifest> manifests_;
  DeepSeekRankSpawnResourcePlan spawn_plan_;
  std::vector<DeepSeekRankPostExecResourcePlan> post_exec_plans_;
  std::uint64_t post_exec_deadline_ns_ = 0;
  DeepSeekRankPostExecResourceChannel* channel_ = nullptr;
  std::optional<DeepSeekRankPostExecResourceCoordinator>
      post_exec_coordinator_;
  bool poisoned_ = false;
};

}  // namespace pih
