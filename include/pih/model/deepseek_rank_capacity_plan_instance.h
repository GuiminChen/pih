#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "pih/model/deepseek_execution_topology.h"
#include "pih/model/deepseek_rank_process_supervisor.h"
#include "pih/model/runtime_profile_payload.h"
#include "pih/model/runtime_profile_supervisor_bootstrap_manifest.h"

namespace pih {

class LinuxDeepSeekRankSpawnController;

inline constexpr std::string_view kDeepSeekRankCapacityPlanInstanceAbi =
    "pih_deepseek_rank_capacity_plan_instance_v1";
inline constexpr std::string_view kDeepSeekRankCapacityImplementedScopeAbi =
    "profile_device_bootstrap_rank_spawn_resource_v1";

class DeepSeekRankCapacityPlanInstance final {
 public:
  static Result<DeepSeekRankCapacityPlanInstance> Compile(
      const RuntimeEngineAdmission& admission,
      const RuntimeProfileSupervisorBootstrapManifest& bootstrap,
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankSpawnResourcePlan& resource_plan,
      std::uint64_t expected_controller_process_identity);

  DeepSeekRankCapacityPlanInstance(
      const DeepSeekRankCapacityPlanInstance&) = delete;
  DeepSeekRankCapacityPlanInstance& operator=(
      const DeepSeekRankCapacityPlanInstance&) = delete;
  DeepSeekRankCapacityPlanInstance(
      DeepSeekRankCapacityPlanInstance&& other) noexcept;
  DeepSeekRankCapacityPlanInstance& operator=(
      DeepSeekRankCapacityPlanInstance&& other) noexcept;

  [[nodiscard]] Status validate_static_binding(
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankSpawnResourcePlan& resource_plan) const;
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] DeepSeekExecutionTopology execution_topology() const noexcept {
    return DeepSeekExecutionTopology::kOneProcessPerRank;
  }
  [[nodiscard]] std::uint64_t expected_controller_process_identity()
      const noexcept {
    return expected_controller_process_identity_;
  }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }
  [[nodiscard]] const Sha256Digest& resource_plan_root() const noexcept {
    return resource_plan_root_;
  }
  [[nodiscard]] const Sha256Digest& instance_root() const noexcept {
    return instance_root_;
  }

 private:
  friend class DeepSeekRankSpawnAuthorization;
  friend class LinuxDeepSeekRankSpawnController;
  DeepSeekRankCapacityPlanInstance(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size,
      std::uint64_t expected_controller_process_identity,
      Sha256Digest manifest_root, Sha256Digest resource_plan_root,
      Sha256Digest instance_root,
      std::shared_ptr<const RuntimeEngineAdmission> admission_anchor) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint64_t expected_controller_process_identity_ = 0;
  Sha256Digest manifest_root_{};
  Sha256Digest resource_plan_root_{};
  Sha256Digest instance_root_{};
  std::shared_ptr<const RuntimeEngineAdmission> admission_anchor_;
};

}  // namespace pih
