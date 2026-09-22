#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "pih/model/deepseek_rank_process_supervisor.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankPostExecResourcePlanAbi =
    "pih_deepseek_rank_post_exec_resource_plan_v1";
inline constexpr std::string_view kDeepSeekRankPostExecResourceReceiptAbi =
    "pih_deepseek_rank_post_exec_resource_receipt_v1";
inline constexpr std::string_view kDeepSeekRankPostExecResourceSealAbi =
    "pih_deepseek_rank_post_exec_resource_seal_v1";

struct DeepSeekRankPostExecResourcePlan final {
  std::uint64_t worker_task_peak = 0;
  std::uint64_t worker_fd_peak = 0;
  std::uint64_t worker_vma_peak = 0;
  std::uint64_t task_emergency_reserve = 0;
  std::uint64_t fd_emergency_reserve = 0;
  std::uint64_t vma_emergency_reserve = 0;
  Sha256Digest os_resource_envelope_root{};
};

struct DeepSeekRankPostExecResourceAuthority final {
  std::uint32_t protocol_version = 0;
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  Sha256Digest manifest_root{};
  Sha256Digest spawn_resource_plan_root{};
  Sha256Digest capacity_plan_instance_root{};
  Sha256Digest os_resource_envelope_root{};
  std::uint64_t deadline_ns = 0;
  DeepSeekRankPostExecResourcePlan resource_plan;
};

struct DeepSeekRankPostExecResourceObservation final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t rank = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t challenge_identity = 0;
  Sha256Digest acknowledged_capacity_plan_instance_root{};
  Sha256Digest acknowledged_os_resource_envelope_root{};
  std::uint64_t task_count = 0;
  std::uint64_t open_fd_count = 0;
  std::uint64_t scm_rights_inflight_fd_count = 0;
  std::uint64_t vma_count = 0;
  std::uint64_t rlimit_nofile_soft = 0;
  std::uint64_t rlimit_nofile_hard = 0;
  std::uint64_t fs_nr_open = 0;
  std::uint64_t vm_max_map_count = 0;
  bool non_dumpable = false;
};

Result<Sha256Digest> compile_deepseek_rank_post_exec_resource_plan_root(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::uint32_t rank,
    const DeepSeekRankPostExecResourcePlan& plan);
Result<Sha256Digest> compile_deepseek_rank_post_exec_resource_plan_set_root(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> plans);

class DeepSeekRankPostExecResourceReceipt final {
 public:
  static Result<DeepSeekRankPostExecResourceReceipt> Compile(
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankSpawnResourcePlan& spawn_plan,
      const DeepSeekRankPostExecResourcePlan& plan,
      const DeepSeekRankExecReady& exec_ready,
      const DeepSeekRankPostExecResourceObservation& observation);

  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::uint64_t process_identity() const noexcept {
    return process_identity_;
  }
  [[nodiscard]] std::uint64_t process_manifest_identity() const noexcept {
    return process_manifest_identity_;
  }
  [[nodiscard]] std::uint64_t pidfd_identity() const noexcept {
    return pidfd_identity_;
  }
  [[nodiscard]] std::uint64_t control_identity() const noexcept {
    return control_identity_;
  }
  [[nodiscard]] std::uint64_t challenge_identity() const noexcept {
    return challenge_identity_;
  }
  [[nodiscard]] const Sha256Digest& capacity_plan_instance_root()
      const noexcept {
    return capacity_plan_instance_root_;
  }
  [[nodiscard]] const Sha256Digest& os_resource_envelope_root()
      const noexcept {
    return os_resource_envelope_root_;
  }
  [[nodiscard]] const Sha256Digest& resource_plan_root() const noexcept {
    return resource_plan_root_;
  }
  [[nodiscard]] const Sha256Digest& receipt_root() const noexcept {
    return receipt_root_;
  }

 private:
  DeepSeekRankPostExecResourceReceipt(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t rank, std::uint32_t world_size,
      std::uint64_t process_manifest_identity,
      std::uint64_t process_identity, std::uint64_t pidfd_identity,
      std::uint64_t control_identity, std::uint64_t challenge_identity,
      Sha256Digest capacity_plan_instance_root,
      Sha256Digest os_resource_envelope_root,
      Sha256Digest resource_plan_root,
      Sha256Digest receipt_root) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t rank_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint64_t process_manifest_identity_ = 0;
  std::uint64_t process_identity_ = 0;
  std::uint64_t pidfd_identity_ = 0;
  std::uint64_t control_identity_ = 0;
  std::uint64_t challenge_identity_ = 0;
  Sha256Digest capacity_plan_instance_root_{};
  Sha256Digest os_resource_envelope_root_{};
  Sha256Digest resource_plan_root_{};
  Sha256Digest receipt_root_{};
};

class DeepSeekRankPostExecResourceSeal final {
 public:
  static Result<DeepSeekRankPostExecResourceSeal> Compile(
      const DeepSeekRankProcessSupervisor& supervisor,
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankSpawnResourcePlan& spawn_plan,
      std::span<const DeepSeekRankPostExecResourcePlan> plans,
      std::span<const DeepSeekRankPostExecResourceReceipt> receipts);

  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] const Sha256Digest& capacity_plan_instance_root()
      const noexcept {
    return capacity_plan_instance_root_;
  }
  [[nodiscard]] const Sha256Digest& os_resource_envelope_root()
      const noexcept {
    return os_resource_envelope_root_;
  }
  [[nodiscard]] const Sha256Digest& receipt_set_root() const noexcept {
    return receipt_set_root_;
  }
  [[nodiscard]] const Sha256Digest& plan_set_root() const noexcept {
    return plan_set_root_;
  }
  [[nodiscard]] const Sha256Digest& seal_root() const noexcept {
    return seal_root_;
  }

 private:
  DeepSeekRankPostExecResourceSeal(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, Sha256Digest capacity_plan_instance_root,
      Sha256Digest os_resource_envelope_root, Sha256Digest plan_set_root,
      Sha256Digest receipt_set_root,
      Sha256Digest seal_root) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  Sha256Digest capacity_plan_instance_root_{};
  Sha256Digest os_resource_envelope_root_{};
  Sha256Digest plan_set_root_{};
  Sha256Digest receipt_set_root_{};
  Sha256Digest seal_root_{};
};

}  // namespace pih
