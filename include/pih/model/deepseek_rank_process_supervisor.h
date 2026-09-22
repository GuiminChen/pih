#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

class DeepSeekRankCapacityPlanInstance;
class RuntimeEngineAdmission;

struct DeepSeekRankProcessManifest final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint64_t physical_device_identity = 0;
  std::uint64_t process_manifest_identity = 0;
  Sha256Digest physical_device_uuid_commitment{};
  std::int32_t startup_device_ordinal = -1;
  std::uint64_t startup_deadline_ns = 0;
};

struct DeepSeekRankProcessHandle final {
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
};

struct DeepSeekRankExecChallenge final {
  std::uint32_t protocol_version = 0;
  DeepSeekRankProcessManifest manifest;
  DeepSeekRankProcessHandle handle;
  std::uint64_t controller_process_identity = 0;
  std::uint64_t challenge_identity = 0;
};

struct DeepSeekRankReadyReceipt final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t rank = 0;
  std::uint64_t physical_device_identity = 0;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  Sha256Digest physical_device_uuid_commitment{};
  std::int32_t startup_device_ordinal = -1;
  std::uint64_t startup_deadline_ns = 0;
};

struct DeepSeekRankExecReady final {
  DeepSeekRankReadyReceipt receipt;
  std::uint64_t challenge_identity = 0;
};

inline constexpr std::string_view kDeepSeekRankSpawnAuthorizationAbi =
    "pih_deepseek_rank_spawn_authorization_v1";
inline constexpr std::string_view kDeepSeekRankSpawnPreflightAbi =
    "pih_deepseek_rank_spawn_preflight_v1";

struct DeepSeekRankSpawnResourcePlan final {
  std::uint32_t worker_processes = 0;
  std::uint32_t emergency_task_reserve = 0;
  std::uint64_t worker_fd_peak = 0;
  std::uint64_t fd_emergency_reserve = 0;
  std::uint64_t worker_vma_peak = 0;
  std::uint64_t vma_emergency_reserve = 0;
  std::uint64_t node_file_handle_reserve = 0;
};

Result<Sha256Digest> compile_deepseek_rank_process_manifest_root(
    std::span<const DeepSeekRankProcessManifest> manifests);
Result<Sha256Digest> compile_deepseek_rank_spawn_resource_plan_root(
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& plan);

struct DeepSeekRankSpawnCgroupPidsObservation final {
  Sha256Digest scope_root{};
  std::uint64_t current_tasks = 0;
  std::optional<std::uint64_t> maximum_tasks;
};

struct DeepSeekRankSpawnResourceObservation final {
  std::uint64_t uid_tasks_current = 0;
  std::optional<std::uint64_t> rlimit_nproc_soft;
  bool rlimit_nproc_enforced = true;
  std::vector<DeepSeekRankSpawnCgroupPidsObservation> cgroup_ancestors;
  std::uint64_t controller_open_fds = 0;
  std::uint64_t rlimit_nofile_soft = 0;
  std::uint64_t rlimit_nofile_hard = 0;
  std::uint64_t fs_nr_open = 0;
  std::uint64_t node_file_allocated = 0;
  std::uint64_t node_file_maximum = 0;
  std::uint64_t vm_max_map_count = 0;
};

class DeepSeekRankSpawnPreflightReceipt final {
 public:
  static Status ValidatePlan(
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankSpawnResourcePlan& plan);
  static Result<DeepSeekRankSpawnPreflightReceipt> Compile(
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankSpawnResourcePlan& plan,
      const DeepSeekRankSpawnResourceObservation& observation);
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }
  [[nodiscard]] std::uint64_t task_increment() const noexcept {
    return task_increment_;
  }
  [[nodiscard]] std::uint64_t controller_fd_increment() const noexcept {
    return controller_fd_increment_;
  }
  [[nodiscard]] std::uint64_t node_file_handle_increment() const noexcept {
    return node_file_handle_increment_;
  }
  [[nodiscard]] const Sha256Digest& resource_plan_root() const noexcept {
    return resource_plan_root_;
  }
  [[nodiscard]] const Sha256Digest& receipt_root() const noexcept {
    return receipt_root_;
  }

 private:
  DeepSeekRankSpawnPreflightReceipt(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, Sha256Digest manifest_root,
      std::uint64_t task_increment, std::uint64_t controller_fd_increment,
      std::uint64_t node_file_handle_increment,
      Sha256Digest resource_plan_root,
      Sha256Digest receipt_root) noexcept;
  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  Sha256Digest manifest_root_{};
  std::uint64_t task_increment_ = 0;
  std::uint64_t controller_fd_increment_ = 0;
  std::uint64_t node_file_handle_increment_ = 0;
  Sha256Digest resource_plan_root_{};
  Sha256Digest receipt_root_{};
};

class DeepSeekRankSpawnAuthorization final {
 public:
  DeepSeekRankSpawnAuthorization(const DeepSeekRankSpawnAuthorization&) =
      delete;
  DeepSeekRankSpawnAuthorization& operator=(
      const DeepSeekRankSpawnAuthorization&) = delete;
  DeepSeekRankSpawnAuthorization(DeepSeekRankSpawnAuthorization&&) noexcept =
      default;
  DeepSeekRankSpawnAuthorization& operator=(
      DeepSeekRankSpawnAuthorization&&) noexcept = default;
  static Result<DeepSeekRankSpawnAuthorization> Create(
      std::span<const DeepSeekRankProcessManifest> manifests,
      DeepSeekRankCapacityPlanInstance capacity_plan_instance,
      const DeepSeekRankSpawnPreflightReceipt& preflight);
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }
  [[nodiscard]] const Sha256Digest& capacity_plan_instance_root()
      const noexcept {
    return capacity_plan_instance_root_;
  }
  [[nodiscard]] const Sha256Digest& prospective_resource_receipt_root()
      const noexcept {
    return prospective_resource_receipt_root_;
  }
  [[nodiscard]] const Sha256Digest& spawn_resource_plan_root()
      const noexcept {
    return spawn_resource_plan_root_;
  }
  [[nodiscard]] const Sha256Digest& authorization_root() const noexcept {
    return authorization_root_;
  }

 private:
  friend class DeepSeekRankProcessSupervisor;
  DeepSeekRankSpawnAuthorization(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, Sha256Digest manifest_root,
      Sha256Digest capacity_plan_instance_root,
      Sha256Digest spawn_resource_plan_root,
      Sha256Digest prospective_resource_receipt_root,
      Sha256Digest authorization_root,
      std::shared_ptr<const RuntimeEngineAdmission>
          capacity_authority_anchor) noexcept;
  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  Sha256Digest manifest_root_{};
  Sha256Digest capacity_plan_instance_root_{};
  Sha256Digest spawn_resource_plan_root_{};
  Sha256Digest prospective_resource_receipt_root_{};
  Sha256Digest authorization_root_{};
  std::shared_ptr<const RuntimeEngineAdmission> capacity_authority_anchor_;
};

enum class DeepSeekRankProcessObservation : std::uint8_t {
  kRunning, kExitedSuccess, kExitedFailure,
};

class DeepSeekRankProcessDriver {
 public:
  virtual ~DeepSeekRankProcessDriver() = default;
  virtual Result<DeepSeekRankProcessHandle> spawn(
      const DeepSeekRankProcessManifest& manifest) = 0;
  virtual Result<DeepSeekRankProcessObservation> observe(
      const DeepSeekRankProcessHandle& handle) = 0;
  virtual Status terminate(const DeepSeekRankProcessHandle& handle) = 0;
  virtual Status send_challenge(
      const DeepSeekRankProcessHandle& handle,
      const DeepSeekRankExecChallenge& challenge) = 0;
  virtual Result<std::optional<DeepSeekRankExecReady>> poll_ready(
      const DeepSeekRankProcessHandle& handle) = 0;
};

class DeepSeekRankProcessSupervisor final {
 public:
  static Result<DeepSeekRankProcessSupervisor> Create(
      std::span<const DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnAuthorization authorization,
      DeepSeekRankProcessDriver& driver);
  Status launch();
  Status dispatch_challenges(
      std::uint64_t controller_process_identity,
      std::span<const std::uint64_t> challenge_identities);
  Status dispatch_challenge(std::uint32_t rank,
                            std::uint64_t controller_process_identity,
                            std::uint64_t challenge_identity);
  Status poll_ready(std::uint32_t rank);
  Status advance_exec_startup(std::uint64_t now_ns);
  Status abort_startup(Status cause) noexcept;
  Status abort_post_exec(Status cause) noexcept;
  Status poll();
  Status expire(std::uint64_t now_ns);
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] bool failed() const noexcept { return failed_; }
  [[nodiscard]] bool rank_exec_ready(std::uint32_t rank) const noexcept {
    return rank < ready_.size() && ready_[rank];
  }
  [[nodiscard]] const DeepSeekRankExecReady* exec_ready(
      std::uint32_t rank) const noexcept {
    return rank < exec_ready_.size() && exec_ready_[rank]
               ? &*exec_ready_[rank]
               : nullptr;
  }
  [[nodiscard]] const DeepSeekRankProcessHandle* process_handle(
      std::uint32_t rank) const noexcept {
    return rank < handles_.size() && handles_[rank] ? &*handles_[rank]
                                                    : nullptr;
  }
  [[nodiscard]] bool capacity_authority_retained() const noexcept {
    return capacity_authority_anchor_ != nullptr;
  }
  // Borrow the exact admission anchor consumed by the rank capacity
  // instance. Later startup joins must use this object instead of accepting a
  // caller-selected second admission with superficially matching roots.
  [[nodiscard]] const RuntimeEngineAdmission* capacity_admission()
      const noexcept {
    return capacity_authority_anchor_.get();
  }
  [[nodiscard]] const Sha256Digest& spawn_authorization_root() const noexcept {
    return spawn_authorization_root_;
  }
  [[nodiscard]] const Sha256Digest& capacity_plan_instance_root()
      const noexcept {
    return capacity_plan_instance_root_;
  }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }
  [[nodiscard]] const Sha256Digest& spawn_resource_plan_root()
      const noexcept {
    return spawn_resource_plan_root_;
  }

 private:
  DeepSeekRankProcessSupervisor(
      std::vector<DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnAuthorization authorization,
      DeepSeekRankProcessDriver& driver) noexcept;
  Status fail(Status cause) noexcept;
  Status accept_ready(DeepSeekRankReadyReceipt receipt,
                      std::uint64_t challenge_identity);
  std::vector<DeepSeekRankProcessManifest> manifests_;
  std::vector<std::optional<DeepSeekRankProcessHandle>> handles_;
  std::vector<bool> ready_;
  std::vector<std::optional<DeepSeekRankExecReady>> exec_ready_;
  std::vector<std::uint64_t> challenges_;
  std::optional<DeepSeekRankSpawnAuthorization> spawn_authorization_;
  std::shared_ptr<const RuntimeEngineAdmission> capacity_authority_anchor_;
  Sha256Digest spawn_authorization_root_{};
  Sha256Digest capacity_plan_instance_root_{};
  Sha256Digest manifest_root_{};
  Sha256Digest spawn_resource_plan_root_{};
  DeepSeekRankProcessDriver* driver_ = nullptr;
  bool launched_ = false;
  bool failed_ = false;
};

}  // namespace pih
