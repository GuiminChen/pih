#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include "pih/model/deepseek_rank_post_mapping_resource_exchange.h"
#include "pih/model/deepseek_rank_materialization_allocation_census.h"
#include "pih/model/runtime_profile_payload.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankMaterializationGrantAbi =
    "pih_deepseek_rank_materialization_grant_v2";
inline constexpr std::string_view kDeepSeekRankMaterializationGrantFrameAbi =
    "pih_deepseek_rank_materialization_grant_frame_v2";
inline constexpr std::size_t kDeepSeekRankMaterializationGrantFrameBytes =
    580;
inline constexpr std::string_view
    kDeepSeekRankMaterializationGrantAckAbi =
        "pih_deepseek_rank_materialization_grant_ack_v1";
inline constexpr std::string_view
    kDeepSeekRankMaterializationGrantAckFrameAbi =
        "pih_deepseek_rank_materialization_grant_ack_frame_v1";
inline constexpr std::size_t
    kDeepSeekRankMaterializationGrantAckFrameBytes = 232;

struct DeepSeekRankMaterializationGrantFields final {
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
  std::int32_t device_ordinal = -1;
  RuntimeProfileGpuFamily gpu_family =
      RuntimeProfileGpuFamily::kRtx4090D24GiB;
  RuntimeProfileResidency residency = RuntimeProfileResidency::kFullResident;
  bool production_eligible = false;
  bool dspark_enabled = false;
  std::uint64_t deadline_ns = 0;
  Sha256Digest profile_envelope_root{};
  Sha256Digest device_observation_root{};
  Sha256Digest kernel_closure_root{};
  Sha256Digest graph_snapshot_root{};
  Sha256Digest capacity_plan_instance_root{};
  Sha256Digest first_resource_seal_root{};
  Sha256Digest metadata_transaction_root{};
  Sha256Digest mapping_owner_root{};
  Sha256Digest report_root{};
  Sha256Digest post_mapping_receipt_root{};
  Sha256Digest post_mapping_seal_root{};
  Sha256Digest mapping_owner_set_root{};
  Sha256Digest receipt_set_root{};
  Sha256Digest allocation_authority_root{};
};

struct DeepSeekRankMaterializationGrantAckFields final {
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
  Sha256Digest grant_root{};
  Sha256Digest report_root{};
  Sha256Digest mapping_owner_root{};
  Sha256Digest post_mapping_seal_root{};
};

Result<Sha256Digest> compile_deepseek_rank_materialization_grant_root(
    const DeepSeekRankMaterializationGrantFields& fields);
Result<std::array<std::byte,
                  kDeepSeekRankMaterializationGrantFrameBytes>>
encode_deepseek_rank_materialization_grant(
    const DeepSeekRankMaterializationGrantFields& fields);
Result<DeepSeekRankMaterializationGrantFields>
decode_deepseek_rank_materialization_grant(
    std::span<const std::byte> frame);

Result<Sha256Digest> compile_deepseek_rank_materialization_grant_ack_root(
    const DeepSeekRankMaterializationGrantAckFields& fields);
Result<std::array<
    std::byte, kDeepSeekRankMaterializationGrantAckFrameBytes>>
encode_deepseek_rank_materialization_grant_ack(
    const DeepSeekRankMaterializationGrantAckFields& fields);
Result<DeepSeekRankMaterializationGrantAckFields>
decode_deepseek_rank_materialization_grant_ack(
    std::span<const std::byte> frame);

Result<DeepSeekRankMaterializationGrantFields>
compile_deepseek_rank_materialization_grant(
    const RuntimeEngineAdmission& admission,
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankPostMappingResourceCoordinator& coordinator,
    std::uint32_t rank, std::uint64_t deadline_ns,
    const DeepSeekRankMaterializationAllocationAuthority&
        allocation_authority);

// Worker-local unforgeable admission. It proves a controller grant was joined
// to the exact local exec identity, stable mapping owner and exact report sent
// by this worker. It still authorizes only materialization startup, not ready.
class DeepSeekRankMaterializationAdmission final {
 public:
  static Result<DeepSeekRankMaterializationAdmission> Accept(
      const DeepSeekRankProcessManifest& manifest,
      const DeepSeekRankExecReady& exec_ready,
      const DeepSeekRankArtifactMappingOwner& mapping_owner,
      const DeepSeekRankPostMappingResourceReport& report,
      DeepSeekRankMaterializationGrantFields grant);

  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return grant_.engine_epoch;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return grant_.worker_generation;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return grant_.world_size;
  }
  [[nodiscard]] std::uint32_t rank() const noexcept { return grant_.rank; }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return grant_.device_ordinal;
  }
  [[nodiscard]] RuntimeProfileGpuFamily gpu_family() const noexcept {
    return grant_.gpu_family;
  }
  [[nodiscard]] RuntimeProfileResidency residency() const noexcept {
    return grant_.residency;
  }
  [[nodiscard]] bool production_eligible() const noexcept {
    return grant_.production_eligible;
  }
  [[nodiscard]] bool dspark_enabled() const noexcept {
    return grant_.dspark_enabled;
  }
  [[nodiscard]] std::uint64_t deadline_ns() const noexcept {
    return grant_.deadline_ns;
  }
  [[nodiscard]] const Sha256Digest& mapping_owner_root() const noexcept {
    return grant_.mapping_owner_root;
  }
  [[nodiscard]] const Sha256Digest& post_mapping_seal_root() const noexcept {
    return grant_.post_mapping_seal_root;
  }
  [[nodiscard]] const Sha256Digest& grant_root() const noexcept {
    return grant_root_;
  }
  [[nodiscard]] const DeepSeekRankMaterializationGrantFields& fields()
      const noexcept {
    return grant_;
  }

 private:
  DeepSeekRankMaterializationAdmission(
      DeepSeekRankMaterializationGrantFields grant,
      Sha256Digest grant_root) noexcept
      : grant_(std::move(grant)), grant_root_(grant_root) {}

  DeepSeekRankMaterializationGrantFields grant_;
  Sha256Digest grant_root_{};
};

struct DeepSeekRankAuthorizedMaterializationInputs final {
  DeepSeekRankAuthorizedMaterializationInputs(
      DeepSeekRankMaterializationAdmission admitted,
      std::unique_ptr<DeepSeekRankArtifactMappingOwner> owner) noexcept
      : admission(std::move(admitted)), mapping_owner(std::move(owner)) {}

  DeepSeekRankAuthorizedMaterializationInputs(
      const DeepSeekRankAuthorizedMaterializationInputs&) = delete;
  DeepSeekRankAuthorizedMaterializationInputs& operator=(
      const DeepSeekRankAuthorizedMaterializationInputs&) = delete;
  DeepSeekRankAuthorizedMaterializationInputs(
      DeepSeekRankAuthorizedMaterializationInputs&&) noexcept = default;
  DeepSeekRankAuthorizedMaterializationInputs& operator=(
      DeepSeekRankAuthorizedMaterializationInputs&&) noexcept = default;

  DeepSeekRankMaterializationAdmission admission;
  std::unique_ptr<DeepSeekRankArtifactMappingOwner> mapping_owner;
};

Result<DeepSeekRankMaterializationGrantAckFields>
compile_deepseek_rank_materialization_grant_ack(
    const DeepSeekRankMaterializationAdmission& admission);

}  // namespace pih
