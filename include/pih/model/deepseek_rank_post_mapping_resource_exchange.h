#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_mapping_owner.h"
#include "pih/model/deepseek_rank_post_exec_resource_collector.h"
#include "pih/model/deepseek_rank_post_mapping_resource_codec.h"
#include "pih/model/deepseek_rank_post_mapping_resource_report.h"

namespace pih {

inline constexpr std::string_view
    kDeepSeekRankPostMappingResourceExchangeAbi =
        "pih_deepseek_rank_post_mapping_resource_exchange_v1";

Result<DeepSeekRankPostMappingResourceAuthority>
compile_deepseek_rank_post_mapping_resource_authority(
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
    const DeepSeekRankPostExecResourceSeal& first_resource_seal,
    const DeepSeekRankArtifactMetadataTransferTransaction&
        metadata_transaction,
    std::uint32_t rank, std::uint64_t deadline_ns);

enum class DeepSeekRankPostMappingResourceReporterWaitEvent : std::uint8_t {
  kAuthorityReadable,
  kObservationWritable,
};

class DeepSeekRankPostMappingResourceReporterOperations {
 public:
  virtual ~DeepSeekRankPostMappingResourceReporterOperations() = default;
  virtual Result<std::optional<std::vector<std::byte>>> receive_authority(
      std::int32_t control_fd) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  virtual Status send_observation(
      std::int32_t control_fd, std::span<const std::byte> frame) = 0;
};

class DeepSeekRankPostMappingResourceReporter final {
 public:
  static Result<DeepSeekRankPostMappingResourceReporter> Create(
      DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
      std::int32_t control_fd,
      const DeepSeekRankArtifactMappingOwner& mapping_owner,
      StableDeepSeekRankPostExecResourceCollector& collector,
      DeepSeekRankPostMappingResourceReporterOperations& operations);

  Status advance();
  [[nodiscard]] bool reported() const noexcept { return reported_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::optional<
      DeepSeekRankPostMappingResourceReporterWaitEvent>
  wait_event() const noexcept;
  [[nodiscard]] std::optional<std::uint64_t> deadline_ns()
      const noexcept;
  [[nodiscard]] const DeepSeekRankPostMappingResourceReport* report()
      const noexcept {
    return reported_ && report_ ? &*report_ : nullptr;
  }

 private:
  DeepSeekRankPostMappingResourceReporter(
      DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
      std::int32_t control_fd,
      const DeepSeekRankArtifactMappingOwner& mapping_owner,
      StableDeepSeekRankPostExecResourceCollector& collector,
      DeepSeekRankPostMappingResourceReporterOperations& operations) noexcept;
  Status fail(Status cause) noexcept;

  DeepSeekRankProcessManifest manifest_;
  DeepSeekRankExecReady exec_ready_;
  std::int32_t control_fd_ = -1;
  const DeepSeekRankArtifactMappingOwner* mapping_owner_ = nullptr;
  StableDeepSeekRankPostExecResourceCollector* collector_ = nullptr;
  DeepSeekRankPostMappingResourceReporterOperations* operations_ = nullptr;
  std::optional<DeepSeekRankPostMappingResourceAuthority> authority_;
  std::optional<std::array<
      std::byte, kDeepSeekRankPostMappingResourceObservationFrameBytes>>
      observation_frame_;
  std::optional<DeepSeekRankPostMappingResourceReport> report_;
  bool reported_ = false;
  bool poisoned_ = false;
};

class DeepSeekRankPostMappingResourceChannel {
 public:
  virtual ~DeepSeekRankPostMappingResourceChannel() = default;
  // kUnavailable means no authority bytes were accepted.
  virtual Status send_authority(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) = 0;
  virtual Result<std::optional<std::vector<std::byte>>> poll_observation(
      const DeepSeekRankProcessHandle& handle) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
};

class DeepSeekRankPostMappingResourceCoordinator final {
 public:
  static Result<DeepSeekRankPostMappingResourceCoordinator> Create(
      DeepSeekRankProcessSupervisor& supervisor,
      std::span<const DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan spawn_plan,
      std::span<const DeepSeekRankPostExecResourcePlan> resource_plans,
      DeepSeekRankPostExecResourceSeal first_resource_seal,
      const DeepSeekRankArtifactMetadataTransferTransaction&
          metadata_transaction,
      std::uint64_t deadline_ns,
      DeepSeekRankPostMappingResourceChannel& channel);

  Status advance();
  [[nodiscard]] bool sealed() const noexcept { return seal_.has_value(); }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::size_t receipt_count() const noexcept;
  [[nodiscard]] std::uint64_t deadline_ns() const noexcept {
    return deadline_ns_;
  }
  [[nodiscard]] const DeepSeekRankPostMappingResourceSeal* seal()
      const noexcept {
    return seal_ ? &*seal_ : nullptr;
  }
  [[nodiscard]] const DeepSeekRankPostMappingResourceReport* report(
      std::uint32_t rank) const noexcept {
    return seal_ && rank < reports_.size() && reports_[rank]
               ? &*reports_[rank]
               : nullptr;
  }
  [[nodiscard]] const DeepSeekRankPostMappingResourceReceipt* receipt(
      std::uint32_t rank) const noexcept {
    return seal_ && rank < receipts_.size() && receipts_[rank]
               ? &*receipts_[rank]
               : nullptr;
  }

 private:
  DeepSeekRankPostMappingResourceCoordinator(
      DeepSeekRankProcessSupervisor& supervisor,
      std::vector<DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan spawn_plan,
      std::vector<DeepSeekRankPostExecResourcePlan> resource_plans,
      DeepSeekRankPostExecResourceSeal first_resource_seal,
      const DeepSeekRankArtifactMetadataTransferTransaction&
          metadata_transaction,
      std::vector<DeepSeekRankPostMappingResourceAuthority> authorities,
      std::vector<std::array<
          std::byte, kDeepSeekRankPostMappingResourceAuthorityFrameBytes>>
          authority_frames,
      std::uint64_t deadline_ns,
      DeepSeekRankPostMappingResourceChannel& channel) noexcept;
  Status fail(Status cause) noexcept;
  Status ensure_before_deadline();

  DeepSeekRankProcessSupervisor* supervisor_ = nullptr;
  std::vector<DeepSeekRankProcessManifest> manifests_;
  DeepSeekRankSpawnResourcePlan spawn_plan_;
  std::vector<DeepSeekRankPostExecResourcePlan> resource_plans_;
  DeepSeekRankPostExecResourceSeal first_resource_seal_;
  const DeepSeekRankArtifactMetadataTransferTransaction*
      metadata_transaction_ = nullptr;
  std::vector<DeepSeekRankPostMappingResourceAuthority> authorities_;
  std::vector<std::array<
      std::byte, kDeepSeekRankPostMappingResourceAuthorityFrameBytes>>
      authority_frames_;
  std::vector<bool> authorities_dispatched_;
  std::vector<std::optional<DeepSeekRankPostMappingResourceReport>>
      reports_;
  std::vector<std::optional<DeepSeekRankPostMappingResourceReceipt>>
      receipts_;
  std::uint64_t deadline_ns_ = 0;
  DeepSeekRankPostMappingResourceChannel* channel_ = nullptr;
  std::optional<DeepSeekRankPostMappingResourceSeal> seal_;
  bool poisoned_ = false;
};

}  // namespace pih
