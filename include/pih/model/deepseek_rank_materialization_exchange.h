#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_materialization_grant.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankMaterializationExchangeAbi =
    "pih_deepseek_rank_materialization_exchange_v1";

enum class DeepSeekRankMaterializationGrantReceiverWaitEvent
    : std::uint8_t {
  kGrantReadable,
  kAckWritable,
};

class DeepSeekRankMaterializationGrantReceiverOperations {
 public:
  virtual ~DeepSeekRankMaterializationGrantReceiverOperations() = default;
  virtual Result<std::optional<std::vector<std::byte>>> receive_grant(
      std::int32_t control_fd) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  // kUnavailable means no ACK bytes were accepted and the identical frame
  // may be retried.
  virtual Status send_ack(
      std::int32_t control_fd, std::span<const std::byte> frame) = 0;
};

// Worker-side delivery gate. A decoded grant remains private until its exact
// ACK is accepted before the grant deadline. Only then may the one-shot
// materialization admission be moved into the authorized builder.
class DeepSeekRankMaterializationGrantReceiver final {
 public:
  static Result<DeepSeekRankMaterializationGrantReceiver> Create(
      DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
      std::int32_t control_fd, std::uint64_t grant_wait_deadline_ns,
      const DeepSeekRankArtifactMappingOwner& mapping_owner,
      const DeepSeekRankPostMappingResourceReport& report,
      DeepSeekRankMaterializationGrantReceiverOperations& operations);

  DeepSeekRankMaterializationGrantReceiver(
      const DeepSeekRankMaterializationGrantReceiver&) = delete;
  DeepSeekRankMaterializationGrantReceiver& operator=(
      const DeepSeekRankMaterializationGrantReceiver&) = delete;
  DeepSeekRankMaterializationGrantReceiver(
      DeepSeekRankMaterializationGrantReceiver&&) noexcept = default;
  DeepSeekRankMaterializationGrantReceiver& operator=(
      DeepSeekRankMaterializationGrantReceiver&&) noexcept = default;

  Status advance();
  [[nodiscard]] bool complete() const noexcept { return complete_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] bool admission_available() const noexcept {
    return complete_ && admission_.has_value();
  }
  [[nodiscard]] std::optional<
      DeepSeekRankMaterializationGrantReceiverWaitEvent>
  wait_event() const noexcept;
  [[nodiscard]] std::uint64_t wait_deadline_ns() const noexcept;
  Result<DeepSeekRankMaterializationAdmission> take_admission();

 private:
  DeepSeekRankMaterializationGrantReceiver(
      DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
      std::int32_t control_fd, std::uint64_t grant_wait_deadline_ns,
      const DeepSeekRankArtifactMappingOwner& mapping_owner,
      const DeepSeekRankPostMappingResourceReport& report,
      DeepSeekRankMaterializationGrantReceiverOperations& operations)
      noexcept;

  Status fail(Status cause) noexcept;
  Status ensure_before_deadline();
  Status flush_ack();

  DeepSeekRankProcessManifest manifest_;
  DeepSeekRankExecReady exec_ready_;
  std::int32_t control_fd_ = -1;
  std::uint64_t grant_wait_deadline_ns_ = 0;
  const DeepSeekRankArtifactMappingOwner* mapping_owner_ = nullptr;
  const DeepSeekRankPostMappingResourceReport* report_ = nullptr;
  DeepSeekRankMaterializationGrantReceiverOperations* operations_ = nullptr;
  std::optional<DeepSeekRankMaterializationAdmission> admission_;
  std::optional<std::array<
      std::byte, kDeepSeekRankMaterializationGrantAckFrameBytes>>
      ack_frame_;
  bool complete_ = false;
  bool poisoned_ = false;
};

class DeepSeekRankMaterializationGrantChannel {
 public:
  virtual ~DeepSeekRankMaterializationGrantChannel() = default;
  // kUnavailable means no grant bytes were accepted and the identical frame
  // may be retried.
  virtual Status send_grant(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) = 0;
  virtual Result<std::optional<std::vector<std::byte>>> poll_ack(
      const DeepSeekRankProcessHandle& handle) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
};

// Controller-side all-rank delivery transaction. It can only be created from
// a sealed post-mapping coordinator, and it completes only after every exact
// rank ACK is received before one common deadline.
class DeepSeekRankMaterializationGrantCoordinator final {
 public:
  static Result<DeepSeekRankMaterializationGrantCoordinator> Create(
      const RuntimeEngineAdmission& admission,
      DeepSeekRankProcessSupervisor& supervisor,
      std::span<const DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankPostMappingResourceCoordinator& post_mapping,
      std::uint64_t deadline_ns,
      std::span<const DeepSeekRankMaterializationAllocationAuthority>
          allocation_authorities,
      DeepSeekRankMaterializationGrantChannel& channel);

  Status advance();
  [[nodiscard]] bool complete() const noexcept { return complete_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::size_t acknowledgment_count() const noexcept;
  [[nodiscard]] std::uint64_t deadline_ns() const noexcept {
    return deadline_ns_;
  }
  [[nodiscard]] const DeepSeekRankMaterializationGrantFields* grant(
      std::uint32_t rank) const noexcept;
  [[nodiscard]] const DeepSeekRankMaterializationGrantAckFields*
  acknowledgment(std::uint32_t rank) const noexcept;

 private:
  friend class DeepSeekRankMaterializationWarmupCoordinator;
  DeepSeekRankMaterializationGrantCoordinator(
      const RuntimeEngineAdmission& admission,
      DeepSeekRankProcessSupervisor& supervisor,
      std::vector<DeepSeekRankProcessManifest> manifests,
      const DeepSeekRankPostMappingResourceCoordinator& post_mapping,
      std::vector<DeepSeekRankMaterializationGrantFields> grants,
      std::vector<std::array<
          std::byte, kDeepSeekRankMaterializationGrantFrameBytes>> frames,
      std::uint64_t deadline_ns,
      DeepSeekRankMaterializationGrantChannel& channel) noexcept;

  Status fail(Status cause) noexcept;
  Status ensure_before_deadline();
  Status validate_ack(
      std::uint32_t rank,
      const DeepSeekRankMaterializationGrantAckFields& ack) const;

  const RuntimeEngineAdmission* admission_ = nullptr;
  DeepSeekRankProcessSupervisor* supervisor_ = nullptr;
  std::vector<DeepSeekRankProcessManifest> manifests_;
  const DeepSeekRankPostMappingResourceCoordinator* post_mapping_ = nullptr;
  std::vector<DeepSeekRankMaterializationGrantFields> grants_;
  std::vector<std::array<
      std::byte, kDeepSeekRankMaterializationGrantFrameBytes>> frames_;
  std::vector<bool> grants_dispatched_;
  std::vector<std::optional<DeepSeekRankMaterializationGrantAckFields>>
      acknowledgments_;
  std::uint64_t deadline_ns_ = 0;
  DeepSeekRankMaterializationGrantChannel* channel_ = nullptr;
  bool complete_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
