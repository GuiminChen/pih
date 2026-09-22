#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_descriptor_batch.h"
#include "pih/model/deepseek_rank_scm_rights_ledger.h"
#include "pih/model/deepseek_rank_worker_handshake.h"

namespace pih {

class DeepSeekRankArtifactAdoptionReceipt;

inline constexpr std::string_view kDeepSeekRankArtifactTransferReceiverAbi =
    "pih_deepseek_rank_artifact_transfer_receiver_v1";

// Linux implementations must populate this object only after exact ancillary
// count, peer, regular-file, read-only, CLOEXEC, fstat and (when required)
// fs-verity validation. Ownership of every descriptor is transferred here.
struct DeepSeekRankArtifactReceivedBatch final {
  DeepSeekRankArtifactDescriptorBatch metadata;
  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
};

class DeepSeekRankArtifactTransferReceiverOperations {
 public:
  virtual ~DeepSeekRankArtifactTransferReceiverOperations() = default;
  virtual Result<std::optional<std::vector<std::byte>>> receive_manifest(
      std::int32_t control_fd) = 0;
  virtual Result<std::optional<DeepSeekRankArtifactReceivedBatch>>
  receive_descriptor_batch(std::int32_t control_fd) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  virtual Status send_ack(
      std::int32_t control_fd, std::span<const std::byte> frame) = 0;
};

enum class DeepSeekRankArtifactTransferReceiverWaitEvent {
  kPacketReadable,
  kAckWritable,
};

// Worker-side owner. It never exposes adopted descriptors: a later sealed
// adoption receipt must consume the complete receiver before mapping.
class DeepSeekRankArtifactTransferReceiver final {
 public:
  static Result<DeepSeekRankArtifactTransferReceiver> Create(
      DeepSeekRankProcessManifest local_manifest,
      DeepSeekRankExecReady exec_ready, std::int32_t control_fd,
      std::uint64_t manifest_wait_deadline_ns,
      DeepSeekRankScmRightsInFlightLedger& scm_rights_ledger,
      DeepSeekRankArtifactTransferReceiverOperations& operations);

  DeepSeekRankArtifactTransferReceiver(
      const DeepSeekRankArtifactTransferReceiver&) = delete;
  DeepSeekRankArtifactTransferReceiver& operator=(
      const DeepSeekRankArtifactTransferReceiver&) = delete;
  DeepSeekRankArtifactTransferReceiver(
      DeepSeekRankArtifactTransferReceiver&&) noexcept = default;
  DeepSeekRankArtifactTransferReceiver& operator=(
      DeepSeekRankArtifactTransferReceiver&&) noexcept = default;

  Status advance();
  [[nodiscard]] bool manifest_received() const noexcept {
    return transfer_manifest_.has_value();
  }
  [[nodiscard]] bool complete() const noexcept { return complete_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::size_t adopted_descriptor_count() const noexcept {
    return adopted_descriptors_.size();
  }
  [[nodiscard]] const Sha256Digest* transaction_root() const noexcept {
    return transaction_root_ ? &*transaction_root_ : nullptr;
  }
  [[nodiscard]] std::optional<DeepSeekRankArtifactTransferReceiverWaitEvent>
  wait_event() const noexcept {
    if (complete_ || poisoned_) return std::nullopt;
    return pending_ack_
               ? DeepSeekRankArtifactTransferReceiverWaitEvent::kAckWritable
               : DeepSeekRankArtifactTransferReceiverWaitEvent::
                     kPacketReadable;
  }
  [[nodiscard]] std::uint64_t wait_deadline_ns() const noexcept {
    return transfer_manifest_ ? transfer_manifest_->fields().deadline_ns
                              : manifest_wait_deadline_ns_;
  }

 private:
  friend class DeepSeekRankArtifactAdoptionReceipt;
  DeepSeekRankArtifactTransferReceiver(
      DeepSeekRankProcessManifest local_manifest,
      DeepSeekRankExecReady exec_ready, std::int32_t control_fd,
      std::uint64_t manifest_wait_deadline_ns,
      DeepSeekRankScmRightsInFlightLedger& scm_rights_ledger,
      DeepSeekRankArtifactTransferReceiverOperations& operations) noexcept;

  Status fail(Status cause) noexcept;
  Status ensure_before_deadline();
  Status validate_manifest(
      const DeepSeekRankArtifactTransferManifest& manifest) const;
  Status accept_batch(DeepSeekRankArtifactReceivedBatch batch);
  Status flush_ack();

  DeepSeekRankProcessManifest local_manifest_;
  DeepSeekRankExecReady exec_ready_;
  std::int32_t control_fd_ = -1;
  std::uint64_t manifest_wait_deadline_ns_ = 0;
  DeepSeekRankScmRightsInFlightLedger* scm_rights_ledger_ = nullptr;
  DeepSeekRankArtifactTransferReceiverOperations* operations_ = nullptr;
  std::optional<DeepSeekRankArtifactTransferManifest> transfer_manifest_;
  std::optional<Sha256Digest> transaction_root_;
  std::vector<DeepSeekRankArtifactDescriptorExpectation>
      adopted_expectations_;
  std::vector<DeepSeekWorkerShardDescriptor> adopted_descriptors_;
  std::uint32_t next_batch_index_ = 0;
  std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>> pending_ack_;
  std::optional<DeepSeekRankScmRightsInFlightLedger::Lease>
      in_flight_lease_;
  bool pending_ack_is_final_ = false;
  bool complete_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
