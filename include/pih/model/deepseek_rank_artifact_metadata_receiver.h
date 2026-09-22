#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_adoption_receipt.h"
#include "pih/model/deepseek_rank_artifact_metadata_chunk.h"

namespace pih {

class DeepSeekRankArtifactMetadataReceipt;

inline constexpr std::string_view kDeepSeekRankArtifactMetadataReceiverAbi =
    "pih_deepseek_rank_artifact_metadata_receiver_v1";

class DeepSeekRankArtifactMetadataReceiverOperations {
 public:
  virtual ~DeepSeekRankArtifactMetadataReceiverOperations() = default;
  virtual Result<std::optional<std::vector<std::byte>>> receive_chunk(
      std::int32_t control_fd) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  // kUnavailable means no ACK byte was accepted; the exact frame is retained.
  virtual Status send_ack(
      std::int32_t control_fd, std::span<const std::byte> frame) = 0;
};

enum class DeepSeekRankArtifactMetadataReceiverWaitEvent {
  kPacketReadable,
  kAckWritable,
};

// Worker-side bounded reassembly owner. It consumes the descriptor adoption
// receipt and withholds the final ACK until the complete blob hash, semantic
// decode, transfer roots and independently supplied D-Spark mode all match.
class DeepSeekRankArtifactMetadataReceiver final {
 public:
  static Result<DeepSeekRankArtifactMetadataReceiver> Create(
      DeepSeekRankArtifactAdoptionReceipt adoption_receipt,
      bool expected_dspark_enabled,
      std::uint64_t maximum_reassembly_bytes,
      DeepSeekRankArtifactMetadataReceiverOperations& operations);

  DeepSeekRankArtifactMetadataReceiver(
      const DeepSeekRankArtifactMetadataReceiver&) = delete;
  DeepSeekRankArtifactMetadataReceiver& operator=(
      const DeepSeekRankArtifactMetadataReceiver&) = delete;
  DeepSeekRankArtifactMetadataReceiver(
      DeepSeekRankArtifactMetadataReceiver&&) noexcept = default;
  DeepSeekRankArtifactMetadataReceiver& operator=(
      DeepSeekRankArtifactMetadataReceiver&&) noexcept = default;

  Status advance();
  [[nodiscard]] bool complete() const noexcept { return complete_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] bool retains_descriptor_owners() const noexcept {
    return adoption_receipt_.retains_descriptor_owners();
  }
  [[nodiscard]] std::uint64_t reassembled_bytes() const noexcept {
    return reassembled_.size();
  }
  [[nodiscard]] std::uint32_t acknowledged_chunk_count() const noexcept {
    return next_chunk_index_;
  }
  [[nodiscard]] const Sha256Digest* metadata_transaction_root()
      const noexcept {
    return metadata_transaction_root_
               ? &*metadata_transaction_root_
               : nullptr;
  }
  [[nodiscard]] std::optional<DeepSeekRankArtifactMetadataReceiverWaitEvent>
  wait_event() const noexcept {
    if (complete_ || poisoned_) return std::nullopt;
    return pending_ack_
               ? DeepSeekRankArtifactMetadataReceiverWaitEvent::kAckWritable
               : DeepSeekRankArtifactMetadataReceiverWaitEvent::
                     kPacketReadable;
  }
  [[nodiscard]] std::uint64_t wait_deadline_ns() const noexcept {
    return adoption_receipt_.transfer_manifest().fields().deadline_ns;
  }

 private:
  friend class DeepSeekRankArtifactMetadataReceipt;

  DeepSeekRankArtifactMetadataReceiver(
      DeepSeekRankArtifactAdoptionReceipt adoption_receipt,
      bool expected_dspark_enabled,
      std::uint64_t maximum_reassembly_bytes,
      DeepSeekRankArtifactMetadataReceiverOperations& operations) noexcept;

  Status fail(Status cause) noexcept;
  Status ensure_before_deadline();
  Status accept_chunk(DeepSeekRankArtifactMetadataChunk chunk);
  Status validate_complete_blob(
      const DeepSeekRankArtifactMetadataChunkFields& final_fields);
  Status flush_ack();

  DeepSeekRankArtifactAdoptionReceipt adoption_receipt_;
  bool expected_dspark_enabled_ = false;
  std::uint64_t maximum_reassembly_bytes_ = 0;
  DeepSeekRankArtifactMetadataReceiverOperations* operations_ = nullptr;
  std::optional<Sha256Digest> metadata_transaction_root_;
  std::optional<Sha256Digest> metadata_root_;
  std::optional<Sha256Digest> blob_sha256_;
  std::optional<std::uint64_t> total_blob_bytes_;
  std::optional<std::uint32_t> chunk_count_;
  std::vector<std::byte> reassembled_;
  std::optional<DeepSeekRankArtifactMetadataBlob> decoded_blob_;
  std::uint32_t next_chunk_index_ = 0;
  std::optional<std::array<
      std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>>
      pending_ack_;
  bool pending_ack_is_final_ = false;
  bool complete_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
