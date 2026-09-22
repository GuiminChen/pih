#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_artifact_prefault_layout.h"
#include "pih/model/deepseek_rank_artifact_metadata_chunk.h"
#include "pih/model/deepseek_rank_artifact_transfer_transaction.h"

namespace pih {

inline constexpr std::string_view
    kDeepSeekRankArtifactMetadataTransferTransactionAbi =
        "pih_deepseek_rank_artifact_metadata_transfer_transaction_v1";

class DeepSeekRankArtifactMetadataTransferOperations {
 public:
  virtual ~DeepSeekRankArtifactMetadataTransferOperations() = default;

  // kUnavailable means the frame was not accepted at all and the identical
  // frame may be retried. Any other failure poisons the full generation.
  virtual Status send_chunk(
      std::uint32_t rank, std::span<const std::byte> frame) = 0;
  virtual Result<std::optional<std::array<
      std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>>>
  poll_ack(std::uint32_t rank) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  virtual Status abort_generation(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      const Status& cause) = 0;
};

// Controller-side continuation of a completed descriptor transaction. It
// retains that transaction and all descriptor/authority owners while sending
// bounded rank-local semantic blobs with one in-flight chunk per rank.
class DeepSeekRankArtifactMetadataTransferTransaction final {
 public:
  static Result<DeepSeekRankArtifactMetadataTransferTransaction> Create(
      DeepSeekRankArtifactTransferTransaction descriptor_transaction,
      DeepSeekRankArtifactMetadataTransferOperations& operations);

  DeepSeekRankArtifactMetadataTransferTransaction(
      const DeepSeekRankArtifactMetadataTransferTransaction&) = delete;
  DeepSeekRankArtifactMetadataTransferTransaction& operator=(
      const DeepSeekRankArtifactMetadataTransferTransaction&) = delete;
  DeepSeekRankArtifactMetadataTransferTransaction(
      DeepSeekRankArtifactMetadataTransferTransaction&&) noexcept = default;
  DeepSeekRankArtifactMetadataTransferTransaction& operator=(
      DeepSeekRankArtifactMetadataTransferTransaction&&) noexcept = default;

  Status advance();
  [[nodiscard]] bool complete() const noexcept { return complete_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return descriptor_transaction_.world_size();
  }
  [[nodiscard]] bool retains_descriptor_transaction() const noexcept {
    return descriptor_transaction_.complete() &&
           descriptor_transaction_.retains_transfer_antecedents();
  }
  [[nodiscard]] const Sha256Digest& transaction_root() const noexcept {
    return transaction_root_;
  }
  [[nodiscard]] const Sha256Digest& descriptor_transaction_root()
      const noexcept {
    return descriptor_transaction_.transaction_root();
  }
  [[nodiscard]] const Sha256Digest& metadata_root(
      std::uint32_t rank) const;
  [[nodiscard]] const Sha256Digest& blob_sha256(
      std::uint32_t rank) const;
  [[nodiscard]] std::uint64_t blob_bytes(std::uint32_t rank) const;
  [[nodiscard]] std::uint32_t chunk_count(std::uint32_t rank) const;
  [[nodiscard]] std::uint64_t acknowledged_bytes(
      std::uint32_t rank) const;
  [[nodiscard]] const Sha256Digest& expected_mapping_owner_root(
      std::uint32_t rank) const;
  [[nodiscard]] std::uint64_t expected_mapped_interval_bytes(
      std::uint32_t rank) const;
  [[nodiscard]] const DeepSeekRankArtifactPrefaultLayout&
  expected_prefault_layout(std::uint32_t rank) const;
  [[nodiscard]] const DeepSeekNodeArtifactPrefaultLayout&
  expected_node_prefault_layout() const noexcept {
    return node_prefault_layout_;
  }
  [[nodiscard]] ArtifactImmutabilityMode expected_immutability_mode(
      std::uint32_t rank) const;
  [[nodiscard]] bool expected_source_catalog_production_eligible(
      std::uint32_t rank) const;
  [[nodiscard]] bool dspark_enabled() const noexcept {
    return descriptor_transaction_.plan_.dspark_enabled();
  }
  [[nodiscard]] std::uint64_t deadline_ns() const noexcept {
    return descriptor_transaction_.plan_.rank_manifest(0).fields().deadline_ns;
  }

 private:
  struct RankPayload final {
    std::vector<std::byte> blob;
    Sha256Digest metadata_root{};
    Sha256Digest blob_sha256{};
    Sha256Digest expected_mapping_owner_root{};
    DeepSeekRankArtifactPrefaultLayout expected_prefault_layout{};
    std::uint64_t expected_mapped_interval_bytes = 0;
    ArtifactImmutabilityMode expected_immutability_mode =
        ArtifactImmutabilityMode::kUncalibrated;
    bool expected_source_catalog_production_eligible = false;
    std::uint32_t chunk_count = 0;
  };
  struct RankState final {
    std::uint32_t next_chunk_index = 0;
    std::optional<DeepSeekRankArtifactMetadataChunk> in_flight_chunk;
    std::vector<std::byte> in_flight_frame;
    bool frame_accepted = false;
    std::uint64_t acknowledged_bytes = 0;
  };

  DeepSeekRankArtifactMetadataTransferTransaction(
      DeepSeekRankArtifactTransferTransaction descriptor_transaction,
      DeepSeekRankArtifactMetadataTransferOperations& operations,
      std::vector<RankPayload> payloads,
      DeepSeekNodeArtifactPrefaultLayout node_prefault_layout,
      Sha256Digest transaction_root) noexcept;

  Status fail(Status cause) noexcept;
  Status ensure_before_deadline();
  Result<DeepSeekRankArtifactMetadataChunk> make_chunk(
      std::uint32_t rank, std::uint32_t chunk_index) const;
  Status validate_ack(
      std::uint32_t rank,
      const DeepSeekRankArtifactMetadataChunkAck& ack) const;

  DeepSeekRankArtifactTransferTransaction descriptor_transaction_;
  DeepSeekRankArtifactMetadataTransferOperations* operations_ = nullptr;
  std::vector<RankPayload> payloads_;
  std::vector<RankState> ranks_;
  DeepSeekNodeArtifactPrefaultLayout node_prefault_layout_{};
  Sha256Digest transaction_root_{};
  bool complete_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
