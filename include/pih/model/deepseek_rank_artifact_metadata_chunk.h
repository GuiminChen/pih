#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_metadata_blob.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankArtifactMetadataChunkAbi =
    "pih_deepseek_rank_artifact_metadata_chunk_v1";
inline constexpr std::string_view kDeepSeekRankArtifactMetadataChunkFrameAbi =
    "pih_deepseek_rank_artifact_metadata_chunk_frame_v1";
inline constexpr std::string_view kDeepSeekRankArtifactMetadataChunkAckAbi =
    "pih_deepseek_rank_artifact_metadata_chunk_ack_v1";
inline constexpr std::string_view
    kDeepSeekRankArtifactMetadataChunkAckFrameAbi =
        "pih_deepseek_rank_artifact_metadata_chunk_ack_frame_v1";
inline constexpr std::uint32_t
    kDeepSeekRankArtifactMetadataChunkPayloadMaximumBytes = 60U * 1024U;
inline constexpr std::size_t
    kDeepSeekRankArtifactMetadataChunkFrameMaximumBytes = 61'764;
inline constexpr std::size_t
    kDeepSeekRankArtifactMetadataChunkAckFrameBytes = 320;

struct DeepSeekRankArtifactMetadataChunkFields final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint32_t chunk_index = 0;
  std::uint32_t chunk_count = 0;
  std::uint64_t payload_offset = 0;
  std::uint32_t payload_bytes = 0;
  std::uint64_t total_blob_bytes = 0;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  Sha256Digest transfer_manifest_root{};
  Sha256Digest descriptor_transfer_transaction_root{};
  Sha256Digest metadata_root{};
  Sha256Digest metadata_transaction_root{};
  Sha256Digest blob_sha256{};
  Sha256Digest chunk_sha256{};
};

class DeepSeekRankArtifactMetadataChunk final {
 public:
  static Result<DeepSeekRankArtifactMetadataChunk> Create(
      DeepSeekRankArtifactMetadataChunkFields fields,
      std::vector<std::byte> payload);

  [[nodiscard]] const DeepSeekRankArtifactMetadataChunkFields& fields()
      const noexcept {
    return fields_;
  }
  [[nodiscard]] std::span<const std::byte> payload() const noexcept {
    return payload_;
  }
  [[nodiscard]] const Sha256Digest& frame_root() const noexcept {
    return frame_root_;
  }

 private:
  DeepSeekRankArtifactMetadataChunk(
      DeepSeekRankArtifactMetadataChunkFields fields,
      std::vector<std::byte> payload, Sha256Digest frame_root) noexcept;

  DeepSeekRankArtifactMetadataChunkFields fields_;
  std::vector<std::byte> payload_;
  Sha256Digest frame_root_{};
};

struct DeepSeekRankArtifactMetadataChunkAckFields final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint32_t chunk_index = 0;
  std::uint32_t chunk_count = 0;
  std::uint64_t cumulative_payload_bytes = 0;
  std::uint64_t total_blob_bytes = 0;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  Sha256Digest transfer_manifest_root{};
  Sha256Digest descriptor_transfer_transaction_root{};
  Sha256Digest metadata_root{};
  Sha256Digest metadata_transaction_root{};
  Sha256Digest blob_sha256{};
  Sha256Digest chunk_sha256{};
};

class DeepSeekRankArtifactMetadataChunkAck final {
 public:
  static Result<DeepSeekRankArtifactMetadataChunkAck> Create(
      DeepSeekRankArtifactMetadataChunkAckFields fields);
  [[nodiscard]] const DeepSeekRankArtifactMetadataChunkAckFields& fields()
      const noexcept {
    return fields_;
  }
  [[nodiscard]] const Sha256Digest& ack_root() const noexcept {
    return ack_root_;
  }

 private:
  DeepSeekRankArtifactMetadataChunkAck(
      DeepSeekRankArtifactMetadataChunkAckFields fields,
      Sha256Digest ack_root) noexcept;
  DeepSeekRankArtifactMetadataChunkAckFields fields_;
  Sha256Digest ack_root_{};
};

Result<std::vector<std::byte>> encode_deepseek_rank_artifact_metadata_chunk(
    const DeepSeekRankArtifactMetadataChunk& chunk);
Result<DeepSeekRankArtifactMetadataChunk>
decode_deepseek_rank_artifact_metadata_chunk(
    std::span<const std::byte> frame);
std::array<std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>
encode_deepseek_rank_artifact_metadata_chunk_ack(
    const DeepSeekRankArtifactMetadataChunkAck& ack);
Result<DeepSeekRankArtifactMetadataChunkAck>
decode_deepseek_rank_artifact_metadata_chunk_ack(
    std::span<const std::byte> frame);

}  // namespace pih
