#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/deepseek_rank_artifact_transfer_manifest.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankArtifactTransferAckAbi =
    "pih_deepseek_rank_artifact_transfer_ack_v1";
inline constexpr std::string_view kDeepSeekRankArtifactTransferAckFrameAbi =
    "pih_deepseek_rank_artifact_transfer_ack_frame_v1";
inline constexpr std::size_t kDeepSeekRankArtifactTransferAckBytes = 288;

// Acknowledges one atomically accepted descriptor batch. The two descriptor
// roots are expected to be recomputed by the worker after adopting and
// inspecting the received descriptors; a successful send alone cannot create
// this object.
struct DeepSeekRankArtifactTransferAckFields final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint32_t batch_index = 0;
  std::uint32_t batch_count = 0;
  std::uint32_t first_descriptor_ordinal = 0;
  std::uint32_t descriptor_count = 0;
  std::uint32_t cumulative_descriptor_count = 0;
  bool final_batch = false;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  Sha256Digest transfer_manifest_root{};
  Sha256Digest artifact_admission_binding_root{};
  Sha256Digest transfer_transaction_root{};
  Sha256Digest descriptor_batch_root{};
  Sha256Digest adopted_descriptor_set_root{};
};

class DeepSeekRankArtifactTransferAck final {
 public:
  static Result<DeepSeekRankArtifactTransferAck> Create(
      DeepSeekRankArtifactTransferAckFields fields);

  [[nodiscard]] const DeepSeekRankArtifactTransferAckFields& fields()
      const noexcept {
    return fields_;
  }
  [[nodiscard]] const Sha256Digest& ack_root() const noexcept {
    return ack_root_;
  }

 private:
  DeepSeekRankArtifactTransferAck(
      DeepSeekRankArtifactTransferAckFields fields,
      Sha256Digest ack_root) noexcept;

  DeepSeekRankArtifactTransferAckFields fields_;
  Sha256Digest ack_root_{};
};

std::array<std::byte, kDeepSeekRankArtifactTransferAckBytes>
encode_deepseek_rank_artifact_transfer_ack(
    const DeepSeekRankArtifactTransferAck& ack);
Result<DeepSeekRankArtifactTransferAck>
decode_deepseek_rank_artifact_transfer_ack(
    std::span<const std::byte> frame);

}  // namespace pih
