#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_transfer_ack.h"
#include "pih/model/deepseek_rank_artifact_transfer_plan.h"

namespace pih {

class DeepSeekRankArtifactMetadataTransferTransaction;

inline constexpr std::string_view
    kDeepSeekRankArtifactTransferTransactionAbi =
        "pih_deepseek_rank_artifact_transfer_transaction_v1";

// Shared controller/worker canonicalization. A worker constructs these values
// only from the metadata accompanying descriptors it has independently
// adopted and inspected.
Result<Sha256Digest>
compile_deepseek_rank_artifact_transfer_descriptor_root(
    const DeepSeekRankArtifactDescriptorExpectation& descriptor);
Result<Sha256Digest>
compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
    std::uint32_t rank, std::uint32_t batch_index,
    std::span<const DeepSeekRankArtifactDescriptorExpectation> descriptors);
Result<Sha256Digest>
compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
    std::uint32_t rank,
    std::span<const DeepSeekRankArtifactDescriptorExpectation> descriptors);

// Ephemeral view valid only for the duration of send_descriptor_batch(). A
// channel must atomically accept every descriptor in the view or return
// kUnavailable without accepting any of them.
struct DeepSeekRankArtifactDescriptorBatchView final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint32_t batch_index = 0;
  std::uint32_t batch_count = 0;
  std::uint32_t first_descriptor_ordinal = 0;
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
  std::span<const DeepSeekRankArtifactDescriptorExpectation> expectations;
  std::span<const DeepSeekWorkerShardDescriptor> descriptors;
};

class DeepSeekRankArtifactTransferOperations {
 public:
  virtual ~DeepSeekRankArtifactTransferOperations() = default;

  // kUnavailable means that no byte was accepted. Any other failure is fatal
  // for the complete worker generation.
  virtual Status send_manifest(
      std::uint32_t rank,
      std::span<const std::byte> manifest_frame) = 0;
  // kUnavailable means that neither payload nor ancillary descriptors were
  // accepted. A successful return means the exact batch is in flight once.
  virtual Status send_descriptor_batch(
      const DeepSeekRankArtifactDescriptorBatchView& batch) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  virtual Result<std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>>>
  poll_ack(std::uint32_t rank) = 0;
  virtual Status abort_generation(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      const Status& cause) = 0;
};

// Owns the only controller-side descriptor inventory for this transfer. It
// allows one in-flight batch per rank, accepts only the next exact ACK, and
// keeps all descriptors alive through success or poison until destruction.
class DeepSeekRankArtifactTransferTransaction final {
 public:
  static Result<DeepSeekRankArtifactTransferTransaction> Create(
      DeepSeekRankArtifactTransferPlan plan,
      DeepSeekRankArtifactTransferOperations& operations);

  DeepSeekRankArtifactTransferTransaction(
      const DeepSeekRankArtifactTransferTransaction&) = delete;
  DeepSeekRankArtifactTransferTransaction& operator=(
      const DeepSeekRankArtifactTransferTransaction&) = delete;
  DeepSeekRankArtifactTransferTransaction(
      DeepSeekRankArtifactTransferTransaction&&) noexcept = default;
  DeepSeekRankArtifactTransferTransaction& operator=(
      DeepSeekRankArtifactTransferTransaction&&) noexcept = default;

  Status advance();
  [[nodiscard]] bool complete() const noexcept { return complete_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return plan_.world_size();
  }
  [[nodiscard]] const Sha256Digest& transaction_root() const noexcept {
    return transaction_root_;
  }
  [[nodiscard]] bool retains_transfer_antecedents() const noexcept {
    return plan_.owns_all_antecedents();
  }
  [[nodiscard]] bool manifest_accepted(std::uint32_t rank) const;
  [[nodiscard]] std::uint32_t acknowledged_descriptor_count(
      std::uint32_t rank) const;

 private:
  friend class DeepSeekRankArtifactMetadataTransferTransaction;

  struct BatchState final {
    std::uint32_t first_descriptor_ordinal = 0;
    std::uint32_t descriptor_count = 0;
    Sha256Digest descriptor_batch_root{};
    Sha256Digest adopted_descriptor_set_root{};
  };
  struct RankState final {
    bool manifest_accepted = false;
    std::uint32_t next_batch_index = 0;
    std::optional<std::uint32_t> in_flight_batch_index;
    std::uint32_t acknowledged_descriptor_count = 0;
  };

  DeepSeekRankArtifactTransferTransaction(
      DeepSeekRankArtifactTransferPlan plan,
      DeepSeekRankArtifactTransferOperations& operations,
      std::vector<std::array<
          std::byte, kDeepSeekRankArtifactTransferManifestBytes>>
          manifest_frames,
      std::vector<std::vector<BatchState>> batches,
      Sha256Digest transaction_root) noexcept;

  Status fail(Status cause) noexcept;
  Status ensure_before_deadline();
  Status validate_ack(
      std::uint32_t rank,
      const DeepSeekRankArtifactTransferAck& ack) const;
  [[nodiscard]] DeepSeekRankArtifactDescriptorBatchView batch_view(
      std::uint32_t rank, std::uint32_t batch_index) const;

  DeepSeekRankArtifactTransferPlan plan_;
  DeepSeekRankArtifactTransferOperations* operations_ = nullptr;
  std::vector<std::array<
      std::byte, kDeepSeekRankArtifactTransferManifestBytes>>
      manifest_frames_;
  std::vector<std::vector<BatchState>> batches_;
  std::vector<RankState> ranks_;
  Sha256Digest transaction_root_{};
  bool complete_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
