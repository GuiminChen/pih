#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_artifact_metadata_transfer_transaction.h"
#include "pih/model/deepseek_rank_materialization_completion.h"
#include "pih/model/deepseek_rank_materialization_exchange.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankMaterializationWarmupAbi =
    "pih_deepseek_rank_materialization_warmup_v1";

Result<Sha256Digest>
compile_deepseek_rank_materialization_warm_grant_set_root(
    std::span<const DeepSeekRankMaterializationGrantFields> grants);

// Controller-only, all-rank proof. Node page bytes are de-duplicated using the
// authoritative metadata projection; process-local selected bytes remain a
// separate sum and therefore cannot masquerade as physical host consumption.
class DeepSeekRankMaterializationWarmSeal final {
 public:
  static Result<DeepSeekRankMaterializationWarmSeal> Compile(
      std::span<const DeepSeekRankMaterializationGrantFields> grants,
      std::span<const DeepSeekRankArtifactPrefaultLayout> rank_layouts,
      const DeepSeekNodeArtifactPrefaultLayout& node_layout,
      const Sha256Digest& metadata_transaction_root,
      std::span<const DeepSeekRankMaterializationCompletionFields>
          completions);

  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::uint64_t completion_monotonic_ns() const noexcept {
    return completion_monotonic_ns_;
  }
  [[nodiscard]] std::uint64_t summed_rank_selected_page_bytes()
      const noexcept {
    return summed_rank_selected_page_bytes_;
  }
  [[nodiscard]] std::uint64_t node_selected_page_union_bytes()
      const noexcept {
    return node_selected_page_union_bytes_;
  }
  [[nodiscard]] std::uint64_t node_duplicate_selected_page_bytes()
      const noexcept {
    return node_duplicate_selected_page_bytes_;
  }
  [[nodiscard]] std::uint64_t fixed_weight_backing_bytes() const noexcept {
    return fixed_weight_backing_bytes_;
  }
  [[nodiscard]] std::uint64_t pinned_staging_bytes() const noexcept {
    return pinned_staging_bytes_;
  }
  [[nodiscard]] bool production_eligible() const noexcept {
    return production_eligible_;
  }
  [[nodiscard]] bool dspark_enabled() const noexcept {
    return dspark_enabled_;
  }
  [[nodiscard]] const Sha256Digest& node_prefault_layout_root()
      const noexcept {
    return node_prefault_layout_root_;
  }
  [[nodiscard]] const Sha256Digest& completion_set_root() const noexcept {
    return completion_set_root_;
  }
  [[nodiscard]] const Sha256Digest& grant_set_root() const noexcept {
    return grant_set_root_;
  }
  [[nodiscard]] const Sha256Digest& seal_root() const noexcept {
    return seal_root_;
  }

 private:
  DeepSeekRankMaterializationWarmSeal(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, std::uint64_t completion_monotonic_ns,
      std::uint64_t summed_rank_selected_page_bytes,
      std::uint64_t node_selected_page_union_bytes,
      std::uint64_t node_duplicate_selected_page_bytes,
      std::uint64_t fixed_weight_backing_bytes,
      std::uint64_t pinned_staging_bytes, bool production_eligible,
      bool dspark_enabled, Sha256Digest node_prefault_layout_root,
      Sha256Digest grant_set_root, Sha256Digest completion_set_root,
      Sha256Digest seal_root) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint64_t completion_monotonic_ns_ = 0;
  std::uint64_t summed_rank_selected_page_bytes_ = 0;
  std::uint64_t node_selected_page_union_bytes_ = 0;
  std::uint64_t node_duplicate_selected_page_bytes_ = 0;
  std::uint64_t fixed_weight_backing_bytes_ = 0;
  std::uint64_t pinned_staging_bytes_ = 0;
  bool production_eligible_ = false;
  bool dspark_enabled_ = false;
  Sha256Digest node_prefault_layout_root_{};
  Sha256Digest grant_set_root_{};
  Sha256Digest completion_set_root_{};
  Sha256Digest seal_root_{};
};

class DeepSeekRankMaterializationWarmupOperations {
 public:
  virtual ~DeepSeekRankMaterializationWarmupOperations() = default;
  virtual Result<std::optional<std::vector<std::byte>>> poll_completion(
      const DeepSeekRankProcessHandle& handle) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  virtual Status abort_generation(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      const Status& cause) = 0;
};

// Consumes completion frames only after every materialization grant was ACKed.
// A malformed/late/missing-process observation poisons the generation through
// the existing supervisor and invokes the platform rollback hook exactly once.
class DeepSeekRankMaterializationWarmupCoordinator final {
 public:
  static Result<DeepSeekRankMaterializationWarmupCoordinator> Create(
      DeepSeekRankMaterializationGrantCoordinator& grants,
      const DeepSeekRankArtifactMetadataTransferTransaction& metadata,
      DeepSeekRankMaterializationWarmupOperations& operations);

  DeepSeekRankMaterializationWarmupCoordinator(
      const DeepSeekRankMaterializationWarmupCoordinator&) = delete;
  DeepSeekRankMaterializationWarmupCoordinator& operator=(
      const DeepSeekRankMaterializationWarmupCoordinator&) = delete;
  DeepSeekRankMaterializationWarmupCoordinator(
      DeepSeekRankMaterializationWarmupCoordinator&&) noexcept = default;
  DeepSeekRankMaterializationWarmupCoordinator& operator=(
      DeepSeekRankMaterializationWarmupCoordinator&&) noexcept = default;

  Status advance();
  [[nodiscard]] bool complete() const noexcept { return seal_.has_value(); }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::size_t completion_count() const noexcept;
  [[nodiscard]] const DeepSeekRankMaterializationWarmSeal* seal()
      const noexcept {
    return seal_ ? &*seal_ : nullptr;
  }

 private:
  DeepSeekRankMaterializationWarmupCoordinator(
      DeepSeekRankMaterializationGrantCoordinator& grants,
      const DeepSeekRankArtifactMetadataTransferTransaction& metadata,
      DeepSeekRankMaterializationWarmupOperations& operations) noexcept;
  Status fail(Status cause) noexcept;
  Status validate_completion(
      std::uint32_t rank,
      const DeepSeekRankMaterializationCompletionFields& completion) const;

  DeepSeekRankMaterializationGrantCoordinator* grants_ = nullptr;
  const DeepSeekRankArtifactMetadataTransferTransaction* metadata_ = nullptr;
  DeepSeekRankMaterializationWarmupOperations* operations_ = nullptr;
  std::vector<std::optional<DeepSeekRankMaterializationCompletionFields>>
      completions_;
  std::optional<DeepSeekRankMaterializationWarmSeal> seal_;
  bool poisoned_ = false;
};

}  // namespace pih
