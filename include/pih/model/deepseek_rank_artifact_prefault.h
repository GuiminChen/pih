#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "pih/model/deepseek_artifact_prefault_layout.h"
#include "pih/model/deepseek_rank_materialization_grant.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankArtifactPrefaultAbi =
    "pih_deepseek_rank_artifact_prefault_v1";

struct DeepSeekRankArtifactPrefaultResourceSnapshot final {
  std::uint64_t monotonic_ns = 0;
  std::uint64_t major_fault_count = 0;
  std::uint64_t vmpte_bytes = 0;
  std::uint64_t cgroup_memory_current_bytes = 0;
  std::uint64_t cgroup_file_bytes = 0;
  std::uint64_t cgroup_anon_bytes = 0;
  std::uint64_t cgroup_kernel_bytes = 0;
};

struct DeepSeekRankArtifactPrefaultRangeObservation final {
  std::uint64_t resident_page_bytes_before = 0;
  std::uint64_t resident_page_bytes_after = 0;
  std::uint64_t touched_page_count = 0;
};

class DeepSeekRankArtifactPrefaultOperations {
 public:
  virtual ~DeepSeekRankArtifactPrefaultOperations() = default;
  virtual Result<DeepSeekRankArtifactPrefaultResourceSnapshot>
  sample_resources() = 0;
  virtual Result<DeepSeekRankArtifactPrefaultRangeObservation>
  prefault_range(std::span<const std::byte> mapping,
                 std::uint64_t page_bytes) = 0;
};

// Immutable proof produced only by the consuming transaction below. It binds
// the exact materialization grant, stable mapping owner and canonical 4 KiB
// selected-page layout to before/after process and cgroup observations.
class DeepSeekRankArtifactPrefaultReceipt final {
 public:
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint64_t page_bytes() const noexcept {
    return page_bytes_;
  }
  [[nodiscard]] std::uint32_t interval_count() const noexcept {
    return interval_count_;
  }
  [[nodiscard]] std::uint64_t mapped_interval_bytes() const noexcept {
    return mapped_interval_bytes_;
  }
  [[nodiscard]] std::uint64_t selected_page_union_bytes() const noexcept {
    return selected_page_union_bytes_;
  }
  [[nodiscard]] std::uint64_t resident_page_bytes_before() const noexcept {
    return resident_page_bytes_before_;
  }
  [[nodiscard]] std::uint64_t resident_page_bytes_after() const noexcept {
    return resident_page_bytes_after_;
  }
  [[nodiscard]] std::uint64_t touched_page_count() const noexcept {
    return touched_page_count_;
  }
  [[nodiscard]] const DeepSeekRankArtifactPrefaultResourceSnapshot& before()
      const noexcept {
    return before_;
  }
  [[nodiscard]] const DeepSeekRankArtifactPrefaultResourceSnapshot& after()
      const noexcept {
    return after_;
  }
  [[nodiscard]] const Sha256Digest& grant_root() const noexcept {
    return grant_root_;
  }
  [[nodiscard]] const Sha256Digest& mapping_owner_root() const noexcept {
    return mapping_owner_root_;
  }
  [[nodiscard]] const Sha256Digest& mapping_plan_root() const noexcept {
    return mapping_plan_root_;
  }
  [[nodiscard]] const Sha256Digest& layout_root() const noexcept {
    return layout_root_;
  }
  [[nodiscard]] const Sha256Digest& receipt_root() const noexcept {
    return receipt_root_;
  }

 private:
  friend class DeepSeekRankArtifactPrefaultTransaction;
  DeepSeekRankArtifactPrefaultReceipt(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, std::uint32_t rank,
      std::uint64_t page_bytes, std::uint32_t interval_count,
      std::uint64_t mapped_interval_bytes,
      std::uint64_t selected_page_union_bytes,
      std::uint64_t resident_page_bytes_before,
      std::uint64_t resident_page_bytes_after,
      std::uint64_t touched_page_count,
      DeepSeekRankArtifactPrefaultResourceSnapshot before,
      DeepSeekRankArtifactPrefaultResourceSnapshot after,
      Sha256Digest grant_root, Sha256Digest mapping_owner_root,
      Sha256Digest mapping_plan_root, Sha256Digest layout_root,
      Sha256Digest receipt_root) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint32_t rank_ = 0;
  std::uint64_t page_bytes_ = 0;
  std::uint32_t interval_count_ = 0;
  std::uint64_t mapped_interval_bytes_ = 0;
  std::uint64_t selected_page_union_bytes_ = 0;
  std::uint64_t resident_page_bytes_before_ = 0;
  std::uint64_t resident_page_bytes_after_ = 0;
  std::uint64_t touched_page_count_ = 0;
  DeepSeekRankArtifactPrefaultResourceSnapshot before_{};
  DeepSeekRankArtifactPrefaultResourceSnapshot after_{};
  Sha256Digest grant_root_{};
  Sha256Digest mapping_owner_root_{};
  Sha256Digest mapping_plan_root_{};
  Sha256Digest layout_root_{};
  Sha256Digest receipt_root_{};
};

struct DeepSeekRankPrefaultedMaterializationInputs final {
  DeepSeekRankPrefaultedMaterializationInputs(
      const DeepSeekRankPrefaultedMaterializationInputs&) = delete;
  DeepSeekRankPrefaultedMaterializationInputs& operator=(
      const DeepSeekRankPrefaultedMaterializationInputs&) = delete;
  DeepSeekRankPrefaultedMaterializationInputs(
      DeepSeekRankPrefaultedMaterializationInputs&&) noexcept = default;
  DeepSeekRankPrefaultedMaterializationInputs& operator=(
      DeepSeekRankPrefaultedMaterializationInputs&&) noexcept = default;

  DeepSeekRankMaterializationAdmission admission;
  std::unique_ptr<DeepSeekRankArtifactMappingOwner> mapping_owner;
  DeepSeekRankArtifactPrefaultReceipt prefault_receipt;

 private:
  friend class DeepSeekRankArtifactPrefaultTransaction;
  DeepSeekRankPrefaultedMaterializationInputs(
      DeepSeekRankMaterializationAdmission admitted,
      std::unique_ptr<DeepSeekRankArtifactMappingOwner> owner,
      DeepSeekRankArtifactPrefaultReceipt receipt) noexcept
      : admission(std::move(admitted)), mapping_owner(std::move(owner)),
        prefault_receipt(std::move(receipt)) {}
};

class DeepSeekRankArtifactPrefaultTransaction final {
 public:
  static constexpr std::uint64_t kPageBytes =
      kDeepSeekArtifactPrefaultPageBytes;

  // Consumes the only raw authorized inputs. On failure their mapping and
  // admission are destroyed; callers cannot retry a partial transaction.
  static Result<DeepSeekRankPrefaultedMaterializationInputs> Run(
      DeepSeekRankAuthorizedMaterializationInputs inputs,
      DeepSeekRankArtifactPrefaultOperations& operations);
};

Status validate_deepseek_rank_prefaulted_materialization_source(
    const DeepSeekRankPrefaultedMaterializationInputs& inputs);

}  // namespace pih
