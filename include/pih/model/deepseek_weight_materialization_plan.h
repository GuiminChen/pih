#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_weight_disposition_manifest.h"

namespace pih {

struct DeepSeekWeightCopyRecord final {
  std::string tensor_name;
  std::string shard_name;
  DType dtype{};
  std::vector<std::uint64_t> shape;
  std::uint64_t source_offset = 0;
  std::uint64_t destination_offset = 0;
  std::uint64_t bytes = 0;
  std::optional<DeepSeekExpertIdentity> expert_identity;
  std::uint32_t logical_layer = UINT32_MAX;
  DeepSeekStorageSemantics storage_semantics{};
  Sha256Digest artifact_root;
  Sha256Digest layout_root;
  Sha256Digest disposition_root;
  Sha256Digest target_logical_root;
  Sha256Digest disposition_record_root;
  Sha256Digest layout_record_root;
  Sha256Digest runtime_record_root;
};

class DeepSeekWeightMaterializationPlan final {
 public:
  static constexpr std::uint64_t kFinalAlignment = 256;

  static Result<DeepSeekWeightMaterializationPlan> Create(
      const DeepSeekRankWeightDispositionManifest& disposition,
      const DeepSeekExpertBundleManifest& experts,
      std::span<const DeepSeekRankTensorRecord> rank_tensors);

  [[nodiscard]] const std::vector<DeepSeekWeightCopyRecord>& copies()
      const noexcept { return copies_; }
  [[nodiscard]] const DeepSeekWeightCopyRecord* find(
      std::string_view tensor_name) const noexcept;
  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
    return backing_bytes_;
  }
  [[nodiscard]] std::uint64_t paged_source_bytes() const noexcept {
    return paged_source_bytes_;
  }
  [[nodiscard]] std::uint32_t owner_rank() const noexcept {
    return owner_rank_;
  }
  [[nodiscard]] bool target_authority_bound() const noexcept {
    return target_authority_bound_;
  }
  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] const Sha256Digest& layout_root() const noexcept {
    return layout_root_;
  }
  [[nodiscard]] const Sha256Digest& disposition_root() const noexcept {
    return disposition_root_;
  }

 private:
  std::vector<DeepSeekWeightCopyRecord> copies_;
  std::uint32_t owner_rank_ = 0;
  std::uint64_t payload_bytes_ = 0;
  std::uint64_t backing_bytes_ = 0;
  std::uint64_t paged_source_bytes_ = 0;
  bool target_authority_bound_ = false;
  Sha256Digest artifact_root_;
  Sha256Digest layout_root_;
  Sha256Digest disposition_root_;
};

}  // namespace pih
