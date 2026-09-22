#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_expert_bundle_manifest.h"

namespace pih {

enum class DeepSeekRoutedExpertResidency : std::uint8_t {
  kHostSpill,
  kFullResident,
};

enum class DeepSeekWeightDispositionKind : std::uint8_t {
  kMaterializeFixed,
  kPagedSource,
};

struct DeepSeekRankWeightDispositionRecord final {
  std::string tensor_name;
  DeepSeekWeightDispositionKind kind{};
  DeepSeekTensorRole role{};
  std::uint32_t owner_rank = 0;
  std::optional<DeepSeekExpertIdentity> expert_identity;
  std::uint64_t source_bytes = 0;
};

class DeepSeekRankWeightDispositionManifest final {
 public:
  static Result<DeepSeekRankWeightDispositionManifest> Create(
      std::uint32_t owner_rank, DeepSeekStageRange owned_layers,
      DeepSeekRoutedExpertResidency expert_residency,
      std::span<const DeepSeekRankTensorRecord> rank_tensors);

  [[nodiscard]] const std::vector<DeepSeekRankWeightDispositionRecord>& records()
      const noexcept { return records_; }
  [[nodiscard]] const DeepSeekRankWeightDispositionRecord* find(
      std::string_view tensor_name) const noexcept;
  [[nodiscard]] std::uint32_t materialize_fixed_record_count() const noexcept {
    return materialize_fixed_record_count_;
  }
  [[nodiscard]] std::uint32_t paged_source_record_count() const noexcept {
    return paged_source_record_count_;
  }
  [[nodiscard]] std::uint64_t materialize_fixed_bytes() const noexcept {
    return materialize_fixed_bytes_;
  }
  [[nodiscard]] std::uint64_t paged_source_bytes() const noexcept {
    return paged_source_bytes_;
  }
  [[nodiscard]] std::uint32_t owner_rank() const noexcept {
    return owner_rank_;
  }

 private:
  std::vector<DeepSeekRankWeightDispositionRecord> records_;
  std::uint32_t owner_rank_ = 0;
  std::uint32_t materialize_fixed_record_count_ = 0;
  std::uint32_t paged_source_record_count_ = 0;
  std::uint64_t materialize_fixed_bytes_ = 0;
  std::uint64_t paged_source_bytes_ = 0;
};

}  // namespace pih
