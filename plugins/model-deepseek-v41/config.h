#pragma once
#include <array>
#include <cstdint>
#include <string_view>
#include "pih/core/sha256.h"

namespace pih::deepseek_v41 {
struct AttentionSharingLayer final {
  std::uint32_t layer = 0;
  std::uint32_t compression_ratio = 0;
  // UINT32_MAX means absent, including the three sliding-window-only MTP layers.
  std::uint32_t kv_source = UINT32_MAX;
  std::uint32_t index_source = UINT32_MAX;
  bool owns_kv = false;
  bool owns_index = false;
  bool produces_candidates = false;
  bool index_uses_candidates = false;
  bool has_engram = false;
  std::uint32_t routed_experts = 0;
  std::uint32_t activated_experts = 0;
};
class FlashConfig final {
 public:
  static constexpr std::size_t kMaximumBytes = 1U << 20;
  static constexpr std::string_view kReferenceRevision = "517ef625df97ec57aadc91b67506a57c20fdc5bb";
  static constexpr std::uint32_t kMainLayers = 40, kMtpLayers = 3, kHiddenSize = 5120;
  static constexpr std::uint32_t kVocabularySize = 129280, kMaximumPositions = 1048576;
  // Parses the frozen nested HF configuration, not the reduced inference JSON.
  // Exact semantic fields/types; formatting and object order may differ. This
  // establishes configuration geometry only, not revision/weight authenticity.
  static Result<FlashConfig> Parse(std::string_view json);
  const Sha256Digest& config_sha256() const noexcept { return config_sha256_; }
  const std::array<AttentionSharingLayer, 43>& attention_sharing() const noexcept { return layers_; }
 private:
  Sha256Digest config_sha256_;
  std::array<AttentionSharingLayer, 43> layers_{};
};
}  // namespace pih::deepseek_v41
