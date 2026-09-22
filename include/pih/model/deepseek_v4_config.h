#pragma once

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <string_view>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

struct DeepSeekV4Config final {
  static constexpr std::size_t kMaxConfigBytes = 1024 * 1024;
  static Result<DeepSeekV4Config> ParseFlash0731(std::string_view json);
  std::uint32_t main_layers = 43;
  // These are separate pinned facts.  The model config publishes one
  // next-token-prediction layer.  The reference inference stack's ordered
  // MTP stage count is validated by DeepSeekV4InferenceConfig below.
  std::uint32_t nextn_predict_layers = 1;
  std::uint32_t hidden_size = 4096;
  std::uint32_t hc_streams = 4;
  std::uint32_t attention_heads = 64;
  std::uint32_t routed_experts = 256;
  std::uint32_t activated_experts = 6;
  std::uint32_t vocabulary_size = 129280;
  std::uint32_t maximum_positions = 1048576;
  double rms_norm_epsilon = 0.000001;

  static DeepSeekV4Config Flash0731() noexcept { return {}; }
  [[nodiscard]] Status validate() const;
};

struct DeepSeekV4InferenceConfig final {
  static constexpr std::size_t kMaxConfigBytes = 1024 * 1024;
  static Result<DeepSeekV4InferenceConfig> ParseFlash0731(
      std::string_view json);
  std::uint32_t mtp_stages = 3;

  static DeepSeekV4InferenceConfig Flash0731() noexcept { return {}; }
  [[nodiscard]] Status validate() const;
};

struct DeepSeekV4ConfigFileReceipt final {
  DeepSeekV4Config config;
  std::uint64_t file_bytes = 0;
  Sha256Digest file_sha256;
};

Result<DeepSeekV4ConfigFileReceipt> load_deepseek_v4_flash_0731_config_file(
    const std::filesystem::path& path);

struct DeepSeekV4InferenceConfigFileReceipt final {
  DeepSeekV4InferenceConfig config;
  std::uint64_t file_bytes = 0;
  Sha256Digest file_sha256;
};

Result<DeepSeekV4InferenceConfigFileReceipt>
load_deepseek_v4_flash_0731_inference_config_file(
    const std::filesystem::path& path);

struct DeepSeekStageRange final {
  std::uint32_t first_layer = 0;
  std::uint32_t last_layer = 0;
  bool operator==(const DeepSeekStageRange&) const = default;
};

struct DeepSeekStagePlan final {
  std::uint32_t rank = 0;
  DeepSeekStageRange layers;
  bool owns_embedding = false;
  bool owns_lm_head = false;
  bool owns_dspark = false;
};

class DeepSeekPipelinePlan final {
 public:
  static Result<DeepSeekPipelinePlan> Create(std::uint32_t world_size,
                                             bool dspark_enabled);
  [[nodiscard]] const DeepSeekStagePlan& rank(std::uint32_t rank) const;
  [[nodiscard]] const std::vector<DeepSeekStageRange>& ranges() const noexcept {
    return ranges_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return static_cast<std::uint32_t>(stages_.size());
  }

 private:
  std::vector<DeepSeekStagePlan> stages_;
  std::vector<DeepSeekStageRange> ranges_;
};

}  // namespace pih
