#pragma once

#include <memory>
#include <string_view>

#include "pih/model/deepseek_rank_artifact_prefault.h"

namespace pih {

inline constexpr std::string_view
    kLinuxDeepSeekRankArtifactPrefaultOperationsAbi =
        "pih_linux_deepseek_rank_artifact_prefault_operations_v1";

class LinuxDeepSeekRankArtifactPrefaultOperations final
    : public DeepSeekRankArtifactPrefaultOperations {
 public:
  static Result<std::unique_ptr<
      LinuxDeepSeekRankArtifactPrefaultOperations>>
  Create();

  ~LinuxDeepSeekRankArtifactPrefaultOperations() override;
  LinuxDeepSeekRankArtifactPrefaultOperations(
      const LinuxDeepSeekRankArtifactPrefaultOperations&) = delete;
  LinuxDeepSeekRankArtifactPrefaultOperations& operator=(
      const LinuxDeepSeekRankArtifactPrefaultOperations&) = delete;
  LinuxDeepSeekRankArtifactPrefaultOperations(
      LinuxDeepSeekRankArtifactPrefaultOperations&&) = delete;
  LinuxDeepSeekRankArtifactPrefaultOperations& operator=(
      LinuxDeepSeekRankArtifactPrefaultOperations&&) = delete;

  Result<DeepSeekRankArtifactPrefaultResourceSnapshot>
  sample_resources() override;
  Result<DeepSeekRankArtifactPrefaultRangeObservation>
  prefault_range(std::span<const std::byte> mapping,
                 std::uint64_t page_bytes) override;

 private:
  struct Impl;
  explicit LinuxDeepSeekRankArtifactPrefaultOperations(
      std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace pih
