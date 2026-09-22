#pragma once
#include "weight_files.h"

namespace pih::deepseek_v41 {
// Parses only after matching the independently admitted digest of raw bytes.
Result<std::vector<ExpectedWeightShard>> ParseWeightManifest(std::string_view json,
    const Sha256Digest& expected_digest, const FlashConfig& config,
    std::uint32_t world, std::uint32_t rank);
inline constexpr std::size_t kWeightManifestMaximumBytes = 128U * 1024;
}  // namespace pih::deepseek_v41
