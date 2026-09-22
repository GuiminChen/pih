#pragma once
#include "engram_hash.h"
#include "engram_launch.h"

namespace pih::deepseek_v41 {
struct TokenInputUploadLaunch final {
  EngramDeviceRegion device_ids, host_ids;
  std::array<EngramDeviceRegion, 2> device_hashes, host_hashes;
  std::uintptr_t stream = 0;
};
Status ValidateTokenInputUpload(const TokenInputUploadLaunch& launch, std::uint32_t tokens);
// Text-only, all tokens participate. Hash state advances before H2D; any later
// failure requires sequence retirement, not retry with the advanced state.
Status UploadTokenInputs(EngramHashState& hashes, std::uint32_t start,
    std::span<const std::uint32_t> tokens, const TokenInputUploadLaunch& launch);
}  // namespace pih::deepseek_v41
