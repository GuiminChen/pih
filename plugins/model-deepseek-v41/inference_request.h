#pragma once
#include "token_ledger.h"

namespace pih::deepseek_v41 {
struct InferenceRequest final {
  TokenSamplingRequest sampling;
  std::array<std::uint32_t, 4096> tokens{};
  std::uint32_t count = 0;
  bool finish = false;
  static constexpr std::size_t kWireBytes = 256 + 4096 * 4;
  Status Validate() const;
  Result<std::array<std::uint8_t, kWireBytes>> Encode() const;
  static Result<InferenceRequest> Decode(std::span<const std::uint8_t> frame);
};
}  // namespace pih::deepseek_v41
