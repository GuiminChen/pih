#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

struct DeepSeekExpertDeviceSpan final {
  std::uintptr_t address = 0;
  std::uint64_t bytes = 0;
};

struct DeepSeekExpertMatrixDeviceView final {
  DeepSeekExpertDeviceSpan packed;
  DeepSeekExpertDeviceSpan scales;
};

struct DeepSeekExpertBundleDeviceView final {
  DeepSeekExpertMatrixDeviceView w1;
  DeepSeekExpertMatrixDeviceView w2;
  DeepSeekExpertMatrixDeviceView w3;
};

class DeepSeekExpertBundleLayout final {
 public:
  static constexpr std::uint32_t kHiddenSize = 4096;
  static constexpr std::uint32_t kIntermediateSize = 2048;
  static constexpr std::uint64_t kPackedBytesPerMatrix = 4194304;
  static constexpr std::uint64_t kScaleBytesPerMatrix = 262144;
  static constexpr std::uint64_t kBundleBytes = 13369344;
  static constexpr std::uint64_t kAlignment = 256;

  static Result<DeepSeekExpertBundleDeviceView> Bind(std::uintptr_t base);
};

}  // namespace pih
