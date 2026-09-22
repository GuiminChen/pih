#include "pih/model/deepseek_expert_bundle_layout.h"

#include <limits>

namespace pih {
namespace {

DeepSeekExpertDeviceSpan span(std::uintptr_t base, std::uint64_t offset,
                              std::uint64_t bytes) {
  return DeepSeekExpertDeviceSpan{base + static_cast<std::uintptr_t>(offset),
                                  bytes};
}

}  // namespace

Result<DeepSeekExpertBundleDeviceView> DeepSeekExpertBundleLayout::Bind(
    std::uintptr_t base) {
  if (base == 0 || base % kAlignment != 0 ||
      base > std::numeric_limits<std::uintptr_t>::max() - kBundleBytes) {
    return Status::InvalidArgument(
        "DeepSeek canonical expert bundle base is invalid");
  }
  constexpr std::uint64_t kW1Packed = 0;
  constexpr std::uint64_t kW1Scales =
      kW1Packed + kPackedBytesPerMatrix;
  constexpr std::uint64_t kW2Packed = kW1Scales + kScaleBytesPerMatrix;
  constexpr std::uint64_t kW2Scales =
      kW2Packed + kPackedBytesPerMatrix;
  constexpr std::uint64_t kW3Packed = kW2Scales + kScaleBytesPerMatrix;
  constexpr std::uint64_t kW3Scales =
      kW3Packed + kPackedBytesPerMatrix;
  static_assert(kW3Scales + kScaleBytesPerMatrix == kBundleBytes);

  return DeepSeekExpertBundleDeviceView{
      .w1 = {span(base, kW1Packed, kPackedBytesPerMatrix),
             span(base, kW1Scales, kScaleBytesPerMatrix)},
      .w2 = {span(base, kW2Packed, kPackedBytesPerMatrix),
             span(base, kW2Scales, kScaleBytesPerMatrix)},
      .w3 = {span(base, kW3Packed, kPackedBytesPerMatrix),
             span(base, kW3Scales, kScaleBytesPerMatrix)},
  };
}

}  // namespace pih
