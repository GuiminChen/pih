#include "pih/model/qwen3_cuda_invariant.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenCudaInvariantTest, FreezesDeviceErrorWireValues) {
  EXPECT_EQ(static_cast<std::uint32_t>(QwenCudaInvariant::kNone), 0);
  EXPECT_EQ(static_cast<std::uint32_t>(QwenCudaInvariant::kKvAppendLaunch), 1);
  EXPECT_EQ(
      static_cast<std::uint32_t>(QwenCudaInvariant::kKvAppendHandleState), 3);
  EXPECT_EQ(static_cast<std::uint32_t>(QwenCudaInvariant::kPagedGqaLaunch), 4);
  EXPECT_EQ(static_cast<std::uint32_t>(
                QwenCudaInvariant::kPagedGqaSoftmaxDenominator),
            8);
  EXPECT_EQ(static_cast<std::uint32_t>(
                QwenCudaInvariant::kPagedGqaNonfiniteOutput),
            9);
  EXPECT_EQ(static_cast<std::uint32_t>(
                QwenCudaInvariant::kArgmaxNonfiniteLogit),
            10);
  EXPECT_EQ(static_cast<std::uint32_t>(QwenCudaInvariant::kW4A16Launch), 11);
  EXPECT_EQ(static_cast<std::uint32_t>(
                QwenCudaInvariant::kW4A16NonfiniteOutput),
            14);
  EXPECT_EQ(static_cast<std::uint32_t>(QwenCudaInvariant::kPackedSampleLaunch),
            15);
  EXPECT_EQ(static_cast<std::uint32_t>(QwenCudaInvariant::kPackedSampleRow),
            16);
  EXPECT_EQ(static_cast<std::uint32_t>(QwenCudaInvariant::kPackedArgmaxLaunch),
            17);
  EXPECT_EQ(static_cast<std::uint32_t>(QwenCudaInvariant::kPackedSamplerLaunch),
            18);
  EXPECT_EQ(
      static_cast<std::uint32_t>(QwenCudaInvariant::kPackedSamplerDescriptor),
      19);
}

TEST(QwenCudaInvariantTest, RejectsUnknownDeviceErrorCodes) {
  for (std::uint32_t code = 0; code <= 20; ++code) {
    EXPECT_TRUE(qwen_cuda_invariant_known(code));
  }
  EXPECT_FALSE(qwen_cuda_invariant_known(21));
  EXPECT_FALSE(qwen_cuda_invariant_known(UINT32_MAX));
}

}  // namespace
}  // namespace pih
