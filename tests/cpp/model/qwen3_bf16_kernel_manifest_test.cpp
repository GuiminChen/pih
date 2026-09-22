#include <array>
#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include "pih/core/sha256.h"
#include "pih/model/qwen3_bf16_kernel_manifest.h"

namespace pih {
namespace {

std::span<const std::byte> bytes(std::string_view value) {
  return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

TEST(QwenBf16KernelManifestTest, FreezesSymbolsAndRuntimeParameterLayouts) {
  const std::string cubin_digest(64, 'c');
  const std::array primitives{QwenBf16Primitive::kEmbedding,
                              QwenBf16Primitive::kResidualAdd,
                              QwenBf16Primitive::kSiluMul,
                              QwenBf16Primitive::kRmsNorm,
                              QwenBf16Primitive::kRope,
                              QwenBf16Primitive::kKvAppend,
                              QwenBf16Primitive::kPagedGqa,
                              QwenBf16Primitive::kRopeAngles,
                              QwenBf16Primitive::kGreedyArgmax,
                              QwenBf16Primitive::kTeacherForcedMetric};
  const std::array<std::size_t, 10> counts{6, 4, 4, 6, 7, 11, 14, 5, 4, 8};
  const std::array<std::uint32_t, 10> totals{48, 32, 32, 44, 56, 76, 84, 40,
                                             32, 56};
  for (std::size_t index = 0; index < primitives.size(); ++index) {
    auto manifest = qwen_bf16_kernel_manifest(primitives[index], cubin_digest);
    ASSERT_TRUE(manifest.ok()) << manifest.status().message();
    EXPECT_EQ(manifest->selected_cubin_sha256(), cubin_digest);
    EXPECT_EQ(manifest->parameter_count(), counts[index]);
    EXPECT_EQ(manifest->total_device_parameter_bytes(), totals[index]);
    EXPECT_TRUE(qwen_bf16_kernel_symbol(primitives[index]).ok());
  }
}

TEST(QwenBf16KernelManifestTest, DescriptorDigestMatchesFrozenIdentity) {
  const std::string cubin_digest(64, 'd');
  for (const auto primitive : {QwenBf16Primitive::kEmbedding,
                               QwenBf16Primitive::kResidualAdd,
                               QwenBf16Primitive::kSiluMul,
                               QwenBf16Primitive::kRmsNorm,
                               QwenBf16Primitive::kRope,
                               QwenBf16Primitive::kKvAppend,
                               QwenBf16Primitive::kPagedGqa,
                               QwenBf16Primitive::kRopeAngles,
                               QwenBf16Primitive::kGreedyArgmax,
                               QwenBf16Primitive::kTeacherForcedMetric}) {
    auto descriptor = qwen_bf16_parameter_abi_descriptor(primitive);
    auto manifest = qwen_bf16_kernel_manifest(primitive, cubin_digest);
    ASSERT_TRUE(descriptor.ok());
    ASSERT_TRUE(manifest.ok());
    auto digest = sha256(bytes(descriptor.value()));
    ASSERT_TRUE(digest.ok());
    EXPECT_EQ(digest->hex(), manifest->parameter_abi_sha256());
  }
}

TEST(QwenBf16KernelManifestTest, RejectsUnknownPrimitiveAndBadCubinIdentity) {
  const auto unknown = static_cast<QwenBf16Primitive>(255);
  EXPECT_FALSE(qwen_bf16_kernel_symbol(unknown).ok());
  EXPECT_FALSE(qwen_bf16_parameter_abi_descriptor(unknown).ok());
  EXPECT_FALSE(qwen_bf16_kernel_manifest(unknown, std::string(64, 'a')).ok());
  EXPECT_FALSE(
      qwen_bf16_kernel_manifest(QwenBf16Primitive::kEmbedding, "bad").ok());
}

}  // namespace
}  // namespace pih
