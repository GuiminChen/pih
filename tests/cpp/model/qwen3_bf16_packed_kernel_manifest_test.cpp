#include "pih/model/qwen3_bf16_packed_kernel_manifest.h"

#include <array>

#include <gtest/gtest.h>

#include "pih/core/sha256.h"

namespace pih {
namespace {

TEST(QwenBf16PackedKernelManifestTest, FreezesSymbolsAndLayouts) {
  const std::array primitives{
      QwenBf16PackedPrimitive::kEmbedding,
      QwenBf16PackedPrimitive::kRopeAngles,
      QwenBf16PackedPrimitive::kKvAppend,
      QwenBf16PackedPrimitive::kPagedGqa,
      QwenBf16PackedPrimitive::kSampleHidden,
      QwenBf16PackedPrimitive::kGreedyArgmax,
      QwenBf16PackedPrimitive::kSampler};
  const std::array<std::size_t, 7> counts{6, 5, 13, 16, 7, 5, 14};
  const std::array<std::uint32_t, 7> totals{48, 40, 96, 108, 44, 32, 100};
  for (std::size_t i = 0; i < primitives.size(); ++i) {
    auto manifest = qwen_bf16_packed_kernel_manifest(
        primitives[i], std::string(64, 'c'));
    ASSERT_TRUE(manifest.ok()) << manifest.status().message();
    EXPECT_EQ(manifest->parameter_count(), counts[i]);
    EXPECT_EQ(manifest->total_device_parameter_bytes(), totals[i]);
    EXPECT_TRUE(qwen_bf16_packed_kernel_symbol(primitives[i]).ok());
  }
}

TEST(QwenBf16PackedKernelManifestTest, DescriptorDigestMatchesManifest) {
  for (const auto primitive : {
           QwenBf16PackedPrimitive::kEmbedding,
           QwenBf16PackedPrimitive::kRopeAngles,
           QwenBf16PackedPrimitive::kKvAppend,
           QwenBf16PackedPrimitive::kPagedGqa,
           QwenBf16PackedPrimitive::kSampleHidden,
           QwenBf16PackedPrimitive::kGreedyArgmax,
           QwenBf16PackedPrimitive::kSampler}) {
    auto descriptor = qwen_bf16_packed_parameter_abi_descriptor(primitive);
    auto manifest = qwen_bf16_packed_kernel_manifest(
        primitive, std::string(64, 'd'));
    ASSERT_TRUE(descriptor.ok() && manifest.ok());
    auto digest = sha256(std::as_bytes(std::span(*descriptor)));
    ASSERT_TRUE(digest.ok());
    EXPECT_EQ(digest->hex(), manifest->parameter_abi_sha256());
  }
}

TEST(QwenBf16PackedKernelManifestTest, RejectsUnknownAndBadCubinIdentity) {
  const auto unknown = static_cast<QwenBf16PackedPrimitive>(255);
  EXPECT_FALSE(qwen_bf16_packed_kernel_symbol(unknown).ok());
  EXPECT_FALSE(qwen_bf16_packed_parameter_abi_descriptor(unknown).ok());
  EXPECT_FALSE(qwen_bf16_packed_kernel_manifest(
      unknown, std::string(64, 'a')).ok());
  EXPECT_FALSE(qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kEmbedding, "bad").ok());
}

}  // namespace
}  // namespace pih
