#include "pih/model/qwen3_bf16_packed_dispatch_plan.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

namespace pih {
namespace {

TensorView packed_view(std::uintptr_t address, DType dtype,
                       std::span<const std::int64_t> shape,
                       std::uint64_t generation = 7) {
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape, {},
                            Device::Create(DeviceType::kCuda, 0).value(),
                            generation).value();
}

ResolvedKernelFunction packed_function(QwenBf16PackedPrimitive primitive) {
  const std::string cubin(64, 'd');
  auto manifest = qwen_bf16_packed_kernel_manifest(primitive, cubin).value();
  return {std::string(qwen_bf16_packed_kernel_symbol(primitive).value()),
          std::string(manifest.logical_id()), cubin,
          std::string(manifest.parameter_abi_sha256()), 91};
}

std::uint64_t packed_u64(const KernelArgumentPacket& packet,
                         std::size_t ordinal) {
  std::uint64_t value = 0;
  std::memcpy(&value, packet.argument_cell(ordinal), sizeof(value));
  return value;
}

std::uint32_t packed_u32(const KernelArgumentPacket& packet,
                         std::size_t ordinal) {
  std::uint32_t value = 0;
  std::memcpy(&value, packet.argument_cell(ordinal), sizeof(value));
  return value;
}

TEST(QwenBf16PackedDispatchPlanTest, BuildsRealPrefixEmbeddingAndRopeAngles) {
  const std::array<std::int64_t, 2> table{151936, 1024};
  const std::array<std::int64_t, 1> token_bytes{32};
  const std::array<std::int64_t, 2> hidden{8, 1024};
  auto embedding = QwenBf16PackedDispatchPlan::CreateEmbedding(
      packed_function(QwenBf16PackedPrimitive::kEmbedding),
      packed_view(0x100000, DType::kBFloat16, table),
      packed_view(0x20000000, DType::kUInt8, token_bytes),
      packed_view(0x21000000, DType::kBFloat16, hidden), 5, 0);
  ASSERT_TRUE(embedding.ok()) << embedding.status().message();
  EXPECT_EQ(packed_u64(embedding->arguments(), 3), 5);
  EXPECT_EQ(embedding->geometry().grid_x(), 20);

  const std::array<std::int64_t, 1> position_bytes{64};
  const std::array<std::int64_t, 2> angles{8, 64};
  auto rope = QwenBf16PackedDispatchPlan::CreateRopeAngles(
      packed_function(QwenBf16PackedPrimitive::kRopeAngles),
      packed_view(0x22000000, DType::kUInt8, position_bytes),
      packed_view(0x23000000, DType::kFloat32, angles),
      packed_view(0x24000000, DType::kFloat32, angles), 5, 0);
  ASSERT_TRUE(rope.ok()) << rope.status().message();
  EXPECT_EQ(packed_u64(rope->arguments(), 3), 5);
  EXPECT_EQ(rope->geometry().grid_x(), 2);
}

TEST(QwenBf16PackedDispatchPlanTest, BuildsOwnerCheckedPackedKvAndGqaPackets) {
  const std::array<std::int64_t, 3> kv_activation{3, 8, 128};
  const std::array<std::int64_t, 3> q_activation{3, 16, 128};
  const std::array<std::int64_t, 1> kv_backing{3'670'016};
  const std::array<std::int64_t, 1> states{32};
  const std::array<std::int64_t, 1> handles{24};
  const std::array<std::int64_t, 1> offsets_u16{6};
  const std::array<std::int64_t, 1> token_u32{12};
  const std::array<std::int64_t, 1> prefix_u32{12};
  const std::array<std::int64_t, 1> sequence_u32{8};
  const std::array<std::int64_t, 1> error{4};
  auto append = QwenBf16PackedDispatchPlan::CreateKvAppend(
      packed_function(QwenBf16PackedPrimitive::kKvAppend),
      packed_view(0x10000, DType::kBFloat16, kv_activation),
      packed_view(0x20000, DType::kBFloat16, kv_activation),
      packed_view(0x100000, DType::kUInt8, kv_backing),
      packed_view(0x500000, DType::kUInt8, states),
      packed_view(0x510000, DType::kUInt8, handles),
      packed_view(0x520000, DType::kUInt8, offsets_u16),
      packed_view(0x530000, DType::kUInt8, token_u32),
      packed_view(0x540000, DType::kUInt8, sequence_u32),
      packed_view(0x550000, DType::kUInt8, error), 27, 3, 2, 2, 0);
  ASSERT_TRUE(append.ok()) << append.status().message();
  EXPECT_EQ(packed_u32(append->arguments(), 9), 27);
  EXPECT_EQ(packed_u64(append->arguments(), 10), 3);
  EXPECT_EQ(packed_u32(append->arguments(), 11), 2);
  EXPECT_EQ(append->geometry().grid_x(), 24);

  auto gqa = QwenBf16PackedDispatchPlan::CreatePagedGqa(
      packed_function(QwenBf16PackedPrimitive::kPagedGqa),
      packed_view(0x600000, DType::kBFloat16, q_activation),
      packed_view(0x700000, DType::kBFloat16, q_activation),
      packed_view(0x100000, DType::kUInt8, kv_backing),
      packed_view(0x500000, DType::kUInt8, states),
      packed_view(0x510000, DType::kUInt8, handles),
      packed_view(0x560000, DType::kUInt8, prefix_u32),
      packed_view(0x530000, DType::kUInt8, token_u32),
      packed_view(0x570000, DType::kUInt8, prefix_u32),
      packed_view(0x580000, DType::kUInt8, sequence_u32),
      packed_view(0x540000, DType::kUInt8, sequence_u32),
      packed_view(0x550000, DType::kUInt8, error), 27, 3, 2,
      1.0F / std::sqrt(128.0F), 2, 0);
  ASSERT_TRUE(gqa.ok()) << gqa.status().message();
  EXPECT_EQ(packed_u32(gqa->arguments(), 11), 27);
  EXPECT_EQ(packed_u32(gqa->arguments(), 12), 3);
  EXPECT_EQ(packed_u32(gqa->arguments(), 13), 2);
  EXPECT_EQ(gqa->geometry().grid_x(), 48);
}

TEST(QwenBf16PackedDispatchPlanTest, RejectsPaddedCountAndIdentityDrift) {
  const std::array<std::int64_t, 2> table{4, 3};
  const std::array<std::int64_t, 1> ids{8};
  const std::array<std::int64_t, 2> output{2, 3};
  auto function = packed_function(QwenBf16PackedPrimitive::kEmbedding);
  EXPECT_FALSE(QwenBf16PackedDispatchPlan::CreateEmbedding(
                   function, packed_view(0x1000, DType::kBFloat16, table),
                   packed_view(0x2000, DType::kUInt8, ids),
                   packed_view(0x3000, DType::kBFloat16, output), 3, 0).ok());
  function.parameter_abi_sha256 = std::string(64, '0');
  EXPECT_FALSE(QwenBf16PackedDispatchPlan::CreateEmbedding(
                   function, packed_view(0x1000, DType::kBFloat16, table),
                   packed_view(0x2000, DType::kUInt8, ids),
                   packed_view(0x3000, DType::kBFloat16, output), 2, 0).ok());
}

TEST(QwenBf16PackedDispatchPlanTest, BuildsSampleGatherAndBatchedArgmax) {
  const std::array<std::int64_t, 2> packed_hidden{8, 1024};
  const std::array<std::int64_t, 1> sample_rows{12};
  const std::array<std::int64_t, 2> sampled_hidden{3, 1024};
  const std::array<std::int64_t, 1> error{4};
  auto gather = QwenBf16PackedDispatchPlan::CreateSampleHidden(
      packed_function(QwenBf16PackedPrimitive::kSampleHidden),
      packed_view(0x10000, DType::kBFloat16, packed_hidden),
      packed_view(0x30000, DType::kUInt8, sample_rows),
      packed_view(0x40000, DType::kBFloat16, sampled_hidden),
      packed_view(0x50000, DType::kUInt8, error), 3, 8, 0);
  ASSERT_TRUE(gather.ok()) << gather.status().message();
  EXPECT_EQ(packed_u32(gather->arguments(), 4), 3);
  EXPECT_EQ(packed_u32(gather->arguments(), 5), 8);
  EXPECT_EQ(packed_u32(gather->arguments(), 6), 1024);
  EXPECT_EQ(gather->geometry().grid_x(), 12);

  const std::array<std::int64_t, 2> logits{3, 151936};
  const std::array<std::int64_t, 1> sampled_ids{12};
  auto argmax = QwenBf16PackedDispatchPlan::CreateGreedyArgmax(
      packed_function(QwenBf16PackedPrimitive::kGreedyArgmax),
      packed_view(0x60000, DType::kFloat32, logits),
      packed_view(0x250000, DType::kUInt8, sampled_ids),
      packed_view(0x260000, DType::kUInt8, error), 3, 0);
  ASSERT_TRUE(argmax.ok()) << argmax.status().message();
  EXPECT_EQ(packed_u32(argmax->arguments(), 3), 3);
  EXPECT_EQ(packed_u32(argmax->arguments(), 4), 151936);
  EXPECT_EQ(argmax->geometry().grid_x(), 3);
  EXPECT_EQ(argmax->geometry().block_x(), 256);
}

TEST(QwenBf16PackedDispatchPlanTest, BuildsMixedSamplerPacket) {
  const std::array<std::int64_t, 2> logits{3, 151936};
  const std::array<std::int64_t, 1> descriptors{3 * 40};
  const std::array<std::int64_t, 1> sample_sequences{3 * 4};
  const std::array<std::int64_t, 1> workspace{3 * 151936 * 4};
  const std::array<std::int64_t, 1> per_sample{3 * 4};
  const std::array<std::int64_t, 1> top_values{3 * 20 * 4};
  const std::array<std::int64_t, 1> error{4};
  auto sampler = QwenBf16PackedDispatchPlan::CreateSampler(
      packed_function(QwenBf16PackedPrimitive::kSampler),
      packed_view(0x100000, DType::kFloat32, logits),
      packed_view(0x300000, DType::kUInt8, descriptors),
      packed_view(0x310000, DType::kUInt8, sample_sequences),
      packed_view(0x400000, DType::kUInt8, workspace),
      packed_view(0x800000, DType::kUInt8, per_sample),
      packed_view(0x810000, DType::kUInt8, per_sample),
      packed_view(0x820000, DType::kUInt8, per_sample),
      packed_view(0x830000, DType::kUInt8, top_values),
      packed_view(0x840000, DType::kUInt8, top_values),
      packed_view(0x850000, DType::kUInt8, per_sample),
      packed_view(0x860000, DType::kUInt8, error), 3, 3, 0);
  ASSERT_TRUE(sampler.ok()) << sampler.status().message();
  EXPECT_EQ(sampler->primitive(), QwenBf16PackedPrimitive::kSampler);
  EXPECT_EQ(packed_u32(sampler->arguments(), 11), 3);
  EXPECT_EQ(packed_u32(sampler->arguments(), 12), 3);
  EXPECT_EQ(packed_u32(sampler->arguments(), 13), 151936);
  EXPECT_EQ(sampler->geometry().grid_x(), 3);
  EXPECT_EQ(sampler->geometry().block_x(), 1);
}

}  // namespace
}  // namespace pih
