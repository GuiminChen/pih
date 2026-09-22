#include "pih/model/deepseek_bound_attention_work_provider.h"

#include <gtest/gtest.h>

#include <array>

namespace pih {
namespace {

DeepSeekDecodeAttentionWork decode_work(
    std::uintptr_t identity, std::span<const std::uint32_t> positions) {
  DeepSeekDecodeAttentionWork work;
  work.attention.query_positions = positions;
  work.recent_writer = reinterpret_cast<DeepSeekRecentStateWriter*>(
      identity * 4 + 1);
  work.update_coordinator =
      reinterpret_cast<DeepSeekCompressedLayerUpdateCoordinator*>(
          identity * 4 + 2);
  work.coordinator = reinterpret_cast<DeepSeekAttentionLayerCoordinator*>(
      identity * 4 + 3);
  work.transaction = reinterpret_cast<DeepSeekAttentionSequenceTransaction*>(
      identity * 4 + 4);
  return work;
}

DeepSeekChunkAttentionWork chunk_work(
    std::uintptr_t identity, std::span<const std::uint32_t> positions,
    std::span<const DeepSeekRecentStateSubmission> recent,
    std::span<const DeepSeekCompressedLayerUpdateSubmission> updates) {
  DeepSeekChunkAttentionWork work;
  work.submission.attention.query_positions = positions;
  work.submission.recent = recent;
  work.submission.updates = updates;
  work.chunk_coordinator = reinterpret_cast<DeepSeekPrefillLayerCoordinator*>(
      identity * 3 + 1);
  work.attention_coordinator =
      reinterpret_cast<DeepSeekAttentionLayerCoordinator*>(identity * 3 + 2);
  work.transaction = reinterpret_cast<DeepSeekAttentionSequenceTransaction*>(
      identity * 3 + 3);
  return work;
}

TEST(DeepSeekBoundAttentionWorkProviderTest,
     DecodeBindsEveryOwnedLayerToExactDescriptor) {
  auto provider = DeepSeekBoundDecodeAttentionWorkProvider::Create({5, 6});
  ASSERT_TRUE(provider.ok());
  const std::array<std::uint32_t, 2> positions{{10, 11}};
  const std::array<DeepSeekBoundDecodeAttentionLayerWork, 2> layers{{
      {6, decode_work(2, positions)}, {5, decode_work(1, positions)}}};
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 2, 2};
  ASSERT_TRUE(provider->bind(descriptor, layers).ok());
  auto resolved = provider->resolve(5, descriptor);
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ((*resolved)->attention.query_positions.data(), positions.data());
  auto stale = descriptor;
  ++stale.plan_sequence;
  EXPECT_FALSE(provider->resolve(5, stale).ok());
  EXPECT_FALSE(provider->resolve(4, descriptor).ok());
}

TEST(DeepSeekBoundAttentionWorkProviderTest,
     FailedDecodeRebindPreservesPriorPlan) {
  auto provider = DeepSeekBoundDecodeAttentionWorkProvider::Create({5, 5});
  ASSERT_TRUE(provider.ok());
  const std::array<std::uint32_t, 1> positions{{10}};
  const std::array<DeepSeekBoundDecodeAttentionLayerWork, 1> valid{{
      {5, decode_work(1, positions)}}};
  const DeepSeekPipelinePlanDescriptor first{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  ASSERT_TRUE(provider->bind(first, valid).ok());
  auto second = first;
  ++second.plan_sequence;
  auto incomplete = decode_work(2, positions);
  incomplete.coordinator = nullptr;
  const std::array<DeepSeekBoundDecodeAttentionLayerWork, 1> invalid{{
      {5, incomplete}}};
  EXPECT_FALSE(provider->bind(second, invalid).ok());
  EXPECT_TRUE(provider->resolve(5, first).ok());
  EXPECT_FALSE(provider->resolve(5, second).ok());
}

TEST(DeepSeekBoundAttentionWorkProviderTest,
     ChunkAcceptsPrefillAndVerifyButRejectsDecode) {
  auto provider = DeepSeekBoundChunkAttentionWorkProvider::Create({5, 5});
  ASSERT_TRUE(provider.ok());
  const std::array<std::uint32_t, 2> positions{{10, 11}};
  const std::array<DeepSeekRecentStateSubmission, 1> recent{};
  const std::array<DeepSeekCompressedLayerUpdateSubmission, 1> updates{};
  const std::array<DeepSeekBoundChunkAttentionLayerWork, 1> layers{{
      {5, chunk_work(1, positions, recent, updates)}}};
  const DeepSeekPipelinePlanDescriptor prefill{
      7, 9, DeepSeekPlanPhase::kPrefill, 2, 1};
  ASSERT_TRUE(provider->bind(prefill, layers).ok());
  EXPECT_TRUE(provider->resolve(5, prefill).ok());
  const DeepSeekPipelinePlanDescriptor verify{
      7, 10, DeepSeekPlanPhase::kVerify, 2, 1};
  ASSERT_TRUE(provider->bind(verify, layers).ok());
  EXPECT_TRUE(provider->resolve(5, verify).ok());
  const DeepSeekPipelinePlanDescriptor decode{
      7, 11, DeepSeekPlanPhase::kDecode, 2, 1};
  EXPECT_FALSE(provider->bind(decode, layers).ok());
  EXPECT_TRUE(provider->resolve(5, verify).ok());
}

TEST(DeepSeekBoundAttentionWorkProviderTest,
     RejectsSyntheticDsparkLayerIdentity) {
  EXPECT_FALSE(DeepSeekBoundDecodeAttentionWorkProvider::Create({43, 43}).ok());
  EXPECT_FALSE(DeepSeekBoundChunkAttentionWorkProvider::Create({43, 43}).ok());
}

}  // namespace
}  // namespace pih
