#include "pih/model/qwen3_bf16_packed_resource_factory.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

QwenBf16DeviceArenaOwner packed_owner(std::uintptr_t base,
                                      std::uint64_t bytes,
                                      std::uint64_t generation) {
  return {base, bytes, generation};
}

struct PackedLayouts final {
  QwenBf16ExecutionArenaLayout execution;
  QwenBf16PackedStepStagingLayout staging;
  QwenBf16PackedResultLayout result;
};

PackedLayouts packed_layouts() {
  return {QwenBf16ExecutionArenaLayout::Create(8, 3).value(),
          QwenBf16PackedStepStagingLayout::CreateBounded(8, 3, 4).value(),
          QwenBf16PackedResultLayout::Create(3).value()};
}

QwenBf16StepDeviceOwners packed_owners(const PackedLayouts& layouts,
                                       std::uint64_t error_generation = 41) {
  return {
      packed_owner(0x100000, layouts.staging.total_bytes(), 41),
      packed_owner(0x200000, layouts.execution.activation_arena_bytes(), 2),
      packed_owner(0x300000, layouts.execution.mlp().arena_bytes(), 3),
      packed_owner(0x400000, layouts.execution.rope_workspace_bytes(), 4),
      packed_owner(0x500000, layouts.execution.logit_workspace_bytes() * 2, 5),
      packed_owner(0x900000, layouts.result.device_error().offset_bytes, 6),
      packed_owner(0xA00000, 4, error_generation),
      packed_owner(0x1000000, 2 * QwenKvSlotPool::kSlotPayloadBytes, 8),
      packed_owner(0x1500000, 2 * sizeof(QwenKvSlotState), 9)};
}

TEST(QwenBf16PackedResourceFactoryTest, BindsClosedPackedResourceSet) {
  auto layouts = packed_layouts();
  auto resources = QwenBf16PackedResourceFactory::Create(
      41, 0, 2, 3, layouts.execution, layouts.staging,
      packed_owners(layouts));
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(resources->execution_bucket_tokens(), 8);
  EXPECT_EQ(resources->sample_count(), 3);
  EXPECT_EQ(resources->sequence_count(), 3);
  auto ids = resources->activations().view(QwenBf16ActivationSlot::kTokenIds);
  auto hidden = resources->activations().view(QwenBf16ActivationSlot::kHidden);
  auto logits = resources->activations().view(QwenBf16ActivationSlot::kLogits);
  auto sampled =
      resources->activations().view(QwenBf16ActivationSlot::kSampledToken);
  auto query_offsets = resources->metadata().view(
      QwenBf16PackedStagingSlot::kQueryStartOffsets);
  ASSERT_TRUE(ids.ok() && hidden.ok() && logits.ok() && sampled.ok() &&
              query_offsets.ok());
  EXPECT_EQ(ids->dtype(), DType::kUInt8);
  EXPECT_EQ(ids->num_elements(), 32);
  EXPECT_EQ(hidden->dim(0), 8);
  EXPECT_EQ(logits->dim(0), 3);
  EXPECT_EQ(sampled->dtype(), DType::kUInt8);
  EXPECT_EQ(sampled->num_elements(), 12);
  EXPECT_EQ(query_offsets->num_elements(), 16);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                resources->sampler_workspace_ids().data()),
            UINT64_C(0x500000) + layouts.execution.logit_workspace_bytes());
  EXPECT_EQ(resources->sampler_workspace_ids().num_elements(),
            UINT64_C(3) * 151936 * 4);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(
                resources->selected_logprobs().data()),
            UINT64_C(0x900000) + layouts.result.selected_logprobs().offset_bytes);
  EXPECT_EQ(resources->selected_logprobs().num_elements(), 12);
  EXPECT_EQ(resources->rng_words().num_elements(), 12);
  EXPECT_EQ(resources->top_token_ids().num_elements(), 3 * 20 * 4);
  EXPECT_EQ(resources->top_logprobs().num_elements(), 3 * 20 * 4);
  EXPECT_EQ(resources->top_counts().num_elements(), 12);
}

TEST(QwenBf16PackedResourceFactoryTest, RejectsCountOwnerAndLayoutDrift) {
  auto layouts = packed_layouts();
  EXPECT_FALSE(QwenBf16PackedResourceFactory::Create(
      41, 0, 2, 4, layouts.execution, layouts.staging,
      packed_owners(layouts)).ok());
  EXPECT_FALSE(QwenBf16PackedResourceFactory::Create(
      41, 0, 2, 3, layouts.execution, layouts.staging,
      packed_owners(layouts, 42)).ok());
  auto short_sampled = packed_owners(layouts);
  short_sampled.sampled_token.bytes =
      layouts.result.device_error().offset_bytes - 1;
  EXPECT_FALSE(QwenBf16PackedResourceFactory::Create(
      41, 0, 2, 3, layouts.execution, layouts.staging,
      short_sampled).ok());
  auto short_logits = packed_owners(layouts);
  short_logits.logits.bytes = layouts.execution.logit_workspace_bytes();
  EXPECT_FALSE(QwenBf16PackedResourceFactory::Create(
      41, 0, 2, 3, layouts.execution, layouts.staging,
      short_logits).ok());
}

}  // namespace
}  // namespace pih
