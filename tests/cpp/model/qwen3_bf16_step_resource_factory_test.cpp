#include "pih/model/qwen3_bf16_step_resource_factory.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

struct Layouts final {
  QwenBf16ExecutionArenaLayout execution;
  QwenBf16StepStagingLayout staging;
};

Layouts layouts(std::uint32_t tokens = 17) {
  const QwenKvBlockHandle handles[] = {{4, 9}, {7, 3}};
  auto table = QwenKvBlockTable::Create(2, 6, 32, handles).value();
  auto append = table.prepare_append(tokens).value();
  std::vector<std::int64_t> ids(tokens, 4);
  auto input = QwenBf16StepInputPlan::Create(ids, 0, table, append).value();
  return {QwenBf16ExecutionArenaLayout::Create(tokens, 1).value(),
          QwenBf16StepStagingLayout::Create(input).value()};
}

QwenBf16DeviceArenaOwner owner(std::uintptr_t base, std::uint64_t bytes,
                               std::uint64_t generation) {
  return {base, bytes, generation};
}

QwenBf16StepDeviceOwners owners(const Layouts& value,
                                std::uint32_t slots = 2) {
  return {
      owner(0x100000, value.staging.total_bytes(), 1),
      owner(0x200000, value.execution.activation_arena_bytes(), 2),
      owner(0x300000, value.execution.mlp().arena_bytes(), 3),
      owner(0x400000, value.execution.rope_workspace_bytes(), 4),
      owner(0x500000, value.execution.logit_workspace_bytes(), 5),
      owner(0x600000, sizeof(std::int64_t), 6),
      owner(0x700000, sizeof(std::uint32_t), 41),
      owner(0x800000,
            static_cast<std::uint64_t>(slots) *
                QwenKvSlotPool::kSlotPayloadBytes,
            8),
      owner(0x900000,
            static_cast<std::uint64_t>(slots) * sizeof(QwenKvSlotState), 9)};
}

TEST(QwenBf16StepResourceFactoryTest, DerivesAllTwentyTypedSlots) {
  auto value = layouts();
  auto resources = QwenBf16StepResourceFactory::Create(
      41, 0, 2, value.execution, value.staging, owners(value));
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(resources->size(), 20);
  auto ids = resources->view(QwenBf16ActivationSlot::kTokenIds);
  auto query = resources->view(QwenBf16ActivationSlot::kQuery);
  auto cosine = resources->view(QwenBf16ActivationSlot::kRopeCosine);
  auto visible = resources->view(QwenBf16ActivationSlot::kKvVisibleHandles);
  auto logits = resources->view(QwenBf16ActivationSlot::kLogits);
  ASSERT_TRUE(ids.ok() && query.ok() && cosine.ok() && visible.ok() &&
              logits.ok());
  EXPECT_EQ(ids->dtype(), DType::kInt64);
  EXPECT_EQ(ids->dim(0), 17);
  EXPECT_EQ(query->rank(), 3);
  EXPECT_EQ(query->dim(0), 17);
  EXPECT_EQ(query->dim(1), 16);
  EXPECT_EQ(query->dim(2), 128);
  EXPECT_EQ(cosine->dtype(), DType::kFloat32);
  EXPECT_EQ(cosine->dim(1), 64);
  EXPECT_EQ(visible->dtype(), DType::kUInt8);
  EXPECT_EQ(visible->dim(0), 2 * sizeof(QwenKvBlockHandle));
  EXPECT_EQ(logits->dtype(), DType::kFloat32);
  EXPECT_EQ(logits->dim(0), 1);
  EXPECT_EQ(logits->dim(1), 151936);
}

TEST(QwenBf16StepResourceFactoryTest, SeparatesArenaGenerations) {
  auto value = layouts(1);
  auto resources = QwenBf16StepResourceFactory::Create(
      41, 0, 2, value.execution, value.staging, owners(value));
  ASSERT_TRUE(resources.ok());
  EXPECT_EQ(resources->view(QwenBf16ActivationSlot::kTokenIds)->generation(),
            1);
  EXPECT_EQ(resources->view(QwenBf16ActivationSlot::kHidden)->generation(),
            2);
  EXPECT_EQ(resources->view(QwenBf16ActivationSlot::kGate)->generation(), 3);
  EXPECT_EQ(resources->view(QwenBf16ActivationSlot::kRopeSine)->generation(),
            4);
  EXPECT_EQ(resources->view(QwenBf16ActivationSlot::kKvBacking)->generation(),
            8);
}

TEST(QwenBf16StepResourceFactoryTest, RejectsAnyShortOrStaleOwner) {
  auto value = layouts();
  auto short_activation = owners(value);
  --short_activation.activations.bytes;
  EXPECT_FALSE(QwenBf16StepResourceFactory::Create(
                   41, 0, 2, value.execution, value.staging,
                   short_activation)
                   .ok());
  auto short_rope = owners(value);
  --short_rope.rope.bytes;
  EXPECT_FALSE(QwenBf16StepResourceFactory::Create(
                   41, 0, 2, value.execution, value.staging, short_rope)
                   .ok());
  auto stale = owners(value);
  stale.device_error.generation = 0;
  EXPECT_FALSE(QwenBf16StepResourceFactory::Create(
                   41, 0, 2, value.execution, value.staging, stale)
                   .ok());
  auto cross_request = owners(value);
  cross_request.device_error.generation = 42;
  EXPECT_FALSE(QwenBf16StepResourceFactory::Create(
                   41, 0, 2, value.execution, value.staging, cross_request)
                   .ok());
  EXPECT_FALSE(QwenBf16StepResourceFactory::Create(
                   0, 0, 2, value.execution, value.staging, owners(value))
                   .ok());
}

}  // namespace
}  // namespace pih
