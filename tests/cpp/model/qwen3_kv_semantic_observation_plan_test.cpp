#include <cstdint>
#include <algorithm>
#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_kv_semantic_observation_plan.h"

namespace pih {
namespace {

QwenKvBlockTable committed_table(std::span<const QwenKvBlockHandle> handles,
                                 std::uint32_t tokens) {
  auto table = QwenKvBlockTable::Create(4, 7, tokens, handles).value();
  const auto append = table.prepare_append(tokens).value();
  EXPECT_TRUE(table.commit_append(append).ok());
  return table;
}

std::vector<QwenKvSlotState> states(
    std::span<const QwenKvBlockHandle> handles, std::uint32_t tokens,
    std::uint32_t slot_count) {
  std::vector<QwenKvSlotState> result(
      slot_count, {1, QwenKvSlotPool::kNoOwner, 0,
                   QwenKvSlotLifecycle::kFreeClean, 0, 0});
  for (std::size_t index = 0; index < handles.size(); ++index) {
    const std::uint32_t first =
        static_cast<std::uint32_t>(index) * QwenKvSlotPool::kTokensPerSlot;
    const auto valid = static_cast<std::uint16_t>(
        std::min<std::uint32_t>(tokens - first,
                                QwenKvSlotPool::kTokensPerSlot));
    result[handles[index].slot] =
        {handles[index].generation, 4, valid,
         QwenKvSlotLifecycle::kOwned, 0, 0};
  }
  return result;
}

TEST(QwenKvSemanticObservationPlanTest,
     CanonicalDestinationIsIndependentOfPhysicalSlots) {
  const std::vector<QwenKvBlockHandle> first_handles{{2, 11}, {7, 13}};
  const std::vector<QwenKvBlockHandle> second_handles{{8, 21}, {1, 23}};
  auto first_table = committed_table(first_handles, 17);
  auto second_table = committed_table(second_handles, 17);
  auto first_states = states(first_handles, 17, 10);
  auto second_states = states(second_handles, 17, 10);
  constexpr std::uint64_t kBacking =
      10 * QwenKvSlotPool::kSlotPayloadBytes;

  auto first = QwenKvSemanticObservationPlan::Create(
      first_table, first_states, kBacking);
  auto second = QwenKvSemanticObservationPlan::Create(
      second_table, second_states, kBacking);

  ASSERT_TRUE(first.ok() && second.ok());
  EXPECT_EQ(first->payload_bytes(),
            17 * QwenKvAddressMapper::kBytesPerToken * 2 * 28);
  ASSERT_EQ(first->slices().size(), 2 * 2 * 28);
  ASSERT_EQ(first->slices().size(), second->slices().size());
  for (std::size_t index = 0; index < first->slices().size(); ++index) {
    EXPECT_EQ(first->slices()[index].destination_offset,
              second->slices()[index].destination_offset);
    EXPECT_EQ(first->slices()[index].bytes, second->slices()[index].bytes);
  }
  EXPECT_NE(first->slices()[0].source_offset,
            second->slices()[0].source_offset);
  EXPECT_EQ(first->slices()[0].bytes,
            16 * QwenKvAddressMapper::kBytesPerToken);
  EXPECT_EQ(first->slices()[1].bytes,
            QwenKvAddressMapper::kBytesPerToken);
}

TEST(QwenKvSemanticObservationPlanTest, RejectsStaleOrNoncanonicalTailState) {
  const std::vector<QwenKvBlockHandle> handles{{2, 11}, {7, 13}};
  auto table = committed_table(handles, 17);
  auto slot_states = states(handles, 17, 10);
  constexpr std::uint64_t kBacking =
      10 * QwenKvSlotPool::kSlotPayloadBytes;

  slot_states[7].valid_tokens = 16;
  EXPECT_FALSE(QwenKvSemanticObservationPlan::Create(
                   table, slot_states, kBacking)
                   .ok());
  slot_states = states(handles, 17, 10);
  slot_states[2].generation = 12;
  EXPECT_FALSE(QwenKvSemanticObservationPlan::Create(
                   table, slot_states, kBacking)
                   .ok());
}

TEST(QwenKvSemanticObservationPlanTest, AllowsCommittedPrefixOfReservation) {
  const std::vector<QwenKvBlockHandle> handles{{2, 11}, {7, 13}, {5, 17}};
  auto table = QwenKvBlockTable::Create(4, 7, 48, handles).value();
  const auto append = table.prepare_append(17).value();
  ASSERT_TRUE(table.commit_append(append).ok());
  const std::array visible_handles{handles[0], handles[1]};
  auto slot_states = states(visible_handles, 17, 10);
  auto plan = QwenKvSemanticObservationPlan::Create(
      table, slot_states, 10 * QwenKvSlotPool::kSlotPayloadBytes);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->slices().size(), 2 * 2 * 28);
}

}  // namespace
}  // namespace pih
