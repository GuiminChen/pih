#include "pih/model/qwen3_kv_block_table.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

std::array<QwenKvBlockHandle, 3> handles() {
  return {{{7, 2}, {3, 4}, {9, 1}}};
}

TEST(QwenKvBlockTableTest, FreezesDescriptorAndSixteenTokenVisibility) {
  EXPECT_EQ(sizeof(QwenKvSequenceDescriptor), 256);
  const auto all = handles();
  auto table = QwenKvBlockTable::Create(5, 8, 33, all);
  ASSERT_TRUE(table.ok()) << table.status().message();
  EXPECT_EQ(table->reserved_handles().size(), 3);
  EXPECT_TRUE(table->visible_handles().empty());

  auto first = table->prepare_append(15);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(first->visible_handle_count_after_commit, 1);
  ASSERT_TRUE(table->commit_append(first.value()).ok());
  EXPECT_EQ(table->visible_handles().size(), 1);

  auto boundary = table->prepare_append(16);
  ASSERT_TRUE(boundary.ok());
  EXPECT_EQ(boundary->visible_handle_count_after_commit, 1);
  ASSERT_TRUE(table->commit_append(boundary.value()).ok());
  EXPECT_EQ(table->visible_handles().size(), 1);

  auto crossed = table->prepare_append(17);
  ASSERT_TRUE(crossed.ok());
  EXPECT_EQ(crossed->visible_handle_count_after_commit, 2);
  ASSERT_TRUE(table->commit_append(crossed.value()).ok());
  EXPECT_EQ(table->visible_handles().size(), 2);
}

TEST(QwenKvBlockTableTest, RollbackLeavesCommittedViewByteStable) {
  const auto all = handles();
  auto table = QwenKvBlockTable::Create(5, 8, 33, all).value();
  auto initial = table.prepare_append(16).value();
  ASSERT_TRUE(table.commit_append(initial).ok());
  const auto generation = table.descriptor().block_table_generation;
  const auto committed = table.descriptor().committed_tokens;
  auto tentative = table.prepare_append(33).value();
  ASSERT_TRUE(table.rollback_append(tentative).ok());
  EXPECT_EQ(table.descriptor().block_table_generation, generation);
  EXPECT_EQ(table.descriptor().committed_tokens, committed);
  EXPECT_EQ(table.visible_handles().size(), 1);
}

TEST(QwenKvBlockTableTest, RejectsStalePlansAndMalformedReservations) {
  const auto all = handles();
  EXPECT_FALSE(QwenKvBlockTable::Create(5, 8, 33, {all.data(), 2}).ok());
  auto duplicate = all;
  duplicate[1].slot = duplicate[0].slot;
  EXPECT_FALSE(QwenKvBlockTable::Create(5, 8, 33, duplicate).ok());

  auto table = QwenKvBlockTable::Create(5, 8, 33, all).value();
  auto stale = table.prepare_append(16).value();
  ASSERT_TRUE(table.commit_append(stale).ok());
  EXPECT_FALSE(table.commit_append(stale).ok());
  EXPECT_FALSE(table.prepare_append(16).ok());
  EXPECT_FALSE(table.prepare_append(34).ok());
}

}  // namespace
}  // namespace pih
