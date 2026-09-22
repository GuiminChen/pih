#include "pih/model/deepseek_attention_page_pool.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekAttentionPagePoolTest, EncodesTypedEightByteHandles) {
  auto handle = DeepSeekBlockHandle::Create(
      DeepSeekStatePoolKind::kRatio4MainBf16, 0x123456, 0x89ABCDEF);
  ASSERT_TRUE(handle.ok());
  EXPECT_EQ(sizeof(*handle), 8U);
  EXPECT_EQ(handle->pool_kind(), DeepSeekStatePoolKind::kRatio4MainBf16);
  EXPECT_EQ(handle->slot(), 0x123456U);
  EXPECT_EQ(handle->generation(), 0x89ABCDEFU);
  auto recent = DeepSeekBlockHandle::Create(
      DeepSeekStatePoolKind::kRecentBf16, 0, 1);
  ASSERT_TRUE(recent.ok());
  EXPECT_EQ(recent->pool_kind(), DeepSeekStatePoolKind::kRecentBf16);
  EXPECT_FALSE(DeepSeekBlockHandle::Create(
                   static_cast<DeepSeekStatePoolKind>(5), 0, 1)
                   .ok());
}

TEST(DeepSeekAttentionPagePoolTest, PublishesAndReleasesPairsAtomically) {
  auto pool = DeepSeekRatio4PagePool::Create(2);
  ASSERT_TRUE(pool.ok());
  auto pair = pool->reserve(7, 2, 11);
  ASSERT_TRUE(pair.ok());
  EXPECT_EQ(pair->main.slot(), pair->index.slot());
  EXPECT_EQ(pair->main.generation(), pair->index.generation());
  EXPECT_EQ(pool->reserved_pairs(), 1U);
  EXPECT_TRUE(pool->publish(*pair).ok());
  EXPECT_EQ(pool->published_pairs(), 1U);
  EXPECT_TRUE(pool->release(*pair).ok());
  EXPECT_EQ(pool->free_pairs(), 2U);
  EXPECT_FALSE(pool->publish(*pair).ok());
}

TEST(DeepSeekAttentionPagePoolTest, RollbackNeverPublishesHalfPair) {
  auto pool = DeepSeekRatio4PagePool::Create(1);
  ASSERT_TRUE(pool.ok());
  auto first = pool->reserve(3, 2, 5);
  ASSERT_TRUE(first.ok());
  EXPECT_FALSE(pool->reserve(3, 2, 5).ok());
  auto corrupted = *first;
  corrupted.index.value ^= 1U;
  EXPECT_FALSE(pool->publish(corrupted).ok());
  EXPECT_EQ(pool->reserved_pairs(), 1U);
  EXPECT_EQ(pool->published_pairs(), 0U);
  EXPECT_TRUE(pool->rollback(*first).ok());
  auto second = pool->reserve(3, 2, 5);
  ASSERT_TRUE(second.ok());
  EXPECT_NE(second->generation, first->generation);
  EXPECT_FALSE(pool->publish(*first).ok());
}

TEST(DeepSeekAttentionPagePoolTest, ExhaustionDoesNotCreatePartialCredit) {
  auto pool = DeepSeekRatio4PagePool::Create(1);
  ASSERT_TRUE(pool.ok());
  ASSERT_TRUE(pool->reserve(1, 2, 0).ok());
  EXPECT_FALSE(pool->reserve(2, 2, 0).ok());
  EXPECT_EQ(pool->free_pairs(), 0U);
  EXPECT_EQ(pool->reserved_pairs(), 1U);
}

TEST(DeepSeekAttentionPagePoolTest, TailCowKeepsCommittedPairUntilSwap) {
  auto pool = DeepSeekRatio4PagePool::Create(2);
  ASSERT_TRUE(pool.ok());
  auto committed = pool->reserve(9, 2, 17);
  ASSERT_TRUE(committed.ok());
  ASSERT_TRUE(pool->publish(*committed).ok());
  auto replacement = pool->reserve_tail_cow(*committed);
  ASSERT_TRUE(replacement.ok());
  EXPECT_EQ(pool->published_pairs(), 1U);
  EXPECT_EQ(pool->reserved_pairs(), 1U);

  auto damaged = *replacement;
  damaged.index.value ^= 1U;
  EXPECT_FALSE(pool->publish_tail_cow(*committed, damaged).ok());
  EXPECT_EQ(pool->published_pairs(), 1U);
  EXPECT_EQ(pool->reserved_pairs(), 1U);
  EXPECT_TRUE(pool->publish_tail_cow(*committed, *replacement).ok());
  EXPECT_EQ(pool->published_pairs(), 1U);
  EXPECT_EQ(pool->reserved_pairs(), 0U);
  EXPECT_EQ(pool->free_pairs(), 1U);
  EXPECT_FALSE(pool->release(*committed).ok());
  EXPECT_TRUE(pool->release(*replacement).ok());
}

TEST(DeepSeekAttentionPagePoolTest, TailCowRollbackLeavesOldPairPublished) {
  auto pool = DeepSeekRatio4PagePool::Create(2);
  ASSERT_TRUE(pool.ok());
  auto committed = pool->reserve(9, 2, 17);
  ASSERT_TRUE(committed.ok());
  ASSERT_TRUE(pool->publish(*committed).ok());
  auto replacement = pool->reserve_tail_cow(*committed);
  ASSERT_TRUE(replacement.ok());
  EXPECT_TRUE(pool->rollback(*replacement).ok());
  EXPECT_EQ(pool->published_pairs(), 1U);
  EXPECT_EQ(pool->free_pairs(), 1U);
  EXPECT_TRUE(pool->release(*committed).ok());
}

TEST(DeepSeekAttentionPagePoolTest, Ratio128UsesIndependentSingleHandles) {
  auto pool = DeepSeekRatio128PagePool::Create(2);
  ASSERT_TRUE(pool.ok());
  auto page = pool->reserve(5, 3, 3);
  ASSERT_TRUE(page.ok());
  EXPECT_EQ(page->handle.pool_kind(),
            DeepSeekStatePoolKind::kRatio128MainBf16);
  EXPECT_TRUE(pool->publish(*page).ok());
  auto replacement = pool->reserve_tail_cow(*page);
  ASSERT_TRUE(replacement.ok());
  EXPECT_TRUE(pool->publish_tail_cow(*page, *replacement).ok());
  EXPECT_EQ(pool->published_pages(), 1U);
  EXPECT_EQ(pool->free_pages(), 1U);
  EXPECT_FALSE(pool->release(*page).ok());
  EXPECT_TRUE(pool->release(*replacement).ok());
}

TEST(DeepSeekAttentionPagePoolTest, Ratio128RollbackRejectsStaleHandle) {
  auto pool = DeepSeekRatio128PagePool::Create(1);
  ASSERT_TRUE(pool.ok());
  auto first = pool->reserve(5, 3, 3);
  ASSERT_TRUE(first.ok());
  EXPECT_TRUE(pool->rollback(*first).ok());
  auto second = pool->reserve(5, 3, 3);
  ASSERT_TRUE(second.ok());
  EXPECT_NE(first->handle.generation(), second->handle.generation());
  EXPECT_FALSE(pool->publish(*first).ok());
  EXPECT_TRUE(pool->publish(*second).ok());
}

TEST(DeepSeekAttentionPagePoolTest,
     ReleasesOnlyPublishedPagesOwnedByOneSequence) {
  auto ratio4 = DeepSeekRatio4PagePool::Create(4).value();
  auto ratio128 = DeepSeekRatio128PagePool::Create(4).value();
  auto first4 = ratio4.reserve(7, 2, 0).value();
  auto other4 = ratio4.reserve(8, 2, 0).value();
  auto first128 = ratio128.reserve(7, 3, 0).value();
  auto other128 = ratio128.reserve(8, 3, 0).value();
  ASSERT_TRUE(ratio4.publish(first4).ok());
  ASSERT_TRUE(ratio4.publish(other4).ok());
  ASSERT_TRUE(ratio128.publish(first128).ok());
  ASSERT_TRUE(ratio128.publish(other128).ok());
  ASSERT_TRUE(ratio4.release_published_sequence(7).ok());
  ASSERT_TRUE(ratio128.release_published_sequence(7).ok());
  EXPECT_EQ(ratio4.published_pairs(), 1U);
  EXPECT_EQ(ratio128.published_pages(), 1U);
  EXPECT_TRUE(ratio4.release(other4).ok());
  EXPECT_TRUE(ratio128.release(other128).ok());
}

TEST(DeepSeekAttentionPagePoolTest,
     SeparatesSameSequenceLogicalPageAcrossLayers) {
  auto ratio4 = DeepSeekRatio4PagePool::Create(3).value();
  auto first = ratio4.reserve(7, 3, 0);
  auto second = ratio4.reserve(7, 4, 0);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_NE(first->main.slot(), second->main.slot());
  EXPECT_FALSE(ratio4.reserve(7, 3, 0).ok());
  EXPECT_EQ(first->layer_id, 3U);

  auto ratio128 = DeepSeekRatio128PagePool::Create(3).value();
  auto wide_first = ratio128.reserve(7, 21, 0);
  auto wide_second = ratio128.reserve(7, 22, 0);
  ASSERT_TRUE(wide_first.ok());
  ASSERT_TRUE(wide_second.ok());
  EXPECT_FALSE(ratio128.reserve(7, 21, 0).ok());
  EXPECT_EQ(wide_second->layer_id, 22U);
}

TEST(DeepSeekAttentionPagePoolTest,
     SequenceReleasePreflightRejectsOutstandingReservations) {
  auto ratio4 = DeepSeekRatio4PagePool::Create(2).value();
  auto ratio128 = DeepSeekRatio128PagePool::Create(2).value();
  ASSERT_TRUE(ratio4.reserve(7, 2, 0).ok());
  ASSERT_TRUE(ratio128.reserve(7, 3, 0).ok());
  EXPECT_FALSE(ratio4.validate_release_published_sequence(7).ok());
  EXPECT_FALSE(ratio128.validate_release_published_sequence(7).ok());
  EXPECT_FALSE(ratio4.release_published_sequence(7).ok());
  EXPECT_FALSE(ratio128.release_published_sequence(7).ok());
}

TEST(DeepSeekAttentionPagePoolTest,
     FindsPublishedTailBySequenceLayerAndLogicalPage) {
  auto ratio4 = DeepSeekRatio4PagePool::Create(3).value();
  auto pair = ratio4.reserve(7, 2, 11).value();
  EXPECT_FALSE(ratio4.find_published(7, 2, 11).ok());
  ASSERT_TRUE(ratio4.publish(pair).ok());
  auto found4 = ratio4.find_published(7, 2, 11);
  ASSERT_TRUE(found4.ok());
  ASSERT_TRUE(found4->has_value());
  EXPECT_EQ(found4->value().main, pair.main);
  EXPECT_FALSE(ratio4.find_published(7, 3, 11)->has_value());

  auto ratio128 = DeepSeekRatio128PagePool::Create(3).value();
  auto page = ratio128.reserve(7, 3, 5).value();
  EXPECT_FALSE(ratio128.find_published(7, 3, 5).ok());
  ASSERT_TRUE(ratio128.publish(page).ok());
  auto found128 = ratio128.find_published(7, 3, 5);
  ASSERT_TRUE(found128.ok());
  ASSERT_TRUE(found128->has_value());
  EXPECT_EQ(found128->value().handle, page.handle);
}

TEST(DeepSeekAttentionPagePoolTest,
     PublishedLookupRejectsInvalidIdentity) {
  auto ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  auto ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  EXPECT_EQ(ratio4.find_published(0, 2, 0).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(ratio128.find_published(7, 43, 0).status().code(),
            StatusCode::kInvalidArgument);
}

} }  // namespace pih
