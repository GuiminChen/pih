#include "pih/model/deepseek_compressed_page_mutation_assembler.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class FixedOps final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

struct Fixture final {
  DeepSeekFixedStateLayout layout =
      DeepSeekFixedStateLayout::Build(std::vector<std::uint32_t>{2, 3}, false)
          .value();
  FixedOps operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x100000, layout.total_bytes()}, {0x200000, layout.total_bytes()}, 8,
      operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(4).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(4).value();
};

DeepSeekCompressedLayerUpdateSubmission seed(std::uint32_t layer,
                                              std::uint32_t ratio,
                                              std::uint32_t position) {
  DeepSeekCompressedLayerUpdateSubmission result;
  result.ratio = ratio;
  result.main_state = {layer, false, 11, 12, 13, 14, 15, 7, 1, position};
  if (ratio == 4) {
    result.index_state = {layer, true, 21, 22, 23, 24, 15, 7, 1, position};
  }
  return result;
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     ReservesPairedRatio4AppendAtFirstSlot) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  auto result = DeepSeekCompressedPageMutationAssembler::BindDecode(
      seed(2, 4, 3), transaction, 31, 32, 41, 42, 1.0e-6F);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_TRUE(result->has_completed_slot);
  EXPECT_FALSE(result->ratio4_slot.tail_cow);
  EXPECT_EQ(result->ratio4_slot.target.layer_id, 2U);
  EXPECT_EQ(result->ratio4_slot.target.logical_page, 0U);
  EXPECT_EQ(result->ratio4_slot.main_compressed_f32, 14U);
  EXPECT_EQ(result->ratio4_slot.index_compressed_f32, 24U);
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     ReservesRatio4TailCowForLaterSlotInPublishedPage) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(9, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 2, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  auto result = DeepSeekCompressedPageMutationAssembler::BindDecode(
      seed(2, 4, 7), transaction, 31, 32, 41, 42, 1.0e-6F,
      &committed, nullptr);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_TRUE(result->ratio4_slot.tail_cow);
  EXPECT_EQ(result->ratio4_slot.committed.main, committed.main);
  EXPECT_NE(result->ratio4_slot.target.main, committed.main);
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     ReservesRatio128AppendAndLeavesRemainderUnmutated) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 0, 1, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  auto remainder = DeepSeekCompressedPageMutationAssembler::BindDecode(
      seed(3, 128, 126), transaction, 31, 0, 41, 0, 1.0e-6F);
  ASSERT_TRUE(remainder.ok());
  EXPECT_FALSE(remainder->has_completed_slot);
  auto boundary = DeepSeekCompressedPageMutationAssembler::BindDecode(
      seed(3, 128, 127), transaction, 31, 0, 41, 0, 1.0e-6F);
  ASSERT_TRUE(boundary.ok()) << boundary.status().message();
  EXPECT_TRUE(boundary->has_completed_slot);
  EXPECT_EQ(boundary->ratio128_slot.target.layer_id, 3U);
  EXPECT_EQ(boundary->ratio128_slot.target.logical_page, 0U);
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     RejectsMissingTailBeforeReservingAnything) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  EXPECT_FALSE(DeepSeekCompressedPageMutationAssembler::BindDecode(
      seed(2, 4, 7), transaction, 31, 32, 41, 42, 1.0e-6F).ok());
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 0U);
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     ReservesRatio128TailCowInsteadOfOverwritingPublishedTail) {
  Fixture fixture;
  auto committed = fixture.ratio128.reserve(9, 3, 0).value();
  ASSERT_TRUE(fixture.ratio128.publish(committed).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 0, 1, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  auto result = DeepSeekCompressedPageMutationAssembler::BindDecode(
      seed(3, 128, 255), transaction, 31, 0, 41, 0, 1.0e-6F,
      nullptr, &committed);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_TRUE(result->ratio128_slot.tail_cow);
  EXPECT_EQ(result->ratio128_slot.committed.handle, committed.handle);
  EXPECT_NE(result->ratio128_slot.target.handle, committed.handle);
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     ChunkReusesOneRatio4CowTargetForMultipleSlotsInTheSamePage) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(9, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates;
  for (std::uint32_t position = 7; position <= 11; ++position) {
    updates.push_back(seed(2, 4, position));
  }
  auto result = DeepSeekCompressedPageMutationAssembler::BindChunk(
      std::move(updates), transaction, 31, 32, 41, 42, 1.0e-6F);
  ASSERT_TRUE(result.ok()) << result.status().message();
  ASSERT_TRUE((*result)[0].has_completed_slot);
  ASSERT_TRUE((*result)[4].has_completed_slot);
  EXPECT_EQ((*result)[0].ratio4_slot.target.main,
            (*result)[4].ratio4_slot.target.main);
  EXPECT_EQ((*result)[0].ratio4_slot.committed.main, committed.main);
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 1U);
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     ChunkBindsTailAndNextAppendAsDistinctPages) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(9, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 2, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates;
  for (std::uint32_t position = 255; position <= 259; ++position) {
    updates.push_back(seed(2, 4, position));
  }
  auto result = DeepSeekCompressedPageMutationAssembler::BindChunk(
      std::move(updates), transaction, 31, 32, 41, 42, 1.0e-6F);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ((*result)[0].ratio4_slot.target.logical_page, 0U);
  EXPECT_EQ((*result)[4].ratio4_slot.target.logical_page, 1U);
  EXPECT_FALSE((*result)[4].ratio4_slot.tail_cow);
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 2U);
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     ChunkCapacityFailureRollsBackEarlierPageReservation) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(9, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates;
  for (std::uint32_t position = 255; position <= 259; ++position) {
    updates.push_back(seed(2, 4, position));
  }
  EXPECT_FALSE(DeepSeekCompressedPageMutationAssembler::BindChunk(
      std::move(updates), transaction, 31, 32, 41, 42, 1.0e-6F).ok());
  EXPECT_EQ(transaction.state(),
            DeepSeekAttentionSequenceTransactionState::kIdle);
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 0U);
}

TEST(DeepSeekCompressedPageMutationAssemblerTest,
     ChunkRejectsMalformedLaterTokenBeforeAnyReservation) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates;
  for (std::uint32_t position = 3; position <= 7; ++position) {
    updates.push_back(seed(2, 4, position));
  }
  updates.back().main_state.output_f32 = 0;
  EXPECT_FALSE(DeepSeekCompressedPageMutationAssembler::BindChunk(
      std::move(updates), transaction, 31, 32, 41, 42, 1.0e-6F).ok());
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 0U);
}

} }  // namespace pih
