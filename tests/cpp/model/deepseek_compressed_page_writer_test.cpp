#include "pih/model/deepseek_compressed_page_writer.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {

class FixedOperations final : public DeepSeekFixedStateBankOperations {
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

class PageOperations final : public DeepSeekCompressedPageOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero");
    accumulated_error = 0;
    return Status::Ok();
  }
  Status copy_d2d_async(std::uintptr_t destination, std::uintptr_t source,
                        std::size_t bytes, std::uintptr_t) override {
    calls.push_back("copy");
    copies.push_back({destination, source, bytes});
    return Status::Ok();
  }
  Status store(DeepSeekCompressorBf16StoreLaunch launch) override {
    calls.push_back("store");
    stores.push_back(launch);
    accumulated_error |= device_error;
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h");
    *host = accumulated_error;
    return Status::Ok();
  }
  struct Copy final {
    std::uintptr_t destination;
    std::uintptr_t source;
    std::size_t bytes;
  };
  std::vector<std::string> calls;
  std::vector<Copy> copies;
  std::vector<DeepSeekCompressorBf16StoreLaunch> stores;
  std::uint32_t device_error = 0;
  std::uint32_t accumulated_error = 0;
};

struct Fixture final {
  FixedOperations fixed_operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x400000, 1024}, {0x500000, 1024}, 9, fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(3).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(3).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 2, 2, banks, ratio4, ratio128).value();
  DeepSeekAttentionPageArena arena = DeepSeekAttentionPageArena::Create(
      {0x100000, 3U * 65536U}, {0x200000, 3U * 16384U},
      {0x300000, 3U * 65536U}, 3, 3).value();
  PageOperations operations;
  std::uint32_t host_error = 7;
  DeepSeekCompressedPageWriter writer =
      DeepSeekCompressedPageWriter::Create(arena, operations, &host_error)
          .value();
};

DeepSeekRatio4SlotSubmission ratio4_submission(
    DeepSeekRatio4PagePair target, std::uint32_t position) {
  DeepSeekRatio4SlotSubmission value;
  value.target = target;
  value.main_compressed_f32 = 11;
  value.index_compressed_f32 = 12;
  value.main_rms_weight_bf16 = 13;
  value.index_rms_weight_bf16 = 14;
  value.main_cos_sin_cache_f32 = 15;
  value.index_cos_sin_cache_f32 = 16;
  value.device_error_flag_u32 = 17;
  value.stream = 7;
  value.absolute_position = position;
  value.rms_epsilon = 1.0e-6F;
  return value;
}

TEST(DeepSeekCompressedPageWriterTest,
     WritesPairedRatio4SlotsAtTheCompressedOrdinal) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_append(2, 0).value();
  ASSERT_TRUE(fixture.writer.write_ratio4(
      ratio4_submission(target, 7), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "store", "store", "d2h"}));
  ASSERT_EQ(fixture.operations.stores.size(), 2U);
  EXPECT_EQ(fixture.operations.stores[0].destination_bf16, 0x100400U);
  EXPECT_EQ(fixture.operations.stores[0].head_dim, 512U);
  EXPECT_EQ(fixture.operations.stores[0].rope_position, 4U);
  EXPECT_EQ(fixture.operations.stores[1].destination_bf16, 0x200100U);
  EXPECT_EQ(fixture.operations.stores[1].head_dim, 128U);
}

TEST(DeepSeekCompressedPageWriterTest,
     CopiesBothCommittedRatio4PagesBeforeTailCowWrite) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(7, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_tail_cow(committed).value();
  auto submission = ratio4_submission(target, 11);
  submission.tail_cow = true;
  submission.committed = committed;
  ASSERT_TRUE(fixture.writer.write_ratio4(submission, fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "copy", "copy", "store",
                                      "store", "d2h"}));
  ASSERT_EQ(fixture.operations.copies.size(), 2U);
  EXPECT_EQ(fixture.operations.copies[0].destination, 0x110000U);
  EXPECT_EQ(fixture.operations.copies[0].source, 0x100000U);
  EXPECT_EQ(fixture.operations.copies[0].bytes, 65536U);
  EXPECT_EQ(fixture.operations.copies[1].destination, 0x204000U);
  EXPECT_EQ(fixture.operations.copies[1].source, 0x200000U);
  EXPECT_EQ(fixture.operations.copies[1].bytes, 16384U);
}

TEST(DeepSeekCompressedPageWriterTest,
     CopiesATailCowTargetOnlyOnceBeforeMultipleSlotWrites) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(7, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_tail_cow(committed).value();
  auto first = ratio4_submission(target, 11);
  first.tail_cow = true;
  first.committed = committed;
  auto second = ratio4_submission(target, 15);
  second.tail_cow = true;
  second.committed = committed;
  ASSERT_TRUE(fixture.writer.write_ratio4(first, fixture.transaction).ok());
  ASSERT_TRUE(fixture.writer.write_ratio4(second, fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.copies.size(), 2U);
  EXPECT_EQ(fixture.operations.stores.size(), 4U);
  EXPECT_EQ(fixture.operations.stores[2].destination_bf16, 0x110C00U);
}

TEST(DeepSeekCompressedPageWriterTest,
     RejectsWrongBoundaryOrLogicalPageBeforeLaunching) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_append(2, 0).value();
  EXPECT_FALSE(fixture.writer.write_ratio4(
      ratio4_submission(target, 6), fixture.transaction).ok());
  auto wrong_page = ratio4_submission(target, 259);
  EXPECT_FALSE(fixture.writer.write_ratio4(wrong_page, fixture.transaction)
                   .ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}

TEST(DeepSeekCompressedPageWriterTest,
     DeviceErrorPreventsPageAndFixedStatePublication) {
  Fixture fixture;
  fixture.operations.device_error = 4;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_append(2, 0).value();
  ASSERT_TRUE(fixture.writer.write_ratio4(
      ratio4_submission(target, 3), fixture.transaction).ok());
  ASSERT_TRUE(fixture.transaction.seal(7).ok());
  EXPECT_EQ(fixture.transaction.poll().value(),
            DeepSeekExpertAsyncStatus::kError);
  EXPECT_FALSE(fixture.transaction.commit().ok());
  EXPECT_EQ(fixture.ratio4.published_pairs(), 0U);
}

TEST(DeepSeekCompressedPageWriterTest,
     WritesRatio128SlotUsingItsIndependentPage) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio128_append(3, 0).value();
  DeepSeekRatio128SlotSubmission submission;
  submission.target = target;
  submission.compressed_f32 = 21;
  submission.rms_weight_bf16 = 22;
  submission.cos_sin_cache_f32 = 23;
  submission.device_error_flag_u32 = 17;
  submission.stream = 7;
  submission.absolute_position = 255;
  submission.rms_epsilon = 1.0e-6F;
  ASSERT_TRUE(fixture.writer.write_ratio128(submission, fixture.transaction)
                  .ok());
  ASSERT_EQ(fixture.operations.stores.size(), 1U);
  EXPECT_EQ(fixture.operations.stores[0].destination_bf16, 0x300400U);
  EXPECT_EQ(fixture.operations.stores[0].head_dim, 512U);
  EXPECT_EQ(fixture.operations.stores[0].rope_position, 128U);
}

TEST(DeepSeekCompressedPageWriterTest,
     RejectsAValidPoolHandleNotOwnedByTheTransaction) {
  Fixture fixture;
  auto foreign = fixture.ratio4.reserve(7, 2, 1).value();
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  EXPECT_FALSE(fixture.writer.write_ratio4(
      ratio4_submission(foreign, 259), fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}

} }  // namespace pih
