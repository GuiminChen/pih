#include "pih/model/deepseek_recent_state_writer.h"

#include <gtest/gtest.h>

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

class RecentOperations final : public DeepSeekRecentStateOperations {
 public:
  Status copy_d2d_async(std::uintptr_t destination, std::uintptr_t source,
                        std::size_t bytes, std::uintptr_t stream) override {
    calls.push_back({destination, source, bytes, stream});
    return fail ? Status::Internal("injected recent write failure")
                : Status::Ok();
  }
  struct Call { std::uintptr_t destination; std::uintptr_t source;
                std::size_t bytes; std::uintptr_t stream; };
  std::vector<Call> calls;
  bool fail = false;
};

struct Fixture final {
  DeepSeekFixedStateLayout layout =
      DeepSeekFixedStateLayout::Build(
          std::vector<std::uint32_t>{0, 2}, false).value();
  FixedOperations fixed;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x100000, layout.total_bytes()}, {0x200000, layout.total_bytes()}, 5,
      fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 0, 0, banks, ratio4, ratio128).value();
  RecentOperations operations;
  DeepSeekRecentStateWriter writer =
      DeepSeekRecentStateWriter::Create(layout, operations).value();
};

TEST(DeepSeekRecentStateWriterTest, WritesOneLatentRowIntoTentativeRing) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.writer.launch({0, 0xABC000, 7, 1, 3},
                                    fixture.transaction).ok());
  ASSERT_EQ(fixture.operations.calls.size(), 1U);
  auto layer = fixture.layout.Resolve(
      0, fixture.transaction.tentative_fixed_state().value()).value();
  EXPECT_EQ(fixture.operations.calls[0].destination,
            layer.recent_bf16.address + 3U * 512U * 2U);
  EXPECT_EQ(fixture.operations.calls[0].source, 0xABC000U);
  EXPECT_EQ(fixture.operations.calls[0].bytes, 512U * 2U);
  EXPECT_EQ(fixture.operations.calls[0].stream, 7U);
}

TEST(DeepSeekRecentStateWriterTest, WrapsAtTheOneHundredTwentyEightRowBoundary) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.writer.launch({2, 0xABC000, 7, 1, 128},
                                    fixture.transaction).ok());
  auto layer = fixture.layout.Resolve(
      2, fixture.transaction.tentative_fixed_state().value()).value();
  EXPECT_EQ(fixture.operations.calls[0].destination,
            layer.recent_bf16.address);
}

TEST(DeepSeekRecentStateWriterTest, RejectsWrongStreamBatchAndUnownedLayer) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  EXPECT_FALSE(fixture.writer.launch({0, 1, 8, 1, 0}, fixture.transaction).ok());
  EXPECT_FALSE(fixture.writer.launch({0, 1, 7, 2, 0}, fixture.transaction).ok());
  EXPECT_FALSE(fixture.writer.launch({1, 1, 7, 1, 0}, fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}

TEST(DeepSeekRecentStateWriterTest, ValidationHasNoWriteSideEffects) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  EXPECT_TRUE(fixture.writer.validate({0, 1, 7, 1, 127},
                                      fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}

TEST(DeepSeekRecentStateWriterTest, OperationFailurePoisonsWriter) {
  Fixture fixture;
  fixture.operations.fail = true;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  EXPECT_FALSE(fixture.writer.launch({0, 1, 7, 1, 0}, fixture.transaction).ok());
  fixture.operations.fail = false;
  EXPECT_FALSE(fixture.writer.launch({0, 1, 7, 1, 1}, fixture.transaction).ok());
}

} }  // namespace pih
