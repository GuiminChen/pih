#include "pih/model/deepseek_compressed_layer_update_coordinator.h"

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

class StateOperations final : public DeepSeekCompressorStateOperations {
 public:
  explicit StateOperations(std::vector<std::string>& calls) : calls_(calls) {}
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls_.push_back("zero");
    return Status::Ok();
  }
  Status pooling(DeepSeekCompressorPoolingLaunch launch) override {
    calls_.push_back(launch.head_dim == 512 ? "main" : "index");
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    calls_.push_back("state-d2h");
    *host = 0;
    return Status::Ok();
  }
 private:
  std::vector<std::string>& calls_;
};

class PageOperations final : public DeepSeekCompressedPageOperations {
 public:
  explicit PageOperations(std::vector<std::string>& calls) : calls_(calls) {}
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls_.push_back("page-zero");
    return Status::Ok();
  }
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::size_t,
                        std::uintptr_t) override {
    calls_.push_back("cow");
    return Status::Ok();
  }
  Status store(DeepSeekCompressorBf16StoreLaunch launch) override {
    calls_.push_back(launch.head_dim == 512 ? "main-store" : "index-store");
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    calls_.push_back("page-d2h");
    *host = 0;
    return Status::Ok();
  }
 private:
  std::vector<std::string>& calls_;
};

struct Fixture final {
  std::vector<std::string> calls;
  StateOperations state_operations{calls};
  PageOperations page_operations{calls};
  DeepSeekFixedStateLayout layout =
      DeepSeekFixedStateLayout::Build(std::vector<std::uint32_t>{2, 3}, false)
          .value();
  FixedOperations fixed_operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x400000, layout.total_bytes()}, {0x500000, layout.total_bytes()}, 9,
      fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(2).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(2).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 1, 1, banks, ratio4, ratio128).value();
  DeepSeekAttentionPageArena arena = DeepSeekAttentionPageArena::Create(
      {0x100000, 2U * 65536U}, {0x200000, 2U * 16384U},
      {0x300000, 2U * 65536U}, 2, 2).value();
  std::uint32_t host_error = 0;
  DeepSeekCompressorStateWriter state_writer =
      DeepSeekCompressorStateWriter::Create(
          layout, state_operations, &host_error).value();
  DeepSeekCompressedPageWriter page_writer =
      DeepSeekCompressedPageWriter::Create(
          arena, page_operations, &host_error).value();
  DeepSeekCompressedLayerUpdateCoordinator coordinator =
      DeepSeekCompressedLayerUpdateCoordinator::Create(
          state_writer, page_writer).value();
};

DeepSeekCompressorStateSubmission state(std::uint32_t layer, bool indexer,
                                        std::uintptr_t output,
                                        std::uint32_t position) {
  return {layer, indexer, 11, 12, 13, output, 17, 7, 1, position};
}

TEST(DeepSeekCompressedLayerUpdateCoordinatorTest,
     OrdersRatio4MainIndexerAndPairedPageStores) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_append(2, 0).value();
  DeepSeekCompressedLayerUpdateSubmission submission;
  submission.ratio = 4;
  submission.main_state = state(2, false, 101, 3);
  submission.index_state = state(2, true, 102, 3);
  submission.has_completed_slot = true;
  submission.ratio4_slot = {target, {}, false, 101, 102, 21, 22, 23, 24,
                            17, 7, 3, 1.0e-6F};
  ASSERT_TRUE(fixture.coordinator.launch(submission, fixture.transaction).ok());
  EXPECT_EQ(fixture.calls,
            std::vector<std::string>({"zero", "main", "state-d2h", "index",
                                      "state-d2h", "main-store", "index-store",
                                      "page-d2h"}));
}

TEST(DeepSeekCompressedLayerUpdateCoordinatorTest,
     UpdatesFixedStateWithoutWritingBeforeRatio4Boundary) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekCompressedLayerUpdateSubmission submission;
  submission.ratio = 4;
  submission.main_state = state(2, false, 101, 2);
  submission.index_state = state(2, true, 102, 2);
  ASSERT_TRUE(fixture.coordinator.launch(submission, fixture.transaction).ok());
  EXPECT_EQ(fixture.calls,
            std::vector<std::string>({"zero", "main", "state-d2h", "index",
                                      "state-d2h"}));
}

TEST(DeepSeekCompressedLayerUpdateCoordinatorTest,
     ResolvesDeferredPublishedTailAfterTransactionBegin) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(7, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekCompressedLayerUpdateSubmission submission;
  submission.ratio = 4;
  submission.main_state = state(2, false, 101, 7);
  submission.index_state = state(2, true, 102, 7);
  submission.defer_page_mutation = true;
  submission.deferred_page = {21, 22, 23, 24, 1.0e-6F};
  ASSERT_TRUE(fixture.coordinator.launch(submission, fixture.transaction).ok());
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 1U);
  EXPECT_EQ(fixture.calls,
            std::vector<std::string>({"zero", "main", "state-d2h", "index",
                                      "state-d2h", "cow", "cow",
                                      "main-store", "index-store",
                                      "page-d2h"}));
}

TEST(DeepSeekCompressedLayerUpdateCoordinatorTest,
     RejectsPageReservedForAnotherLayerBeforeAnyWrite) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_append(3, 0).value();
  DeepSeekCompressedLayerUpdateSubmission submission;
  submission.ratio = 4;
  submission.main_state = state(2, false, 101, 3);
  submission.index_state = state(2, true, 102, 3);
  submission.has_completed_slot = true;
  submission.ratio4_slot = {target, {}, false, 101, 102, 21, 22, 23, 24,
                            17, 7, 3, 1.0e-6F};
  EXPECT_FALSE(fixture.coordinator.launch(
      submission, fixture.transaction).ok());
  EXPECT_TRUE(fixture.calls.empty());
}

TEST(DeepSeekCompressedLayerUpdateCoordinatorTest,
     ValidationHasNoCompressorOrPageSideEffects) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekCompressedLayerUpdateSubmission submission;
  submission.ratio = 4;
  submission.main_state = state(2, false, 101, 2);
  submission.index_state = state(2, true, 102, 2);
  EXPECT_TRUE(fixture.coordinator.validate(submission, fixture.transaction)
                  .ok());
  EXPECT_TRUE(fixture.calls.empty());
}

TEST(DeepSeekCompressedLayerUpdateCoordinatorTest,
     RejectsSlotPresenceThatDisagreesWithTheCompressionBoundary) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekCompressedLayerUpdateSubmission submission;
  submission.ratio = 128;
  submission.main_state = state(3, false, 101, 126);
  submission.has_completed_slot = true;
  EXPECT_FALSE(fixture.coordinator.launch(submission, fixture.transaction)
                   .ok());
  EXPECT_TRUE(fixture.calls.empty());
}

TEST(DeepSeekCompressedLayerUpdateCoordinatorTest,
     RejectsMalformedIndexerBeforeLaunchingMainStateUpdate) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekCompressedLayerUpdateSubmission submission;
  submission.ratio = 4;
  submission.main_state = state(2, false, 101, 2);
  submission.index_state = state(2, true, 102, 2);
  submission.index_state.kv_projection_f32 = 0;
  EXPECT_FALSE(fixture.coordinator.launch(submission, fixture.transaction)
                   .ok());
  EXPECT_TRUE(fixture.calls.empty());
}

} }  // namespace pih
