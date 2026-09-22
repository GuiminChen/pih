#include "pih/model/deepseek_sparse_attention_driver.h"

#include <gtest/gtest.h>

#include <algorithm>
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

class SparseOperations final : public DeepSeekSparseAttentionOperations {
 public:
  Status validate_host_staging(
      const DeepSeekSparseAttentionHostStaging&) override {
    return Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero");
    accumulated_error = 0;
    return Status::Ok();
  }
  Status copy_h2d_async(std::uintptr_t, const void*, std::size_t bytes,
                        std::uintptr_t) override {
    calls.push_back("h2d");
    copied_bytes.push_back(bytes);
    return Status::Ok();
  }
  Status attention(DeepSeekSparseAttentionLaunch value) override {
    calls.push_back("attention");
    launch = value;
    accumulated_error |= device_error;
    return fail_attention ? Status::Internal("injected attention failure")
                          : Status::Ok();
  }
  Status copy_d2h_async(void* host, std::uintptr_t, std::size_t,
                        std::uintptr_t) override {
    calls.push_back("d2h");
    *static_cast<std::uint32_t*>(host) = accumulated_error;
    return Status::Ok();
  }

  std::vector<std::string> calls;
  DeepSeekSparseAttentionLaunch launch;
  std::size_t index_bytes = 0;
  std::vector<std::size_t> copied_bytes;
  std::uint32_t device_error = 0;
  std::uint32_t accumulated_error = 0;
  bool fail_attention = false;
};

struct SparseFixture final {
  FixedOperations fixed_operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 1024}, {0x2000, 1024}, 5, fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(2).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(2).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 1, 1, banks, ratio4, ratio128).value();
  SparseOperations operations;
  std::vector<std::int32_t> host_indices = std::vector<std::int32_t>(512);
  std::vector<std::uint32_t> host_page_slots = std::vector<std::uint32_t>(8);
  std::uint32_t host_error = 0;
  DeepSeekSparseAttentionDriver driver =
      DeepSeekSparseAttentionDriver::Create(
          operations, {host_indices.data(),
                       static_cast<std::uint32_t>(host_indices.size()),
                       &host_error, host_page_slots.data(),
                       static_cast<std::uint32_t>(host_page_slots.size())}).value();
};

DeepSeekSparseAttentionSubmission submission(
    const DeepSeekSparseIndexMatrix& matrix) {
  return {&matrix, 1, 2, 3, 4, 5, 6, 7, 64, 256};
}

TEST(DeepSeekSparseAttentionDriverTest,
     OrdersIndexCopyKernelAndErrorBeforeTransactionSeal) {
  SparseFixture fixture;
  DeepSeekSparseIndexMatrix matrix{{10, 11, -1, 12}, 1, 4};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.driver.launch(submission(matrix), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "h2d", "attention", "d2h"}));
  ASSERT_EQ(fixture.operations.copied_bytes.size(), 1U);
  EXPECT_EQ(fixture.operations.copied_bytes[0], 4U * sizeof(std::int32_t));
  EXPECT_EQ(fixture.operations.launch.query_count, 1U);
  EXPECT_EQ(fixture.operations.launch.index_count, 4U);
  EXPECT_EQ(fixture.host_indices[2], -1);
  ASSERT_TRUE(fixture.transaction.seal(7).ok());
  EXPECT_EQ(*fixture.transaction.poll(), DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_TRUE(fixture.transaction.commit().ok());
}

TEST(DeepSeekSparseAttentionDriverTest,
     StagesPagedMainTableBeforeKernelLaunch) {
  SparseFixture fixture;
  DeepSeekSparseIndexMatrix matrix{{0, 128, 192}, 1, 3};
  auto value = submission(matrix);
  value.compressed_kv_bf16 = 20;
  value.device_page_slots_u32 = 21;
  value.recent_physical_offset = 0;
  value.compressed_physical_offset = 128;
  value.compressed_slot_count = 65;
  const std::vector<std::uint32_t> slots{1, 0};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.driver.launch_paged(
      value, slots, 2, fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "h2d", "h2d",
                                      "attention", "d2h"}));
  EXPECT_EQ(fixture.host_page_slots[0], 1U);
  EXPECT_EQ(fixture.host_page_slots[1], 0U);
  ASSERT_EQ(fixture.operations.copied_bytes.size(), 2U);
  EXPECT_EQ(fixture.operations.copied_bytes[1], 2U * sizeof(std::uint32_t));
  EXPECT_EQ(fixture.operations.launch.compressed_kv_bf16, 20U);
  EXPECT_EQ(fixture.operations.launch.page_slots_u32, 21U);
  EXPECT_EQ(fixture.operations.launch.logical_page_count, 2U);
  EXPECT_EQ(fixture.operations.launch.physical_page_count, 2U);
}

TEST(DeepSeekSparseAttentionDriverTest, ValidationHasNoCudaSideEffects) {
  SparseFixture fixture;
  DeepSeekSparseIndexMatrix matrix{{10, 11, -1, 12}, 1, 4};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto value = submission(matrix);
  value.index_matrix = nullptr;
  EXPECT_TRUE(fixture.driver.validate(value, matrix, fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
  EXPECT_EQ(fixture.host_error, 0U);
}

TEST(DeepSeekSparseAttentionDriverTest,
     DeviceErrorPoisonsTransactionAfterSuccessfulEvent) {
  SparseFixture fixture;
  DeepSeekSparseIndexMatrix matrix{{10, 11}, 1, 2};
  fixture.operations.device_error = 8;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.driver.launch(submission(matrix), fixture.transaction).ok());
  ASSERT_TRUE(fixture.transaction.seal(7).ok());
  EXPECT_EQ(*fixture.transaction.poll(), DeepSeekExpertAsyncStatus::kError);
  EXPECT_FALSE(fixture.transaction.commit().ok());
}

TEST(DeepSeekSparseAttentionDriverTest,
     AccumulatesErrorsAcrossLayerLaunchesInOneEpoch) {
  SparseFixture fixture;
  DeepSeekSparseIndexMatrix matrix{{10}, 1, 1};
  fixture.operations.device_error = 2;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.driver.launch(submission(matrix), fixture.transaction).ok());
  fixture.operations.device_error = 0;
  ASSERT_TRUE(fixture.driver.launch(submission(matrix), fixture.transaction).ok());
  EXPECT_EQ(std::count(fixture.operations.calls.begin(),
                       fixture.operations.calls.end(), "zero"), 1);
  EXPECT_EQ(fixture.host_error, 2U);
  ASSERT_TRUE(fixture.transaction.seal(7).ok());
  EXPECT_EQ(*fixture.transaction.poll(), DeepSeekExpertAsyncStatus::kError);
}

TEST(DeepSeekSparseAttentionDriverTest,
     RejectsStreamAndErrorBufferDriftWithinEpoch) {
  SparseFixture fixture;
  DeepSeekSparseIndexMatrix matrix{{10}, 1, 1};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto wrong_stream = submission(matrix);
  wrong_stream.stream = 8;
  EXPECT_FALSE(fixture.driver.launch(wrong_stream, fixture.transaction).ok());
  ASSERT_TRUE(fixture.driver.launch(submission(matrix), fixture.transaction).ok());
  auto wrong_error = submission(matrix);
  wrong_error.device_error_flag_u32 = 99;
  EXPECT_FALSE(fixture.driver.launch(wrong_error, fixture.transaction).ok());
}

TEST(DeepSeekSparseAttentionDriverTest, LaunchFailurePoisonsDriver) {
  SparseFixture fixture;
  DeepSeekSparseIndexMatrix matrix{{10}, 1, 1};
  fixture.operations.fail_attention = true;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  EXPECT_FALSE(fixture.driver.launch(submission(matrix), fixture.transaction).ok());
  fixture.operations.fail_attention = false;
  EXPECT_FALSE(fixture.driver.launch(submission(matrix), fixture.transaction).ok());
}

} }  // namespace pih
