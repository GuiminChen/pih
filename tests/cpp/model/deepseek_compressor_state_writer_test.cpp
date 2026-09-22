#include "pih/model/deepseek_compressor_state_writer.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {

class FixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override {
    return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class CompressorOperations final : public DeepSeekCompressorStateOperations {
 public:
  Status validate_host_error(std::uint32_t* value) override {
    validated_host = value;
    return Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t device,
                        std::uintptr_t stream) override {
    calls.push_back("zero");
    zero_device = device;
    zero_stream = stream;
    accumulated_error = 0;
    return Status::Ok();
  }
  Status pooling(DeepSeekCompressorPoolingLaunch value) override {
    calls.push_back("pooling");
    launches.push_back(value);
    accumulated_error |= device_error;
    return fail_pooling ? Status::Internal("injected pooling failure")
                        : Status::Ok();
  }
  Status projection(DeepSeekCompressorBf16ProjectionLaunch value) override {
    calls.push_back("projection");
    projections.push_back(value);
    accumulated_error |= projection_error;
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h");
    *host = accumulated_error;
    return Status::Ok();
  }

  std::uint32_t* validated_host = nullptr;
  std::vector<std::string> calls;
  std::vector<DeepSeekCompressorPoolingLaunch> launches;
  std::vector<DeepSeekCompressorBf16ProjectionLaunch> projections;
  std::uintptr_t zero_device = 0;
  std::uintptr_t zero_stream = 0;
  std::uint32_t device_error = 0;
  std::uint32_t projection_error = 0;
  std::uint32_t accumulated_error = 0;
  bool fail_pooling = false;
};

struct Fixture final {
  DeepSeekFixedStateLayout layout =
      DeepSeekFixedStateLayout::Build(std::vector<std::uint32_t>{2, 3}, false)
          .value();
  FixedOperations fixed_operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x100000, layout.total_bytes()}, {0x200000, layout.total_bytes()}, 5,
      fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 0, 0, banks, ratio4, ratio128).value();
  CompressorOperations operations;
  std::uint32_t host_error = 9;
  DeepSeekCompressorStateWriter writer =
      DeepSeekCompressorStateWriter::Create(layout, operations, &host_error)
          .value();
};

DeepSeekCompressorStateSubmission submission(std::uint32_t layer,
                                              bool indexer = false) {
  return {layer, indexer, 11, 12, 13, 14, 15, 7, 1, 511};
}

TEST(DeepSeekCompressorStateWriterTest,
     ResolvesMainAndIndexerStateInTheTentativeBank) {
  Fixture fixture;
  ASSERT_EQ(fixture.operations.validated_host, &fixture.host_error);
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.writer.launch(submission(2), fixture.transaction).ok());
  ASSERT_TRUE(
      fixture.writer.launch(submission(2, true), fixture.transaction).ok());

  ASSERT_EQ(fixture.operations.launches.size(), 2U);
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "pooling", "d2h", "pooling",
                                      "d2h"}));
  EXPECT_EQ(fixture.operations.zero_device, 15U);
  EXPECT_EQ(fixture.operations.zero_stream, 7U);
  EXPECT_EQ(fixture.operations.launches[0].kv_state_f32, 0x220000U);
  EXPECT_EQ(fixture.operations.launches[0].score_state_f32, 0x228000U);
  EXPECT_EQ(fixture.operations.launches[0].ratio, 4U);
  EXPECT_EQ(fixture.operations.launches[0].head_dim, 512U);
  EXPECT_EQ(fixture.operations.launches[1].kv_state_f32, 0x230000U);
  EXPECT_EQ(fixture.operations.launches[1].score_state_f32, 0x232000U);
  EXPECT_EQ(fixture.operations.launches[1].head_dim, 128U);
}

TEST(DeepSeekCompressorStateWriterTest,
     PreservesProjectionErrorsAndOrdersProjectionBeforePooling) {
  Fixture fixture;
  fixture.operations.projection_error = 4;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto projected = submission(2);
  projected.projection = {21, 22, 23, projected.kv_projection_f32,
                          projected.gate_projection_f32,
                          projected.device_error_flag_u32, projected.stream,
                          1, 4, 512, 4096};
  ASSERT_TRUE(fixture.writer.launch(projected, fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "projection", "pooling",
                                      "d2h"}));
  EXPECT_EQ(fixture.host_error, 4U);
}

TEST(DeepSeekCompressorStateWriterTest,
     ResolvesRatio128MainStateAndRejectsAnIndexer) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.writer.launch(submission(3), fixture.transaction).ok());
  ASSERT_EQ(fixture.operations.launches.size(), 1U);
  EXPECT_EQ(fixture.operations.launches[0].kv_state_f32, 0x254000U);
  EXPECT_EQ(fixture.operations.launches[0].score_state_f32, 0x294000U);
  EXPECT_EQ(fixture.operations.launches[0].ratio, 128U);
  EXPECT_EQ(fixture.operations.launches[0].head_dim, 512U);
  EXPECT_FALSE(fixture.writer.launch(submission(3, true), fixture.transaction)
                   .ok());
}

TEST(DeepSeekCompressorStateWriterTest,
     DeviceErrorPreventsTentativeGenerationPublication) {
  Fixture fixture;
  fixture.operations.device_error = 8;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  ASSERT_TRUE(fixture.writer.launch(submission(2), fixture.transaction).ok());
  ASSERT_TRUE(fixture.transaction.seal(7).ok());
  EXPECT_EQ(fixture.transaction.poll().value(),
            DeepSeekExpertAsyncStatus::kError);
  EXPECT_FALSE(fixture.transaction.commit().ok());
  EXPECT_EQ(fixture.banks.generation(), 1U);
}

TEST(DeepSeekCompressorStateWriterTest,
     PoolingFailurePoisonsWriterForThePrepareEpoch) {
  Fixture fixture;
  fixture.operations.fail_pooling = true;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  EXPECT_FALSE(fixture.writer.launch(submission(2), fixture.transaction).ok());
  fixture.operations.fail_pooling = false;
  EXPECT_FALSE(fixture.writer.launch(submission(2), fixture.transaction).ok());
}

TEST(DeepSeekCompressorStateWriterTest,
     RejectsBatchedWritesIntoSequencePrivateState) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto batched = submission(2);
  batched.batch_count = 2;
  EXPECT_FALSE(fixture.writer.launch(batched, fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.launches.empty());
}

} }  // namespace pih
