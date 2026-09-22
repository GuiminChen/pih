#include "pih/backend/cuda/completion_event_slot.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class FakeEventDriver final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle event, DriverStreamHandle stream) override {
    ++record_calls;
    last_event = event;
    last_stream = stream;
    return record_result;
  }
  Result<CudaEventQueryResult> query(DriverEventHandle event) override {
    ++query_calls;
    last_event = event;
    if (!query_status.ok()) return query_status;
    return query_result;
  }

  Status record_result = Status::Ok();
  Status query_status = Status::Ok();
  CudaEventQueryResult query_result = CudaEventQueryResult::kSuccess;
  int record_calls = 0;
  int query_calls = 0;
  DriverEventHandle last_event = 0;
  DriverStreamHandle last_stream = 0;
};

class Evidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    ++calls;
    return result;
  }
  int calls = 0;
  CompletionPublicationEvidence result{true, 0, false};
};

CudaCompletionFrontier frontier(std::uint64_t event_generation) {
  return CudaCompletionFrontier::Create(
             {1, 0, 2, CudaCompletionPhase::kDecode, 3}, event_generation,
             100, 200)
      .value();
}

TEST(CompletionEventSlotTest, RejectsQueryBeforeRecordAndReusesAfterRelease) {
  auto slot = CompletionEventSlot::Create(41, 43).value();
  FakeEventDriver driver;
  Evidence evidence;
  auto first = frontier(5);
  EXPECT_FALSE(slot.poll(driver, first, evidence).ok());
  EXPECT_EQ(driver.query_calls, 0);
  ASSERT_TRUE(slot.record(driver, 47, 5).ok());
  EXPECT_EQ(driver.last_event, 41);
  EXPECT_EQ(driver.last_stream, 47);
  EXPECT_FALSE(slot.record(driver, 47, 6).ok());
  ASSERT_TRUE(slot.poll(driver, first, evidence).ok());
  EXPECT_EQ(slot.state(), CompletionEventSlotState::kCompleted);
  EXPECT_FALSE(slot.release(4).ok());
  ASSERT_TRUE(slot.release(5).ok());
  EXPECT_EQ(slot.state(), CompletionEventSlotState::kIdle);
  EXPECT_FALSE(slot.record(driver, 47, 5).ok());
  EXPECT_TRUE(slot.record(driver, 47, 6).ok());
}

TEST(CompletionEventSlotTest, NotReadyPreservesRecordedGeneration) {
  auto slot = CompletionEventSlot::Create(41, 43).value();
  FakeEventDriver driver;
  Evidence evidence;
  driver.query_result = CudaEventQueryResult::kNotReady;
  ASSERT_TRUE(slot.record(driver, 47, 7).ok());
  auto value = frontier(7);
  EXPECT_EQ(slot.poll(driver, value, evidence).code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(evidence.calls, 0);
  EXPECT_EQ(slot.state(), CompletionEventSlotState::kRecorded);
  EXPECT_FALSE(value.publication_authorized());
  driver.query_result = CudaEventQueryResult::kSuccess;
  EXPECT_TRUE(slot.poll(driver, value, evidence).ok());
  EXPECT_EQ(evidence.calls, 1);
}

TEST(CompletionEventSlotTest, DriverAndDeviceFailuresPermanentlyPoisonSlot) {
  FakeEventDriver driver;
  driver.record_result = Status::Internal("record failure");
  auto record_failed = CompletionEventSlot::Create(41, 43).value();
  EXPECT_FALSE(record_failed.record(driver, 47, 1).ok());
  EXPECT_EQ(record_failed.state(), CompletionEventSlotState::kPoisoned);

  driver.record_result = Status::Ok();
  Evidence evidence;
  evidence.result.device_error_code = 6;
  auto device_failed = CompletionEventSlot::Create(51, 53).value();
  ASSERT_TRUE(device_failed.record(driver, 57, 2).ok());
  auto value = frontier(2);
  EXPECT_FALSE(device_failed.poll(driver, value, evidence).ok());
  EXPECT_EQ(device_failed.state(), CompletionEventSlotState::kPoisoned);
  EXPECT_FALSE(device_failed.release(2).ok());
}

TEST(CompletionEventSlotTest, RejectsNullIdentity) {
  EXPECT_FALSE(CompletionEventSlot::Create(0, 1).ok());
  EXPECT_FALSE(CompletionEventSlot::Create(1, 0).ok());
}

}  // namespace
}  // namespace pih
