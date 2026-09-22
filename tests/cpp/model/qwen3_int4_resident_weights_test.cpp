#include <cstdint>

#include <gtest/gtest.h>

#include "pih/model/qwen3_int4_resident_weights.h"
#include "pih/model/qwen3_int4_weight_startup.h"
#include "pih/model/qwen3_int4_engine_resident_publisher.h"

namespace pih {
namespace {

class VirtualCudaAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,std::uint64_t alignment) override {
    ++allocations; return Allocation{reinterpret_cast<void*>(0x200000000ULL),
      bytes,alignment,37,Device::Create(DeviceType::kCuda,0).value()};
  }
  void deallocate(Allocation) noexcept override { ++releases; }
  int allocations=0,releases=0;
};

class CopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 0x55; }
  Status copy(CudaCopyKind,std::uintptr_t,std::uintptr_t,std::uint64_t,
              DriverStreamHandle) override {
    ++calls; if(calls==fail_on)return Status::Unavailable("copy failure");
    return Status::Ok();
  }
  int calls=0,fail_on=0;
};

class EventDriver final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle,DriverStreamHandle) override {
    ++records; return record_result;
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    ++queries; return query_result;
  }
  int records=0,queries=0; Status record_result=Status::Ok();
  CudaEventQueryResult query_result=CudaEventQueryResult::kSuccess;
};

class Evidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override { return value; }
  CompletionPublicationEvidence value{true,0,false};
};

CudaCopyEndpoint source() {
  return {0x100000000ULL,QwenInt4ArtifactLayout::kOfficialFileBytes,0,71,73,
          CudaCopyMemoryType::kRegisteredPinnedHost,3,0};
}

Result<QwenInt4ResidentWeights> create(VirtualCudaAllocator& allocator) {
  return QwenInt4ResidentWeights::Create(
      QwenInt4ArtifactLayout::CreateOfficialPureW4().value(),
      QwenInt4LinearShapeLedger::CreateOfficial().value(),1,allocator,source(),
      3,79,0x55,19,29,31,101,1000,2000);
}

TEST(QwenInt4ResidentWeightsTest, PublishesOnlyAfterCleanEventFrontier) {
  VirtualCudaAllocator allocator;
  {
    auto resident=create(allocator); ASSERT_TRUE(resident.ok())<<resident.status().message();
    EXPECT_EQ(allocator.allocations,1); EXPECT_FALSE(resident->resources().ok());
    CopyDriver copies; EventDriver events; Evidence evidence;
    ASSERT_TRUE(resident->submit(copies,events).ok());
    EXPECT_EQ(copies.calls,506); EXPECT_EQ(events.records,1);
    EXPECT_FALSE(resident->resources().ok());
    ASSERT_TRUE(resident->poll(events,evidence).ok());
    EXPECT_EQ(resident->state(),QwenInt4ResidentWeightState::kPublished);
    auto resources=resident->resources(); ASSERT_TRUE(resources.ok());
    EXPECT_EQ(resources.value()->resident_bytes(),538'378'240U);
    EXPECT_EQ(resources.value()->owning_rank(),3);
  }
  EXPECT_EQ(allocator.releases,1);
}

TEST(QwenInt4ResidentWeightsTest, NotReadyDoesNotPublishAndCanBePolledAgain) {
  VirtualCudaAllocator allocator; auto resident=create(allocator).value();
  CopyDriver copies; EventDriver events; Evidence evidence;
  ASSERT_TRUE(resident.submit(copies,events).ok());
  events.query_result=CudaEventQueryResult::kNotReady;
  EXPECT_FALSE(resident.poll(events,evidence).ok());
  EXPECT_EQ(resident.state(),QwenInt4ResidentWeightState::kSubmitted);
  EXPECT_FALSE(resident.resources().ok());
  events.query_result=CudaEventQueryResult::kSuccess;
  EXPECT_TRUE(resident.poll(events,evidence).ok());
  EXPECT_TRUE(resident.resources().ok());
}

TEST(QwenInt4ResidentWeightsTest, CopyAndDeviceFailureNeverPublish) {
  VirtualCudaAllocator allocator;
  auto copy_failed=create(allocator).value(); CopyDriver copies; copies.fail_on=2;
  EventDriver events; Evidence evidence;
  EXPECT_FALSE(copy_failed.submit(copies,events).ok());
  EXPECT_EQ(events.records,0); EXPECT_FALSE(copy_failed.resources().ok());
  auto device_failed=create(allocator).value(); CopyDriver good;
  ASSERT_TRUE(device_failed.submit(good,events).ok());
  evidence.value.device_error_code=11;
  EXPECT_FALSE(device_failed.poll(events,evidence).ok());
  EXPECT_EQ(device_failed.state(),QwenInt4ResidentWeightState::kPoisoned);
  EXPECT_FALSE(device_failed.resources().ok());
}

TEST(QwenInt4ResidentWeightsTest, DeadlinePoisonsPendingUpload) {
  VirtualCudaAllocator allocator;auto resident=create(allocator).value();
  CopyDriver copies;EventDriver events;Evidence evidence;
  ASSERT_TRUE(resident.submit(copies,events).ok());
  events.query_result=CudaEventQueryResult::kNotReady;
  EXPECT_EQ(resident.expire(1999).code(),StatusCode::kUnavailable);
  EXPECT_FALSE(resident.expire(2000).ok());
  EXPECT_EQ(resident.state(),QwenInt4ResidentWeightState::kPoisoned);
  EXPECT_FALSE(resident.poll(events,evidence).ok());
  EXPECT_FALSE(resident.resources().ok());
}

class StartupClock final:public QwenInt4StartupClock{public:
 Result<std::uint64_t> now_ns()override{return now++;}std::uint64_t now=1500;};
class StartupWaiter final:public QwenInt4StartupWaiter{public:
 Status wait()override{++calls;return Status::Ok();}int calls=0;};
class DelayedEventDriver final:public CompletionEventDriver{public:
 Status record(DriverEventHandle,DriverStreamHandle)override{return Status::Ok();}
 Result<CudaEventQueryResult> query(DriverEventHandle)override{
  return queries++==0?CudaEventQueryResult::kNotReady:CudaEventQueryResult::kSuccess;}
 int queries=0;};

TEST(QwenInt4WeightStartupTest, WaitsAndPublishesOnlyCompletedWeights){
 VirtualCudaAllocator allocator;auto resident=create(allocator).value();
 CopyDriver copies;DelayedEventDriver events;Evidence evidence;
 StartupClock clock;StartupWaiter waiter;
 ASSERT_TRUE(QwenInt4WeightStartup::Publish(resident,copies,events,evidence,clock,waiter).ok());
 EXPECT_EQ(events.queries,2);EXPECT_EQ(waiter.calls,1);
 EXPECT_TRUE(resident.resources().ok());
 EXPECT_FALSE(QwenInt4WeightStartup::Publish(resident,copies,events,evidence,clock,waiter).ok());
}
TEST(QwenInt4EngineResidentPublisherTest, ReturnsOnlyPublishedModelPlanWeights){
 Qwen3Config config{1024,3072,28,16,8,128,151936,40960,1'000'000.0,0.000001,151643,151645};
 auto model=QwenInt4EngineModelPlan::Create(config).value();VirtualCudaAllocator allocator;
 CopyDriver copies;DelayedEventDriver events;Evidence evidence;StartupClock clock;StartupWaiter waiter;
 auto resident=QwenInt4EngineResidentPublisher::Publish(model,allocator,source(),1,3,79,
  0x55,19,29,31,101,1000,2000,copies,events,evidence,clock,waiter);
 ASSERT_TRUE(resident.ok())<<resident.status().message();
 EXPECT_TRUE(resident->resources().ok());EXPECT_EQ(events.queries,2);EXPECT_EQ(waiter.calls,1);
}

}  // namespace
}  // namespace pih
