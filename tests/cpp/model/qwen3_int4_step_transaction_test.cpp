#include "pih/model/qwen3_int4_step_transaction.h"

#include <array>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

struct Trace { std::vector<char> events; };

CudaCopyEndpoint endpoint(std::uintptr_t address,std::uint64_t bytes,
                          std::uint64_t owner,CudaCopyMemoryType type) {
  return {address,bytes,0,owner,3,type,0,0};
}

QwenBf16StepStagingLayout staging() {
  const QwenKvBlockHandle handles[]={{4,9}};
  auto table=QwenKvBlockTable::Create(2,6,16,handles).value();
  auto append=table.prepare_append(1).value();
  const std::int64_t token[]={4};
  auto input=QwenBf16StepInputPlan::Create(token,0,table,append).value();
  return QwenBf16StepStagingLayout::Create(input).value();
}

class Copies final : public TypedCopyDriver {
 public:
  explicit Copies(Trace& trace):trace_(&trace){}
  std::uintptr_t context_identity() const noexcept override{return 17;}
  Status copy(CudaCopyKind kind,std::uintptr_t,std::uintptr_t,std::uint64_t,
              DriverStreamHandle) override {
    trace_->events.push_back(kind==CudaCopyKind::kHostToDevice?'U':'D');
    return Status::Ok();
  }
 private: Trace* trace_;
};
class Compute final : public QwenInt4StepCompute {
 public:
  explicit Compute(Trace& trace):trace_(&trace){}
  Status submit(QwenBf16DeviceErrorClearDriver&,KernelLaunchDriver&,
                QwenInt4LmHeadExecutionDriver&,DriverStreamHandle) override {
    trace_->events.push_back('C'); return result;
  }
  Status result=Status::Ok();
 private: Trace* trace_;
};
class Clear final : public QwenBf16DeviceErrorClearDriver {
 public: Status clear_u32_async(const TensorView&,std::int32_t,
      DriverStreamHandle) override{return Status::Ok();}
};
class Kernels final : public KernelLaunchDriver {
 public: Status launch(DriverFunctionHandle,const KernelLaunchGeometry&,
      DriverStreamHandle,void**) override{return Status::Ok();}
};
class LmHead final : public QwenInt4LmHeadExecutionDriver {
 public: Status execute(const QwenInt4LmHeadBinding&,
      DriverStreamHandle) override{return Status::Ok();}
};
class Events final : public CompletionEventDriver {
 public:
  explicit Events(Trace& trace):trace_(&trace){}
  Status record(DriverEventHandle,DriverStreamHandle) override {
    trace_->events.push_back('E');return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override{return query_result;}
  CudaEventQueryResult query_result=CudaEventQueryResult::kNotReady;
 private: Trace* trace_;
};
class Health final : public QwenBf16StepHealthProvider {
 public: Result<QwenBf16StepHealth> collect() override {
    ++calls;return QwenBf16StepHealth{true,false};
  }
  int calls=0;
};

Result<QwenInt4StepTransaction> make_transaction(
    Compute& compute,Health& health,std::span<std::byte> result) {
  auto layout=staging();
  auto upload=QwenBf16StepUpload::Create(layout,
      endpoint(0x100000,layout.total_bytes(),11,CudaCopyMemoryType::kRegisteredPinnedHost),
      endpoint(0x200000,layout.total_bytes(),12,CudaCopyMemoryType::kDevice),
      17,19,23,100).value();
  auto readback=QwenBf16StepReadback::Create(
      endpoint(0x300000,8,13,CudaCopyMemoryType::kDevice),
      endpoint(0x400000,4,14,CudaCopyMemoryType::kDevice),
      endpoint(reinterpret_cast<std::uintptr_t>(result.data()),result.size(),15,
               CudaCopyMemoryType::kRegisteredPinnedHost),17,19,23,200).value();
  auto slot=CompletionEventSlot::Create(31,17).value();
  auto frontier=CudaCompletionFrontier::Create(
      {1,0,2,CudaCompletionPhase::kDecode,3},23,100,200).value();
  return QwenInt4StepTransaction::Create(std::move(upload),compute,
      std::move(readback),std::move(slot),std::move(frontier),result,19,23,health);
}

template<typename T>
void write(std::span<std::byte> result,QwenBf16ArenaSpan span,T value) {
  std::memcpy(result.data()+span.offset_bytes,&value,sizeof(value));
}

TEST(QwenInt4StepTransactionTest, PublishesOnlyAfterOrderedMixedStepCompletion) {
  alignas(256) std::array<std::byte,QwenBf16StepResultLayout::kTotalBytes> result{};
  Trace trace;Compute compute(trace);Health health;
  auto transaction=make_transaction(compute,health,result).value();
  Copies copies(trace);Clear clear;Kernels kernels;LmHead lm;Events events(trace);
  ASSERT_TRUE(transaction.submit(copies,clear,kernels,lm,events).ok());
  EXPECT_EQ(trace.events,(std::vector<char>{'U','U','U','U','U','C','D','D','E'}));
  EXPECT_EQ(transaction.poll(events).status().code(),StatusCode::kUnavailable);
  EXPECT_EQ(health.calls,0);
  write(std::span<std::byte>(result),QwenBf16StepResultLayout::sampled_token(),
        std::int64_t{42});
  write(std::span<std::byte>(result),QwenBf16StepResultLayout::device_error(),
        std::uint32_t{0});
  events.query_result=CudaEventQueryResult::kSuccess;
  auto token=transaction.poll(events);
  ASSERT_TRUE(token.ok());EXPECT_EQ(*token,42);EXPECT_EQ(health.calls,1);
  EXPECT_TRUE(transaction.release_completion().ok());
}

TEST(QwenInt4StepTransactionTest, ComputeFailurePoisonsBeforeReadback) {
  alignas(256) std::array<std::byte,QwenBf16StepResultLayout::kTotalBytes> result{};
  Trace trace;Compute compute(trace);compute.result=Status::Internal("injected");
  Health health;auto transaction=make_transaction(compute,health,result).value();
  Copies copies(trace);Clear clear;Kernels kernels;LmHead lm;Events events(trace);
  EXPECT_FALSE(transaction.submit(copies,clear,kernels,lm,events).ok());
  EXPECT_EQ(trace.events,(std::vector<char>{'U','U','U','U','U','C'}));
  EXPECT_EQ(transaction.state(),QwenInt4StepTransactionState::kPoisoned);
}

}  // namespace
}  // namespace pih
