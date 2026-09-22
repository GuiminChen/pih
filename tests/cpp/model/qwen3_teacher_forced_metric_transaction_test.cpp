#include <array>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_metric_transaction.h"

namespace pih { namespace {
struct Trace { std::vector<char> values; };
CudaCopyEndpoint ep(std::uintptr_t p,std::uint64_t n,std::uint64_t o,CudaCopyMemoryType t){return {p,n,0,o,3,t,0,0};}
TensorView tv(std::uintptr_t p,DType d,std::span<const std::int64_t> s){return TensorView::Create(reinterpret_cast<void*>(p),d,s,{},Device::Create(DeviceType::kCuda,0).value(),3).value();}
ResolvedKernelFunction fn(){std::string d(64,'c');auto m=qwen_bf16_kernel_manifest(QwenBf16Primitive::kTeacherForcedMetric,d).value();return {std::string(qwen_bf16_kernel_symbol(QwenBf16Primitive::kTeacherForcedMetric).value()),std::string(m.logical_id()),d,std::string(m.parameter_abi_sha256()),73};}
class Copies:public TypedCopyDriver{public:Copies(Trace& t):t(&t){}std::uintptr_t context_identity()const noexcept override{return 17;}Status copy(CudaCopyKind k,std::uintptr_t,std::uintptr_t,std::uint64_t,DriverStreamHandle)override{t->values.push_back(k==CudaCopyKind::kHostToDevice?'U':'D');return Status::Ok();}Trace* t;};
class Clear:public QwenBf16DeviceErrorClearDriver{public:Clear(Trace&t):t(&t){}Status clear_u32_async(const TensorView&,std::int32_t,DriverStreamHandle)override{t->values.push_back('C');return Status::Ok();}Trace*t;};
class Kernels:public KernelLaunchDriver{public:Kernels(Trace&t):t(&t){}Status launch(DriverFunctionHandle,const KernelLaunchGeometry&,DriverStreamHandle,void**)override{t->values.push_back('K');return Status::Ok();}Trace*t;};
class Events:public CompletionEventDriver{public:Events(Trace&t):t(&t){}Status record(DriverEventHandle,DriverStreamHandle)override{t->values.push_back('E');return Status::Ok();}Result<CudaEventQueryResult> query(DriverEventHandle)override{return ready?CudaEventQueryResult::kSuccess:CudaEventQueryResult::kNotReady;}Trace*t;bool ready=false;};
class Health:public QwenBf16StepHealthProvider{public:Result<QwenBf16StepHealth> collect()override{return QwenBf16StepHealth{true,false};}};

TEST(QwenTeacherForcedMetricTransactionTest, OrdersAndAuthorizesCompactPublication){
  auto layout=QwenTeacherForcedMetricResultLayout::Create(1).value();
  alignas(256) std::array<std::byte,1024> result{};
  auto transfer=QwenTeacherForcedMetricTransfer::Create(
      layout,1,
      ep(0x080000,4,5,CudaCopyMemoryType::kRegisteredPinnedHost),
      ep(0x180000,4,6,CudaCopyMemoryType::kDevice),
      ep(0x100000,4,1,CudaCopyMemoryType::kRegisteredPinnedHost),
      ep(0x200000,4,2,CudaCopyMemoryType::kDevice),
      ep(0x300000,1024,3,CudaCopyMemoryType::kDevice),
      ep(reinterpret_cast<std::uintptr_t>(result.data()),1024,4,
         CudaCopyMemoryType::kRegisteredPinnedHost),
      17,19,23,100,101,102).value();
  std::array<std::int64_t,2> ls{1,151936};std::array<std::int64_t,1> one{1};
  auto error=tv(0x300000+layout.device_error().offset_bytes,DType::kUInt32,one);
  auto plan=QwenTeacherForcedMetricPlan::Create(fn(),tv(0x400000,DType::kFloat32,ls),tv(0x200000,DType::kUInt32,one),tv(0x300000,DType::kUInt32,one),tv(0x300000+layout.target_nll().offset_bytes,DType::kFloat64,one),tv(0x300000+layout.nonfinite_rows().offset_bytes,DType::kUInt32,one),error,0).value();
  auto slot=CompletionEventSlot::Create(31,17).value();auto frontier=CudaCompletionFrontier::Create({1,0,2,CudaCompletionPhase::kPrefill,3},23,100,200).value();Health health;
  auto tx=QwenTeacherForcedMetricTransaction::Create(std::move(transfer),std::move(plan),error,layout,std::move(slot),std::move(frontier),result,19,23,0,health).value();
  Trace trace;Copies copies(trace);Clear clear(trace);Kernels kernels(trace);Events events(trace);
  ASSERT_TRUE(tx.submit(copies,clear,kernels,events).ok());EXPECT_EQ(trace.values,(std::vector<char>{'U','U','C','K','D','E'}));EXPECT_EQ(tx.poll(events).status().code(),StatusCode::kUnavailable);
  std::uint32_t zero=0,token=7;double nll=1.25;std::memcpy(result.data()+layout.device_error().offset_bytes,&zero,4);std::memcpy(result.data()+layout.argmax_tokens().offset_bytes,&token,4);std::memcpy(result.data()+layout.target_nll().offset_bytes,&nll,8);std::memcpy(result.data()+layout.nonfinite_rows().offset_bytes,&zero,4);events.ready=true;
  auto parsed=tx.poll(events);ASSERT_TRUE(parsed.ok())<<parsed.status().message();EXPECT_DOUBLE_EQ(parsed->nll_sum,1.25);EXPECT_EQ(parsed->rows[0].argmax_token,7U);EXPECT_TRUE(tx.release_completion().ok());
}
} }  // namespace pih
