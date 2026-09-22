#include "pih/model/qwen3_int4_engine_runtime_arenas.h"
#include <cstdlib>
#include <malloc.h>
#include <gtest/gtest.h>
namespace pih{namespace{
class HeapAllocator:public Allocator{public:
 explicit HeapAllocator(Device device):device_(device){}
 Result<Allocation> allocate(std::uint64_t bytes,std::uint64_t alignment)override{
  ++attempts;if(fail_on!=0&&attempts==fail_on)
   return Status::ResourceExhausted("injected device allocation failure");
  void* p=_aligned_malloc(static_cast<std::size_t>(bytes),static_cast<std::size_t>(alignment));
  if(!p)return Status::ResourceExhausted("heap");++allocs;return Allocation{p,bytes,alignment,static_cast<std::uint64_t>(allocs),device_};}
 void deallocate(Allocation a)noexcept override{_aligned_free(a.data);++frees;}
 int attempts=0,fail_on=0,allocs=0,frees=0;Device device_;
};
class PinnedHeap final:public RegisteredPinnedAllocator{public:
 Result<Allocation> allocate(std::uint64_t bytes,std::uint64_t alignment)override{
  ++attempts;if(fail_on!=0&&attempts==fail_on)
   return Status::ResourceExhausted("injected pinned allocation failure");
  void* p=_aligned_malloc(static_cast<std::size_t>(bytes),static_cast<std::size_t>(alignment));
  if(!p)return Status::ResourceExhausted("heap");++allocs;return Allocation{p,bytes,alignment,static_cast<std::uint64_t>(allocs),Device::Create(DeviceType::kCpu,0).value()};}
 void deallocate(Allocation a)noexcept override{_aligned_free(a.data);++frees;}
 int attempts=0,fail_on=0,allocs=0,frees=0;
};
class Placement final:public PinnedPlacementVerifier{public:
 Status verify(const void*,std::uint64_t,std::int32_t node)override{
  ++calls;if(fail_on!=0&&calls==fail_on)return Status::Internal("injected placement failure");
  return node==3?Status::Ok():Status::Internal("node");}
 int calls=0,fail_on=0;};
Qwen3Config config(){return {1024,3072,28,16,8,128,151936,40960,1'000'000.0,0.000001,151643,151645};}
TEST(QwenInt4EngineRuntimeArenasTest, AllocatesAdmittedDeviceAndPinnedOwners){
 auto bootstrap=QwenInt4EngineBootstrapPlan::Create(config(),4096,40960,2560,
  UINT64_C(64)<<20,UINT64_C(24)<<30,UINT64_C(2)<<30,UINT64_C(1)<<30).value();
 HeapAllocator device(Device::Create(DeviceType::kCuda,0).value());PinnedHeap pinned;Placement placement;
 {auto arenas=QwenInt4EngineRuntimeArenas::Allocate(bootstrap,device,pinned,placement,3,0);
  ASSERT_TRUE(arenas.ok())<<arenas.status().message();EXPECT_EQ(device.allocs,10);EXPECT_EQ(pinned.allocs,2);
  EXPECT_EQ(placement.calls,2);EXPECT_EQ(arenas->device().total_bytes(),bootstrap.resources().runtime().total_device_bytes()-bootstrap.resources().runtime().resident_weight_bytes());}
 EXPECT_EQ(device.frees,10);EXPECT_EQ(pinned.frees,2);
}
TEST(QwenInt4EngineRuntimeArenasTest, ProjectsExactTypedBackendEndpoints){
 auto bootstrap=QwenInt4EngineBootstrapPlan::Create(config(),4096,40960,2560,
  UINT64_C(64)<<20,UINT64_C(24)<<30,UINT64_C(2)<<30,UINT64_C(1)<<30).value();
 HeapAllocator device(Device::Create(DeviceType::kCuda,0).value());PinnedHeap pinned;Placement placement;
 auto arenas=QwenInt4EngineRuntimeArenas::Allocate(bootstrap,device,pinned,placement,3,0).value();
 auto projected=arenas.backend_arenas({101,102,103,104,105},0);
 ASSERT_TRUE(projected.ok())<<projected.status().message();
 const auto owners=arenas.device().step_owners();
 EXPECT_EQ(projected->device_staging.allocation_base,owners.step_staging.base);
 EXPECT_EQ(projected->sampled_token.allocation_base,owners.sampled_token.base);
 EXPECT_EQ(projected->device_error.allocation_base,owners.device_error.base);
 EXPECT_EQ(projected->pinned_staging.device_or_numa,3);
 EXPECT_EQ(projected->pinned_result.owner_id,105U);
 EXPECT_EQ(projected->pinned_staging_backing.data(),arenas.pinned().staging().data());
}
TEST(QwenInt4EngineRuntimeArenasTest, RollsBackEveryPartialArenaOwner){
 auto bootstrap=QwenInt4EngineBootstrapPlan::Create(config(),4096,40960,2560,
  UINT64_C(64)<<20,UINT64_C(24)<<30,UINT64_C(2)<<30,UINT64_C(1)<<30).value();
 {
  HeapAllocator device(Device::Create(DeviceType::kCuda,0).value());
  PinnedHeap pinned;Placement placement;device.fail_on=5;
  auto arenas=QwenInt4EngineRuntimeArenas::Allocate(
   bootstrap,device,pinned,placement,3,0);
  EXPECT_EQ(arenas.status().code(),StatusCode::kResourceExhausted);
  EXPECT_EQ(device.allocs,4);EXPECT_EQ(device.frees,4);
  EXPECT_EQ(pinned.allocs,0);EXPECT_EQ(pinned.frees,0);
 }
 {
  HeapAllocator device(Device::Create(DeviceType::kCuda,0).value());
  PinnedHeap pinned;Placement placement;pinned.fail_on=2;
  auto arenas=QwenInt4EngineRuntimeArenas::Allocate(
   bootstrap,device,pinned,placement,3,0);
  EXPECT_EQ(arenas.status().code(),StatusCode::kResourceExhausted);
  EXPECT_EQ(device.allocs,10);EXPECT_EQ(device.frees,10);
  EXPECT_EQ(pinned.allocs,1);EXPECT_EQ(pinned.frees,1);
 }
 {
  HeapAllocator device(Device::Create(DeviceType::kCuda,0).value());
  PinnedHeap pinned;Placement placement;placement.fail_on=2;
  auto arenas=QwenInt4EngineRuntimeArenas::Allocate(
   bootstrap,device,pinned,placement,3,0);
  EXPECT_EQ(arenas.status().code(),StatusCode::kInternal);
  EXPECT_EQ(device.allocs,10);EXPECT_EQ(device.frees,10);
  EXPECT_EQ(pinned.allocs,2);EXPECT_EQ(pinned.frees,2);
 }
}
}}
