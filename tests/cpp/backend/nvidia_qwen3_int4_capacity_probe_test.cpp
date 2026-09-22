#include "pih/model/nvidia_qwen3_int4_capacity_probe.h"
#include <gtest/gtest.h>
namespace pih{namespace{
Qwen3Config config(){return {1024,3072,28,16,8,128,151936,40960,1'000'000.0,0.000001,151643,151645};}
TEST(NvidiaQwenInt4CapacityProbeTest, RejectsIdentityBeforeCudaAccess){
 EXPECT_FALSE(NvidiaQwenInt4CapacityProbe::Admit(config(),-1,4096,40960,2560,0,1,0).ok());
 EXPECT_FALSE(NvidiaQwenInt4CapacityProbe::Admit(config(),0,4096,40960,2560,0,0,0).ok());
}
TEST(NvidiaQwenInt4CapacityProbeTest, AcceptsFrozenPlanBeforeCudaAccess){
 auto model=QwenInt4EngineModelPlan::Create(config()).value();
 auto admitted=NvidiaQwenInt4CapacityProbe::Admit(
  std::move(model),-1,4096,40960,2560,0,1,0);
 ASSERT_FALSE(admitted.ok());
 EXPECT_EQ(admitted.status().code(),StatusCode::kInvalidArgument);
}
}}
