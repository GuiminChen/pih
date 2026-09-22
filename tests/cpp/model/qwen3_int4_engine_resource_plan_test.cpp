#include "pih/model/qwen3_int4_engine_resource_plan.h"
#include <gtest/gtest.h>
namespace pih{namespace{
TEST(QwenInt4EngineResourcePlanTest, FreezesCompactDeviceAndStartupPinnedPeaks){
 auto plan=QwenInt4EngineResourcePlan::Create(4096,40960,2560,UINT64_C(64)<<20);
 ASSERT_TRUE(plan.ok())<<plan.status().message();
 EXPECT_EQ(plan->resident_weight_bytes(),QwenInt4ArtifactLayout::kOfficialLogicalPayloadBytes);
 EXPECT_EQ(plan->canonical_pinned_bytes(),QwenInt4ArtifactLayout::kOfficialFileBytes);
 EXPECT_EQ(plan->steady_pinned_bytes(),plan->runtime().step_staging_bytes()+plan->runtime().pinned_result_bytes());
 EXPECT_EQ(plan->startup_pinned_peak_bytes(),plan->steady_pinned_bytes()+plan->canonical_pinned_bytes());
 EXPECT_EQ(plan->total_device_bytes(),plan->runtime().total_device_bytes()-
   plan->runtime().resident_weight_bytes()+plan->resident_weight_bytes());
 EXPECT_LT(plan->total_device_bytes(),UINT64_C(24)<<30);
}
TEST(QwenInt4EngineResourcePlanTest, PreservesRuntimeAdmissionBounds){
 EXPECT_FALSE(QwenInt4EngineResourcePlan::Create(0,40960,2560,0).ok());
 EXPECT_FALSE(QwenInt4EngineResourcePlan::Create(4096,40960,2559,0).ok());
}
TEST(QwenInt4EngineResourcePlanTest, FundsFusedContinuousBatchResources){
 auto single=QwenInt4EngineResourcePlan::Create(
   4096,40960,2560,UINT64_C(64)<<20,1).value();
 auto packed=QwenInt4EngineResourcePlan::Create(
   4096,40960,2560,UINT64_C(64)<<20,32);
 ASSERT_TRUE(packed.ok())<<packed.status().message();
 EXPECT_EQ(packed->runtime().maximum_batch_sequences(),32U);
 EXPECT_GT(packed->steady_pinned_bytes(),single.steady_pinned_bytes());
 EXPECT_GT(packed->runtime().sampled_token_bytes(),
           single.runtime().sampled_token_bytes());
 EXPECT_LT(packed->total_device_bytes(),UINT64_C(24)<<30);
}
TEST(QwenInt4EngineResourcePlanTest, AdmitsOnlyFullyFundedStartupPeak){
 auto plan=QwenInt4EngineResourcePlan::Create(4096,40960,2560,UINT64_C(64)<<20).value();
 EXPECT_TRUE(plan.admit(UINT64_C(24)<<30,UINT64_C(2)<<30,UINT64_C(1)<<30).ok());
 EXPECT_FALSE(plan.admit(plan.total_device_bytes(),UINT64_C(2)<<30,1).ok());
 EXPECT_FALSE(plan.admit(UINT64_C(24)<<30,plan.startup_pinned_peak_bytes()-1,0).ok());
 EXPECT_FALSE(plan.admit(UINT64_C(24)<<30,UINT64_C(2)<<30,UINT64_MAX).ok());
}
}}
