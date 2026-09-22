#include "pih/model/qwen3_int4_engine_bootstrap_plan.h"
#include <gtest/gtest.h>
namespace pih{namespace{
Qwen3Config config(){return {1024,3072,28,16,8,128,151936,40960,1'000'000.0,
 0.000001,151643,151645};}
TEST(QwenInt4EngineBootstrapPlanTest, AtomicallyAdmitsOfficial4090ClassPlan){
 auto plan=QwenInt4EngineBootstrapPlan::Create(config(),4096,40960,2560,
  UINT64_C(64)<<20,UINT64_C(24)<<30,UINT64_C(2)<<30,UINT64_C(1)<<30);
 ASSERT_TRUE(plan.ok())<<plan.status().message();
 EXPECT_EQ(plan->model().commands().size(),509U);
 EXPECT_LT(plan->resources().total_device_bytes(),UINT64_C(23)<<30);
}
TEST(QwenInt4EngineBootstrapPlanTest, PublishesNothingForModelOrCapacityDrift){
 auto drift=config();drift.layers=27;
 EXPECT_FALSE(QwenInt4EngineBootstrapPlan::Create(drift,4096,40960,2560,0,
  UINT64_C(24)<<30,UINT64_C(2)<<30,0).ok());
 EXPECT_FALSE(QwenInt4EngineBootstrapPlan::Create(config(),4096,40960,2560,0,
  1,UINT64_C(2)<<30,0).ok());
}
TEST(QwenInt4EngineBootstrapPlanTest, DerivesBackendIdentityFromRuntimeOwner){
 auto plan=QwenInt4EngineBootstrapPlan::Create(config(),4096,40960,2560,0,
  UINT64_C(24)<<30,UINT64_C(2)<<30,0).value();
 CudaRuntimeResourceIdentity runtime{0,0,7,0,17,19,23,29,31,37,41};
 auto identity=plan.backend_identity(runtime,5,1000);
 ASSERT_TRUE(identity.ok())<<identity.status().message();
 EXPECT_EQ(identity->epoch,5U);EXPECT_EQ(identity->context_identity,17U);
 EXPECT_EQ(identity->stream,19U);EXPECT_EQ(identity->event,23U);
 EXPECT_EQ(identity->slot_count,2560U);EXPECT_EQ(identity->owning_rank,0);
 EXPECT_FLOAT_EQ(identity->attention_scale,0.088388346F);
 runtime.rank=1;EXPECT_FALSE(plan.backend_identity(runtime,5,1000).ok());
}
TEST(QwenInt4EngineBootstrapPlanTest,ConsumesPreviouslyFrozenModelPlan){
 auto model=QwenInt4EngineModelPlan::Create(config()).value();
 EXPECT_EQ(model.config().layers,28U);
 auto plan=QwenInt4EngineBootstrapPlan::Create(std::move(model),4096,40960,2560,0,
  UINT64_C(24)<<30,UINT64_C(2)<<30,0);
 ASSERT_TRUE(plan.ok());EXPECT_EQ(plan->model().config().vocabulary_size,151936U);
}
}}
