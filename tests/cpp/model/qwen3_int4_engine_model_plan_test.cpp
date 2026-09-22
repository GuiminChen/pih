#include "pih/model/qwen3_int4_engine_model_plan.h"
#include <gtest/gtest.h>
namespace pih{namespace{
Qwen3Config official(){return {1024,3072,28,16,8,128,151936,40960,1'000'000.0,
 0.000001,151643,151645};}
TEST(QwenInt4EngineModelPlanTest, FreezesCompleteOfficialMixedGraph){
 auto plan=QwenInt4EngineModelPlan::Create(official());
 ASSERT_TRUE(plan.ok())<<plan.status().message();
 EXPECT_EQ(plan->layout().records().size(),507U);
 EXPECT_EQ(plan->ledger().records().size(),196U);
 EXPECT_EQ(plan->commands().size(),509U);
 std::size_t kernels=0,linears=0;
 for(const auto& command:plan->commands())
  command.backend==QwenBf16CommandBackend::kKernel?++kernels:++linears;
 EXPECT_EQ(kernels,312U);EXPECT_EQ(linears,197U);
}
TEST(QwenInt4EngineModelPlanTest, RejectsArchitectureDriftBeforeRuntime){
 auto config=official();config.hidden_size=2048;
 EXPECT_FALSE(QwenInt4EngineModelPlan::Create(config).ok());
}
}}
