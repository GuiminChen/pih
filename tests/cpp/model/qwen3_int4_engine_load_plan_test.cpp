#include "pih/model/qwen3_int4_engine_load_plan.h"

#include <gtest/gtest.h>

namespace pih { namespace {
TEST(QwenInt4EngineLoadPlanTest, FreezesAuthenticatedConfigBeforeCudaStartup) {
 constexpr std::string_view config=R"({"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"torch_dtype":"bfloat16","hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"vocab_size":151936,"max_position_embeddings":40960,"rope_theta":1000000.0,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":true,"use_cache":true,"bos_token_id":151643,"eos_token_id":151645})";
 auto plan=QwenInt4EngineLoadPlan::CreateFromJson(config,"kernels",0);
 ASSERT_TRUE(plan.ok())<<plan.status().message();
 EXPECT_EQ(plan->model().config().vocabulary_size,151936U);
 EXPECT_EQ(plan->cubin_root(),std::filesystem::path("kernels"));
 EXPECT_EQ(plan->device_ordinal(),0);
 auto model=std::move(*plan).take_model();
 EXPECT_EQ(model.config().layers,28U);
}
TEST(QwenInt4EngineLoadPlanTest, RejectsInvalidSnapshotIdentity) {
 auto plan=QwenInt4EngineLoadPlan::CreateFromJson("{}",{},0);
 ASSERT_FALSE(plan.ok());EXPECT_EQ(plan.status().code(),StatusCode::kInvalidArgument);
}
}}
