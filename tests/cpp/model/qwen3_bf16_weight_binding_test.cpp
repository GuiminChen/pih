#include "pih/model/qwen3_bf16_weight_binding.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenBf16WeightBindingPlanTest, BindsEveryManifestTensorExactlyOnce) {
  const Qwen3Config config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                           1'000'000.0, 0.000001, 151643, 151645};
  auto schedule = QwenBf16ExecutionSchedule::Create(config);
  ASSERT_TRUE(schedule.ok());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  EXPECT_EQ(bindings->bound_weight_count(), 311);
  EXPECT_EQ(bindings->tensor(0)->name, "model.embed_tokens.weight");
  EXPECT_EQ(bindings->tensor(2)->name,
            "model.layers.0.input_layernorm.weight");
  EXPECT_EQ(bindings->tensor(3)->name,
            "model.layers.0.self_attn.q_proj.weight");
  EXPECT_EQ(bindings->tensor(478)->name, "model.norm.weight");
  EXPECT_EQ(bindings->tensor(479)->name, "lm_head.weight");
  EXPECT_EQ(bindings->tensor(480), nullptr);
}

TEST(QwenBf16WeightBindingPlanTest, LeavesWeightlessKernelsExplicitlyUnbound) {
  const Qwen3Config config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                           1'000'000.0, 0.000001, 151643, 151645};
  auto schedule = QwenBf16ExecutionSchedule::Create(config);
  ASSERT_TRUE(schedule.ok());
  auto bindings = QwenBf16WeightBindingPlan::Create(*schedule);
  ASSERT_TRUE(bindings.ok());
  EXPECT_EQ(bindings->tensor(8), nullptr);   // RoPE.
  EXPECT_EQ(bindings->tensor(9), nullptr);   // KV append.
  EXPECT_EQ(bindings->tensor(10), nullptr);  // paged GQA.
  EXPECT_EQ(bindings->tensor(12), nullptr);  // residual.
}

}  // namespace
}  // namespace pih
