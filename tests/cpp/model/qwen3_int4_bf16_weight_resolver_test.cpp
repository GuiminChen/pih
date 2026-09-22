#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "pih/model/qwen3_int4_bf16_weight_resolver.h"

namespace pih {
namespace {

QwenBf16CommandBuffer commands() {
  Qwen3Config config{1024,3072,28,16,8,128,151936,40960,1'000'000.0,
                     0.000001,151643,151645};
  auto schedule=QwenBf16ExecutionSchedule::Create(config).value();
  auto bindings=QwenBf16WeightBindingPlan::Create(schedule).value();
  return QwenBf16CommandBuffer::Create(schedule,bindings).value();
}

QwenInt4WeightResourceSet weights() {
  auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto ledger=QwenInt4LinearShapeLedger::CreateOfficial().value();
  const std::array<std::int64_t,1> shape{
      static_cast<std::int64_t>(layout.logical_payload_bytes())};
  auto backing=TensorView::Create(reinterpret_cast<void*>(0x20000000000ULL),
      DType::kUInt8,shape,{},Device::Create(DeviceType::kCuda,0).value(),17)
      .value();
  return QwenInt4WeightResourceSet::Create(layout,ledger,backing,0).value();
}

TEST(QwenInt4Bf16WeightResolverTest, ResolvesExactly115RetainedRoles) {
  auto c=commands(); auto w=weights();
  std::size_t retained=0,quantized=0;
  for(const auto& command:c) {
    if(!command.has_weight())continue;
    auto resolved=QwenInt4Bf16WeightResolver::Resolve(command,w);
    if(command.backend==QwenBf16CommandBackend::kLinear &&
       command.linear_kind!=QwenBf16LinearKind::kLmHead) {
      EXPECT_FALSE(resolved.ok()); ++quantized;
    } else {
      ASSERT_TRUE(resolved.ok())<<resolved.status().message();
      EXPECT_EQ(resolved->dtype(),DType::kBFloat16); ++retained;
    }
  }
  EXPECT_EQ(retained,115U); EXPECT_EQ(quantized,196U);
}

TEST(QwenInt4Bf16WeightResolverTest, TiedHeadReusesEmbeddingPointer) {
  auto c=commands(); auto w=weights();
  const TensorView* embedding=nullptr; const TensorView* head=nullptr;
  TensorView embedding_value=w.view("model.embed_tokens.weight").value();
  TensorView head_value=w.view("lm_head.weight").value();
  for(const auto& command:c) {
    if(!command.has_weight())continue;
    if(command.execution_step.operation==QwenBf16ExecutionOp::kEmbedding)
      embedding=&embedding_value;
    if(command.execution_step.operation==QwenBf16ExecutionOp::kLmHead)
      head=&head_value;
  }
  ASSERT_NE(embedding,nullptr); ASSERT_NE(head,nullptr);
  EXPECT_EQ(embedding->data(),head->data());
}

TEST(QwenInt4Bf16WeightResolverTest, RejectsNoWeightAndIndexDrift) {
  auto c=commands(); auto w=weights();
  auto no_weight=c[1]; no_weight.tensor_index=QwenBf16PreparedCommand::kNoTensor;
  EXPECT_FALSE(QwenInt4Bf16WeightResolver::Resolve(no_weight,w).ok());
  auto invalid=c[0]; invalid.tensor_index=Qwen3Manifest::kOfficialTensorCount;
  EXPECT_FALSE(QwenInt4Bf16WeightResolver::Resolve(invalid,w).ok());
}

}  // namespace
}  // namespace pih
