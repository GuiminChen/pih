#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_int4_lm_head_materializer.h"

namespace pih {
namespace {

TensorView view(std::uintptr_t address,DType dtype,
                std::span<const std::int64_t> shape,
                std::uint64_t generation=9) {
  return TensorView::Create(reinterpret_cast<void*>(address),dtype,shape,{},
      Device::Create(DeviceType::kCuda,0).value(),generation).value();
}

QwenBf16ResourceSet activations() {
  std::vector<QwenBf16SlotResource> entries;
  entries.reserve(QwenBf16ResourceSet::kSlotCount);
  const std::array<std::int64_t,1> one{1};
  const std::array<std::int64_t,2> hidden{3,1024},logits{1,151936};
  std::uintptr_t address=0x100000000ULL;
  for(std::size_t i=0;i<QwenBf16ResourceSet::kSlotCount;++i) {
    auto slot=static_cast<QwenBf16ActivationSlot>(i);
    DType dtype=DType::kUInt8; std::span<const std::int64_t> shape=one;
    if(slot==QwenBf16ActivationSlot::kNormalized) {
      dtype=DType::kBFloat16; shape=hidden;
    } else if(slot==QwenBf16ActivationSlot::kLogits) {
      dtype=DType::kFloat32; shape=logits;
    }
    entries.push_back({slot,view(address,dtype,shape)}); address+=0x1000000;
  }
  return QwenBf16ResourceSet::Create(9,0,entries).value();
}

QwenInt4WeightResourceSet weights() {
  auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto ledger=QwenInt4LinearShapeLedger::CreateOfficial().value();
  const std::array<std::int64_t,1> shape{
      static_cast<std::int64_t>(layout.logical_payload_bytes())};
  return QwenInt4WeightResourceSet::Create(layout,ledger,
      view(0x20000000000ULL,DType::kUInt8,shape,17),0).value();
}

QwenBf16PreparedCommand lm_head_command() {
  Qwen3Config config{1024,3072,28,16,8,128,151936,40960,1'000'000.0,
                     0.000001,151643,151645};
  auto schedule=QwenBf16ExecutionSchedule::Create(config).value();
  auto bindings=QwenBf16WeightBindingPlan::Create(schedule).value();
  auto commands=QwenBf16CommandBuffer::Create(schedule,bindings).value();
  for(const auto& command:commands)
    if(command.linear_kind==QwenBf16LinearKind::kLmHead &&
       command.backend==QwenBf16CommandBackend::kLinear)return command;
  return commands[0];
}

TEST(QwenInt4LmHeadMaterializerTest, BindsLastTokenAndTiedEmbedding) {
  auto a=activations(); auto w=weights();
  auto binding=QwenInt4LmHeadMaterializer::Create(
      lm_head_command(),a,w,9);
  ASSERT_TRUE(binding.ok())<<binding.status().message();
  EXPECT_EQ(binding->input().dim(0),1U);
  EXPECT_EQ(binding->input().data(),reinterpret_cast<void*>(
      0x100000000ULL+
      static_cast<std::size_t>(QwenBf16ActivationSlot::kNormalized)*0x1000000ULL+
      2ULL*1024ULL*2ULL));
  EXPECT_EQ(binding->weight().data(),
            w.view("model.embed_tokens.weight").value().data());
  EXPECT_EQ(binding->output().dtype(),DType::kFloat32);
  EXPECT_EQ(binding->output().dim(1),151936U);
}

TEST(QwenInt4LmHeadMaterializerTest, RejectsGenerationAndCommandDrift) {
  auto a=activations(); auto w=weights(); auto command=lm_head_command();
  EXPECT_FALSE(QwenInt4LmHeadMaterializer::Create(command,a,w,8).ok());
  command.linear_kind=QwenBf16LinearKind::kQuery;
  EXPECT_FALSE(QwenInt4LmHeadMaterializer::Create(command,a,w,9).ok());
}

}  // namespace
}  // namespace pih
