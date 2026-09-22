#include "pih/model/deepseek_dense_mhc_stage_submission_assembler.h"

#include <gtest/gtest.h>
#include <cstdlib>
#include <vector>
#if !defined(_WIN32)
static void* _aligned_malloc(std::size_t bytes, std::size_t alignment) {
  void* result = nullptr;
  return posix_memalign(&result, alignment, bytes) == 0 ? result : nullptr;
}
static void _aligned_free(void* pointer) { std::free(pointer); }
#endif

namespace pih {
namespace {

class StageDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("stage device");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 2).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

class StageRopeOperations final : public DeepSeekRopeTableOperations {
 public:
  bool completion_proven = false;
  std::vector<std::uintptr_t> synchronized;
  Status synchronize(std::uintptr_t stream) override {
    synchronized.push_back(stream);
    return completion_proven ? Status::Ok() : Status::Unavailable("test completion pending");
  }
  Status initialize(DeepSeekRopeTableLaunch launch) override {
    return validate_deepseek_rope_table_launch(launch);
  }
};

DeepSeekAttentionWeightBindings attention_weights(std::uintptr_t base) {
  return {base + 1, base + 2, base + 3, base + 4, base + 5,
          base + 6, base + 7, base + 8, base + 9, base + 10,
          base + 11, base + 12, base + 13, 81};
}
DeepSeekMhcWeightBindings mhc_weights(std::uintptr_t base) {
  return {base + 1, base + 2, base + 3, base + 4,
          base + 5, base + 6, base + 7, base + 8, 81};
}

TEST(DeepSeekDenseMhcStageSubmissionAssemblerTest,
     ChainsExactOwnedLayersThroughOneResidualCarrier) {
  StageDeviceAllocator allocator;
  auto attention = DeepSeekAttentionProjectionDeviceResources::Allocate(
      allocator, 8, 91, 2).value();
  auto mhc = DeepSeekMhcDeviceResources::Allocate(
      allocator, 8, 91, 2).value();
  StageRopeOperations rope_operations;
  rope_operations.completion_proven = true;
  auto rope = DeepSeekRopeTableDeviceResources::AllocateDeferred(
      allocator, {0, {7, 8}, false, false, false}, 104, 0x903,
      rope_operations, 91, 2).value();
  ASSERT_TRUE(rope.ensure_initialized().ok());
  const std::array layers{
      DeepSeekDenseMhcLayerWeightInput{7, attention_weights(0x100),
                                      mhc_weights(0x200)},
      DeepSeekDenseMhcLayerWeightInput{8, attention_weights(0x300),
                                      mhc_weights(0x400)}};
  auto assembled = DeepSeekDenseMhcStageSubmissionAssembler::Assemble(
      {7, 8}, layers, attention, mhc, rope, 4, 0x901,
      104, 0x903);
  ASSERT_TRUE(assembled.ok()) << assembled.status().message();
  ASSERT_EQ(assembled->layers.size(), 2U);
  EXPECT_EQ(assembled->layers[0].mhc_attention.residual_bf16, 0x901U);
  EXPECT_EQ(assembled->layers[1].mhc_attention.residual_bf16,
            mhc.view().residual_a_bf16);
  EXPECT_EQ(assembled->final_residual_bf16,
            mhc.view().residual_a_bf16);
  EXPECT_EQ(assembled->layers[0].layer, 7U);
  EXPECT_EQ(assembled->layers[1].layer, 8U);
  EXPECT_EQ(assembled->layers[0].attention_input.q_rope.frequencies_f32,
            rope.view().yarn_f32);
}

TEST(DeepSeekDenseMhcStageSubmissionAssemblerTest,
     RejectsMissingReorderedOrMixedGenerationCoverage) {
  StageDeviceAllocator allocator;
  auto attention = DeepSeekAttentionProjectionDeviceResources::Allocate(
      allocator, 8, 91, 2).value();
  auto mhc = DeepSeekMhcDeviceResources::Allocate(
      allocator, 8, 91, 2).value();
  StageRopeOperations rope_operations;
  rope_operations.completion_proven = true;
  auto rope = DeepSeekRopeTableDeviceResources::AllocateDeferred(
      allocator, {0, {7, 8}, false, false, false}, 104, 0x903,
      rope_operations, 91, 2).value();
  ASSERT_TRUE(rope.ensure_initialized().ok());
  std::array layers{
      DeepSeekDenseMhcLayerWeightInput{8, attention_weights(0x100),
                                      mhc_weights(0x200)},
      DeepSeekDenseMhcLayerWeightInput{7, attention_weights(0x300),
                                      mhc_weights(0x400)}};
  EXPECT_FALSE(DeepSeekDenseMhcStageSubmissionAssembler::Assemble(
                   {7, 8}, layers, attention, mhc, rope, 4, 0x901,
                   104, 0x903)
                   .ok());
  layers[0].layer = 7;
  layers[1].layer = 8;
  layers[1].mhc.generation = 82;
  EXPECT_FALSE(DeepSeekDenseMhcStageSubmissionAssembler::Assemble(
                   {7, 8}, layers, attention, mhc, rope, 4, 0x901,
                   104, 0x903)
                   .ok());
}

}  // namespace
}  // namespace pih
