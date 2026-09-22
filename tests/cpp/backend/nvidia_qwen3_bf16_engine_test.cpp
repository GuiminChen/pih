#include "pih/model/nvidia_qwen3_bf16_engine.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(NvidiaQwen3Bf16EngineTest, RejectsUnauthenticatedSnapshotBeforeCudaInitialization) {
  auto engine = NvidiaQwen3Bf16Engine::LoadPinnedSnapshot(
      {}, {}, {}, "kernels", 0, {}, {}, {}, 89);
  ASSERT_FALSE(engine.ok());
  EXPECT_EQ(engine.status().code(), StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pih
