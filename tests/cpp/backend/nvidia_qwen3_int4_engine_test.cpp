#include "pih/model/nvidia_qwen3_int4_engine.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(NvidiaQwen3Int4EngineTest, RejectsUnauthenticatedSnapshotBeforeCudaInitialization) {
  auto engine = NvidiaQwen3Int4Engine::LoadPinnedSnapshot(
      {}, {}, {}, "kernels", 0,
      {}, {}, {}, 89);
  ASSERT_FALSE(engine.ok());
  EXPECT_EQ(engine.status().code(), StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pih
