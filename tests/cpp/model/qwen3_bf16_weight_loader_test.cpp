#include "pih/model/qwen3_bf16_weight_loader.h"

#include <gtest/gtest.h>

#include "pih/model/qwen3_manifest.h"

namespace pih {
namespace {

TEST(QwenBf16WeightLoaderTest, RequirementsExactlyMirrorFrozenManifest) {
  const auto requirements = QwenBf16WeightLoader::Requirements();
  const auto& expected = Qwen3Manifest::expected_tensors();
  ASSERT_EQ(requirements.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(requirements[index].name, expected[index].name);
    EXPECT_EQ(requirements[index].dtype, DType::kBFloat16);
    EXPECT_EQ(requirements[index].shape, expected[index].shape);
  }
}

}  // namespace
}  // namespace pih
