#include "../../../plugins/offline-qwen/qwen3_int4_safetensors_source.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(QwenInt4SafetensorsSourceTest, VerifiesSealedGeometryAndPayload) {
  const std::array bytes{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  const std::vector<std::uint64_t> shape{2};
  const ImmutableTensorBytes tensor{DType::kBFloat16, shape, bytes};
  const SafetensorRecord physical{"weight", DType::kBFloat16, shape, 100, 104};
  const QwenInt4ObservedSourceRecord binding{
      "weight", DType::kBFloat16, shape, 100, 104, sha256(bytes).value()};
  auto verified = verify_qwen_int4_bound_tensor(tensor, physical, binding);
  ASSERT_TRUE(verified.ok()) << verified.status().message();
  EXPECT_EQ(verified->size(), bytes.size());
}

TEST(QwenInt4SafetensorsSourceTest, RejectsRangeShapeAndDigestMutation) {
  const std::array bytes{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
  const std::vector<std::uint64_t> shape{2};
  const ImmutableTensorBytes tensor{DType::kBFloat16, shape, bytes};
  const SafetensorRecord physical{"weight", DType::kBFloat16, shape, 100, 104};
  auto binding = QwenInt4ObservedSourceRecord{
      "weight", DType::kBFloat16, shape, 100, 104, sha256(bytes).value()};
  ++binding.file_begin;
  EXPECT_FALSE(verify_qwen_int4_bound_tensor(tensor, physical, binding).ok());
  binding.file_begin = 100;
  binding.payload_sha256.bytes[0] ^= std::byte{1};
  EXPECT_FALSE(verify_qwen_int4_bound_tensor(tensor, physical, binding).ok());
}

}  // namespace
}  // namespace pih
