#include "pih/model/deepseek_compressor_projection_device_resources.h"

#include "pih/core/checked_math.h"

namespace pih { namespace {

constexpr std::uint64_t kMainProjectionWidth = 1024;
constexpr std::uint64_t kMainOutputWidth = 512;
constexpr std::uint64_t kIndexProjectionWidth = 256;
constexpr std::uint64_t kIndexOutputWidth = 128;
constexpr std::uint64_t kIndexWeightsWidth = 64;
constexpr std::uint64_t kFloatsPerQuery =
    2 * kMainProjectionWidth + kMainOutputWidth +
    2 * kIndexProjectionWidth + kIndexOutputWidth + kIndexWeightsWidth;

}  // namespace

Result<DeepSeekCompressorProjectionDeviceResources>
DeepSeekCompressorProjectionDeviceResources::Allocate(
    Allocator& allocator, std::uint32_t maximum_queries,
    std::uint64_t context_identity, std::int32_t device_ordinal) {
  if (maximum_queries == 0 || maximum_queries > 4096 ||
      context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument(
        "DeepSeek compressor projection resource identity is invalid");
  }
  auto elements = checked_mul_u64(maximum_queries, kFloatsPerQuery);
  if (!elements.ok()) return elements.status();
  auto bytes = checked_mul_u64(*elements, sizeof(float));
  if (!bytes.ok()) return bytes.status();
  auto storage = Buffer::Allocate(allocator, *bytes, 256);
  if (!storage.ok()) return storage.status();
  if (storage->data() == nullptr || storage->size_bytes() != *bytes ||
      storage->generation() == 0 ||
      storage->device().type() != DeviceType::kCuda ||
      storage->device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek compressor allocator returned invalid device storage");
  }
  return DeepSeekCompressorProjectionDeviceResources(
      std::move(*storage), maximum_queries, context_identity, device_ordinal);
}

Result<DeepSeekCompressorProjectionSlice>
DeepSeekCompressorProjectionDeviceResources::slice(
    std::uint32_t query_ordinal) const {
  if (query_ordinal >= maximum_queries_ || storage_.data() == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek compressor projection query ordinal is invalid");
  }
  const auto base = reinterpret_cast<std::uintptr_t>(storage_.data());
  const auto section = [this, base](std::uint64_t preceding_width,
                                    std::uint64_t row_width,
                                    std::uint32_t row) {
    return base + (preceding_width * maximum_queries_ +
                   static_cast<std::uint64_t>(row) * row_width) *
                      sizeof(float);
  };
  std::uint64_t preceding = 0;
  DeepSeekCompressorProjectionSlice result;
  result.main_kv_f32 = section(preceding, kMainProjectionWidth, query_ordinal);
  preceding += kMainProjectionWidth;
  result.main_gate_f32 = section(preceding, kMainProjectionWidth, query_ordinal);
  preceding += kMainProjectionWidth;
  result.main_output_f32 = section(preceding, kMainOutputWidth, query_ordinal);
  preceding += kMainOutputWidth;
  result.index_kv_f32 = section(preceding, kIndexProjectionWidth, query_ordinal);
  preceding += kIndexProjectionWidth;
  result.index_gate_f32 = section(preceding, kIndexProjectionWidth, query_ordinal);
  preceding += kIndexProjectionWidth;
  result.index_output_f32 = section(preceding, kIndexOutputWidth, query_ordinal);
  preceding += kIndexOutputWidth;
  result.index_weights_f32 = section(preceding, kIndexWeightsWidth, query_ordinal);
  return result;
}

}  // namespace pih
