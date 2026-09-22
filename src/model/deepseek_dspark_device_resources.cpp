#include "pih/model/deepseek_dspark_device_resources.h"

#include <array>

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekDsparkDeviceResources> DeepSeekDsparkDeviceResources::Allocate(
    DeepSeekStagePlan stage, std::uint32_t maximum_tokens,
    Allocator& allocator,
    std::uint64_t context_identity, std::int32_t device_ordinal) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 || maximum_tokens == 0 ||
      maximum_tokens > 4096 || context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument("DeepSeek DSpark device identity is invalid");
  }
  if (!stage.owns_dspark) {
    return DeepSeekDsparkDeviceResources(nullptr, {}, maximum_tokens);
  }
  if (!stage.owns_lm_head || stage.layers.last_layer != 42) {
    return Status::InvalidArgument(
        "DeepSeek DSpark owner must be the final pipeline stage");
  }
  constexpr std::array<std::uint64_t, 18> bytes_per_unit{
      6U * sizeof(std::uint32_t),
      5U * 4U * 4096U * sizeof(std::uint16_t),
      12288U,
      96U,
      4096U * sizeof(std::uint16_t),
      4096U * sizeof(std::uint16_t),
      5U * 4096U * sizeof(std::uint16_t),
      5U * 4096U * sizeof(std::uint16_t),
      5U * 129280U * sizeof(float),
      5U * 129280U * sizeof(float),
      5U * 256U * sizeof(std::uint16_t),
      5U * sizeof(float),
      sizeof(std::uint32_t),
      3U * 4096U * sizeof(std::uint16_t),
      512U * sizeof(std::uint16_t),
      5U * 4U * 4096U * sizeof(std::uint16_t),
      5U * 4U * 4096U * sizeof(std::uint16_t),
      5U * sizeof(std::uint32_t)};
  constexpr std::array<bool, 18> token_scaled{
      false, false, true, true, true, true, false,
      false, false, false, false, false, false, true, true,
      false, false, false};
  std::array<std::uint64_t, bytes_per_unit.size()> offsets{};
  std::uint64_t cursor = 0;
  for (std::size_t index = 0; index < bytes_per_unit.size(); ++index) {
    auto aligned = checked_align_up_u64(cursor, 256);
    if (!aligned.ok()) return aligned.status();
    offsets[index] = *aligned;
    Result<std::uint64_t> bytes = token_scaled[index]
        ? checked_mul_u64(maximum_tokens, bytes_per_unit[index])
        : Result<std::uint64_t>(bytes_per_unit[index]);
    if (!bytes.ok()) return bytes.status();
    auto end = checked_add_u64(*aligned, *bytes);
    if (!end.ok()) return end.status();
    cursor = *end;
  }
  auto total = checked_align_up_u64(cursor, 256);
  if (!total.ok()) return total.status();
  auto backing = Buffer::Allocate(allocator, *total, 256);
  if (!backing.ok()) return backing.status();
  if (backing->data() == nullptr || backing->generation() == 0 ||
      backing->device().type() != DeviceType::kCuda ||
      backing->device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark allocator returned invalid device backing");
  }
  const auto base = reinterpret_cast<std::uintptr_t>(backing->data());
  const DeepSeekDsparkDeviceView view{
      base + offsets[0], base + offsets[1], base + offsets[2],
      base + offsets[3], base + offsets[4], base + offsets[5],
      base + offsets[6], base + offsets[7], base + offsets[8],
      base + offsets[9], base + offsets[10], base + offsets[11],
      base + offsets[12], base + offsets[13], base + offsets[14],
      base + offsets[15], base + offsets[16], base + offsets[17]};
  return DeepSeekDsparkDeviceResources(
      std::make_unique<Buffer>(std::move(*backing)), view, maximum_tokens);
}

}  // namespace pih
