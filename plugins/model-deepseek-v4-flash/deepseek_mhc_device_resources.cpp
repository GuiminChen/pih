#include "pih/model/deepseek_mhc_device_resources.h"

#include <array>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<std::uint64_t> reserve(std::uint64_t& cursor,
                              std::uint32_t maximum_tokens,
                              std::uint64_t bytes_per_token) {
  auto bytes = checked_mul_u64(maximum_tokens, bytes_per_token);
  if (!bytes.ok()) return bytes.status();
  auto offset = checked_align_up_u64(cursor, 256);
  if (!offset.ok()) return offset.status();
  auto end = checked_add_u64(*offset, *bytes);
  if (!end.ok()) return end.status();
  cursor = *end;
  return *offset;
}

}  // namespace

Result<DeepSeekMhcDeviceResources> DeepSeekMhcDeviceResources::Allocate(
    Allocator& allocator, std::uint32_t maximum_tokens,
    std::uint64_t context_identity, std::int32_t device_ordinal) {
  if (maximum_tokens == 0 || maximum_tokens > 4096 ||
      context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument("DeepSeek mHC device identity is invalid");
  }
  constexpr std::array<std::uint64_t, 6> bytes_per_token{
      4 * 4096 * sizeof(std::uint16_t),
      4 * 4096 * sizeof(std::uint16_t),
      4096 * sizeof(std::uint16_t),
      4096 * sizeof(std::uint16_t),
      4 * sizeof(float),
      16 * sizeof(float)};
  std::array<std::uint64_t, 6> offsets{};
  std::uint64_t cursor = 0;
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    auto offset = reserve(cursor, maximum_tokens, bytes_per_token[index]);
    if (!offset.ok()) return offset.status();
    offsets[index] = *offset;
  }
  auto total = checked_align_up_u64(cursor, 256);
  if (!total.ok()) return total.status();
  auto backing = Buffer::Allocate(allocator, *total, 256);
  if (!backing.ok()) return backing.status();
  if (backing->data() == nullptr || backing->size_bytes() != *total ||
      backing->generation() == 0 ||
      backing->device().type() != DeviceType::kCuda ||
      backing->device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek mHC allocator returned invalid device backing");
  }
  const auto base = reinterpret_cast<std::uintptr_t>(backing->data());
  const DeepSeekMhcDeviceView view{
      base + offsets[0], base + offsets[1], base + offsets[2],
      base + offsets[3], base + offsets[4], base + offsets[5]};
  return DeepSeekMhcDeviceResources(
      std::make_unique<Buffer>(std::move(*backing)), view, maximum_tokens,
      context_identity, device_ordinal);
}


}  // namespace pih
