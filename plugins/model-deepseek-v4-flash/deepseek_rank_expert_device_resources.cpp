#include "pih/model/deepseek_rank_expert_device_resources.h"

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Status validate_device_buffer(const Buffer& buffer, std::uint64_t bytes,
                              std::int32_t device_ordinal) {
  if (buffer.size_bytes() != bytes || buffer.generation() == 0 ||
      buffer.data() == nullptr || buffer.device().type() != DeviceType::kCuda ||
      buffer.device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek expert allocator returned an invalid device allocation");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekRankExpertDeviceResources>
DeepSeekRankExpertDeviceResources::Allocate(
    Allocator& allocator, std::uint32_t slot_count,
    std::uint32_t maximum_tokens, std::uint64_t context_identity,
    std::int32_t device_ordinal,
    Allocator* host_spill_slot_allocator) {
  if (context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument(
        "DeepSeek rank expert device resource identity is invalid");
  }
  auto layout = DeepSeekExpertComputeArenaLayout::Create(maximum_tokens);
  if (!layout.ok()) return layout.status();
  auto source_bytes = checked_mul_u64(maximum_tokens, UINT64_C(4096) * 2);
  if (!source_bytes.ok()) return source_bytes.status();
  auto accumulator_bytes = checked_mul_u64(maximum_tokens, UINT64_C(4096) * 4);
  if (!accumulator_bytes.ok()) return accumulator_bytes.status();

  std::vector<Buffer> slot_backings;
  std::vector<std::uintptr_t> slot_bases;
  slot_backings.reserve(slot_count);
  slot_bases.reserve(slot_count);
  auto& slot_allocator = host_spill_slot_allocator == nullptr
      ? allocator
      : *host_spill_slot_allocator;
  for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
    auto backing = Buffer::Allocate(
        slot_allocator, DeepSeekExpertPager::kBundleBytes,
        DeepSeekExpertComputeArenaLayout::kAlignment);
    if (!backing.ok()) return backing.status();
    const Status valid = validate_device_buffer(
        *backing, DeepSeekExpertPager::kBundleBytes, device_ordinal);
    if (!valid.ok()) return valid;
    slot_bases.push_back(reinterpret_cast<std::uintptr_t>(backing->data()));
    slot_backings.push_back(std::move(*backing));
  }
  auto compute_backing = Buffer::Allocate(
      allocator, layout->required_bytes(),
      DeepSeekExpertComputeArenaLayout::kAlignment);
  if (!compute_backing.ok()) return compute_backing.status();
  auto source_hidden = Buffer::Allocate(
      allocator, *source_bytes, DeepSeekExpertComputeArenaLayout::kAlignment);
  if (!source_hidden.ok()) return source_hidden.status();
  auto accumulator = Buffer::Allocate(
      allocator, *accumulator_bytes,
      DeepSeekExpertComputeArenaLayout::kAlignment);
  if (!accumulator.ok()) return accumulator.status();
  for (const auto* entry : {&*compute_backing, &*source_hidden, &*accumulator}) {
    const Status valid = validate_device_buffer(
        *entry, entry->size_bytes(), device_ordinal);
    if (!valid.ok()) return valid;
  }
  auto arena = layout->bind(
      reinterpret_cast<std::uintptr_t>(compute_backing->data()),
      compute_backing->size_bytes());
  if (!arena.ok()) return arena.status();
  auto slots = slot_bases.empty()
      ? DeepSeekExpertSlotTable::CreateResidentOnly(
            context_identity, static_cast<std::uint32_t>(device_ordinal))
      : DeepSeekExpertSlotTable::Create(
            slot_bases, context_identity,
            static_cast<std::uint32_t>(device_ordinal));
  if (!slots.ok()) return slots.status();
  return DeepSeekRankExpertDeviceResources(
      std::move(slot_backings), std::move(slot_bases), std::move(*slots),
      std::move(*compute_backing), *arena, std::move(*source_hidden),
      std::move(*accumulator), maximum_tokens);
}

}  // namespace pih
