#include "pih/model/deepseek_rank_attention_state_pool.h"

namespace pih {
namespace {

Result<Buffer> allocate_page_storage(Allocator& allocator,
                                     std::uint64_t bytes,
                                     std::int32_t device_ordinal) {
  auto buffer = Buffer::Allocate(allocator, bytes, 65536);
  if (!buffer.ok()) return buffer.status();
  if (buffer->data() == nullptr || buffer->size_bytes() != bytes ||
      buffer->generation() == 0 ||
      buffer->device().type() != DeviceType::kCuda ||
      buffer->device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek rank page allocator returned invalid device storage");
  }
  return buffer;
}

}  // namespace

Result<DeepSeekRankAttentionStatePool>
DeepSeekRankAttentionStatePool::Allocate(
    Allocator& allocator, DeepSeekStagePlan stage,
    std::uint32_t maximum_sequences,
    std::uint32_t reserved_tokens_per_sequence,
    std::uintptr_t completion_event,
    DeepSeekFixedStateBankOperations& fixed_operations,
    std::uint64_t context_identity, std::int32_t device_ordinal) {
  auto layout = DeepSeekAttentionPhysicalLayout::Compile(
      stage, maximum_sequences, reserved_tokens_per_sequence);
  if (!layout.ok()) return layout.status();
  auto page_storage = allocate_page_storage(
      allocator, layout->allocation_bytes(), device_ordinal);
  if (!page_storage.ok()) return page_storage.status();
  auto ratio4_pool = DeepSeekRatio4PagePool::Create(
      layout->ratio4_pair_count());
  if (!ratio4_pool.ok()) return ratio4_pool.status();
  auto ratio128_pool = DeepSeekRatio128PagePool::Create(
      layout->ratio128_page_count());
  if (!ratio128_pool.ok()) return ratio128_pool.status();
  const auto base = reinterpret_cast<std::uintptr_t>(page_storage->data());
  const auto& ratio4_main = layout->extent(
      DeepSeekAttentionPhysicalExtentKind::kRatio4Main);
  const auto& ratio4_index = layout->extent(
      DeepSeekAttentionPhysicalExtentKind::kRatio4Index);
  const auto& ratio128_main = layout->extent(
      DeepSeekAttentionPhysicalExtentKind::kRatio128Main);
  auto page_arena = DeepSeekAttentionPageArena::Create(
      {base + ratio4_main.offset_bytes, ratio4_main.bytes},
      {base + ratio4_index.offset_bytes, ratio4_index.bytes},
      {base + ratio128_main.offset_bytes, ratio128_main.bytes},
      layout->ratio4_pair_count(), layout->ratio128_page_count());
  if (!page_arena.ok()) return page_arena.status();
  std::vector<DeepSeekAttentionSequenceDeviceResources> sequences;
  sequences.reserve(maximum_sequences);
  for (std::uint32_t slot = 0; slot < maximum_sequences; ++slot) {
    auto sequence = DeepSeekAttentionSequenceDeviceResources::Allocate(
        allocator, layout->fixed_layout(), completion_event,
        fixed_operations, context_identity, device_ordinal);
    if (!sequence.ok()) return sequence.status();
    sequences.push_back(std::move(*sequence));
  }
  const auto ratio4_per_sequence =
      layout->ratio4_page_pairs_per_sequence();
  const auto ratio128_per_sequence =
      layout->ratio128_pages_per_sequence();
  return DeepSeekRankAttentionStatePool(
      std::move(*layout), std::move(*page_storage),
      std::move(*ratio4_pool), std::move(*ratio128_pool), *page_arena,
      std::move(sequences),
      reserved_tokens_per_sequence, ratio4_per_sequence,
      ratio128_per_sequence);
}

}  // namespace pih
