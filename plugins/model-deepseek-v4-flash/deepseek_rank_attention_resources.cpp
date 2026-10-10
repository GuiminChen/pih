#include "pih/model/deepseek_rank_attention_resources.h"

#include <new>
#include <utility>

namespace pih {

DeepSeekRankAttentionResources& DeepSeekRankAttentionResources::operator=(
    DeepSeekRankAttentionResources&& other) noexcept {
  if (this != &other) {
    this->~DeepSeekRankAttentionResources();
    ::new (static_cast<void*>(this))
        DeepSeekRankAttentionResources(std::move(other));
  }
  return *this;
}

Result<DeepSeekRankAttentionResources>
DeepSeekRankAttentionResources::Allocate(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_queries,
    std::uint32_t maximum_ratio4_logical_pages,
    DeepSeekFixedStateLayout fixed_layout,
    DeepSeekAttentionPageArena page_arena,
    DeepSeekAttentionRuntimeOperations operations,
    RegisteredPinnedAllocator& pinned_allocator,
    Allocator& device_allocator, std::uintptr_t completion_event,
    std::uint64_t context_identity, std::int32_t device_ordinal,
    std::uint32_t maximum_sequences) {
  if (maximum_sequences == 0 || maximum_sequences > 4096)
    return Status::InvalidArgument("DeepSeek attention lane capacity invalid");
  auto host = DeepSeekAttentionHostStagingResources::Allocate(
      pinned_allocator, maximum_queries, maximum_ratio4_logical_pages);
  if (!host.ok()) return host.status();
  auto device = DeepSeekAttentionDeviceScratchResources::Allocate(
      device_allocator, maximum_queries, maximum_ratio4_logical_pages,
      context_identity, device_ordinal);
  if (!device.ok()) return device.status();
  auto compressor_projection =
      DeepSeekCompressorProjectionDeviceResources::Allocate(
          device_allocator, maximum_queries, context_identity,
          device_ordinal);
  if (!compressor_projection.ok()) return compressor_projection.status();
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || completion_event == 0) {
    return Status::InvalidArgument(
        "DeepSeek rank attention assembly identity is invalid");
  }
  std::vector<DeepSeekAttentionLayerRuntimeInput> inputs;
  inputs.reserve(owned_layers.last_layer - owned_layers.first_layer + 1);
  for (std::uint32_t layer = owned_layers.first_layer;
       layer <= owned_layers.last_layer; ++layer) {
    inputs.push_back({layer, fixed_layout, page_arena,
                      host->compressor_error(), host->page_error(),
                      host->indexer_projection_error(), host->index(),
                      completion_event, host->sparse()});
  }
  auto runtime = DeepSeekAttentionRuntimeResources::Create(
      owned_layers, operations, std::move(inputs));
  if (!runtime.ok()) return runtime.status();
  auto factory = DeepSeekAttentionWorkFactory::Create(
      owned_layers, runtime->borrow_layer_resources());
  if (!factory.ok()) return factory.status();
  DeepSeekRankAttentionResources result(
      std::make_unique<DeepSeekAttentionHostStagingResources>(
          std::move(*host)),
      std::make_unique<DeepSeekAttentionDeviceScratchResources>(
          std::move(*device)),
      std::make_unique<DeepSeekCompressorProjectionDeviceResources>(
          std::move(*compressor_projection)),
      std::make_unique<DeepSeekAttentionRuntimeResources>(
          std::move(*runtime)),
      std::make_unique<DeepSeekAttentionWorkFactory>(std::move(*factory)));
  auto all_layers = result.runtime_->borrow_layer_resources();
  for (std::uint32_t sequence = 1; sequence < maximum_sequences; ++sequence) {
    auto lane = Allocate(owned_layers, maximum_queries,
        maximum_ratio4_logical_pages, fixed_layout, page_arena, operations,
        pinned_allocator, device_allocator, completion_event, context_identity,
        device_ordinal, 1);
    if (!lane.ok()) return lane.status();
    auto owned_lane = std::make_unique<DeepSeekRankAttentionResources>(std::move(*lane));
    for (auto layer : owned_lane->runtime_->borrow_layer_resources()) {
      layer.sequence_index = sequence;
      all_layers.push_back(layer);
    }
    result.additional_lanes_.push_back(std::move(owned_lane));
  }
  auto packed = DeepSeekAttentionWorkFactory::Create(owned_layers, std::move(all_layers));
  if (!packed.ok()) return packed.status();
  result.work_factory_ = std::make_unique<DeepSeekAttentionWorkFactory>(std::move(*packed));
  return result;
}

}  // namespace pih
