#include "pih/model/deepseek_attention_runtime_resources.h"

#include <algorithm>
#include <array>

namespace pih {

struct DeepSeekAttentionRuntimeResources::LayerOwner final {
  std::uint32_t layer = 0;
  std::unique_ptr<DeepSeekRecentStateWriter> recent;
  std::unique_ptr<DeepSeekCompressorStateWriter> compressor;
  std::unique_ptr<DeepSeekCompressedPageWriter> page;
  std::unique_ptr<DeepSeekCompressedLayerUpdateCoordinator> update;
  std::unique_ptr<DeepSeekIndexerProjectionCoordinator> indexer_projection;
  std::unique_ptr<DeepSeekIndexSelectionDriver> selection;
  std::unique_ptr<DeepSeekSparseAttentionDriver> sparse;
  std::unique_ptr<DeepSeekAttentionLayerCoordinator> attention;
  std::unique_ptr<DeepSeekPrefillLayerCoordinator> prefill;
};

DeepSeekAttentionRuntimeResources::~DeepSeekAttentionRuntimeResources() =
    default;
DeepSeekAttentionRuntimeResources::DeepSeekAttentionRuntimeResources(
    DeepSeekAttentionRuntimeResources&&) noexcept = default;
DeepSeekAttentionRuntimeResources&
DeepSeekAttentionRuntimeResources::operator=(
    DeepSeekAttentionRuntimeResources&&) noexcept = default;

namespace {

bool complete(DeepSeekAttentionRuntimeOperations value) {
  return value.recent != nullptr && value.compressor != nullptr &&
         value.page != nullptr && value.indexer_projection != nullptr &&
         value.index_selection != nullptr && value.sparse_attention != nullptr;
}

template <typename T>
Result<std::unique_ptr<T>> own(Result<T> result) {
  if (!result.ok()) return result.status();
  return std::make_unique<T>(std::move(*result));
}

}  // namespace

Result<DeepSeekAttentionRuntimeResources>
DeepSeekAttentionRuntimeResources::Create(
    DeepSeekStageRange owned_layers,
    DeepSeekAttentionRuntimeOperations operations,
    std::vector<DeepSeekAttentionLayerRuntimeInput> layers) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || !complete(operations) ||
      layers.size() != owned_layers.last_layer - owned_layers.first_layer + 1) {
    return Status::InvalidArgument(
        "DeepSeek attention runtime resource coverage is invalid");
  }
  std::array<bool, 43> seen{};
  DeepSeekAttentionRuntimeResources result;
  result.owned_layers_ = owned_layers;
  result.layers_.reserve(layers.size());
  for (auto& input : layers) {
    if (input.layer < owned_layers.first_layer ||
        input.layer > owned_layers.last_layer || seen[input.layer] ||
        input.compressor_error_flag == nullptr ||
        input.page_error_flag == nullptr ||
        input.indexer_projection_error_flag == nullptr ||
        input.index_completion_event == 0 ||
        std::ranges::none_of(
            input.fixed_layout.descriptors(),
            [&input](const auto& descriptor) {
              return descriptor.layer_id == input.layer;
            })) {
      return Status::InvalidArgument(
          "DeepSeek attention layer runtime identity is invalid");
    }
    seen[input.layer] = true;
    auto owner = std::make_unique<LayerOwner>();
    owner->layer = input.layer;
    auto recent = own(DeepSeekRecentStateWriter::Create(
        input.fixed_layout, *operations.recent));
    if (!recent.ok()) return recent.status();
    owner->recent = std::move(*recent);
    auto compressor = own(DeepSeekCompressorStateWriter::Create(
        input.fixed_layout, *operations.compressor,
        input.compressor_error_flag));
    if (!compressor.ok()) return compressor.status();
    owner->compressor = std::move(*compressor);
    auto page = own(DeepSeekCompressedPageWriter::Create(
        input.page_arena, *operations.page, input.page_error_flag));
    if (!page.ok()) return page.status();
    owner->page = std::move(*page);
    auto update = own(DeepSeekCompressedLayerUpdateCoordinator::Create(
        *owner->compressor, *owner->page));
    if (!update.ok()) return update.status();
    owner->update = std::move(*update);
    auto indexer_projection = own(DeepSeekIndexerProjectionCoordinator::Create(
        *operations.indexer_projection,
        input.indexer_projection_error_flag));
    if (!indexer_projection.ok()) return indexer_projection.status();
    owner->indexer_projection = std::move(*indexer_projection);
    auto selection = own(DeepSeekIndexSelectionDriver::Create(
        *operations.index_selection, input.index_staging,
        input.index_completion_event));
    if (!selection.ok()) return selection.status();
    owner->selection = std::move(*selection);
    auto sparse = own(DeepSeekSparseAttentionDriver::Create(
        *operations.sparse_attention, input.sparse_staging));
    if (!sparse.ok()) return sparse.status();
    owner->sparse = std::move(*sparse);
    auto attention = own(DeepSeekAttentionLayerCoordinator::CreatePaged(
        input.layer, input.fixed_layout, input.page_arena,
        *owner->indexer_projection, *owner->selection,
        *owner->sparse));
    if (!attention.ok()) return attention.status();
    owner->attention = std::move(*attention);
    auto prefill = own(DeepSeekPrefillLayerCoordinator::Create(
        *owner->recent, *owner->update, *owner->selection, *owner->sparse,
        *owner->attention));
    if (!prefill.ok()) return prefill.status();
    owner->prefill = std::move(*prefill);
    result.layers_.push_back(std::move(owner));
  }
  return result;
}

std::vector<DeepSeekAttentionLayerRuntimeResources>
DeepSeekAttentionRuntimeResources::borrow_layer_resources() noexcept {
  std::vector<DeepSeekAttentionLayerRuntimeResources> result;
  result.reserve(layers_.size());
  for (auto& layer : layers_) {
    result.push_back({layer->layer, layer->recent.get(), layer->update.get(),
                      layer->attention.get(), layer->prefill.get()});
  }
  return result;
}

}  // namespace pih
