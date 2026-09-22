#include "pih/model/deepseek_dense_mhc_runtime_resources.h"

#include <cstring>

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekDenseMhcRuntimeResources>
DeepSeekDenseMhcRuntimeResources::Allocate(
    DeepSeekStageRange owned_layers,
    DeepSeekDenseMhcRuntimeOperations operations,
    RegisteredPinnedAllocator& allocator) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || operations.input == nullptr ||
      operations.output == nullptr || operations.mhc == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek dense mHC runtime identity is invalid");
  }
  const auto layer_count = owned_layers.last_layer -
                           owned_layers.first_layer + 1;
  auto error_count = checked_mul_u64(layer_count, UINT64_C(4));
  if (!error_count.ok()) return error_count.status();
  auto bytes = checked_mul_u64(*error_count, sizeof(std::uint32_t));
  if (!bytes.ok()) return bytes.status();
  auto host_errors = Buffer::Allocate(allocator, *bytes, kAlignment);
  if (!host_errors.ok()) return host_errors.status();
  if (host_errors->data() == nullptr || host_errors->generation() == 0 ||
      host_errors->device().type() != DeviceType::kCpu) {
    return Status::FailedPrecondition(
        "DeepSeek dense mHC pinned error allocation is invalid");
  }
  std::memset(host_errors->data(), 0, static_cast<std::size_t>(*bytes));
  DeepSeekDenseMhcRuntimeResources result(std::move(*host_errors));
  result.owned_layers_ = owned_layers;
  result.layer_count_ = layer_count;
  auto* errors = static_cast<std::uint32_t*>(result.host_errors_.data());
  for (std::uint32_t layer = owned_layers.first_layer;
       layer <= owned_layers.last_layer; ++layer) {
    const auto offset = (layer - owned_layers.first_layer) * 4;
    auto input = DeepSeekAttentionProjectionCoordinator::Create(
        *operations.input, errors + offset);
    if (!input.ok()) return input.status();
    auto output = DeepSeekAttentionOutputProjectionCoordinator::Create(
        *operations.output, errors + offset + 1);
    if (!output.ok()) return output.status();
    auto attention = DeepSeekMhcSequenceExecutor::Create(
        *operations.mhc, errors + offset + 2);
    if (!attention.ok()) return attention.status();
    auto feed_forward = DeepSeekMhcSequenceExecutor::Create(
        *operations.mhc, errors + offset + 3);
    if (!feed_forward.ok()) return feed_forward.status();
    result.input_[layer] = std::move(*input);
    result.output_[layer] = std::move(*output);
    result.mhc_attention_[layer] = std::move(*attention);
    result.mhc_feed_forward_[layer] = std::move(*feed_forward);
  }
  return result;
}

Result<DeepSeekDenseMhcLayerRuntimeBorrow>
DeepSeekDenseMhcRuntimeResources::borrow(std::uint32_t layer) noexcept {
  if (layer >= input_.size() || !input_[layer].has_value() ||
      !output_[layer].has_value() || !mhc_attention_[layer].has_value() ||
      !mhc_feed_forward_[layer].has_value()) {
    return Status::InvalidArgument(
        "DeepSeek dense mHC runtime layer is not owned");
  }
  return DeepSeekDenseMhcLayerRuntimeBorrow{
      &*input_[layer], &*output_[layer], &*mhc_attention_[layer],
      &*mhc_feed_forward_[layer]};
}

}  // namespace pih
