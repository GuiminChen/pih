#include "pih/model/qwen3_bf16_resource_set.h"

namespace pih {
namespace {

bool has_contiguous_shape(const TensorView& view,
                          const ExpectedTensor& expected) {
  if (view.rank() != expected.shape.size()) return false;
  std::uint64_t stride = 1;
  for (std::size_t reverse = expected.shape.size(); reverse > 0; --reverse) {
    const auto axis = reverse - 1;
    if (view.dim(axis) != expected.shape[axis] ||
        view.stride(axis) != stride) {
      return false;
    }
    stride *= expected.shape[axis];
  }
  return true;
}

}  // namespace

Result<QwenBf16ResourceSet> QwenBf16ResourceSet::Create(
    std::uint64_t request_generation, std::int32_t owning_rank,
    std::span<const QwenBf16SlotResource> resources) {
  if (request_generation == 0 || owning_rank < 0 ||
      resources.size() != kSlotCount) {
    return Status::InvalidArgument(
        "Qwen BF16 resource set identity or cardinality is invalid");
  }

  QwenBf16ResourceSet set;
  set.request_generation_ = request_generation;
  set.owning_rank_ = owning_rank;
  for (const auto& resource : resources) {
    const auto index = static_cast<std::size_t>(resource.slot);
    if (index >= kSlotCount || set.resources_[index].has_value()) {
      return Status::InvalidArgument(
          "Qwen BF16 resource set contains an unknown or duplicate slot");
    }
    if (resource.view.device().type() != DeviceType::kCuda ||
        resource.view.device().index() != owning_rank ||
        resource.view.generation() == 0) {
      return Status::InvalidArgument(
          "Qwen BF16 resource view device or generation is invalid");
    }
    set.resources_[index] = resource.view;
  }
  return set;
}

Result<TensorView> QwenBf16ResourceSet::view(
    QwenBf16ActivationSlot slot) const {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= kSlotCount || !resources_[index].has_value()) {
    return Status::InvalidArgument("Qwen BF16 resource slot is not bound");
  }
  return *resources_[index];
}

Result<QwenBf16WeightResourceSet> QwenBf16WeightResourceSet::Create(
    std::int32_t owning_rank, std::span<const TensorView> weights) {
  const auto& expected = Qwen3Manifest::expected_tensors();
  if (owning_rank < 0 || weights.size() != kWeightCount ||
      expected.size() != kWeightCount) {
    return Status::InvalidArgument(
        "Qwen BF16 weight resource identity or cardinality is invalid");
  }

  QwenBf16WeightResourceSet set;
  set.owning_rank_ = owning_rank;
  for (std::size_t index = 0; index < kWeightCount; ++index) {
    const auto& weight = weights[index];
    if (weight.dtype() != DType::kBFloat16 ||
        weight.device().type() != DeviceType::kCuda ||
        weight.device().index() != owning_rank || weight.generation() == 0 ||
        !has_contiguous_shape(weight, expected[index])) {
      return Status::InvalidArgument(
          "Qwen BF16 weight resource does not match the frozen manifest");
    }
    set.weights_[index] = weight;
  }
  return set;
}

Result<TensorView> QwenBf16WeightResourceSet::view(
    std::size_t tensor_index) const {
  if (tensor_index >= kWeightCount || !weights_[tensor_index].has_value()) {
    return Status::InvalidArgument("Qwen BF16 weight index is not bound");
  }
  return *weights_[tensor_index];
}

}  // namespace pih
