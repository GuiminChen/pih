#include "pih/model/qwen3_bf16_weight_loader.h"

#include "pih/model/qwen3_manifest.h"

namespace pih {

std::vector<WeightRequirement> QwenBf16WeightLoader::Requirements() {
  const auto& manifest = Qwen3Manifest::expected_tensors();
  std::vector<WeightRequirement> requirements;
  requirements.reserve(manifest.size());
  for (const auto& tensor : manifest) {
    requirements.push_back(
        WeightRequirement{tensor.name, DType::kBFloat16, tensor.shape});
  }
  return requirements;
}

Result<QwenBf16WeightResourceSet> QwenBf16WeightLoader::Bind(
    const ResidentWeightSet& resident, std::int32_t owning_rank) {
  const auto& manifest = Qwen3Manifest::expected_tensors();
  if (owning_rank < 0 || resident.size() != manifest.size()) {
    return Status::InvalidArgument(
        "Qwen resident weight identity or cardinality is invalid");
  }
  std::vector<TensorView> views;
  views.reserve(manifest.size());
  for (const auto& expected : manifest) {
    auto tensor = resident.tensor(expected.name);
    if (!tensor.ok()) return tensor.status();
    views.push_back(std::move(*tensor));
  }
  return QwenBf16WeightResourceSet::Create(owning_rank, views);
}

}  // namespace pih
