#include "pih/model/deepseek_rank_weight_disposition_manifest.h"

#include <algorithm>
#include <map>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

std::string expert_tensor_name(std::uint32_t layer, std::uint32_t expert,
                               std::string_view matrix,
                               std::string_view suffix) {
  const auto block = "layers." + std::to_string(layer);
  return block + ".ffn.experts." +
         std::to_string(expert) + "." + std::string(matrix) + "." +
         std::string(suffix);
}

}  // namespace

Result<DeepSeekRankWeightDispositionManifest>
DeepSeekRankWeightDispositionManifest::Create(
    std::uint32_t owner_rank, DeepSeekStageRange owned_layers,
    DeepSeekRoutedExpertResidency expert_residency,
    std::span<const DeepSeekRankTensorRecord> rank_tensors) {
  if (owner_rank >= 4 || rank_tensors.empty()) {
    return Status::InvalidArgument("DeepSeek rank weight disposition identity is invalid");
  }
  auto expert_manifest = DeepSeekExpertBundleManifest::Create(
      owned_layers, rank_tensors);
  if (!expert_manifest.ok()) return expert_manifest.status();

  std::map<std::string, DeepSeekExpertIdentity, std::less<>> expert_names;
  for (std::uint32_t layer = owned_layers.first_layer;
       layer <= owned_layers.last_layer; ++layer) {
    for (std::uint32_t expert = 0; expert < 256; ++expert) {
      const DeepSeekExpertIdentity identity{
          static_cast<std::uint16_t>(layer),
          static_cast<std::uint16_t>(expert)};
      for (const auto* matrix : {"w1", "w2", "w3"}) {
        for (const auto* suffix : {"weight", "scale"}) {
          expert_names.emplace(
              expert_tensor_name(layer, expert, matrix, suffix), identity);
        }
      }
    }
  }

  DeepSeekRankWeightDispositionManifest result;
  result.owner_rank_ = owner_rank;
  result.records_.reserve(rank_tensors.size());
  for (const auto& tensor : rank_tensors) {
    if (tensor.tensor_name.empty() || tensor.file_begin >= tensor.file_end) {
      return Status::InvalidArgument("DeepSeek disposition source record is invalid");
    }
    const auto bytes = tensor.file_end - tensor.file_begin;
    const auto expert = expert_names.find(tensor.tensor_name);
    const bool paged = expert != expert_names.end() &&
                       expert_residency == DeepSeekRoutedExpertResidency::kHostSpill;
    DeepSeekRankWeightDispositionRecord disposition{
        tensor.tensor_name,
        paged ? DeepSeekWeightDispositionKind::kPagedSource
              : DeepSeekWeightDispositionKind::kMaterializeFixed,
        tensor.role, owner_rank,
        expert == expert_names.end()
            ? std::optional<DeepSeekExpertIdentity>{}
            : std::optional<DeepSeekExpertIdentity>{expert->second},
        bytes};
    auto* byte_total = paged ? &result.paged_source_bytes_
                             : &result.materialize_fixed_bytes_;
    auto total = checked_add_u64(*byte_total, bytes);
    if (!total.ok()) return total.status();
    *byte_total = *total;
    if (paged) {
      ++result.paged_source_record_count_;
    } else {
      ++result.materialize_fixed_record_count_;
    }
    result.records_.push_back(std::move(disposition));
  }
  std::sort(result.records_.begin(), result.records_.end(),
            [](const auto& left, const auto& right) {
              return left.tensor_name < right.tensor_name;
            });
  for (std::size_t i = 1; i < result.records_.size(); ++i) {
    if (result.records_[i - 1].tensor_name == result.records_[i].tensor_name) {
      return Status::InvalidArgument("DeepSeek disposition contains duplicate source records");
    }
  }
  if (expert_residency == DeepSeekRoutedExpertResidency::kHostSpill &&
      (result.paged_source_record_count_ != expert_names.size() ||
       result.paged_source_bytes_ != expert_manifest->payload_bytes())) {
    return Status::Internal("DeepSeek paged expert disposition ledger is inconsistent");
  }
  if (expert_residency == DeepSeekRoutedExpertResidency::kFullResident &&
      (result.paged_source_record_count_ != 0 || result.paged_source_bytes_ != 0)) {
    return Status::Internal("DeepSeek full-resident disposition retained paged owners");
  }
  return result;
}

const DeepSeekRankWeightDispositionRecord*
DeepSeekRankWeightDispositionManifest::find(
    std::string_view tensor_name) const noexcept {
  const auto found = std::lower_bound(
      records_.begin(), records_.end(), tensor_name,
      [](const auto& record, std::string_view name) {
        return record.tensor_name < name;
      });
  return found != records_.end() && found->tensor_name == tensor_name
             ? &*found
             : nullptr;
}

}  // namespace pih
