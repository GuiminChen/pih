#include "pih/model/deepseek_expert_bundle_manifest.h"

#include <algorithm>
#include <map>
#include <string>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

constexpr std::uint32_t kExpertsPerLayer = 256;

bool is_routed_expert_name(std::string_view name) {
  return name.starts_with("layers.") &&
         name.find(".ffn.experts.") != std::string_view::npos;
}

std::string tensor_name(std::uint32_t layer, std::uint32_t expert,
                        std::string_view matrix, std::string_view suffix) {
  const auto prefix = "layers." + std::to_string(layer);
  return prefix + ".ffn.experts." +
         std::to_string(expert) + "." + std::string(matrix) + "." +
         std::string(suffix);
}

bool shape_equals(const std::vector<std::uint64_t>& actual,
                  std::uint64_t rows, std::uint64_t columns) {
  return actual.size() == 2 && actual[0] == rows && actual[1] == columns;
}

struct SourceRange final {
  std::string_view shard;
  std::uint64_t begin;
  std::uint64_t end;
};

}  // namespace

Result<DeepSeekExpertBundleManifest> DeepSeekExpertBundleManifest::Create(
    DeepSeekStageRange owned_layers,
    std::span<const DeepSeekRankTensorRecord> rank_tensors) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer >= 43) {
    return Status::InvalidArgument("DeepSeek expert manifest ownership is invalid");
  }
  std::map<std::string_view, const DeepSeekRankTensorRecord*, std::less<>> by_name;
  std::size_t routed_record_count = 0;
  for (const auto& tensor : rank_tensors) {
    if (!by_name.emplace(tensor.tensor_name, &tensor).second) {
      return Status::InvalidArgument("DeepSeek rank tensor manifest has duplicate names");
    }
    if (is_routed_expert_name(tensor.tensor_name)) ++routed_record_count;
  }
  const auto layer_count = owned_layers.last_layer - owned_layers.first_layer + 1U;
  const auto expected_bundle_count = static_cast<std::uint64_t>(layer_count) *
                                     kExpertsPerLayer;
  const auto expected_record_count = expected_bundle_count * 6U;
  if (routed_record_count != expected_record_count) {
    return Status::InvalidArgument("DeepSeek routed expert tensor manifest is not total");
  }

  DeepSeekExpertBundleManifest result;
  result.bundles_.reserve(static_cast<std::size_t>(expected_bundle_count));
  std::vector<SourceRange> source_ranges;
  source_ranges.reserve(static_cast<std::size_t>(expected_record_count));
  for (std::uint32_t layer = owned_layers.first_layer;
       layer <= owned_layers.last_layer; ++layer) {
    for (std::uint32_t expert = 0; expert < kExpertsPerLayer; ++expert) {
      DeepSeekExpertMappedBundleRecord bundle;
      bundle.identity = {static_cast<std::uint16_t>(layer),
                         static_cast<std::uint16_t>(expert)};
      std::size_t segment_index = 0;
      for (const auto* matrix : {"w1", "w2", "w3"}) {
        const bool w2 = std::string_view(matrix) == "w2";
        for (const auto* suffix : {"weight", "scale"}) {
          const bool scale = std::string_view(suffix) == "scale";
          const auto name = tensor_name(layer, expert, matrix, suffix);
          const auto found = by_name.find(name);
          if (found == by_name.end()) {
            return Status::InvalidArgument("DeepSeek expert bundle tensor is missing");
          }
          const auto& tensor = *found->second;
          const auto expected_dtype = scale ? DType::kFloat8E8M0 : DType::kInt8;
          const auto expected_bytes = scale
              ? DeepSeekExpertBundleLayout::kScaleBytesPerMatrix
              : DeepSeekExpertBundleLayout::kPackedBytesPerMatrix;
          const bool shape_ok = scale
              ? (w2 ? shape_equals(tensor.shape, 4096, 64)
                    : shape_equals(tensor.shape, 2048, 128))
              : (w2 ? shape_equals(tensor.shape, 4096, 1024)
                    : shape_equals(tensor.shape, 2048, 2048));
          if (tensor.role != DeepSeekTensorRole::kMainLayer ||
              tensor.shard_name.empty() || tensor.dtype != expected_dtype ||
              tensor.file_begin >= tensor.file_end ||
              tensor.file_end - tensor.file_begin != expected_bytes || !shape_ok) {
            return Status::InvalidArgument("DeepSeek expert bundle tensor layout is invalid");
          }
          bundle.segments[segment_index++] =
              {tensor.shard_name, tensor.file_begin, expected_bytes};
          source_ranges.push_back(
              {tensor.shard_name, tensor.file_begin, tensor.file_end});
        }
      }
      result.bundles_.push_back(std::move(bundle));
    }
  }
  std::sort(source_ranges.begin(), source_ranges.end(), [](const auto& left,
                                                            const auto& right) {
    return left.shard < right.shard ||
           (left.shard == right.shard && left.begin < right.begin);
  });
  for (std::size_t i = 1; i < source_ranges.size(); ++i) {
    if (source_ranges[i - 1].shard == source_ranges[i].shard &&
        source_ranges[i - 1].end > source_ranges[i].begin) {
      return Status::InvalidArgument("DeepSeek expert source tensor ranges overlap");
    }
  }
  auto payload = checked_mul_u64(expected_bundle_count,
                                 DeepSeekExpertBundleLayout::kBundleBytes);
  if (!payload.ok()) return payload.status();
  result.payload_bytes_ = *payload;
  return result;
}

}  // namespace pih
