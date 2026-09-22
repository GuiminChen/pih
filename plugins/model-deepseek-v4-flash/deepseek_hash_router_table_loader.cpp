#include "pih/model/deepseek_hash_router_table_loader.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "pih/core/checked_math.h"

namespace pih {

Result<std::vector<DeepSeekHashRouterLayerTable>>
DeepSeekHashRouterTableLoader::Load(
    DeepSeekStageRange owned_layers, std::uint32_t vocabulary_size,
    const DeepSeekWeightByteSource& source) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer >= 43 || vocabulary_size == 0) {
    return Status::InvalidArgument(
        "DeepSeek hash router table identity is invalid");
  }
  const auto last = std::min(owned_layers.last_layer, 2U);
  std::vector<DeepSeekHashRouterLayerTable> result;
  if (owned_layers.first_layer > last) return result;
  auto elements = checked_mul_u64(
      vocabulary_size, DeepSeekExpertSubwavePlan::kRoutesPerToken);
  if (!elements.ok()) return elements.status();
  auto expected_bytes = checked_mul_u64(*elements, sizeof(std::int64_t));
  if (!expected_bytes.ok()) return expected_bytes.status();
  result.reserve(last - owned_layers.first_layer + 1U);
  for (std::uint32_t layer = owned_layers.first_layer; layer <= last;
       ++layer) {
    const auto name = "layers." + std::to_string(layer) +
                      ".ffn.gate.tid2eid";
    auto bytes = source.resolve_weight_bytes(name);
    if (!bytes.ok()) return bytes.status();
    if (bytes->size_bytes() != *expected_bytes) {
      return Status::InvalidArgument(
          "DeepSeek hash router table byte extent is invalid");
    }
    DeepSeekHashRouterLayerTable table;
    table.layer = layer;
    table.token_to_experts.resize(static_cast<std::size_t>(*elements));
    for (std::uint32_t token = 0; token < vocabulary_size; ++token) {
      std::array<bool, DeepSeekExpertSubwavePlan::kExpertCount> seen{};
      for (std::uint32_t ordinal = 0;
           ordinal < DeepSeekExpertSubwavePlan::kRoutesPerToken; ++ordinal) {
        const auto index = static_cast<std::size_t>(token) *
                               DeepSeekExpertSubwavePlan::kRoutesPerToken +
                           ordinal;
        std::int64_t expert = -1;
        std::memcpy(&expert,
                    bytes->data() + index * sizeof(std::int64_t),
                    sizeof(expert));
        if (expert < 0 ||
            expert >= static_cast<std::int64_t>(
                          DeepSeekExpertSubwavePlan::kExpertCount) ||
            seen[static_cast<std::size_t>(expert)]) {
          return Status::InvalidArgument(
              "DeepSeek hash router table contains an invalid expert row");
        }
        seen[static_cast<std::size_t>(expert)] = true;
        table.token_to_experts[index] = static_cast<std::uint16_t>(expert);
      }
    }
    result.push_back(std::move(table));
  }
  return result;
}

}  // namespace pih
