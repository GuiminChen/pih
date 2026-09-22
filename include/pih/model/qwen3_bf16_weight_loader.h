#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/qwen3_bf16_resource_set.h"
#include "pih/model/resident_weight_set.h"

namespace pih {

class QwenBf16WeightLoader final {
 public:
  static std::vector<WeightRequirement> Requirements();
  static Result<QwenBf16WeightResourceSet> Bind(
      const ResidentWeightSet& resident, std::int32_t owning_rank);
};

}  // namespace pih
