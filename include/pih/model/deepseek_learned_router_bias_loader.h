#pragma once

#include <vector>

#include "pih/model/deepseek_learned_router_work_factory.h"
#include "pih/model/deepseek_mapped_tensor_source.h"

namespace pih {

class DeepSeekLearnedRouterBiasLoader final {
 public:
  static Result<std::vector<DeepSeekLearnedRouterLayerBias>> Load(
      DeepSeekStageRange owned_layers,
      const DeepSeekWeightByteSource& source);
};

}  // namespace pih
