#pragma once

#include <vector>

#include "pih/model/deepseek_hash_router_work_factory.h"
#include "pih/model/deepseek_mapped_tensor_source.h"

namespace pih {

class DeepSeekHashRouterTableLoader final {
 public:
  static Result<std::vector<DeepSeekHashRouterLayerTable>> Load(
      DeepSeekStageRange owned_layers, std::uint32_t vocabulary_size,
      const DeepSeekWeightByteSource& source);
};

}  // namespace pih
