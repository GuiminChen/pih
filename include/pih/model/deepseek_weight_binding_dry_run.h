#pragma once

#include "pih/model/deepseek_weight_finalizer.h"

namespace pih {

class DeepSeekWeightBindingDryRun final
    : public DeepSeekWeightConsumerDryRun {
 public:
  static constexpr std::string_view kAbi =
      "deepseek_final_weight_binding_consumer_v1";

  Result<Receipt> run(
      const DeepSeekWeightMaterializationPlan& plan,
      const DeepSeekResidentWeightArena& arena,
      const Sha256Digest& expected_layout_digest) override;
};

}  // namespace pih
