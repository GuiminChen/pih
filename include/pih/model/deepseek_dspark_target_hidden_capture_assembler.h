#pragma once

#include <span>

#include "pih/model/deepseek_dense_mhc_plan_input_assembler.h"
#include "pih/model/deepseek_dspark_device_resources.h"

namespace pih {

// Binds the official D-Spark target-hidden taps to the feed-forward outputs of
// target layers 40, 41 and 42.  The operation runs after each mHC post mix,
// before the shared error channel is copied back to the host.
class DeepSeekDsparkTargetHiddenCaptureAssembler final {
 public:
  static Status Bind(
      DeepSeekStagePlan stage,
      const DeepSeekDsparkDeviceResources& dspark_resources,
      std::uint32_t token_count,
      std::span<DeepSeekDenseMhcLayerSubmissionInput> layers);
};

}  // namespace pih
