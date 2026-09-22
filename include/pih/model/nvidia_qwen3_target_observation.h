#pragma once

#include "pih/model/qwen3_target_observation.h"

namespace pih {

Result<QwenTargetObservation> observe_nvidia_qwen_target(
    QwenNumericalRunIdentity identity);

}  // namespace pih
