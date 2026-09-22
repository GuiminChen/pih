#pragma once

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/deepseek_pipeline_transaction.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

// Compiles the complete, pinned per-rank layer topology into a stable root.
// Moved-from or otherwise non-canonical plans fail closed.
Result<Sha256Digest> compile_deepseek_pipeline_plan_root(
    const DeepSeekPipelinePlan& plan);

// Compiles both requested ceilings and every derived byte/rank ceiling. The
// mutable capacity value is first checked against a canonical reconstruction,
// so callers cannot authorize hand-edited derived fields.
Result<Sha256Digest> compile_deepseek_pipeline_capacity_root(
    const DeepSeekPipelineCapacity& capacity);

}  // namespace pih
