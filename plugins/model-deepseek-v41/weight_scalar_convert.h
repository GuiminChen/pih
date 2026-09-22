#pragma once
#include "weight_inventory.h"
#include "weight_source_wo_a.h"

namespace pih::deepseek_v41 {
// Metadata-only admission, sharing the conversion's exact dtype/shape rules.
Status ValidateScalarWeight(const RuntimeWeight& target, const SafetensorRecord& source);
// Elementwise canonical conversion, not quantization or scaled dequantization.
// Reader uses source-tensor-relative offsets; writer uses output-tensor offsets.
// Caller supplies an inventory-derived target and an authenticated source record.
Status ConvertScalarWeight(const RuntimeWeight& target, const SafetensorRecord& source,
    const WoATensorReader& read, const WoATensorWriter& write);
}  // namespace pih::deepseek_v41
