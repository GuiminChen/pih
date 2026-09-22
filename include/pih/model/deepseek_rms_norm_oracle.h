#pragma once

#include <span>

#include "pih/core/bfloat16.h"
#include "pih/core/status.h"

namespace pih {

Status deepseek_rms_norm_oracle(
    std::span<const BFloat16> input, std::span<const BFloat16> weight,
    std::uint32_t rows, std::uint32_t hidden_size,
    std::span<BFloat16> output);

}  // namespace pih
