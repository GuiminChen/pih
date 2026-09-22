#pragma once

#include <span>

#include "pih/core/sha256.h"

namespace pih {

Result<Sha256Digest> deepseek_gpu_uuid_commitment(
    std::span<const std::byte> raw_uuid);

}  // namespace pih
