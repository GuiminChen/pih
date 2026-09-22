#pragma once

#include "pih/core/sha256.h"

namespace pih {

Result<Sha256Digest> nvidia_physical_device_identity(
    std::int32_t device_ordinal);

}  // namespace pih
