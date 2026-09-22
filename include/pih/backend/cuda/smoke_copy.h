#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

Status cuda_smoke_copy(std::uint64_t bytes);

}  // namespace pih
