#pragma once

#include <string_view>

#include <cuda.h>

#include "pih/core/status.h"

namespace pih {

Status cuda_driver_status(CUresult result, std::string_view operation);

}  // namespace pih
