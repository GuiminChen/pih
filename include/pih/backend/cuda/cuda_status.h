#pragma once

#include <string_view>

#include <cuda_runtime_api.h>

#include "pih/core/status.h"

namespace pih {

Status cuda_status(cudaError_t error, std::string_view operation);

}  // namespace pih
