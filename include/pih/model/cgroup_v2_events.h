#pragma once

#include <string_view>

#include "pih/core/result.h"

namespace pih {

Result<bool> parse_cgroup_v2_populated(std::string_view events);

}  // namespace pih
