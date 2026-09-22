#pragma once

#include "pih/plugin_sdk/status_bridge.h"

namespace pih {

inline Status deepseek_kernel_status(const pih_status_v1& result) {
  return plugin_status(result);
}

}  // namespace pih
