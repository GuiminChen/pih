#pragma once

#include "pih/contracts/execution_controller_v1.h"

namespace pih::execution_plugin {

const pih_execution_controller_api_v1* ControllerApi() noexcept;
void SetControllerReady(bool ready) noexcept;
bool ControllerHandlesEmpty() noexcept;

}  // namespace pih::execution_plugin
