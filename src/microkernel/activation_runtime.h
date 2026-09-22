#pragma once

#include <vector>

#include "microkernel/plugin_loader.h"

namespace pih::microkernel {

using RegistrationBarrier = void (*)(void* context,
                                     bool registration_complete);

void ActivatePluginStack(std::vector<PluginInstance>& plugins,
                         RegistrationBarrier registration_barrier,
                         void* registration_barrier_context);
void ShutdownPluginStackChecked(std::vector<PluginInstance>& plugins);

}  // namespace pih::microkernel
