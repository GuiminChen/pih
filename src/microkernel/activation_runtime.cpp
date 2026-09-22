#include "microkernel/activation_runtime.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

namespace pih::microkernel {
namespace {

[[noreturn]] void ThrowLifecycleFailure(const PluginInstance& plugin,
                                        const pih_status_v1& status,
                                        const char* phase) {
  std::string failure = phase;
  failure += ":";
  failure += plugin.id();
  if (!pih_status_is_valid_v1(&status)) {
    failure += ":status_abi_invalid";
  } else {
    failure += ":status=";
    failure += std::to_string(status.code);
    const auto* end = std::find(std::begin(status.message),
                                std::end(status.message), '\0');
    if (end != std::begin(status.message)) {
      failure += ":";
      failure.append(std::begin(status.message), end);
    }
  }
  throw std::runtime_error(failure);
}

void RequireOk(const PluginInstance& plugin,
               pih_lifecycle_callback_v1 callback, const char* phase) {
  const auto& api = plugin.api();
  const auto status = callback(api.context);
  if (!pih_status_is_ok_v1(&status)) {
    ThrowLifecycleFailure(plugin, status, phase);
  }
}

bool BestEffortOk(pih_lifecycle_callback_v1 callback,
                  void* context,
                  std::chrono::steady_clock::time_point deadline) noexcept {
  try {
    for (;;) {
      const auto status = callback(context);
      if (pih_status_is_ok_v1(&status)) return true;
      if (!pih_status_is_valid_v1(&status) ||
          status.code != PIH_STATUS_UNAVAILABLE_V1 ||
          std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (...) {
    return false;
  }
}

void RollBack(std::vector<PluginInstance>& plugins, std::size_t ready_count,
              std::size_t started_count, std::size_t registered_count) noexcept {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::minutes(5);
  std::size_t safe_floor = 0;
  while (ready_count != 0) {
    --ready_count;
    const auto& api = plugins[ready_count].api();
    if (!BestEffortOk(api.lifecycle.drain, api.context, deadline)) {
      safe_floor = ready_count + 1;
      break;
    }
  }
  std::size_t dispose_floor = safe_floor;
  while (started_count > safe_floor) {
    --started_count;
    const auto& api = plugins[started_count].api();
    if (!BestEffortOk(api.lifecycle.stop, api.context, deadline)) {
      dispose_floor = started_count + 1;
      break;
    }
  }
  while (registered_count > dispose_floor) {
    --registered_count;
    const auto& api = plugins[registered_count].api();
    if (!BestEffortOk(api.lifecycle.dispose, api.context, deadline)) break;
  }
}

}  // namespace

void ActivatePluginStack(std::vector<PluginInstance>& plugins,
                         RegistrationBarrier registration_barrier,
                         void* registration_barrier_context) {
  if (registration_barrier == nullptr ||
      registration_barrier_context == nullptr) {
    throw std::invalid_argument("registration_barrier_missing");
  }
  std::size_t registered_count = 0;
  std::size_t started_count = 0;
  std::size_t ready_count = 0;
  try {
    for (auto& plugin : plugins) {
      const auto& api = plugin.api();
      RequireOk(plugin, api.lifecycle.register_plugin,
                "plugin_register_failed");
      ++registered_count;
    }
    registration_barrier(registration_barrier_context, true);
    for (auto& plugin : plugins) {
      const auto& api = plugin.api();
      RequireOk(plugin, api.lifecycle.configure, "plugin_configure_failed");
    }
    for (auto& plugin : plugins) {
      const auto& api = plugin.api();
      RequireOk(plugin, api.lifecycle.start, "plugin_start_failed");
      ++started_count;
    }
    for (auto& plugin : plugins) {
      const auto& api = plugin.api();
      RequireOk(plugin, api.lifecycle.ready, "plugin_ready_failed");
      ++ready_count;
    }
  } catch (...) {
    const auto activation_failure = std::current_exception();
    try {
      registration_barrier(registration_barrier_context, false);
    } catch (...) {
      // Preserve the phase failure. The process cannot continue if the
      // registration boundary itself could not be closed.
    }
    RollBack(plugins, ready_count, started_count, registered_count);
    std::rethrow_exception(activation_failure);
  }
}

void ShutdownPluginStackChecked(std::vector<PluginInstance>& plugins) {
  std::exception_ptr first_failure;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::minutes(5);
  const auto run_phase = [&](const PluginInstance& plugin,
                             pih_lifecycle_callback_v1 callback,
                             const char* phase) -> bool {
    try {
      for (;;) {
        const auto status = callback(plugin.api().context);
        if (pih_status_is_ok_v1(&status)) return true;
        if (!pih_status_is_valid_v1(&status) ||
            status.code != PIH_STATUS_UNAVAILABLE_V1 ||
            std::chrono::steady_clock::now() >= deadline) {
          ThrowLifecycleFailure(plugin, status, phase);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } catch (...) {
      if (first_failure == nullptr) first_failure = std::current_exception();
      return false;
    }
  };
  std::size_t drained_count = 0;
  for (auto plugin = plugins.rbegin(); plugin != plugins.rend(); ++plugin) {
    if (!run_phase(*plugin, plugin->api().lifecycle.drain,
                   "plugin_drain_failed")) {
      break;
    }
    ++drained_count;
  }
  std::size_t stopped_count = 0;
  for (auto plugin = plugins.rbegin(); stopped_count != drained_count;
       ++plugin) {
    if (!run_phase(*plugin, plugin->api().lifecycle.stop,
                   "plugin_stop_failed")) {
      break;
    }
    ++stopped_count;
  }
  std::size_t disposed_count = 0;
  for (auto plugin = plugins.rbegin(); disposed_count != stopped_count;
       ++plugin) {
    if (!run_phase(*plugin, plugin->api().lifecycle.dispose,
                   "plugin_dispose_failed")) {
      break;
    }
    ++disposed_count;
  }
  if (first_failure != nullptr) std::rethrow_exception(first_failure);
}

}  // namespace pih::microkernel
