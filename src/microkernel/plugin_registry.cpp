#include "microkernel/plugin_registry.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace pih::microkernel {
namespace {

constexpr std::size_t kMaximumPluginIdentifierBytes = 256;

bool ValidPluginId(const std::string& plugin_id) {
  if (plugin_id.empty() ||
      plugin_id.size() > kMaximumPluginIdentifierBytes ||
      plugin_id.front() == '.' ||
      plugin_id.back() == '.') {
    return false;
  }
  for (const char byte : plugin_id) {
    if (!((byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
          byte == '.' || byte == '-')) {
      return false;
    }
  }
  return plugin_id.find("..") == std::string::npos;
}

}  // namespace

void PluginRegistry::Add(std::string plugin_id) {
  if (sealed_) {
    throw std::logic_error("plugin_registry_sealed");
  }
  if (!ValidPluginId(plugin_id)) {
    throw std::invalid_argument("plugin_id_invalid");
  }
  if (!ids_.insert(std::move(plugin_id)).second) {
    throw std::invalid_argument("plugin_id_duplicate");
  }
}

void PluginRegistry::Seal() {
  if (sealed_) {
    throw std::logic_error("plugin_registry_already_sealed");
  }
  if (ids_.empty()) {
    throw std::logic_error("plugin_registry_empty");
  }
  sealed_ = true;
}

bool PluginRegistry::Contains(const std::string& plugin_id) const {
  return ids_.find(plugin_id) != ids_.end();
}

}  // namespace pih::microkernel
