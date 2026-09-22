#pragma once

#include <string>
#include <unordered_set>

namespace pih::microkernel {

class PluginRegistry final {
 public:
  void Add(std::string plugin_id);
  void Seal();
  [[nodiscard]] bool Contains(const std::string& plugin_id) const;
  [[nodiscard]] bool sealed() const noexcept { return sealed_; }

 private:
  std::unordered_set<std::string> ids_;
  bool sealed_{};
};

}  // namespace pih::microkernel
