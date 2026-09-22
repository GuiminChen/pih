#pragma once

#include <string>
#include <string_view>

#include "pih/plugin_sdk/abi.h"

namespace pih::microkernel {

class PluginInstance final {
 public:
  PluginInstance(void* native_handle, pih_plugin_api_v1 api);
  PluginInstance(const PluginInstance&) = delete;
  PluginInstance& operator=(const PluginInstance&) = delete;
  PluginInstance(PluginInstance&& other) noexcept;
  PluginInstance& operator=(PluginInstance&&) = delete;
  ~PluginInstance() = default;

  [[nodiscard]] const std::string& id() const noexcept { return id_; }
  [[nodiscard]] const std::string& version() const noexcept { return version_; }
  [[nodiscard]] const pih_plugin_api_v1& api() const noexcept { return api_; }

 private:
  void* native_handle_{};
  pih_plugin_api_v1 api_{};
  std::string id_;
  std::string version_;
};

class PluginLoader final {
 public:
  PluginLoader() = default;
  PluginLoader(const PluginLoader&) = delete;
  PluginLoader& operator=(const PluginLoader&) = delete;

  [[nodiscard]] PluginInstance Load(const std::string& path,
                                    const pih_host_api_v1& host,
                                    std::string_view expected_sha256_hex = {});
  void Seal();
  [[nodiscard]] bool sealed() const noexcept { return sealed_; }

 private:
  bool sealed_{};
};

}  // namespace pih::microkernel
