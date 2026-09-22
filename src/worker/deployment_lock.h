#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pih::worker {

struct LockedKernelPack final {
  std::string pack_id;
  std::string pack_version;
  std::string pack_abi;
  std::string architecture;
  std::string binary;
  std::string binary_sha256_hex;
};

struct LockedPlugin final {
  std::string plugin_id;
  std::string plugin_version;
  std::string entrypoint;
  std::string entrypoint_sha256_hex;
};

struct LockedCapability final {
  std::string capability_id;
  std::string provider_id;
  std::string contract_id;
  std::uint32_t threading_model{};
  std::uint32_t scope{};
  std::uint32_t cardinality{};
};

struct EngineLockConfig final {
  std::string capability_id;
  std::string contract_id;
  uint64_t activation_epoch{};
  std::string configuration_json;
  std::string smoke_http_body_json;
};

struct DevelopmentLock final {
  std::vector<LockedKernelPack> kernel_packs;
  std::vector<LockedPlugin> plugins;
  std::vector<LockedCapability> capabilities;
  std::string deployment_root;
  bool has_engine{};
  EngineLockConfig engine;
};

DevelopmentLock LoadDevelopmentLock(const std::string& absolute_path);
// Parse an already authenticated snapshot; absolute_path supplies only the
// deployment-root context. Does not reopen the lock file itself.
DevelopmentLock ParseDevelopmentLock(const std::string& bytes, const std::string& absolute_path);

}  // namespace pih::worker
