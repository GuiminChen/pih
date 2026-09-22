#pragma once

#include "pih/model/engine_network_namespace_inventory.h"

namespace pih {

enum class EngineNetworkSocketLifecycle : std::uint8_t {
  kActive,
  kTimeWait,
};

struct EngineNetworkSocketCensusRow final {
  bool ownership_complete = false;
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
  EngineNetworkSocketLifecycle lifecycle =
      EngineNetworkSocketLifecycle::kActive;
  bool kernel_backing_complete = false;
  std::uint64_t kernel_backing_bytes = 0;
};

struct EngineNetworkNamespaceRawSnapshot final {
  std::uint64_t namespace_identity = 0;
  bool namespace_handle_present = false;
  bool namespace_destroyed = false;
  std::vector<EngineNetworkSocketCensusRow> rows;
};

class EngineNetworkNamespaceCensusBackend {
 public:
  virtual ~EngineNetworkNamespaceCensusBackend() = default;
  virtual Result<EngineNetworkNamespaceRawSnapshot> capture(
      std::uint64_t namespace_identity) = 0;
};

class EngineNetworkNamespaceCensusCompiler final
    : public EngineNetworkNamespaceOperations {
 public:
  static Result<EngineNetworkNamespaceCensusCompiler> Create(
      std::uint64_t namespace_identity,
      EngineNetworkNamespaceCensusBackend& backend);

  Result<EngineNetworkNamespaceSnapshot> capture(
      std::uint64_t namespace_identity) override;

 private:
  EngineNetworkNamespaceCensusCompiler(
      std::uint64_t namespace_identity,
      EngineNetworkNamespaceCensusBackend& backend) noexcept
      : namespace_identity_(namespace_identity), backend_(&backend) {}

  std::uint64_t namespace_identity_ = 0;
  EngineNetworkNamespaceCensusBackend* backend_ = nullptr;
};

}  // namespace pih
