#pragma once

#include <span>

#include "pih/model/engine_network_namespace_census_compiler.h"

namespace pih {

struct EngineNetworkPortOwnerBinding final {
  std::uint16_t first_port = 0;
  std::uint16_t last_port = 0;
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
};

struct EngineNetworkPortCensusRow final {
  std::uint16_t local_port = 0;
  EngineNetworkSocketLifecycle lifecycle =
      EngineNetworkSocketLifecycle::kActive;
  bool kernel_backing_complete = false;
  std::uint64_t kernel_backing_bytes = 0;
};

struct EngineNetworkPortRawSnapshot final {
  std::uint64_t namespace_identity = 0;
  bool namespace_handle_present = false;
  bool namespace_destroyed = false;
  std::vector<EngineNetworkPortCensusRow> rows;
};

class EngineNetworkPortCensusBackend {
 public:
  virtual ~EngineNetworkPortCensusBackend() = default;
  virtual Result<EngineNetworkPortRawSnapshot> capture(
      std::uint64_t namespace_identity) = 0;
};

class EngineNetworkPortOwnerClassifier final
    : public EngineNetworkNamespaceCensusBackend {
 public:
  static Result<EngineNetworkPortOwnerClassifier> Create(
      std::uint64_t namespace_identity,
      std::span<const EngineNetworkPortOwnerBinding> bindings,
      EngineNetworkPortCensusBackend& backend);

  Result<EngineNetworkNamespaceRawSnapshot> capture(
      std::uint64_t namespace_identity) override;

 private:
  EngineNetworkPortOwnerClassifier(
      std::uint64_t namespace_identity,
      std::vector<EngineNetworkPortOwnerBinding> bindings,
      EngineNetworkPortCensusBackend& backend) noexcept
      : namespace_identity_(namespace_identity), bindings_(std::move(bindings)),
        backend_(&backend) {}

  std::uint64_t namespace_identity_ = 0;
  std::vector<EngineNetworkPortOwnerBinding> bindings_;
  EngineNetworkPortCensusBackend* backend_ = nullptr;
};

}  // namespace pih
