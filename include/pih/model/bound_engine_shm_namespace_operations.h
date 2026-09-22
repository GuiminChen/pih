#pragma once

#include <span>
#include <string>
#include <string_view>

#include "pih/model/engine_shm_namespace_inventory.h"

namespace pih {

struct EngineShmNamespaceHandleBinding final {
  std::uint64_t object_identity = 0;
  std::int32_t directory_descriptor = -1;
  std::string basename;
};

struct EngineShmNamespaceStat final {
  bool present = false;
  bool regular_file = false;
  std::uint64_t device_identity = 0;
  std::uint64_t inode_identity = 0;
  std::uint64_t size_bytes = 0;
};

class EngineShmNamespaceStatBackend {
 public:
  virtual ~EngineShmNamespaceStatBackend() = default;
  virtual Result<EngineShmNamespaceStat> stat_at(
      std::int32_t directory_descriptor, std::string_view basename) = 0;
};

class BoundEngineShmNamespaceOperations final
    : public EngineShmNamespaceOperations {
 public:
  static Result<BoundEngineShmNamespaceOperations> Create(
      std::span<const EngineShmNamespaceHandleBinding> bindings,
      EngineShmNamespaceStatBackend& backend);

  Result<EngineShmObjectObservation> observe(
      std::uint64_t object_identity) override;

 private:
  BoundEngineShmNamespaceOperations(
      std::vector<EngineShmNamespaceHandleBinding> bindings,
      EngineShmNamespaceStatBackend& backend) noexcept
      : bindings_(std::move(bindings)), backend_(&backend) {}

  std::vector<EngineShmNamespaceHandleBinding> bindings_;
  EngineShmNamespaceStatBackend* backend_ = nullptr;
};

}  // namespace pih
