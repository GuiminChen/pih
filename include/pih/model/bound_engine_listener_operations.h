#pragma once

#include <span>

#include "pih/model/engine_listener_inventory.h"

namespace pih {

struct EngineListenerHandleBinding final {
  std::uint64_t listener_identity = 0;
  std::int32_t descriptor = -1;
};

struct EngineListenerKernelState final {
  bool open = false;
  bool socket = false;
  bool kernel_listening = false;
  std::uint64_t device_identity = 0;
  std::uint64_t inode_identity = 0;
};

class EngineListenerKernelBackend {
 public:
  virtual ~EngineListenerKernelBackend() = default;
  virtual Result<EngineListenerKernelState> inspect(
      std::int32_t descriptor) = 0;
};

class EngineListenerAcceptAuthority {
 public:
  virtual ~EngineListenerAcceptAuthority() = default;
  virtual Result<bool> enabled(std::uint64_t listener_identity) = 0;
};

class BoundEngineListenerOperations final : public EngineListenerOperations {
 public:
  static Result<BoundEngineListenerOperations> Create(
      std::span<const EngineListenerHandleBinding> bindings,
      EngineListenerKernelBackend& kernel,
      EngineListenerAcceptAuthority& authority);

  Result<EngineListenerObservation> observe(
      std::uint64_t listener_identity) override;

 private:
  BoundEngineListenerOperations(
      std::vector<EngineListenerHandleBinding> bindings,
      EngineListenerKernelBackend& kernel,
      EngineListenerAcceptAuthority& authority) noexcept
      : bindings_(std::move(bindings)), kernel_(&kernel),
        authority_(&authority) {}

  std::vector<EngineListenerHandleBinding> bindings_;
  EngineListenerKernelBackend* kernel_ = nullptr;
  EngineListenerAcceptAuthority* authority_ = nullptr;
};

}  // namespace pih
