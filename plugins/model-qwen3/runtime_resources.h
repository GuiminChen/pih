#pragma once

#include "pih/backend/cuda/cuda_runtime_resources.h"

namespace pih::qwen_plugin {
// Qwen owns compute, KV scrub and diagnostic lanes only. Native handles are
// created and retired through the bound CUDA resource capability.
class RuntimeResources final {
 public:
  static Result<RuntimeResources> Create(std::int32_t ordinal, std::uint32_t rank,
      std::uint64_t generation, std::uint32_t flags, CudaRuntimeResourceDriver& driver);
  ~RuntimeResources();
  RuntimeResources(const RuntimeResources&) = delete;
  RuntimeResources& operator=(const RuntimeResources&) = delete;
  RuntimeResources(RuntimeResources&& other) noexcept;
  RuntimeResources& operator=(RuntimeResources&& other) noexcept;
  const CudaRuntimeResourceIdentity& identity() const noexcept { return identity_; }
 private:
  explicit RuntimeResources(CudaRuntimeResourceDriver& driver) : driver_(&driver) {}
  void Reset() noexcept;
  CudaRuntimeResourceIdentity identity_{};
  CudaRuntimeResourceDriver* driver_{};
};
}  // namespace pih::qwen_plugin
