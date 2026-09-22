#pragma once

#include "pih/backend/cuda/typed_copy_plan.h"

namespace pih {

class NvidiaTypedCopyDriver final : public TypedCopyDriver {
 public:
  static Result<NvidiaTypedCopyDriver> Create();

  [[nodiscard]] std::uintptr_t context_identity() const noexcept override {
    return context_identity_;
  }
  Status copy(CudaCopyKind kind, std::uintptr_t destination,
              std::uintptr_t source, std::uint64_t bytes,
              DriverStreamHandle stream) override;

 private:
  explicit NvidiaTypedCopyDriver(std::uintptr_t context_identity)
      : context_identity_(context_identity) {}
  Status require_current_context() const;

  std::uintptr_t context_identity_;
};

}  // namespace pih
