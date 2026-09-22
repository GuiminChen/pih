#pragma once

#include <cstdint>

#include "pih/backend/cuda/completion_frontier.h"
#include "pih/backend/cuda/verified_kernel_launcher.h"

namespace pih {

using DriverEventHandle = std::uintptr_t;

class CompletionEventDriver {
 public:
  virtual ~CompletionEventDriver() = default;
  virtual Status record(DriverEventHandle event,
                        DriverStreamHandle stream) = 0;
  virtual Result<CudaEventQueryResult> query(DriverEventHandle event) = 0;
};

struct CompletionPublicationEvidence final {
  bool submit_thread_last_error_clean;
  std::uint32_t device_error_code;
  bool engine_poisoned;
};

class CompletionEvidenceProvider {
 public:
  virtual ~CompletionEvidenceProvider() = default;
  virtual Result<CompletionPublicationEvidence> collect() = 0;
};

enum class CompletionEventSlotState : std::uint8_t {
  kIdle = 0,
  kRecorded,
  kCompleted,
  kPoisoned,
};

class CompletionEventSlot final {
 public:
  static Result<CompletionEventSlot> Create(DriverEventHandle event,
                                             std::uintptr_t context_identity);

  Status record(CompletionEventDriver& driver, DriverStreamHandle stream,
                std::uint64_t event_generation);
  Status poll(CompletionEventDriver& driver, CudaCompletionFrontier& frontier,
              CompletionEvidenceProvider& evidence_provider);
  Status release(std::uint64_t event_generation);

  [[nodiscard]] CompletionEventSlotState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::uint64_t event_generation() const noexcept {
    return event_generation_;
  }
  [[nodiscard]] std::uintptr_t context_identity() const noexcept {
    return context_identity_;
  }

 private:
  CompletionEventSlot(DriverEventHandle event, std::uintptr_t context_identity)
      : event_(event), context_identity_(context_identity) {}

  DriverEventHandle event_;
  std::uintptr_t context_identity_;
  std::uint64_t event_generation_ = 0;
  CompletionEventSlotState state_ = CompletionEventSlotState::kIdle;
};

}  // namespace pih
