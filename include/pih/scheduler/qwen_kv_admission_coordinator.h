#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pih/model/qwen3_kv_admission_transaction.h"
#include "pih/scheduler/controller_packed_driver.h"
#include "pih/scheduler/controller_runtime.h"

namespace pih {

class QwenKvAdmissionCoordinator final
    : public ControllerAdmissionParticipant {
 public:
  static Result<QwenKvAdmissionCoordinator> Create(
      QwenKvSlotPool& pool, ControllerPackedDriver& driver,
      std::uint32_t maximum_sequences,
      QwenKvCompletionEvent initial_last_use_event);

  Status reserve(const ControllerAdmissionView& admission) override;
  Status publish(std::uint64_t sequence_generation) override;
  Status rollback(std::uint64_t request_generation) override;
  Status drain(std::uint64_t sequence_generation,
               QwenBf16KvRecycler& recycler);

  [[nodiscard]] std::uint32_t active_admission_count() const noexcept;

 private:
  QwenKvAdmissionCoordinator(
      QwenKvSlotPool& pool, ControllerPackedDriver& driver,
      std::uint32_t maximum_sequences,
      QwenKvCompletionEvent initial_last_use_event)
      : pool_(&pool), driver_(&driver), admissions_(maximum_sequences),
        initial_last_use_event_(initial_last_use_event) {}

  std::optional<std::uint32_t> find(
      std::uint64_t sequence_generation) const noexcept;

  QwenKvSlotPool* pool_;
  ControllerPackedDriver* driver_;
  std::vector<std::optional<QwenKvAdmissionTransaction>> admissions_;
  QwenKvCompletionEvent initial_last_use_event_{};
};

}  // namespace pih
