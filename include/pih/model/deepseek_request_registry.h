#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "pih/model/deepseek_request_lifecycle.h"

namespace pih {

class DeepSeekRequestRegistry final {
 public:
  Status submit(std::uint64_t request_id, std::uint64_t request_generation);
  [[nodiscard]] Status validate_prepare(
      std::uint64_t request_id, std::uint64_t request_generation,
      std::uint64_t plan_sequence) const;
  Status prepare(std::uint64_t request_id, std::uint64_t request_generation,
                 std::uint64_t plan_sequence);
  [[nodiscard]] Status validate_commit(
      std::uint64_t request_id, std::uint64_t request_generation,
      std::uint64_t plan_sequence) const;
  Status commit(std::uint64_t request_id, std::uint64_t request_generation,
                std::uint64_t plan_sequence);
  [[nodiscard]] Status validate_abort_prepare(
      std::uint64_t request_id, std::uint64_t request_generation,
      std::uint64_t plan_sequence) const;
  Status abort_prepare(std::uint64_t request_id,
                       std::uint64_t request_generation,
                       std::uint64_t plan_sequence);
  [[nodiscard]] Status validate_cancel(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Status cancel(std::uint64_t request_id, std::uint64_t request_generation);
  [[nodiscard]] Status validate_backend_complete(
      std::uint64_t request_id, std::uint64_t request_generation,
      std::uint64_t plan_sequence) const;
  Status backend_complete(std::uint64_t request_id,
                          std::uint64_t request_generation,
                          std::uint64_t plan_sequence,
                          bool terminal = true);
  [[nodiscard]] Status validate_retire(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Status retire(std::uint64_t request_id, std::uint64_t request_generation);
  Status fail(std::uint64_t request_id, std::uint64_t request_generation);
  Status finish(std::uint64_t request_id, std::uint64_t request_generation);
  void fail_all() noexcept;
  void cancel_all() noexcept;

  Result<DeepSeekRequestState> state(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  [[nodiscard]] std::size_t active_count() const noexcept {
    return requests_.size();
  }

 private:
  Result<DeepSeekRequestLifecycle*> find(
      std::uint64_t request_id, std::uint64_t request_generation);
  Result<const DeepSeekRequestLifecycle*> find(
      std::uint64_t request_id, std::uint64_t request_generation) const;

  std::unordered_map<std::uint64_t, DeepSeekRequestLifecycle> requests_;
};

}  // namespace pih
