#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

enum class ControllerSamplingMode : std::uint8_t {
  kGreedy,
  kStochastic,
};

struct ControllerRequestSampling final {
  static constexpr std::uint32_t kMaximumStopTokenIds = 16;
  ControllerSamplingMode mode = ControllerSamplingMode::kGreedy;
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::optional<std::uint32_t> top_k;
  std::uint64_t effective_seed = 0;
  std::uint64_t sample_ordinal = 0;
  std::uint32_t minimum_new_tokens = 0;
  bool logprobs_enabled = false;
  std::uint32_t top_logprobs_count = 0;
  std::array<std::uint32_t, kMaximumStopTokenIds> stop_token_ids{};
  std::uint32_t stop_token_count = 0;
  bool operator==(const ControllerRequestSampling&) const = default;
};

struct ControllerRequestArenaLimits final {
  std::uint32_t maximum_requests;
  std::uint32_t maximum_prompt_tokens_per_request;
  std::uint32_t maximum_context_tokens;
};

struct ControllerRequestReceipt final {
  std::uint32_t slot_index;
  std::uint64_t slot_generation;
  std::uint64_t request_generation;
  Sha256Digest payload_digest;
};

struct ControllerRequestView final {
  std::uint32_t slot_index;
  std::uint64_t slot_generation;
  std::uint64_t request_generation;
  std::span<const std::uint32_t> prompt_token_ids;
  std::uint32_t maximum_new_tokens;
  ControllerRequestSampling sampling;
  Sha256Digest payload_digest;
};

Result<Sha256Digest> controller_request_payload_digest(
    std::uint64_t request_generation,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t maximum_new_tokens,
    const ControllerRequestSampling& sampling = {});

class ControllerRequestArena final {
 public:
  static constexpr std::string_view kAbi = "fixed_controller_request_arena_v1";

  static Result<ControllerRequestArena> Create(
      ControllerRequestArenaLimits limits);
  ControllerRequestArena(const ControllerRequestArena&) = delete;
  ControllerRequestArena& operator=(const ControllerRequestArena&) = delete;
  ControllerRequestArena(ControllerRequestArena&& other) noexcept;
  ControllerRequestArena& operator=(ControllerRequestArena&&) = delete;

  Result<ControllerRequestReceipt> publish(
      std::uint64_t request_generation,
      std::span<const std::uint32_t> prompt_token_ids,
      std::uint32_t maximum_new_tokens,
      const ControllerRequestSampling& sampling = {});
  Result<ControllerRequestView> claim(
      std::uint64_t request_generation, Sha256Digest payload_digest);
  Status rollback_published(const ControllerRequestReceipt& receipt);
  Status release_claimed(std::uint32_t slot_index,
                         std::uint64_t slot_generation);

  [[nodiscard]] std::uint32_t published_count() const;
  [[nodiscard]] std::uint32_t claimed_count() const;

 private:
  enum class SlotState : std::uint8_t { kFree, kPublished, kClaimed };
  struct Slot final {
    SlotState state = SlotState::kFree;
    std::uint64_t generation = 0;
    std::uint64_t request_generation = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t maximum_new_tokens = 0;
    ControllerRequestSampling sampling{};
    Sha256Digest payload_digest{};
  };

  ControllerRequestArena(ControllerRequestArenaLimits limits,
                         std::vector<std::uint32_t> tokens);
  void clear_slot(std::uint32_t slot_index) noexcept;

  mutable std::mutex mutex_;
  ControllerRequestArenaLimits limits_{};
  std::vector<std::uint32_t> tokens_;
  std::vector<Slot> slots_;
};

}  // namespace pih
