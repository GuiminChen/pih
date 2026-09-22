#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

struct OutputSlotArenaLimits final {
  std::uint32_t maximum_sequences;
  std::uint32_t shared_delta_slots;
  std::uint32_t terminal_slots_per_sequence;
  std::uint32_t slot_bytes;
  std::uint32_t maximum_chain_slots;
  std::uint32_t per_sequence_dynamic_slots;
};

enum class OutputSlotState : std::uint8_t {
  kFree = 0,
  kBuildOwned,
  kWriterOwned,
  kOrphanDrain,
  kScrubPending,
};

struct OutputFrameLease final {
  std::uint32_t sequence_slot;
  std::uint64_t sequence_generation;
  std::uint64_t lease_generation;
  std::uint32_t slot_count;
  std::uint64_t frame_bytes;
  bool terminal;
  std::array<std::uint32_t, 64> slot_indices{};
};

class OutputSlotArena final {
 public:
  static constexpr std::string_view kAbi =
      "per_sequence_terminal_plus_shared_delta_v1";

  static Result<OutputSlotArena> Create(OutputSlotArenaLimits limits);
  OutputSlotArena(const OutputSlotArena&) = delete;
  OutputSlotArena& operator=(const OutputSlotArena&) = delete;
  OutputSlotArena(OutputSlotArena&&) noexcept = default;

  Result<std::uint32_t> admit(std::uint64_t sequence_generation);
  Status release_sequence(std::uint32_t sequence_slot,
                          std::uint64_t sequence_generation);
  Result<OutputFrameLease> reserve_frame(
      std::uint32_t sequence_slot, std::uint64_t sequence_generation,
      std::span<const std::byte> frame, bool terminal);
  Status abort_build(const OutputFrameLease& lease);
  Status publish(const OutputFrameLease& lease);
  Status advance_writer(const OutputFrameLease& lease,
                        std::uint64_t accepted_bytes);
  Status orphan(const OutputFrameLease& lease);
  Status finish_orphan(const OutputFrameLease& lease);
  Status scrub(const OutputFrameLease& lease);

  [[nodiscard]] std::uint32_t free_shared_slots() const noexcept;
  [[nodiscard]] std::uint32_t state_count(OutputSlotState state) const noexcept;

 private:
  struct Sequence final {
    bool admitted = false;
    std::uint64_t generation = 0;
    std::uint32_t dynamic_slots = 0;
  };
  struct Slot final {
    OutputSlotState state = OutputSlotState::kFree;
    std::uint64_t lease_generation = 0;
    std::uint32_t sequence_slot = 0;
    std::uint64_t frame_bytes = 0;
    std::uint64_t accepted_bytes = 0;
    std::uint32_t chain_count = 0;
    std::uint32_t chain_ordinal = 0;
    bool terminal = false;
  };

  OutputSlotArena(OutputSlotArenaLimits limits, std::vector<std::byte> bytes);
  Status validate(const OutputFrameLease& lease,
                  OutputSlotState expected) const;
  void transition(const OutputFrameLease& lease, OutputSlotState state);

  OutputSlotArenaLimits limits_{};
  std::vector<std::byte> bytes_;
  std::vector<Sequence> sequences_;
  std::vector<Slot> slots_;
  std::uint64_t next_lease_generation_ = 1;
};

}  // namespace pih
