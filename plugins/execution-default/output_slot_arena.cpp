#include "pih/scheduler/output_slot_arena.h"

#include <algorithm>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

OutputSlotArena::OutputSlotArena(OutputSlotArenaLimits limits,
                                 std::vector<std::byte> bytes)
    : limits_(limits),
      bytes_(std::move(bytes)),
      sequences_(limits.maximum_sequences),
      slots_(limits.shared_delta_slots +
             limits.maximum_sequences * limits.terminal_slots_per_sequence) {}

Result<OutputSlotArena> OutputSlotArena::Create(OutputSlotArenaLimits limits) {
  if (limits.maximum_sequences == 0 || limits.shared_delta_slots == 0 ||
      limits.terminal_slots_per_sequence == 0 || limits.slot_bytes == 0 ||
      limits.maximum_chain_slots == 0 || limits.maximum_chain_slots > 64 ||
      limits.per_sequence_dynamic_slots == 0 ||
      limits.per_sequence_dynamic_slots > limits.shared_delta_slots) {
    return Status::InvalidArgument("output slot arena limits are invalid");
  }
  auto terminal = checked_mul_u64(limits.maximum_sequences,
                                  limits.terminal_slots_per_sequence);
  if (!terminal.ok()) return terminal.status();
  auto total_slots = checked_add_u64(limits.shared_delta_slots, *terminal);
  if (!total_slots.ok() || *total_slots > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("output slot count overflows");
  }
  auto bytes = checked_mul_u64(*total_slots, limits.slot_bytes);
  if (!bytes.ok() || *bytes > std::vector<std::byte>().max_size()) {
    return Status::ResourceExhausted("output slot backing is too large");
  }
  try {
    return OutputSlotArena(limits, std::vector<std::byte>(*bytes));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("output slot backing allocation failed");
  }
}

Result<std::uint32_t> OutputSlotArena::admit(std::uint64_t generation) {
  if (generation == 0) return Status::InvalidArgument("sequence generation is zero");
  for (const auto& sequence : sequences_) {
    if (sequence.admitted && sequence.generation == generation) {
      return Status::FailedPrecondition("sequence generation is already admitted");
    }
  }
  for (std::uint32_t i = 0; i < sequences_.size(); ++i) {
    if (!sequences_[i].admitted) {
      sequences_[i] = {true, generation, 0};
      return i;
    }
  }
  return Status::ResourceExhausted("output sequence arena is full");
}

Status OutputSlotArena::release_sequence(std::uint32_t index,
                                         std::uint64_t generation) {
  if (index >= sequences_.size() || !sequences_[index].admitted ||
      sequences_[index].generation != generation) {
    return Status::FailedPrecondition("output sequence lease is stale");
  }
  for (const auto& slot : slots_) {
    if (slot.state != OutputSlotState::kFree && slot.sequence_slot == index) {
      return Status::FailedPrecondition("output sequence still owns slots");
    }
  }
  sequences_[index] = {};
  return Status::Ok();
}

Result<OutputFrameLease> OutputSlotArena::reserve_frame(
    std::uint32_t index, std::uint64_t generation,
    std::span<const std::byte> frame, bool terminal) {
  if (index >= sequences_.size() || !sequences_[index].admitted ||
      sequences_[index].generation != generation || frame.empty()) {
    return Status::FailedPrecondition("output frame sequence is stale or empty");
  }
  const std::uint64_t count64 =
      1U + (frame.size() - 1U) / limits_.slot_bytes;
  const std::uint32_t count = static_cast<std::uint32_t>(count64);
  if (count64 == 0 || count64 > limits_.maximum_chain_slots ||
      (terminal && count64 > limits_.terminal_slots_per_sequence) ||
      (!terminal && sequences_[index].dynamic_slots + count64 >
                        limits_.per_sequence_dynamic_slots)) {
    return Status::ResourceExhausted("output frame exceeds its slot envelope");
  }
  if (next_lease_generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return Status::FailedPrecondition("output lease generation wrapped");
  }
  const std::uint32_t begin = terminal
      ? limits_.shared_delta_slots + index * limits_.terminal_slots_per_sequence
      : 0;
  const std::uint32_t end = terminal
      ? begin + limits_.terminal_slots_per_sequence
      : limits_.shared_delta_slots;
  OutputFrameLease lease{index, generation, next_lease_generation_++, count,
                         frame.size(), terminal, {}};
  std::uint32_t found = 0;
  for (std::uint32_t slot_index = begin;
       slot_index < end && found < count; ++slot_index) {
    if (slots_[slot_index].state != OutputSlotState::kFree) continue;
    lease.slot_indices[found++] = slot_index;
  }
  if (found != count) return Status::ResourceExhausted("output slot credits are exhausted");
  std::size_t copied = 0;
  for (std::uint32_t i = 0; i < lease.slot_count; ++i) {
    const auto slot_index = lease.slot_indices[i];
    auto& slot = slots_[slot_index];
    slot.state = OutputSlotState::kBuildOwned;
    slot.lease_generation = lease.lease_generation;
    slot.sequence_slot = index;
    slot.frame_bytes = frame.size();
    slot.accepted_bytes = 0;
    slot.chain_count = lease.slot_count;
    slot.chain_ordinal = i;
    slot.terminal = terminal;
    const auto amount = std::min<std::size_t>(limits_.slot_bytes,
                                              frame.size() - copied);
    std::copy_n(frame.begin() + copied, amount,
                bytes_.begin() + static_cast<std::size_t>(slot_index) * limits_.slot_bytes);
    copied += amount;
  }
  if (!terminal) sequences_[index].dynamic_slots += lease.slot_count;
  return lease;
}

Status OutputSlotArena::validate(const OutputFrameLease& lease,
                                 OutputSlotState expected) const {
  if (lease.sequence_slot >= sequences_.size() || lease.slot_count == 0 ||
      lease.slot_count > 64 || !sequences_[lease.sequence_slot].admitted ||
      sequences_[lease.sequence_slot].generation != lease.sequence_generation) {
    return Status::FailedPrecondition("output frame lease is stale");
  }
  for (std::uint32_t i = 0; i < lease.slot_count; ++i) {
    const auto index = lease.slot_indices[i];
    if (index >= slots_.size() || slots_[index].state != expected ||
        slots_[index].lease_generation != lease.lease_generation ||
        slots_[index].sequence_slot != lease.sequence_slot ||
        slots_[index].frame_bytes != lease.frame_bytes ||
        slots_[index].chain_count != lease.slot_count ||
        slots_[index].chain_ordinal != i ||
        slots_[index].terminal != lease.terminal) {
      return Status::FailedPrecondition("output frame slot ownership drifted");
    }
  }
  return Status::Ok();
}

void OutputSlotArena::transition(const OutputFrameLease& lease,
                                 OutputSlotState state) {
  for (std::uint32_t i = 0; i < lease.slot_count; ++i) {
    slots_[lease.slot_indices[i]].state = state;
  }
}

Status OutputSlotArena::abort_build(const OutputFrameLease& lease) {
  auto status = validate(lease, OutputSlotState::kBuildOwned);
  if (!status.ok()) return status;
  for (std::uint32_t i = 0; i < lease.slot_count; ++i) {
    const auto index = lease.slot_indices[i];
    std::fill_n(bytes_.begin() + static_cast<std::size_t>(index) * limits_.slot_bytes,
                limits_.slot_bytes, std::byte{0});
    slots_[index] = {};
  }
  if (!lease.terminal) sequences_[lease.sequence_slot].dynamic_slots -= lease.slot_count;
  return Status::Ok();
}

Status OutputSlotArena::publish(const OutputFrameLease& lease) {
  auto status = validate(lease, OutputSlotState::kBuildOwned);
  if (status.ok()) transition(lease, OutputSlotState::kWriterOwned);
  return status;
}

Status OutputSlotArena::advance_writer(const OutputFrameLease& lease,
                                       std::uint64_t accepted) {
  auto status = validate(lease, OutputSlotState::kWriterOwned);
  if (!status.ok()) return status;
  auto& head = slots_[lease.slot_indices[0]];
  if (accepted == 0 || accepted > lease.frame_bytes - head.accepted_bytes) {
    return Status::InvalidArgument("partial write progress is invalid");
  }
  head.accepted_bytes += accepted;
  if (head.accepted_bytes == lease.frame_bytes) {
    transition(lease, OutputSlotState::kScrubPending);
  }
  return Status::Ok();
}

Status OutputSlotArena::orphan(const OutputFrameLease& lease) {
  auto status = validate(lease, OutputSlotState::kWriterOwned);
  if (status.ok()) transition(lease, OutputSlotState::kOrphanDrain);
  return status;
}
Status OutputSlotArena::finish_orphan(const OutputFrameLease& lease) {
  auto status = validate(lease, OutputSlotState::kOrphanDrain);
  if (status.ok()) transition(lease, OutputSlotState::kScrubPending);
  return status;
}
Status OutputSlotArena::scrub(const OutputFrameLease& lease) {
  auto status = validate(lease, OutputSlotState::kScrubPending);
  if (!status.ok()) return status;
  for (std::uint32_t i = 0; i < lease.slot_count; ++i) {
    const auto index = lease.slot_indices[i];
    std::fill_n(bytes_.begin() + static_cast<std::size_t>(index) * limits_.slot_bytes,
                limits_.slot_bytes, std::byte{0});
    slots_[index] = {};
  }
  if (!lease.terminal) sequences_[lease.sequence_slot].dynamic_slots -= lease.slot_count;
  return Status::Ok();
}

std::uint32_t OutputSlotArena::free_shared_slots() const noexcept {
  return static_cast<std::uint32_t>(std::count_if(
      slots_.begin(), slots_.begin() + limits_.shared_delta_slots,
      [](const Slot& slot) { return slot.state == OutputSlotState::kFree; }));
}
std::uint32_t OutputSlotArena::state_count(OutputSlotState state) const noexcept {
  return static_cast<std::uint32_t>(std::count_if(
      slots_.begin(), slots_.end(),
      [state](const Slot& slot) { return slot.state == state; }));
}

}  // namespace pih
