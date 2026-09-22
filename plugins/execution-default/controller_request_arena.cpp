#include "pih/scheduler/controller_request_arena.h"

#include <array>
#include <bit>
#include <cmath>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {
constexpr std::string_view kDigestDomain = "pih-controller-request-v1";

Status update_u32(Sha256& digest, std::uint32_t value) {
  const std::array wire{static_cast<std::byte>(value),
                        static_cast<std::byte>(value >> 8U),
                        static_cast<std::byte>(value >> 16U),
                        static_cast<std::byte>(value >> 24U)};
  return digest.update(wire);
}
Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t i = 0; i < wire.size(); ++i) {
    wire[i] = static_cast<std::byte>(value >> (i * 8U));
  }
  return digest.update(wire);
}

Status validate_sampling(const ControllerRequestSampling& sampling,
                         std::uint32_t maximum_new_tokens) {
  if (sampling.sample_ordinal != 0 ||
      sampling.minimum_new_tokens > maximum_new_tokens ||
      sampling.top_logprobs_count > 20U ||
      (!sampling.logprobs_enabled && sampling.top_logprobs_count != 0U) ||
      sampling.stop_token_count >
          ControllerRequestSampling::kMaximumStopTokenIds) {
    return Status::InvalidArgument(
        "controller request sampling descriptor is invalid");
  }
  for (std::uint32_t i = 0; i < sampling.stop_token_ids.size(); ++i) {
    if (i >= sampling.stop_token_count) {
      if (sampling.stop_token_ids[i] != 0)
        return Status::InvalidArgument(
            "controller stop-token descriptor tail is not canonical");
      continue;
    }
    if (sampling.stop_token_ids[i] >= 151936U)
      return Status::InvalidArgument(
          "controller stop-token id is outside the vocabulary");
    for (std::uint32_t previous = 0; previous < i; ++previous) {
      if (sampling.stop_token_ids[previous] == sampling.stop_token_ids[i])
        return Status::InvalidArgument(
            "controller stop-token descriptor contains duplicates");
    }
  }
  if (sampling.mode == ControllerSamplingMode::kGreedy) {
    if (sampling.temperature != 0.0F || sampling.top_p != 1.0F ||
        sampling.top_k.has_value()) {
      return Status::InvalidArgument(
          "controller greedy sampling parameters conflict");
    }
  } else if (sampling.mode != ControllerSamplingMode::kStochastic ||
             !std::isfinite(sampling.temperature) ||
             sampling.temperature <= 0.0F || sampling.temperature > 2.0F ||
             !std::isfinite(sampling.top_p) || sampling.top_p <= 0.0F ||
             sampling.top_p > 1.0F ||
             (sampling.top_k.has_value() &&
              (*sampling.top_k == 0 || *sampling.top_k > 151936U))) {
    return Status::InvalidArgument(
        "controller stochastic sampling parameters are invalid");
  }
  return Status::Ok();
}
}  // namespace

Result<Sha256Digest> controller_request_payload_digest(
    std::uint64_t request_generation,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t maximum_new_tokens,
    const ControllerRequestSampling& sampling) {
  if (request_generation == 0 || prompt_token_ids.empty() ||
      prompt_token_ids.size() > std::numeric_limits<std::uint32_t>::max() ||
      maximum_new_tokens == 0 ||
      !validate_sampling(sampling, maximum_new_tokens).ok()) {
    return Status::InvalidArgument("controller request digest input is invalid");
  }
  Sha256 digest;
  Status status = digest.update(std::as_bytes(std::span(kDigestDomain)));
  if (status.ok()) status = update_u64(digest, request_generation);
  if (status.ok()) {
    status = update_u32(digest,
                        static_cast<std::uint32_t>(prompt_token_ids.size()));
  }
  if (status.ok()) status = update_u32(digest, maximum_new_tokens);
  if (status.ok()) status = update_u32(
      digest, static_cast<std::uint32_t>(sampling.mode));
  if (status.ok()) status = update_u32(
      digest, std::bit_cast<std::uint32_t>(sampling.temperature));
  if (status.ok()) status = update_u32(
      digest, std::bit_cast<std::uint32_t>(sampling.top_p));
  if (status.ok()) status = update_u32(digest, sampling.top_k.has_value());
  if (status.ok()) status = update_u32(digest, sampling.top_k.value_or(0));
  if (status.ok()) status = update_u64(digest, sampling.effective_seed);
  if (status.ok()) status = update_u64(digest, sampling.sample_ordinal);
  if (status.ok()) status = update_u32(digest, sampling.minimum_new_tokens);
  if (status.ok()) status = update_u32(digest, sampling.logprobs_enabled);
  if (status.ok()) status = update_u32(digest, sampling.top_logprobs_count);
  if (status.ok()) status = update_u32(digest, sampling.stop_token_count);
  for (std::uint32_t i = 0;
       status.ok() && i < sampling.stop_token_count; ++i) {
    status = update_u32(digest, sampling.stop_token_ids[i]);
  }
  for (const auto token : prompt_token_ids) {
    if (status.ok()) status = update_u32(digest, token);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

ControllerRequestArena::ControllerRequestArena(
    ControllerRequestArenaLimits limits, std::vector<std::uint32_t> tokens)
    : limits_(limits),
      tokens_(std::move(tokens)),
      slots_(limits.maximum_requests) {}

ControllerRequestArena::ControllerRequestArena(
    ControllerRequestArena&& other) noexcept {
  std::scoped_lock lock(other.mutex_);
  limits_ = other.limits_;
  tokens_ = std::move(other.tokens_);
  slots_ = std::move(other.slots_);
}

Result<ControllerRequestArena> ControllerRequestArena::Create(
    ControllerRequestArenaLimits limits) {
  if (limits.maximum_requests == 0 ||
      limits.maximum_prompt_tokens_per_request == 0 ||
      limits.maximum_context_tokens == 0 ||
      limits.maximum_prompt_tokens_per_request > limits.maximum_context_tokens) {
    return Status::InvalidArgument("controller request arena limits are invalid");
  }
  auto capacity = checked_mul_u64(
      limits.maximum_requests, limits.maximum_prompt_tokens_per_request);
  if (!capacity.ok() || *capacity > std::vector<std::uint32_t>().max_size()) {
    return Status::ResourceExhausted("controller request token arena is too large");
  }
  try {
    return ControllerRequestArena(
        limits, std::vector<std::uint32_t>(static_cast<std::size_t>(*capacity)));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("controller request token allocation failed");
  }
}

Result<ControllerRequestReceipt> ControllerRequestArena::publish(
    std::uint64_t request_generation,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t maximum_new_tokens,
    const ControllerRequestSampling& sampling) {
  if (request_generation == 0 || prompt_token_ids.empty() ||
      prompt_token_ids.size() > limits_.maximum_prompt_tokens_per_request ||
      maximum_new_tokens == 0 ||
      prompt_token_ids.size() >
          limits_.maximum_context_tokens -
              std::min(maximum_new_tokens, limits_.maximum_context_tokens) ||
      maximum_new_tokens > limits_.maximum_context_tokens ||
      !validate_sampling(sampling, maximum_new_tokens).ok()) {
    return Status::InvalidArgument("controller request exceeds its token profile");
  }
  for (const auto token : prompt_token_ids) {
    if (token >= 151936) {
      return Status::InvalidArgument("controller request has invalid token id");
    }
  }
  auto digest = controller_request_payload_digest(
      request_generation, prompt_token_ids, maximum_new_tokens, sampling);
  if (!digest.ok()) return digest.status();
  std::lock_guard lock(mutex_);
  Slot* free_slot = nullptr;
  std::uint32_t free_index = 0;
  for (std::uint32_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].state != SlotState::kFree &&
        slots_[i].request_generation == request_generation) {
      return Status::FailedPrecondition("request generation is already resident");
    }
    if (free_slot == nullptr && slots_[i].state == SlotState::kFree) {
      free_slot = &slots_[i];
      free_index = i;
    }
  }
  if (free_slot == nullptr) {
    return Status::ResourceExhausted("controller request arena is full");
  }
  if (free_slot->generation == std::numeric_limits<std::uint64_t>::max()) {
    return Status::FailedPrecondition("controller request slot generation wrapped");
  }
  const auto offset = static_cast<std::size_t>(free_index) *
                      limits_.maximum_prompt_tokens_per_request;
  std::copy(prompt_token_ids.begin(), prompt_token_ids.end(),
            tokens_.begin() + offset);
  ++free_slot->generation;
  free_slot->request_generation = request_generation;
  free_slot->prompt_tokens = static_cast<std::uint32_t>(prompt_token_ids.size());
  free_slot->maximum_new_tokens = maximum_new_tokens;
  free_slot->sampling = sampling;
  free_slot->payload_digest = *digest;
  free_slot->state = SlotState::kPublished;
  return ControllerRequestReceipt{free_index, free_slot->generation,
                                  request_generation, *digest};
}

Result<ControllerRequestView> ControllerRequestArena::claim(
    std::uint64_t request_generation, Sha256Digest payload_digest) {
  std::lock_guard lock(mutex_);
  for (std::uint32_t i = 0; i < slots_.size(); ++i) {
    auto& slot = slots_[i];
    if (slot.state == SlotState::kPublished &&
        slot.request_generation == request_generation) {
      if (slot.payload_digest != payload_digest) {
        return Status::FailedPrecondition("controller request digest drifted");
      }
      slot.state = SlotState::kClaimed;
      const auto offset = static_cast<std::size_t>(i) *
                          limits_.maximum_prompt_tokens_per_request;
      return ControllerRequestView{
          i, slot.generation, slot.request_generation,
          std::span(tokens_).subspan(offset, slot.prompt_tokens),
          slot.maximum_new_tokens, slot.sampling, slot.payload_digest};
    }
  }
  return Status::FailedPrecondition("controller request is not published");
}

void ControllerRequestArena::clear_slot(std::uint32_t slot_index) noexcept {
  auto& slot = slots_[slot_index];
  const auto offset = static_cast<std::size_t>(slot_index) *
                      limits_.maximum_prompt_tokens_per_request;
  std::fill_n(tokens_.begin() + offset,
              limits_.maximum_prompt_tokens_per_request, 0U);
  slot.state = SlotState::kFree;
  slot.request_generation = 0;
  slot.prompt_tokens = 0;
  slot.maximum_new_tokens = 0;
  slot.sampling = {};
  slot.payload_digest = {};
}

Status ControllerRequestArena::rollback_published(
    const ControllerRequestReceipt& receipt) {
  std::lock_guard lock(mutex_);
  if (receipt.slot_index >= slots_.size()) {
    return Status::InvalidArgument("controller request receipt index is invalid");
  }
  auto& slot = slots_[receipt.slot_index];
  if (slot.state != SlotState::kPublished ||
      slot.generation != receipt.slot_generation ||
      slot.request_generation != receipt.request_generation ||
      slot.payload_digest != receipt.payload_digest) {
    return Status::FailedPrecondition("controller request receipt is stale");
  }
  clear_slot(receipt.slot_index);
  return Status::Ok();
}

Status ControllerRequestArena::release_claimed(std::uint32_t slot_index,
                                                std::uint64_t slot_generation) {
  std::lock_guard lock(mutex_);
  if (slot_index >= slots_.size()) {
    return Status::InvalidArgument("controller request slot index is invalid");
  }
  auto& slot = slots_[slot_index];
  if (slot.state != SlotState::kClaimed || slot.generation != slot_generation) {
    return Status::FailedPrecondition("controller request claim is stale");
  }
  clear_slot(slot_index);
  return Status::Ok();
}

std::uint32_t ControllerRequestArena::published_count() const {
  std::lock_guard lock(mutex_);
  std::uint32_t count = 0;
  for (const auto& slot : slots_) count += slot.state == SlotState::kPublished;
  return count;
}

std::uint32_t ControllerRequestArena::claimed_count() const {
  std::lock_guard lock(mutex_);
  std::uint32_t count = 0;
  for (const auto& slot : slots_) count += slot.state == SlotState::kClaimed;
  return count;
}

}  // namespace pih
