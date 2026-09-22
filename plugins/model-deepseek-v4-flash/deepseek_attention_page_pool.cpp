#include "pih/model/deepseek_attention_page_pool.h"

#include <limits>

namespace pih {
namespace {

constexpr std::uint64_t kKindShift = 56;
constexpr std::uint64_t kSlotShift = 32;
constexpr std::uint64_t kSlotMask = 0x00FFFFFFULL;

bool known_kind(DeepSeekStatePoolKind kind) {
  return kind >= DeepSeekStatePoolKind::kRecentBf16 &&
         kind <= DeepSeekStatePoolKind::kRatio128MainBf16;
}

}  // namespace

Result<DeepSeekBlockHandle> DeepSeekBlockHandle::Create(
    DeepSeekStatePoolKind kind, std::uint32_t slot,
    std::uint32_t generation) {
  if (!known_kind(kind) || slot > kSlotMask || generation == 0) {
    return Status::InvalidArgument("DeepSeek block handle is invalid");
  }
  return DeepSeekBlockHandle{
      (static_cast<std::uint64_t>(kind) << kKindShift) |
      (static_cast<std::uint64_t>(slot) << kSlotShift) | generation};
}

DeepSeekStatePoolKind DeepSeekBlockHandle::pool_kind() const noexcept {
  return static_cast<DeepSeekStatePoolKind>(value >> kKindShift);
}

std::uint32_t DeepSeekBlockHandle::slot() const noexcept {
  return static_cast<std::uint32_t>((value >> kSlotShift) & kSlotMask);
}

std::uint32_t DeepSeekBlockHandle::generation() const noexcept {
  return static_cast<std::uint32_t>(value);
}

Result<DeepSeekRatio4PagePool> DeepSeekRatio4PagePool::Create(
    std::uint32_t page_pairs) {
  if (page_pairs == 0 || page_pairs > kSlotMask + 1U) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 page pair count is invalid");
  }
  return DeepSeekRatio4PagePool(std::vector<Slot>(page_pairs));
}

Result<DeepSeekRatio4PagePair> DeepSeekRatio4PagePool::reserve(
    std::uint32_t sequence, std::uint32_t layer_id,
    std::uint32_t logical_page) {
  if (sequence == 0 || layer_id >= 43) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 page identity is invalid");
  }
  if (generation_exhausted_) {
    return Status::FailedPrecondition(
        "DeepSeek attention page generation requires epoch rebuild");
  }
  for (const auto& slot : slots_) {
    if (slot.main != DeepSeekAttentionPageState::kFree &&
        slot.sequence == sequence && slot.layer_id == layer_id &&
        slot.logical_page == logical_page) {
      return Status::FailedPrecondition(
          "DeepSeek logical ratio-4 page already exists");
    }
  }
  return reserve_free(sequence, layer_id, logical_page);
}

Result<DeepSeekRatio4PagePair> DeepSeekRatio4PagePool::reserve_free(
    std::uint32_t sequence, std::uint32_t layer_id,
    std::uint32_t logical_page) {
  for (std::uint32_t ordinal = 0; ordinal < slots_.size(); ++ordinal) {
    auto& slot = slots_[ordinal];
    if (slot.main != DeepSeekAttentionPageState::kFree ||
        slot.index != DeepSeekAttentionPageState::kFree) {
      continue;
    }
    if (slot.generation == std::numeric_limits<std::uint32_t>::max()) {
      generation_exhausted_ = true;
      return Status::FailedPrecondition(
          "DeepSeek attention page generation requires epoch rebuild");
    }
    ++slot.generation;
    auto main = DeepSeekBlockHandle::Create(
        DeepSeekStatePoolKind::kRatio4MainBf16, ordinal, slot.generation);
    auto index = DeepSeekBlockHandle::Create(
        DeepSeekStatePoolKind::kRatio4IndexBf16, ordinal, slot.generation);
    if (!main.ok() || !index.ok()) {
      generation_exhausted_ = true;
      return Status::Internal("DeepSeek ratio-4 handle construction failed");
    }
    slot.sequence = sequence;
    slot.layer_id = layer_id;
    slot.logical_page = logical_page;
    slot.main = DeepSeekAttentionPageState::kReserved;
    slot.index = DeepSeekAttentionPageState::kReserved;
    --free_pairs_;
    ++reserved_pairs_;
    return DeepSeekRatio4PagePair{*main, *index, sequence, layer_id,
                                  logical_page, slot.generation};
  }
  return Status::ResourceExhausted(
      "DeepSeek ratio-4 page pair pool is exhausted");
}

Result<DeepSeekRatio4PagePair> DeepSeekRatio4PagePool::reserve_tail_cow(
    const DeepSeekRatio4PagePair& committed) {
  auto source = validate_pair(committed,
                              DeepSeekAttentionPageState::kPublished);
  if (!source.ok()) return source.status();
  if (generation_exhausted_) {
    return Status::FailedPrecondition(
        "DeepSeek attention page generation requires epoch rebuild");
  }
  return reserve_free(committed.sequence, committed.layer_id,
                      committed.logical_page);
}

Result<std::uint32_t> DeepSeekRatio4PagePool::validate_pair(
    const DeepSeekRatio4PagePair& pair,
    DeepSeekAttentionPageState expected) const {
  if (pair.main.pool_kind() != DeepSeekStatePoolKind::kRatio4MainBf16 ||
      pair.index.pool_kind() != DeepSeekStatePoolKind::kRatio4IndexBf16 ||
      pair.main.slot() != pair.index.slot() ||
      pair.main.generation() != pair.index.generation() ||
      pair.generation != pair.main.generation() ||
      pair.main.slot() >= slots_.size()) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 page pair handle is invalid");
  }
  const auto& slot = slots_[pair.main.slot()];
  if (slot.generation != pair.generation || slot.sequence != pair.sequence ||
      slot.layer_id != pair.layer_id ||
      slot.logical_page != pair.logical_page || slot.main != expected ||
      slot.index != expected) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-4 page pair generation or state is stale");
  }
  return pair.main.slot();
}

Status DeepSeekRatio4PagePool::publish(
    const DeepSeekRatio4PagePair& pair) {
  auto ordinal = validate_pair(pair, DeepSeekAttentionPageState::kReserved);
  if (!ordinal.ok()) return ordinal.status();
  auto& slot = slots_[*ordinal];
  slot.main = DeepSeekAttentionPageState::kPublished;
  slot.index = DeepSeekAttentionPageState::kPublished;
  --reserved_pairs_;
  ++published_pairs_;
  return Status::Ok();
}

Status DeepSeekRatio4PagePool::validate_publish(
    const DeepSeekRatio4PagePair& pair) const {
  auto value = validate_pair(pair, DeepSeekAttentionPageState::kReserved);
  return value.ok() ? Status::Ok() : value.status();
}

Result<std::optional<DeepSeekRatio4PagePair>>
DeepSeekRatio4PagePool::find_published(
    std::uint32_t sequence, std::uint32_t layer_id,
    std::uint32_t logical_page) const {
  if (sequence == 0 || layer_id >= 43) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 published page identity is invalid");
  }
  for (std::uint32_t ordinal = 0; ordinal < slots_.size(); ++ordinal) {
    const auto& slot = slots_[ordinal];
    if (slot.sequence != sequence || slot.layer_id != layer_id ||
        slot.logical_page != logical_page ||
        slot.main == DeepSeekAttentionPageState::kFree) {
      continue;
    }
    if (slot.main != DeepSeekAttentionPageState::kPublished ||
        slot.index != DeepSeekAttentionPageState::kPublished) {
      return Status::FailedPrecondition(
          "DeepSeek ratio-4 logical page is not stably published");
    }
    auto main = DeepSeekBlockHandle::Create(
        DeepSeekStatePoolKind::kRatio4MainBf16, ordinal, slot.generation);
    auto index = DeepSeekBlockHandle::Create(
        DeepSeekStatePoolKind::kRatio4IndexBf16, ordinal, slot.generation);
    if (!main.ok() || !index.ok()) {
      return Status::Internal(
          "DeepSeek ratio-4 published handle reconstruction failed");
    }
    return std::optional<DeepSeekRatio4PagePair>(DeepSeekRatio4PagePair{
        *main, *index, sequence, layer_id, logical_page, slot.generation});
  }
  return std::optional<DeepSeekRatio4PagePair>{};
}

Status DeepSeekRatio4PagePool::validate_publish_tail_cow(
    const DeepSeekRatio4PagePair& committed,
    const DeepSeekRatio4PagePair& replacement) const {
  auto source = validate_pair(committed,
                              DeepSeekAttentionPageState::kPublished);
  if (!source.ok()) return source.status();
  auto target = validate_pair(replacement,
                              DeepSeekAttentionPageState::kReserved);
  if (!target.ok()) return target.status();
  if (*source == *target || committed.sequence != replacement.sequence ||
      committed.layer_id != replacement.layer_id ||
      committed.logical_page != replacement.logical_page) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 tail COW identity is invalid");
  }
  return Status::Ok();
}

Status DeepSeekRatio4PagePool::publish_tail_cow(
    const DeepSeekRatio4PagePair& committed,
    const DeepSeekRatio4PagePair& replacement) {
  auto source = validate_pair(committed,
                              DeepSeekAttentionPageState::kPublished);
  if (!source.ok()) return source.status();
  auto target = validate_pair(replacement,
                              DeepSeekAttentionPageState::kReserved);
  if (!target.ok()) return target.status();
  if (*source == *target || committed.sequence != replacement.sequence ||
      committed.layer_id != replacement.layer_id ||
      committed.logical_page != replacement.logical_page) {
    return Status::InvalidArgument(
        "DeepSeek ratio-4 tail COW identity is invalid");
  }
  auto& old_slot = slots_[*source];
  auto& new_slot = slots_[*target];
  old_slot.main = DeepSeekAttentionPageState::kFree;
  old_slot.index = DeepSeekAttentionPageState::kFree;
  new_slot.main = DeepSeekAttentionPageState::kPublished;
  new_slot.index = DeepSeekAttentionPageState::kPublished;
  --reserved_pairs_;
  ++free_pairs_;
  return Status::Ok();
}

Status DeepSeekRatio4PagePool::rollback(
    const DeepSeekRatio4PagePair& pair) {
  auto ordinal = validate_pair(pair, DeepSeekAttentionPageState::kReserved);
  if (!ordinal.ok()) return ordinal.status();
  auto& slot = slots_[*ordinal];
  slot.main = DeepSeekAttentionPageState::kFree;
  slot.index = DeepSeekAttentionPageState::kFree;
  --reserved_pairs_;
  ++free_pairs_;
  return Status::Ok();
}

Status DeepSeekRatio4PagePool::release(
    const DeepSeekRatio4PagePair& pair) {
  auto ordinal = validate_pair(pair, DeepSeekAttentionPageState::kPublished);
  if (!ordinal.ok()) return ordinal.status();
  auto& slot = slots_[*ordinal];
  slot.main = DeepSeekAttentionPageState::kFree;
  slot.index = DeepSeekAttentionPageState::kFree;
  --published_pairs_;
  ++free_pairs_;
  return Status::Ok();
}

Status DeepSeekRatio4PagePool::validate_release_published_sequence(
    std::uint32_t sequence) const {
  if (sequence == 0) {
    return Status::InvalidArgument("DeepSeek sequence identity is invalid");
  }
  for (const auto& slot : slots_) {
    if (slot.sequence == sequence &&
        (slot.main == DeepSeekAttentionPageState::kReserved ||
         slot.index == DeepSeekAttentionPageState::kReserved)) {
      return Status::FailedPrecondition(
          "DeepSeek sequence still owns reserved ratio-4 pages");
    }
  }
  return Status::Ok();
}

Status DeepSeekRatio4PagePool::release_published_sequence(
    std::uint32_t sequence) {
  auto status = validate_release_published_sequence(sequence);
  if (!status.ok()) return status;
  for (auto& slot : slots_) {
    if (slot.sequence == sequence &&
        slot.main == DeepSeekAttentionPageState::kPublished) {
      slot.main = DeepSeekAttentionPageState::kFree;
      slot.index = DeepSeekAttentionPageState::kFree;
      --published_pairs_;
      ++free_pairs_;
    }
  }
  return Status::Ok();
}

Result<DeepSeekRatio128PagePool> DeepSeekRatio128PagePool::Create(
    std::uint32_t pages) {
  if (pages == 0 || pages > kSlotMask + 1U) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 page count is invalid");
  }
  return DeepSeekRatio128PagePool(std::vector<Slot>(pages));
}

Result<DeepSeekRatio128Page> DeepSeekRatio128PagePool::reserve(
    std::uint32_t sequence, std::uint32_t layer_id,
    std::uint32_t logical_page) {
  if (sequence == 0 || layer_id >= 43) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 page identity is invalid");
  }
  if (generation_exhausted_) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 generation requires epoch rebuild");
  }
  for (const auto& slot : slots_) {
    if (slot.state != DeepSeekAttentionPageState::kFree &&
        slot.sequence == sequence && slot.layer_id == layer_id &&
        slot.logical_page == logical_page) {
      return Status::FailedPrecondition(
          "DeepSeek logical ratio-128 page already exists");
    }
  }
  return reserve_free(sequence, layer_id, logical_page);
}

Result<DeepSeekRatio128Page> DeepSeekRatio128PagePool::reserve_free(
    std::uint32_t sequence, std::uint32_t layer_id,
    std::uint32_t logical_page) {
  for (std::uint32_t ordinal = 0; ordinal < slots_.size(); ++ordinal) {
    auto& slot = slots_[ordinal];
    if (slot.state != DeepSeekAttentionPageState::kFree) continue;
    if (slot.generation == std::numeric_limits<std::uint32_t>::max()) {
      generation_exhausted_ = true;
      return Status::FailedPrecondition(
          "DeepSeek ratio-128 generation requires epoch rebuild");
    }
    ++slot.generation;
    auto handle = DeepSeekBlockHandle::Create(
        DeepSeekStatePoolKind::kRatio128MainBf16, ordinal, slot.generation);
    if (!handle.ok()) return handle.status();
    slot.sequence = sequence;
    slot.layer_id = layer_id;
    slot.logical_page = logical_page;
    slot.state = DeepSeekAttentionPageState::kReserved;
    --free_;
    ++reserved_;
    return DeepSeekRatio128Page{*handle, sequence, layer_id, logical_page};
  }
  return Status::ResourceExhausted(
      "DeepSeek ratio-128 page pool is exhausted");
}

Result<std::uint32_t> DeepSeekRatio128PagePool::validate(
    const DeepSeekRatio128Page& page,
    DeepSeekAttentionPageState expected) const {
  if (page.handle.pool_kind() !=
          DeepSeekStatePoolKind::kRatio128MainBf16 ||
      page.handle.slot() >= slots_.size()) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 page handle is invalid");
  }
  const auto& slot = slots_[page.handle.slot()];
  if (slot.generation != page.handle.generation() ||
      slot.sequence != page.sequence || slot.layer_id != page.layer_id ||
      slot.logical_page != page.logical_page ||
      slot.state != expected) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 page generation or state is stale");
  }
  return page.handle.slot();
}

Result<DeepSeekRatio128Page> DeepSeekRatio128PagePool::reserve_tail_cow(
    const DeepSeekRatio128Page& committed) {
  auto source = validate(committed, DeepSeekAttentionPageState::kPublished);
  if (!source.ok()) return source.status();
  if (generation_exhausted_) {
    return Status::FailedPrecondition(
        "DeepSeek ratio-128 generation requires epoch rebuild");
  }
  return reserve_free(committed.sequence, committed.layer_id,
                      committed.logical_page);
}

Status DeepSeekRatio128PagePool::publish(const DeepSeekRatio128Page& page) {
  auto slot = validate(page, DeepSeekAttentionPageState::kReserved);
  if (!slot.ok()) return slot.status();
  slots_[*slot].state = DeepSeekAttentionPageState::kPublished;
  --reserved_;
  ++published_;
  return Status::Ok();
}

Status DeepSeekRatio128PagePool::validate_publish(
    const DeepSeekRatio128Page& page) const {
  auto value = validate(page, DeepSeekAttentionPageState::kReserved);
  return value.ok() ? Status::Ok() : value.status();
}

Result<std::optional<DeepSeekRatio128Page>>
DeepSeekRatio128PagePool::find_published(
    std::uint32_t sequence, std::uint32_t layer_id,
    std::uint32_t logical_page) const {
  if (sequence == 0 || layer_id >= 43) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 published page identity is invalid");
  }
  for (std::uint32_t ordinal = 0; ordinal < slots_.size(); ++ordinal) {
    const auto& slot = slots_[ordinal];
    if (slot.sequence != sequence || slot.layer_id != layer_id ||
        slot.logical_page != logical_page ||
        slot.state == DeepSeekAttentionPageState::kFree) {
      continue;
    }
    if (slot.state != DeepSeekAttentionPageState::kPublished) {
      return Status::FailedPrecondition(
          "DeepSeek ratio-128 logical page is not stably published");
    }
    auto handle = DeepSeekBlockHandle::Create(
        DeepSeekStatePoolKind::kRatio128MainBf16, ordinal, slot.generation);
    if (!handle.ok()) {
      return Status::Internal(
          "DeepSeek ratio-128 published handle reconstruction failed");
    }
    return std::optional<DeepSeekRatio128Page>(DeepSeekRatio128Page{
        *handle, sequence, layer_id, logical_page});
  }
  return std::optional<DeepSeekRatio128Page>{};
}

Status DeepSeekRatio128PagePool::validate_publish_tail_cow(
    const DeepSeekRatio128Page& committed,
    const DeepSeekRatio128Page& replacement) const {
  auto source = validate(committed, DeepSeekAttentionPageState::kPublished);
  if (!source.ok()) return source.status();
  auto target = validate(replacement, DeepSeekAttentionPageState::kReserved);
  if (!target.ok()) return target.status();
  if (*source == *target || committed.sequence != replacement.sequence ||
      committed.layer_id != replacement.layer_id ||
      committed.logical_page != replacement.logical_page) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 tail COW identity is invalid");
  }
  return Status::Ok();
}

Status DeepSeekRatio128PagePool::publish_tail_cow(
    const DeepSeekRatio128Page& committed,
    const DeepSeekRatio128Page& replacement) {
  auto source = validate(committed, DeepSeekAttentionPageState::kPublished);
  if (!source.ok()) return source.status();
  auto target = validate(replacement, DeepSeekAttentionPageState::kReserved);
  if (!target.ok()) return target.status();
  if (*source == *target || committed.sequence != replacement.sequence ||
      committed.layer_id != replacement.layer_id ||
      committed.logical_page != replacement.logical_page) {
    return Status::InvalidArgument(
        "DeepSeek ratio-128 tail COW identity is invalid");
  }
  slots_[*source].state = DeepSeekAttentionPageState::kFree;
  slots_[*target].state = DeepSeekAttentionPageState::kPublished;
  --reserved_;
  ++free_;
  return Status::Ok();
}

Status DeepSeekRatio128PagePool::rollback(const DeepSeekRatio128Page& page) {
  auto slot = validate(page, DeepSeekAttentionPageState::kReserved);
  if (!slot.ok()) return slot.status();
  slots_[*slot].state = DeepSeekAttentionPageState::kFree;
  --reserved_;
  ++free_;
  return Status::Ok();
}

Status DeepSeekRatio128PagePool::release(const DeepSeekRatio128Page& page) {
  auto slot = validate(page, DeepSeekAttentionPageState::kPublished);
  if (!slot.ok()) return slot.status();
  slots_[*slot].state = DeepSeekAttentionPageState::kFree;
  --published_;
  ++free_;
  return Status::Ok();
}

Status DeepSeekRatio128PagePool::validate_release_published_sequence(
    std::uint32_t sequence) const {
  if (sequence == 0) {
    return Status::InvalidArgument("DeepSeek sequence identity is invalid");
  }
  for (const auto& slot : slots_) {
    if (slot.sequence == sequence &&
        slot.state == DeepSeekAttentionPageState::kReserved) {
      return Status::FailedPrecondition(
          "DeepSeek sequence still owns reserved ratio-128 pages");
    }
  }
  return Status::Ok();
}

Status DeepSeekRatio128PagePool::release_published_sequence(
    std::uint32_t sequence) {
  auto status = validate_release_published_sequence(sequence);
  if (!status.ok()) return status;
  for (auto& slot : slots_) {
    if (slot.sequence == sequence &&
        slot.state == DeepSeekAttentionPageState::kPublished) {
      slot.state = DeepSeekAttentionPageState::kFree;
      --published_;
      ++free_;
    }
  }
  return Status::Ok();
}

}  // namespace pih
