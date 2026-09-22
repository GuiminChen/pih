#include "pih/model/deepseek_weight_finalizer.h"

#include <array>
#include <span>

namespace pih {
namespace {

Status add_u64(Sha256& hash, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t i = 0; i < wire.size(); ++i) {
    wire[i] = static_cast<std::byte>((value >> (i * 8U)) & 0xffU);
  }
  return hash.update(wire);
}

Status add_string(Sha256& hash, std::string_view value) {
  auto status = add_u64(hash, value.size());
  return status.ok() ? hash.update(std::as_bytes(std::span(value))) : status;
}

}  // namespace

Result<DeepSeekWeightFinalizer> DeepSeekWeightFinalizer::Create(
    const DeepSeekWeightMaterializationPlan& plan,
    const DeepSeekResidentWeightArena& arena) {
  if (plan.copies().empty() || arena.generation() == 0 ||
      arena.base_address() == 0 || arena.backing_bytes() != plan.backing_bytes() ||
      arena.payload_bytes() != plan.payload_bytes()) {
    return Status::InvalidArgument("DeepSeek final weight owner identity is invalid");
  }
  for (const auto& copy : plan.copies()) {
    auto tensor = arena.tensor(copy.tensor_name);
    if (!tensor.ok() || tensor->dtype() != copy.dtype ||
        tensor->byte_span() != copy.bytes ||
        tensor->generation() != arena.generation() ||
        reinterpret_cast<std::uintptr_t>(tensor->data()) !=
            arena.base_address() + copy.destination_offset ||
        tensor->rank() != copy.shape.size()) {
      return Status::FailedPrecondition(
          "DeepSeek final tensor does not match materialization plan");
    }
    for (std::size_t axis = 0; axis < copy.shape.size(); ++axis) {
      if (tensor->dim(axis) != copy.shape[axis]) {
        return Status::FailedPrecondition(
            "DeepSeek final tensor shape does not match materialization plan");
      }
    }
  }
  DeepSeekWeightFinalizer finalizer(plan, arena, {});
  auto layout = finalizer.current_layout_digest();
  if (!layout.ok()) return layout.status();
  finalizer.layout_digest_ = *layout;
  return finalizer;
}

Status DeepSeekWeightFinalizer::run_dry_run(
    DeepSeekWeightConsumerDryRun& dry_run) {
  if (state_ != DeepSeekWeightFinalizeState::kFinalValidated) {
    return Status::FailedPrecondition("DeepSeek weight dry-run state is invalid");
  }
  auto receipt = dry_run.run(*plan_, *arena_, layout_digest_);
  if (!receipt.ok()) {
    state_ = DeepSeekWeightFinalizeState::kFailed;
    return receipt.status();
  }
  if (receipt->consumer_path_count == 0 ||
      receipt->tensor_binding_count != plan_->copies().size() ||
      receipt->arena_generation != arena_->generation() ||
      receipt->layout_digest != layout_digest_) {
    state_ = DeepSeekWeightFinalizeState::kFailed;
    return Status::FailedPrecondition(
        "DeepSeek weight consumer dry-run receipt is incomplete");
  }
  state_ = DeepSeekWeightFinalizeState::kDryRunPassed;
  return Status::Ok();
}

Result<Sha256Digest> DeepSeekWeightFinalizer::current_layout_digest() const {
  static constexpr std::string_view kDomain =
      "pih.deepseek.weight-final-layout.v1";
  Sha256 hash;
  Status status = add_string(hash, kDomain);
  if (status.ok()) status = add_u64(hash, arena_->backing_bytes());
  if (status.ok()) status = add_u64(hash, arena_->payload_bytes());
  if (status.ok()) status = add_u64(hash, plan_->paged_source_bytes());
  if (status.ok()) status = add_u64(hash, plan_->target_authority_bound());
  if (status.ok()) status = hash.update(plan_->artifact_root().bytes);
  if (status.ok()) status = hash.update(plan_->layout_root().bytes);
  if (status.ok()) status = hash.update(plan_->disposition_root().bytes);
  if (status.ok()) status = add_u64(hash, plan_->copies().size());
  for (const auto& copy : plan_->copies()) {
    if (status.ok()) status = add_string(hash, copy.tensor_name);
    if (status.ok()) status = add_u64(hash, static_cast<std::uint8_t>(copy.dtype));
    if (status.ok()) status = add_u64(hash, copy.destination_offset);
    if (status.ok()) status = add_u64(hash, copy.bytes);
    if (status.ok()) status = add_u64(hash, copy.shape.size());
    for (const auto dimension : copy.shape) {
      if (status.ok()) status = add_u64(hash, dimension);
    }
    if (status.ok()) status = add_u64(hash, copy.expert_identity.has_value());
    if (copy.expert_identity.has_value()) {
      if (status.ok()) status = add_u64(hash, copy.expert_identity->layer);
      if (status.ok()) status = add_u64(hash, copy.expert_identity->expert);
    }
    if (status.ok()) status = add_u64(hash, copy.logical_layer);
    if (status.ok()) {
      status = add_u64(
          hash, static_cast<std::uint8_t>(copy.storage_semantics));
    }
    if (status.ok()) status = hash.update(copy.target_logical_root.bytes);
    if (status.ok()) status = hash.update(copy.disposition_record_root.bytes);
    if (status.ok()) status = hash.update(copy.layout_record_root.bytes);
    if (status.ok()) status = hash.update(copy.runtime_record_root.bytes);
  }
  return status.ok() ? hash.finalize() : Result<Sha256Digest>(status);
}

Result<Sha256Digest> DeepSeekWeightFinalizer::current_pointer_digest(
    const Sha256Digest& layout_digest) const {
  static constexpr std::string_view kDomain =
      "pih.deepseek.weight-final-pointer-generation.v1";
  Sha256 hash;
  Status status = add_string(hash, kDomain);
  if (status.ok()) status = hash.update(layout_digest.bytes);
  if (status.ok()) status = add_u64(hash, arena_->generation());
  if (status.ok()) status = add_u64(hash, arena_->base_address());
  return status.ok() ? hash.finalize() : Result<Sha256Digest>(status);
}

Status DeepSeekWeightFinalizer::seal() {
  if (state_ != DeepSeekWeightFinalizeState::kDryRunPassed) {
    return Status::FailedPrecondition("DeepSeek weight finalize is not sealable");
  }
  auto layout = current_layout_digest();
  if (!layout.ok()) {
    state_ = DeepSeekWeightFinalizeState::kFailed;
    return layout.status();
  }
  if (*layout != layout_digest_) {
    state_ = DeepSeekWeightFinalizeState::kFailed;
    return Status::FailedPrecondition(
        "DeepSeek weight layout changed before seal");
  }
  auto pointer = current_pointer_digest(layout_digest_);
  if (!pointer.ok()) {
    state_ = DeepSeekWeightFinalizeState::kFailed;
    return pointer.status();
  }
  seal_digest_ = *pointer;
  state_ = DeepSeekWeightFinalizeState::kSealed;
  return Status::Ok();
}

Status DeepSeekWeightFinalizer::verify_unchanged() const {
  if (state_ != DeepSeekWeightFinalizeState::kSealed) {
    return Status::FailedPrecondition("DeepSeek weight final owner is not sealed");
  }
  auto layout = current_layout_digest();
  if (!layout.ok()) return layout.status();
  auto pointer = current_pointer_digest(*layout);
  if (!pointer.ok()) return pointer.status();
  return *layout == layout_digest_ && *pointer == seal_digest_
             ? Status::Ok()
             : Status::FailedPrecondition(
                   "DeepSeek weight final owner changed after seal");
}

}  // namespace pih
