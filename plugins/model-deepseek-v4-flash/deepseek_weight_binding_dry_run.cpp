#include "pih/model/deepseek_weight_binding_dry_run.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

namespace pih {
namespace {

bool zero_digest(const Sha256Digest& digest) {
  return std::ranges::all_of(
      digest.bytes, [](std::byte value) { return value == std::byte{0}; });
}

}  // namespace

Result<DeepSeekWeightConsumerDryRun::Receipt>
DeepSeekWeightBindingDryRun::run(
    const DeepSeekWeightMaterializationPlan& plan,
    const DeepSeekResidentWeightArena& arena,
    const Sha256Digest& expected_layout_digest) {
  if (plan.copies().empty() || arena.device().type() != DeviceType::kCuda ||
      arena.device().index() < 0 || arena.generation() == 0 ||
      arena.base_address() == 0 || arena.backing_bytes() != plan.backing_bytes() ||
      arena.payload_bytes() != plan.payload_bytes()) {
    return Status::FailedPrecondition(
        "DeepSeek final weight binding consumer has no canonical CUDA owner");
  }
  std::unordered_set<std::string_view> names;
  std::unordered_set<std::string> runtime_roots;
  names.reserve(plan.copies().size());
  runtime_roots.reserve(plan.copies().size());
  for (const auto& copy : plan.copies()) {
    if (!names.emplace(copy.tensor_name).second) {
      return Status::FailedPrecondition(
          "DeepSeek final weight binding consumer saw a duplicate tensor");
    }
    if (plan.target_authority_bound() &&
        (copy.artifact_root != plan.artifact_root() ||
         copy.layout_root != plan.layout_root() ||
         copy.disposition_root != plan.disposition_root() ||
         zero_digest(copy.target_logical_root) ||
         zero_digest(copy.disposition_record_root) ||
         zero_digest(copy.layout_record_root) ||
         zero_digest(copy.runtime_record_root) ||
         copy.storage_semantics == DeepSeekStorageSemantics{} ||
         !runtime_roots.emplace(copy.runtime_record_root.hex()).second)) {
      return Status::FailedPrecondition(
          "DeepSeek final weight binding target authority drifted");
    }
    auto tensor = arena.tensor(copy.tensor_name);
    if (!tensor.ok() || tensor->device() != arena.device() ||
        tensor->generation() != arena.generation() ||
        tensor->dtype() != copy.dtype || tensor->rank() != copy.shape.size() ||
        tensor->byte_span() != copy.bytes ||
        reinterpret_cast<std::uintptr_t>(tensor->data()) !=
            arena.base_address() + copy.destination_offset) {
      return Status::FailedPrecondition(
          "DeepSeek final weight binding consumer identity drifted");
    }
    for (std::size_t axis = 0; axis < copy.shape.size(); ++axis) {
      if (tensor->dim(axis) != copy.shape[axis]) {
        return Status::FailedPrecondition(
            "DeepSeek final weight binding consumer shape drifted");
      }
    }
  }
  return Receipt{1, plan.copies().size(), arena.generation(),
                 expected_layout_digest};
}

}  // namespace pih
