#include "pih/model/deepseek_endpoint_device_resources.h"

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<std::uint64_t> reserve(std::uint64_t& cursor,
                              std::uint64_t bytes) {
  auto aligned = checked_align_up_u64(cursor, 256);
  if (!aligned.ok()) return aligned.status();
  auto end = checked_add_u64(*aligned, bytes);
  if (!end.ok()) return end.status();
  cursor = *end;
  return *aligned;
}

}  // namespace

Result<DeepSeekEndpointDeviceResources>
DeepSeekEndpointDeviceResources::Allocate(
    DeepSeekStagePlan stage, std::uint32_t maximum_tokens,
    Allocator& allocator, std::uint64_t context_identity,
    std::int32_t device_ordinal) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43 || maximum_tokens == 0 ||
      maximum_tokens > 4096 || context_identity == 0 ||
      device_ordinal < 0) {
    return Status::InvalidArgument(
        "DeepSeek endpoint device identity is invalid");
  }
  if (!stage.owns_embedding && !stage.owns_lm_head) {
    return DeepSeekEndpointDeviceResources(nullptr, {}, maximum_tokens);
  }
  std::uint64_t cursor = 0;
  std::uint64_t embedding_offset = 0;
  std::uint64_t head_offset = 0;
  std::uint64_t norm_offset = 0;
  std::uint64_t logits_offset = 0;
  std::uint64_t sampled_token_offset = 0;
  std::uint64_t selected_logprob_offset = 0;
  std::uint64_t rng_word_offset = 0;
  std::uint64_t sampling_workspace_values_offset = 0;
  std::uint64_t sampling_workspace_ids_offset = 0;
  std::uint64_t top_logprobs_ids_offset = 0;
  std::uint64_t top_logprobs_offset = 0;
  if (stage.owns_embedding) {
    auto elements = checked_mul_u64(maximum_tokens, UINT64_C(4 * 4096));
    if (!elements.ok()) return elements.status();
    auto bytes = checked_mul_u64(*elements, sizeof(std::uint16_t));
    if (!bytes.ok()) return bytes.status();
    auto offset = reserve(cursor, *bytes);
    if (!offset.ok()) return offset.status();
    embedding_offset = *offset;
  }
  if (stage.owns_lm_head) {
    auto offset = reserve(cursor, UINT64_C(4096) * sizeof(std::uint16_t));
    if (!offset.ok()) return offset.status();
    head_offset = *offset;
    offset = reserve(cursor, UINT64_C(4096) * sizeof(std::uint16_t));
    if (!offset.ok()) return offset.status();
    norm_offset = *offset;
    offset = reserve(cursor, UINT64_C(129280) * sizeof(float));
    if (!offset.ok()) return offset.status();
    logits_offset = *offset;
    offset = reserve(cursor, sizeof(std::uint32_t));
    if (!offset.ok()) return offset.status();
    sampled_token_offset = *offset;
    offset = reserve(cursor, sizeof(float));
    if (!offset.ok()) return offset.status();
    selected_logprob_offset = *offset;
    offset = reserve(cursor, sizeof(std::uint32_t));
    if (!offset.ok()) return offset.status();
    rng_word_offset = *offset;
    offset = reserve(cursor, UINT64_C(129280) * sizeof(float));
    if (!offset.ok()) return offset.status();
    sampling_workspace_values_offset = *offset;
    offset = reserve(cursor, UINT64_C(129280) * sizeof(std::uint32_t));
    if (!offset.ok()) return offset.status();
    sampling_workspace_ids_offset = *offset;
    offset = reserve(cursor, UINT64_C(20) * sizeof(std::uint32_t));
    if (!offset.ok()) return offset.status();
    top_logprobs_ids_offset = *offset;
    offset = reserve(cursor, UINT64_C(20) * sizeof(float));
    if (!offset.ok()) return offset.status();
    top_logprobs_offset = *offset;
  }
  auto error_offset = reserve(cursor, sizeof(std::uint32_t));
  if (!error_offset.ok()) return error_offset.status();
  auto total = checked_align_up_u64(cursor, 256);
  if (!total.ok()) return total.status();
  auto backing = Buffer::Allocate(allocator, *total, 256);
  if (!backing.ok()) return backing.status();
  if (backing->data() == nullptr || backing->generation() == 0 ||
      backing->device().type() != DeviceType::kCuda ||
      backing->device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint allocator returned invalid device backing");
  }
  const auto base = reinterpret_cast<std::uintptr_t>(backing->data());
  const DeepSeekEndpointDeviceView view{
      stage.owns_embedding ? base + embedding_offset : 0,
      stage.owns_lm_head ? base + head_offset : 0,
      stage.owns_lm_head ? base + norm_offset : 0,
      stage.owns_lm_head ? base + logits_offset : 0,
      stage.owns_lm_head ? base + sampled_token_offset : 0,
      base + *error_offset,
      stage.owns_lm_head ? base + selected_logprob_offset : 0,
      stage.owns_lm_head ? base + rng_word_offset : 0,
      stage.owns_lm_head ? base + sampling_workspace_values_offset : 0,
      stage.owns_lm_head ? base + sampling_workspace_ids_offset : 0,
      stage.owns_lm_head ? base + top_logprobs_ids_offset : 0,
      stage.owns_lm_head ? base + top_logprobs_offset : 0};
  return DeepSeekEndpointDeviceResources(
      std::make_unique<Buffer>(std::move(*backing)), view, maximum_tokens);
}

}  // namespace pih
