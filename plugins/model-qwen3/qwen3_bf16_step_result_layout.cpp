#include "pih/model/qwen3_bf16_step_result_layout.h"

#include <cstring>

#include "pih/model/qwen3_cuda_invariant.h"

namespace pih {

Status QwenBf16StepResultLayout::initialize(
    std::span<std::byte> backing) {
  if (backing.size() != kTotalBytes) {
    return Status::InvalidArgument("Qwen step result backing is invalid");
  }
  std::memset(backing.data(), 0xff, backing.size());
  return Status::Ok();
}

Result<std::int64_t> QwenBf16StepResultLayout::parse(
    std::span<const std::byte> backing, bool publication_authorized) {
  if (!publication_authorized || backing.size() != kTotalBytes) {
    return Status::FailedPrecondition(
        "Qwen step result is not authorized for publication");
  }
  std::uint32_t error = UINT32_MAX;
  std::int64_t token = -1;
  std::memcpy(&error, backing.data() + device_error().offset_bytes,
              sizeof(error));
  std::memcpy(&token, backing.data() + sampled_token().offset_bytes,
              sizeof(token));
  if (!qwen_cuda_invariant_known(error)) {
    return Status::Internal("Qwen device returned an unknown invariant code");
  }
  if (error != static_cast<std::uint32_t>(QwenCudaInvariant::kNone)) {
    return Status::Internal("Qwen device invariant rejected token publication");
  }
  if (token < 0 || token >= kVocabularySize) {
    return Status::Internal("Qwen sampled token is outside the vocabulary");
  }
  return token;
}

}  // namespace pih
