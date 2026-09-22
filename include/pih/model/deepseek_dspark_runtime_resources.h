#pragma once

#include <array>
#include <memory>
#include <optional>
#include <span>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/buffer.h"
#include "pih/model/deepseek_dspark_embed_coordinator.h"
#include "pih/model/deepseek_dspark_head_executor.h"
#include "pih/model/deepseek_dspark_prefill_stage_operation.h"
#include "pih/model/deepseek_dspark_decode_stage_operation.h"
#include "pih/model/deepseek_dspark_moe_stage_operation.h"
#include "pih/model/deepseek_stage_mapping_plan.h"

namespace pih {

class DeepSeekDsparkRuntimeResources final {
 public:
  static Result<DeepSeekDsparkRuntimeResources> Allocate(
      DeepSeekStagePlan stage, DeepSeekDsparkEmbedOperations* embed_operations,
      DeepSeekDsparkHeadOperations* head_operations,
      DeepSeekDsparkPrefillStageOperations* prefill_operations,
      DeepSeekDsparkDecodeStageOperations* decode_operations,
      DeepSeekDsparkMoeStageOperations* moe_operations,
      RegisteredPinnedAllocator& allocator);
  [[nodiscard]] DeepSeekDsparkEmbedCoordinator* embed_coordinator() noexcept {
    return embed_.get();
  }
  [[nodiscard]] DeepSeekDsparkHeadExecutor* head_executor() noexcept {
    return head_.get();
  }
  [[nodiscard]] DeepSeekDsparkMtpStageOperation* prefill_operation(
      DeepSeekDsparkStageId stage) noexcept;
  [[nodiscard]] DeepSeekDsparkMtpStageOperation* decode_attention_operation(
      DeepSeekDsparkStageId stage) noexcept;
  [[nodiscard]] DeepSeekDsparkMtpStageOperation* moe_operation(
      DeepSeekDsparkStageId stage) noexcept;
  [[nodiscard]] std::span<float> router_host_scores() noexcept;
  [[nodiscard]] std::span<float> router_host_bias() noexcept;

 private:
  DeepSeekDsparkRuntimeResources(
      std::unique_ptr<Buffer> host_error,
      std::unique_ptr<DeepSeekDsparkEmbedCoordinator> embed,
      std::unique_ptr<DeepSeekDsparkHeadExecutor> head,
      std::array<std::optional<DeepSeekDsparkPrefillStageOperation>, 3>
          prefill,
      std::array<std::optional<DeepSeekDsparkDecodeStageOperation>, 3>
          decode_attention,
      std::array<std::optional<DeepSeekDsparkMoeStageOperation>, 3>
          moe) noexcept
      : host_error_(std::move(host_error)), embed_(std::move(embed)),
        head_(std::move(head)), prefill_(std::move(prefill)),
        decode_attention_(std::move(decode_attention)),
        moe_(std::move(moe)) {}
  std::unique_ptr<Buffer> host_error_;
  std::unique_ptr<DeepSeekDsparkEmbedCoordinator> embed_;
  std::unique_ptr<DeepSeekDsparkHeadExecutor> head_;
  std::array<std::optional<DeepSeekDsparkPrefillStageOperation>, 3> prefill_;
  std::array<std::optional<DeepSeekDsparkDecodeStageOperation>, 3>
      decode_attention_;
  std::array<std::optional<DeepSeekDsparkMoeStageOperation>, 3> moe_;
};

}  // namespace pih
