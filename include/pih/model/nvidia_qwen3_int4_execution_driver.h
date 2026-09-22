#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pih/backend/cuda/gemm_plan.h"
#include "pih/model/qwen3_bf16_execution_prelude.h"
#include "pih/model/qwen3_int4_prepared_execution.h"

namespace pih {

class NvidiaQwenInt4ExecutionDriver final
    : public QwenInt4LmHeadExecutionDriver {
 public:
  static Result<NvidiaQwenInt4ExecutionDriver> Create(
      void* workspace,std::uint64_t workspace_bytes,
      std::uint64_t maximum_workspace_bytes,std::int32_t owning_rank,
      std::uint32_t maximum_lm_head_rows=1);

  NvidiaQwenInt4ExecutionDriver(const NvidiaQwenInt4ExecutionDriver&)=delete;
  NvidiaQwenInt4ExecutionDriver& operator=(const NvidiaQwenInt4ExecutionDriver&)=delete;
  NvidiaQwenInt4ExecutionDriver(NvidiaQwenInt4ExecutionDriver&&) noexcept=default;
  NvidiaQwenInt4ExecutionDriver& operator=(NvidiaQwenInt4ExecutionDriver&&) noexcept=default;

  Status execute(const QwenInt4LmHeadBinding& binding,
                 DriverStreamHandle stream) override;
  [[nodiscard]] std::uintptr_t context_identity() const noexcept{return context_identity_;}
  [[nodiscard]] std::int32_t owning_rank() const noexcept{return owning_rank_;}
  [[nodiscard]] std::uint64_t required_workspace_bytes() const noexcept{
    return required_workspace_bytes_;
  }
  [[nodiscard]] std::uint32_t maximum_lm_head_rows() const noexcept{
    return static_cast<std::uint32_t>(plans_.size());
  }
 private:
  NvidiaQwenInt4ExecutionDriver(std::vector<std::unique_ptr<GemmPlan>> plans,
      std::uint64_t required_workspace_bytes,void* workspace,
      std::uint64_t workspace_bytes,std::int32_t owning_rank,
      std::uintptr_t context_identity)
      :plans_(std::move(plans)),required_workspace_bytes_(required_workspace_bytes),
       workspace_(workspace),workspace_bytes_(workspace_bytes),
       owning_rank_(owning_rank),context_identity_(context_identity){}
  Status require_current_owner() const;
  std::vector<std::unique_ptr<GemmPlan>> plans_;
  std::uint64_t required_workspace_bytes_=0;
  void* workspace_;
  std::uint64_t workspace_bytes_;
  std::int32_t owning_rank_;
  std::uintptr_t context_identity_;
};

}  // namespace pih
