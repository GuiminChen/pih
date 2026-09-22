#pragma once

#include <span>
#include <variant>
#include <vector>

#include "pih/model/qwen3_bf16_packed_kernel_materializer.h"
#include "pih/model/qwen3_bf16_kernel_bundle.h"
#include "pih/model/qwen3_int4_packed_linear_materializer.h"
#include "pih/model/qwen3_int4_packed_lm_head_materializer.h"
#include "pih/model/qwen3_int4_prepared_execution.h"

namespace pih {

enum class QwenInt4PackedPreparedExecutionState : std::uint8_t {
  kPrepared=1,
  kRunning=2,
  kCompleted=3,
  kPoisoned=4,
};

class QwenInt4PackedPreparedExecution final {
 public:
  static constexpr std::size_t kMaximumCommandCount=
      QwenBf16CommandBuffer::kCommandCount+1;
  using Command=std::variant<QwenBf16DispatchPlan,
      QwenBf16PackedDispatchPlan,QwenInt4DispatchPlan,
      QwenInt4LmHeadBinding>;

  static Result<QwenInt4PackedPreparedExecution> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> int4_functions,
      std::span<const ResolvedKernelFunction> packed_functions,
      const QwenBf16PackedResourceSet& resources,
      const QwenInt4WeightResourceSet& weights,
      const QwenBf16PackedKernelContext& context);

  QwenInt4PackedPreparedExecution(
      const QwenInt4PackedPreparedExecution&)=delete;
  QwenInt4PackedPreparedExecution& operator=(
      const QwenInt4PackedPreparedExecution&)=delete;
  QwenInt4PackedPreparedExecution(
      QwenInt4PackedPreparedExecution&&) noexcept=default;

  Status run(QwenBf16DeviceErrorClearDriver& clear_driver,
             KernelLaunchDriver& kernel_driver,
             QwenInt4LmHeadExecutionDriver& lm_head_driver,
             DriverStreamHandle stream);

  [[nodiscard]] std::size_t size() const noexcept{return commands_.size();}
  [[nodiscard]] std::size_t legacy_kernel_count() const noexcept{
    return legacy_kernel_count_;}
  [[nodiscard]] std::size_t packed_kernel_count() const noexcept{
    return packed_kernel_count_;}
  [[nodiscard]] std::size_t int4_linear_count() const noexcept{
    return int4_linear_count_;}
  [[nodiscard]] std::size_t lm_head_count() const noexcept{
    return lm_head_count_;}
  [[nodiscard]] QwenInt4PackedPreparedExecutionState state() const noexcept{
    return state_;}

 private:
  QwenInt4PackedPreparedExecution(QwenBf16ExecutionPrelude prelude,
      std::vector<Command> commands,std::size_t legacy,std::size_t packed,
      std::size_t int4,std::size_t lm)
      :prelude_(std::move(prelude)),commands_(std::move(commands)),
       legacy_kernel_count_(legacy),packed_kernel_count_(packed),
       int4_linear_count_(int4),lm_head_count_(lm){}

  QwenBf16ExecutionPrelude prelude_;
  std::vector<Command> commands_;
  std::size_t legacy_kernel_count_=0;
  std::size_t packed_kernel_count_=0;
  std::size_t int4_linear_count_=0;
  std::size_t lm_head_count_=0;
  std::size_t next_command_=0;
  QwenInt4PackedPreparedExecutionState state_=
      QwenInt4PackedPreparedExecutionState::kPrepared;
};

}  // namespace pih
