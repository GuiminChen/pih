#pragma once

#include <cstddef>
#include <span>
#include <variant>
#include <vector>

#include "pih/model/qwen3_bf16_execution_prelude.h"
#include "pih/model/qwen3_bf16_kernel_materializer.h"
#include "pih/model/qwen3_int4_kernel_bundle.h"
#include "pih/model/qwen3_int4_linear_materializer.h"
#include "pih/model/qwen3_int4_lm_head_materializer.h"

namespace pih {

enum class QwenInt4PreparedExecutionState : std::uint8_t {
  kPrepared,
  kRunning,
  kCompleted,
  kPoisoned,
};

class QwenInt4LmHeadExecutionDriver {
 public:
  virtual ~QwenInt4LmHeadExecutionDriver() = default;
  virtual Status execute(const QwenInt4LmHeadBinding& binding,
                         DriverStreamHandle stream) = 0;
};

class QwenInt4PreparedExecution final {
 public:
  using Command=std::variant<QwenBf16DispatchPlan,QwenInt4DispatchPlan,
                             QwenInt4LmHeadBinding>;
  static Result<QwenInt4PreparedExecution> Create(
      const QwenBf16CommandBuffer& commands,
      std::span<const ResolvedKernelFunction> functions,
      const QwenBf16ResourceSet& activations,
      const QwenInt4WeightResourceSet& weights,
      const QwenBf16KernelContext& context);

  QwenInt4PreparedExecution(const QwenInt4PreparedExecution&)=delete;
  QwenInt4PreparedExecution& operator=(const QwenInt4PreparedExecution&)=delete;
  QwenInt4PreparedExecution(QwenInt4PreparedExecution&&) noexcept=default;
  QwenInt4PreparedExecution& operator=(QwenInt4PreparedExecution&&) noexcept=default;

  Status run(QwenBf16DeviceErrorClearDriver& clear_driver,
             KernelLaunchDriver& kernel_driver,
             QwenInt4LmHeadExecutionDriver& lm_head_driver,
             DriverStreamHandle stream);
  [[nodiscard]] std::size_t size() const noexcept{return commands_.size();}
  [[nodiscard]] std::size_t bf16_kernel_count() const noexcept{return bf16_kernel_count_;}
  [[nodiscard]] std::size_t int4_linear_count() const noexcept{return int4_linear_count_;}
  [[nodiscard]] std::size_t lm_head_count() const noexcept{return lm_head_count_;}
  [[nodiscard]] QwenInt4PreparedExecutionState state() const noexcept{return state_;}

 private:
  QwenInt4PreparedExecution(QwenBf16ExecutionPrelude prelude,
      std::vector<Command> commands,std::size_t bf16,std::size_t int4,
      std::size_t lm)
      :prelude_(std::move(prelude)),commands_(std::move(commands)),
       bf16_kernel_count_(bf16),int4_linear_count_(int4),lm_head_count_(lm){}
  QwenBf16ExecutionPrelude prelude_;
  std::vector<Command> commands_;
  std::size_t bf16_kernel_count_,int4_linear_count_,lm_head_count_;
  std::size_t next_command_=0;
  QwenInt4PreparedExecutionState state_=QwenInt4PreparedExecutionState::kPrepared;
};

}  // namespace pih
