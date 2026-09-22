#pragma once
#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_config.h"
#include "pih/model/qwen3_int4_artifact_layout.h"
#include "pih/model/qwen3_int4_linear_shape_ledger.h"
namespace pih{
class QwenInt4EngineModelPlan final{public:
 static Result<QwenInt4EngineModelPlan>Create(const Qwen3Config& config);
 [[nodiscard]]const QwenInt4ArtifactLayout& layout()const noexcept{return layout_;}
 [[nodiscard]]const Qwen3Config& config()const noexcept{return config_;}
 [[nodiscard]]const QwenInt4LinearShapeLedger& ledger()const noexcept{return ledger_;}
 [[nodiscard]]const QwenBf16ExecutionSchedule& schedule()const noexcept{return schedule_;}
 [[nodiscard]]const QwenBf16WeightBindingPlan& bindings()const noexcept{return bindings_;}
 [[nodiscard]]const QwenBf16CommandBuffer& commands()const noexcept{return commands_;}
 private:QwenInt4EngineModelPlan(Qwen3Config config,QwenInt4ArtifactLayout layout,QwenInt4LinearShapeLedger ledger,
  QwenBf16ExecutionSchedule schedule,QwenBf16WeightBindingPlan bindings,QwenBf16CommandBuffer commands)
  :config_(config),layout_(std::move(layout)),ledger_(std::move(ledger)),schedule_(std::move(schedule)),
   bindings_(std::move(bindings)),commands_(std::move(commands)){}
 Qwen3Config config_;QwenInt4ArtifactLayout layout_;QwenInt4LinearShapeLedger ledger_;
 QwenBf16ExecutionSchedule schedule_;QwenBf16WeightBindingPlan bindings_;QwenBf16CommandBuffer commands_;
};}
