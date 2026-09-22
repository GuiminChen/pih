#include "pih/model/qwen3_int4_engine_model_plan.h"
namespace pih{
Result<QwenInt4EngineModelPlan> QwenInt4EngineModelPlan::Create(const Qwen3Config& c){
 if(c.hidden_size!=1024||c.intermediate_size!=3072||c.layers!=28||
    c.attention_heads!=16||c.kv_heads!=8||c.head_dim!=128||
    c.vocabulary_size!=151936||c.maximum_positions!=40960||
    c.rope_theta!=1'000'000.0||c.rms_norm_epsilon!=0.000001||
    c.bos_token_id!=151643||c.eos_token_id!=151645)
  return Status::InvalidArgument("Qwen INT4 engine requires official Qwen3-0.6B config");
 auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4();if(!layout.ok())return layout.status();
 auto ledger=QwenInt4LinearShapeLedger::CreateOfficial();if(!ledger.ok())return ledger.status();
 auto schedule=QwenBf16ExecutionSchedule::Create(c);if(!schedule.ok())return schedule.status();
 auto bindings=QwenBf16WeightBindingPlan::Create(*schedule);if(!bindings.ok())return bindings.status();
 auto commands=QwenBf16CommandBuffer::Create(*schedule,*bindings);if(!commands.ok())return commands.status();
 if(commands->size()!=509)return Status::Internal("Qwen INT4 engine command count drifted");
 return QwenInt4EngineModelPlan(c,std::move(*layout),std::move(*ledger),
  std::move(*schedule),std::move(*bindings),std::move(*commands));
}}
