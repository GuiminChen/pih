#include "pih/model/qwen3_int4_engine_load_plan.h"

namespace pih {
Result<QwenInt4EngineLoadPlan> QwenInt4EngineLoadPlan::CreateFromJson(
    std::string_view config_json,
    const std::filesystem::path& cubin_root,std::int32_t device_ordinal) {
 if(config_json.empty()||config_json.size()>Qwen3Config::kMaxConfigBytes||
    cubin_root.empty()||device_ordinal<0)
  return Status::InvalidArgument("Qwen INT4 config snapshot or load identity invalid");
 auto config=Qwen3Config::Parse(config_json);if(!config.ok())return config.status();
 auto model=QwenInt4EngineModelPlan::Create(*config);if(!model.ok())return model.status();
 return QwenInt4EngineLoadPlan(std::move(*model),cubin_root,device_ordinal);
}
}
