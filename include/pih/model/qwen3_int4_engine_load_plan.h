#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>

#include "pih/model/qwen3_int4_engine_model_plan.h"

namespace pih {
class QwenInt4EngineLoadPlan final {
 public:
  static Result<QwenInt4EngineLoadPlan> CreateFromJson(
      std::string_view config_json,
      const std::filesystem::path& cubin_root,
      std::int32_t device_ordinal);
  [[nodiscard]] const QwenInt4EngineModelPlan& model() const noexcept { return model_; }
  [[nodiscard]] const std::filesystem::path& cubin_root() const noexcept { return cubin_root_; }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept { return device_ordinal_; }
  [[nodiscard]] QwenInt4EngineModelPlan take_model() && noexcept {
    return std::move(model_);
  }
 private:
  QwenInt4EngineLoadPlan(QwenInt4EngineModelPlan model,
      std::filesystem::path cubin_root,
      std::int32_t device_ordinal)
      :model_(std::move(model)),cubin_root_(std::move(cubin_root)),device_ordinal_(device_ordinal){}
  QwenInt4EngineModelPlan model_;
  std::filesystem::path cubin_root_;
  std::int32_t device_ordinal_;
};
}
