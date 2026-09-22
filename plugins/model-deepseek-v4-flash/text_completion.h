#pragma once
#include "generation.h"
#include "../common/deepseek_semantic_artifact.h"
#include "pih/contracts/text_inference_v2.h"

namespace pih::deepseek_plugin {
// Caller owns the activation lock and the engine/semantic artifact lifetime.
Result<std::string> CompleteText(DeepSeekEngine& engine,
    const plugin_text::DeepSeekSemanticArtifacts& semantic,
    uint32_t context_capacity, uint32_t prefill_chunk, uint32_t generation_timeout_ms,
    uint64_t& next_request, uint64_t& next_generation, uint64_t& next_plan,
    bool chat, std::string_view request, const pih_text_output_sink_v2& sink);
}
