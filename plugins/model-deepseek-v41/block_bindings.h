#pragma once
#include "indexed_sources.h"
#include "prepared_block.h"
#include "step_phases.h"

namespace pih::deepseek_v41 {
// Wire producer-consumer edges before weight binding. These are NOT admission:
// caller must subsequently run ValidatePreparedBlockSources/ValidateSourcePhases.
Result<PreparedBlockLaunch> WirePreparedBlockSources(const IndexedSourcesLaunch& sources, PreparedBlockLaunch block);
Result<IndexedSourcesLaunch> WireSourcePhases(const FlashConfig& config, const StepPhasesLaunch& phases,
    IndexedSourcesLaunch sources);
// Binds produced Q/window/index/coefficient regions into an otherwise complete
// block template. Shared caches/indices without a local producer must already
// be supplied and admitted by the caller. Does not enqueue or prove completion.
Result<PreparedBlockLaunch> BindPreparedBlockSources(const FlashConfig& config,
    const IndexedSourcesLaunch& sources, PreparedBlockLaunch block);
Status ValidatePreparedBlockSources(const FlashConfig& config,
    const IndexedSourcesLaunch& sources, const PreparedBlockLaunch& block);
// Additional source/index-to-block lifetime admission. Does not replace each
// execution stage's validation, resource admission or distributed agreement.
Status ValidateBlockCrossStageBuffers(const FlashConfig& config, const IndexedSourcesLaunch& sources,
    const PreparedBlockLaunch& block, std::span<const ExpertWeights> weights, EngramDeviceRegion workspace,
    const StepPhasesLaunch* phases = nullptr, const EngramLaunch* engram = nullptr,
    std::span<const EngramDeviceRegion> retained = {});
Status ValidateSourcePhases(const FlashConfig& config, const StepPhasesLaunch& phases, const IndexedSourcesLaunch& sources);
Result<IndexedSourcesLaunch> BindSourcePhases(const FlashConfig& config, const StepPhasesLaunch& phases,
    IndexedSourcesLaunch sources);
}  // namespace pih::deepseek_v41
