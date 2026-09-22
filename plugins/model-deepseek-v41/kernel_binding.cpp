#include "kernel_binding.h"
#include "kernel_wire.h"
#include "pih/plugin_sdk/status_bridge.h"
#include <cstring>
namespace pih::deepseek_v41 {
namespace {
thread_local const pih_v41_sm103_kernels_api_v1* bound = nullptr;
template<class T> Status Submit(uint32_t operation, T x) {
  if (!bound) return Status::FailedPrecondition("V4.1 SM103 kernel capability is not bound");
  kernel_wire::Encoder encoder; encoder.command.operation = operation; encoder.One(x);
  if (!encoder.valid) return Status::Internal("V4.1 kernel command exceeds ABI bounds");
  return plugin_status(bound->launch(&encoder.command));
}
}
Status BindSm103Kernels(const pih_v41_sm103_kernels_api_v1& api) {
  if (bound || api.struct_size != sizeof(api) || api.contract_version != PIH_V41_KERNEL_ABI_V1 ||
      api.target_sm != 103 || api.operation_count != 40 || !api.admit_device || !api.launch || !api.identity ||
      api.identity->struct_size != sizeof(pih_kernel_pack_identity_v1) ||
      api.identity->abi_version != PIH_KERNEL_PACK_ABI_VERSION_V1 ||
      !api.identity->pack_id || !api.identity->pack_abi || !api.identity->architecture ||
      std::strcmp(api.identity->pack_id, "pih.kernels.deepseek-v41.sm103") ||
      std::strcmp(api.identity->pack_abi, "pih.deepseek-v41-sm103-kernels.v1") ||
      std::strcmp(api.identity->architecture, "sm103"))
    return Status::FailedPrecondition("V4.1 SM103 kernel capability identity invalid or already bound");
  bound = &api; return Status::Ok();
}
void UnbindSm103Kernels() noexcept { bound = nullptr; }
Status LaunchFp8Linear(const Fp8LinearLaunch& x) { return Submit(1, x); }
Status LaunchFp4Linear(const Fp4LinearLaunch& x) { return Submit(2, x); }
Status LaunchEngramLookup(const EngramLookupLaunch& x) { return Submit(3, x); }
Status LaunchEngramGate(const EngramGateLaunch& x) { return Submit(4, x); }
Status LaunchEngramProjection(const EngramProjectionLaunch& x) { return Submit(5, x); }
Status LaunchEngramSingleRank(const EngramLaunch& x) { return Submit(6, x); }
Status LaunchRmsNorm(const RmsNormLaunch& x) { return Submit(7, x); }
Status LaunchMhcMix(const MhcMixLaunch& x) { return Submit(8, x); }
Status LaunchMhcPre(const MhcPreLaunch& x) { return Submit(9, x); }
Status LaunchMhcPost(const MhcPostLaunch& x) { return Submit(10, x); }
Status LaunchRopeTable(const RopeTableLaunch& x) { return Submit(11, x); }
Status LaunchRopeApply(const RopeApplyLaunch& x) { return Submit(12, x); }
Status LaunchRopeSequence(const RopeSequenceLaunch& x) { return Submit(13, x); }
Status LaunchExpertDispatch(const ExpertDispatchLaunch& x) { return Submit(14, x); }
Status LaunchExpertGather(const ExpertGatherLaunch& x) { return Submit(15, x); }
Status LaunchExpertScatter(const ExpertScatterLaunch& x) { return Submit(16, x); }
Status LaunchExpertActivation(const ExpertActivationLaunch& x) { return Submit(17, x); }
Status LaunchExpertMerge(const ExpertMergeLaunch& x) { return Submit(18, x); }
Status LaunchIndexerQuantize(const IndexerQuantizeLaunch& x) { return Submit(19, x); }
Status LaunchIndexerKeyProjection(const IndexerKeyProjectionLaunch& x) { return Submit(20, x); }
Status LaunchIndexerKeyCache(const IndexerKeyCacheLaunch& x) { return Submit(21, x); }
Status LaunchIndexerWeights(const IndexerWeightsLaunch& x) { return Submit(22, x); }
Status LaunchIndexerScore(const IndexerScoreLaunch& x) { return Submit(23, x); }
Status LaunchIndexerSelect(const IndexerSelectLaunch& x) { return Submit(24, x); }
Status LaunchIndexerCandidates(const IndexerCandidatesLaunch& x) { return Submit(25, x); }
Status LaunchGroupedOutput(const GroupedOutputLaunch& x) { return Submit(26, x); }
Status LaunchSparseAttention(const SparseAttentionLaunch& x) { return Submit(27, x); }
Status LaunchAttentionAssembly(const AttentionAssemblyLaunch& x) { return Submit(28, x); }
Status LaunchCompressorPool(const CompressorPoolLaunch& x) { return Submit(29, x); }
Status LaunchCompressorProjection(const CompressorProjectionLaunch& x) { return Submit(30, x); }
Status LaunchCompressedKv(const CompressedKvLaunch& x) { return Submit(31, x); }
Status LaunchWindowKv(const WindowKvLaunch& x) { return Submit(32, x); }
Status LaunchHeadProjection(const HeadProjectionLaunch& x) { return Submit(33, x); }
Status LaunchTokenEmbeddingLookup(const TokenEmbeddingLaunch& x) { return Submit(34, x); }
Status LaunchTokenEmbeddingExpand(const TokenEmbeddingLaunch& x) { return Submit(35, x); }
Status LaunchRouter(const RouterLaunch& x) { return Submit(36, x); }
Status LaunchSampling(const SamplingLaunch& x) { return Submit(37, x); }
Status PromoteAttentionOutput(const AttentionLocalOutputLaunch& x) { return Submit(38, x); }
Status RoundAttentionOutput(const AttentionLocalOutputLaunch& x) { return Submit(39, x); }
Status LaunchMhcInitialPre(EngramDeviceRegion pre, uint32_t tokens, uintptr_t stream) {
  return Submit(40, kernel_wire::MhcInitialLaunch{pre, tokens, stream});
}
}
