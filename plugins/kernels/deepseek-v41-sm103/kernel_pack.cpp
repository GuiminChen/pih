#include "kernel_wire.h"
#include "pih/plugin_sdk/abi.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <cuda_runtime_api.h>
namespace {
using namespace pih;
using namespace pih::deepseek_v41;
thread_local int admitted_device = -1;
constexpr pih_kernel_pack_identity_v1 identity{
  sizeof(identity), PIH_KERNEL_PACK_ABI_VERSION_V1, "pih.kernels.deepseek-v41.sm103",
  "1.0.0", "pih.deepseek-v41-sm103-kernels.v1", "sm103"};
pih_status_v1 Convert(const Status& status) {
  pih_status_v1 result{}; result.struct_size = sizeof(result);
  result.abi_version = PIH_STATUS_ABI_VERSION_V1;
  switch (status.code()) {
    case StatusCode::kOk: result.code = PIH_STATUS_OK_V1; break;
    case StatusCode::kInvalidArgument: result.code = PIH_STATUS_INVALID_ARGUMENT_V1; break;
    case StatusCode::kFailedPrecondition: result.code = PIH_STATUS_FAILED_PRECONDITION_V1; break;
    case StatusCode::kResourceExhausted: result.code = PIH_STATUS_RESOURCE_EXHAUSTED_V1; break;
    case StatusCode::kUnavailable: result.code = PIH_STATUS_UNAVAILABLE_V1; break;
    case StatusCode::kDeadlineExceeded: result.code = PIH_STATUS_DEADLINE_EXCEEDED_V1; break;
    default: result.code = PIH_STATUS_INTERNAL_V1; break;
  }
  const auto message = status.message();
  std::memcpy(result.message, message.data(), std::min(message.size(), sizeof(result.message)-1));
  return result;
}
template<class T, class F> Status Decode(const pih_v41_kernel_command_v1& command, F f) {
  T x{}; kernel_wire::Decoder decoder{command}; decoder.One(x);
  if (!decoder.valid || decoder.cursor != command.word_count)
    return Status::InvalidArgument("SM103 kernel field encoding invalid");
  return f(x);
}
pih_status_v1 Launch(const pih_v41_kernel_command_v1* command) noexcept {
  try {
    if (!command || command->struct_size != sizeof(*command) || command->abi_version != PIH_V41_KERNEL_ABI_V1 ||
        !command->word_count || command->word_count > PIH_V41_KERNEL_WORDS_V1 ||
        !std::all_of(command->words + command->word_count, command->words + PIH_V41_KERNEL_WORDS_V1,
            [](uint64_t word) { return word == 0; }))
      return Convert(Status::InvalidArgument("SM103 kernel command header invalid"));
    int current_device = -1;
    if (admitted_device < 0 || cudaGetDevice(&current_device) != cudaSuccess ||
        current_device != admitted_device)
      return Convert(Status::FailedPrecondition("SM103 launch requires the admitted rank-local B300 device"));
    switch (command->operation) {
      case 1: return Convert(Decode<Fp8LinearLaunch>(*command, LaunchFp8Linear));
      case 2: return Convert(Decode<Fp4LinearLaunch>(*command, LaunchFp4Linear));
      case 3: return Convert(Decode<EngramLookupLaunch>(*command, LaunchEngramLookup));
      case 4: return Convert(Decode<EngramGateLaunch>(*command, LaunchEngramGate));
      case 5: return Convert(Decode<EngramProjectionLaunch>(*command, LaunchEngramProjection));
      case 6: return Convert(Decode<EngramLaunch>(*command, LaunchEngramSingleRank));
      case 7: return Convert(Decode<RmsNormLaunch>(*command, LaunchRmsNorm));
      case 8: return Convert(Decode<MhcMixLaunch>(*command, LaunchMhcMix));
      case 9: return Convert(Decode<MhcPreLaunch>(*command, LaunchMhcPre));
      case 10: return Convert(Decode<MhcPostLaunch>(*command, LaunchMhcPost));
      case 11: return Convert(Decode<RopeTableLaunch>(*command, LaunchRopeTable));
      case 12: return Convert(Decode<RopeApplyLaunch>(*command, LaunchRopeApply));
      case 13: return Convert(Decode<RopeSequenceLaunch>(*command, LaunchRopeSequence));
      case 14: return Convert(Decode<ExpertDispatchLaunch>(*command, LaunchExpertDispatch));
      case 15: return Convert(Decode<ExpertGatherLaunch>(*command, LaunchExpertGather));
      case 16: return Convert(Decode<ExpertScatterLaunch>(*command, LaunchExpertScatter));
      case 17: return Convert(Decode<ExpertActivationLaunch>(*command, LaunchExpertActivation));
      case 18: return Convert(Decode<ExpertMergeLaunch>(*command, LaunchExpertMerge));
      case 19: return Convert(Decode<IndexerQuantizeLaunch>(*command, LaunchIndexerQuantize));
      case 20: return Convert(Decode<IndexerKeyProjectionLaunch>(*command, LaunchIndexerKeyProjection));
      case 21: return Convert(Decode<IndexerKeyCacheLaunch>(*command, LaunchIndexerKeyCache));
      case 22: return Convert(Decode<IndexerWeightsLaunch>(*command, LaunchIndexerWeights));
      case 23: return Convert(Decode<IndexerScoreLaunch>(*command, LaunchIndexerScore));
      case 24: return Convert(Decode<IndexerSelectLaunch>(*command, LaunchIndexerSelect));
      case 25: return Convert(Decode<IndexerCandidatesLaunch>(*command, LaunchIndexerCandidates));
      case 26: return Convert(Decode<GroupedOutputLaunch>(*command, LaunchGroupedOutput));
      case 27: return Convert(Decode<SparseAttentionLaunch>(*command, LaunchSparseAttention));
      case 28: return Convert(Decode<AttentionAssemblyLaunch>(*command, LaunchAttentionAssembly));
      case 29: return Convert(Decode<CompressorPoolLaunch>(*command, LaunchCompressorPool));
      case 30: return Convert(Decode<CompressorProjectionLaunch>(*command, LaunchCompressorProjection));
      case 31: return Convert(Decode<CompressedKvLaunch>(*command, LaunchCompressedKv));
      case 32: return Convert(Decode<WindowKvLaunch>(*command, LaunchWindowKv));
      case 33: return Convert(Decode<HeadProjectionLaunch>(*command, LaunchHeadProjection));
      case 34: return Convert(Decode<TokenEmbeddingLaunch>(*command, LaunchTokenEmbeddingLookup));
      case 35: return Convert(Decode<TokenEmbeddingLaunch>(*command, LaunchTokenEmbeddingExpand));
      case 36: return Convert(Decode<RouterLaunch>(*command, LaunchRouter));
      case 37: return Convert(Decode<SamplingLaunch>(*command, LaunchSampling));
      case 38: return Convert(Decode<AttentionLocalOutputLaunch>(*command, PromoteAttentionOutput));
      case 39: return Convert(Decode<AttentionLocalOutputLaunch>(*command, RoundAttentionOutput));
      case 40: return Convert(Decode<kernel_wire::MhcInitialLaunch>(*command,
          [](const auto& x) { return LaunchMhcInitialPre(x.pre, x.tokens, x.stream); }));
      default: return Convert(Status::InvalidArgument("SM103 kernel operation unsupported"));
    }
  } catch (const std::bad_alloc&) { return Convert(Status::ResourceExhausted("SM103 kernel allocation failed")); }
  catch (...) { return Convert(Status::Internal("SM103 kernel exception")); }
}
pih_status_v1 AdmitDevice(int32_t ordinal) noexcept {
  admitted_device = -1;
  try {
    cudaDeviceProp properties{};
    // Backend preparation authenticates the ordinal and SM before this call.
    // The rank's primary context is retained and bound later by WorkerHandles;
    // cudaGetDevice() here would observe an unrelated default device for ranks
    // whose ordinal is not zero.
    if (ordinal < 0 || cudaGetDeviceProperties(&properties, ordinal) != cudaSuccess)
      return Convert(Status::FailedPrecondition("SM103 Pack cannot inspect the rank-local device"));
    if (properties.major != 10 || properties.minor != 3 || std::strcmp(properties.name, "NVIDIA B300"))
      return Convert(Status::FailedPrecondition("V4.1 profile requires exact NVIDIA B300 SM103"));
    admitted_device = ordinal;
    return Convert(Status::Ok());
  } catch (...) { return Convert(Status::Internal("B300 device admission failed")); }
}
constexpr pih_v41_sm103_kernels_api_v1 contract{
  sizeof(contract), PIH_V41_KERNEL_ABI_V1, &identity, 103, 40, AdmitDevice, Launch};
// Embedded device images have no external artifacts, but use the same
// mandatory host binding contract. No driver or device call occurs here.
pih_status_v1 BindOrigin(const char* path) noexcept {
  pih_status_v1 result{};
  result.struct_size = sizeof(result); result.abi_version = PIH_STATUS_ABI_VERSION_V1;
  result.code = path && path[0] == '/' ? PIH_STATUS_OK_V1 : PIH_STATUS_INVALID_ARGUMENT_V1;
  return result;
}
constexpr pih_kernel_pack_api_v1 api{sizeof(api), PIH_KERNEL_PACK_ABI_VERSION_V1, &identity, &contract, BindOrigin};
}
extern "C" PIH_PLUGIN_EXPORT const pih_kernel_pack_identity_v1* pih_kernel_pack_identity_v1_get() { return &identity; }
extern "C" PIH_PLUGIN_EXPORT const pih_kernel_pack_api_v1* pih_kernel_pack_api_v1_get() { return &api; }
