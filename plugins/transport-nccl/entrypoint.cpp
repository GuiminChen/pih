#include "pih/contracts/transport_collective_v1.h"
#include "pih/plugin_sdk/abi.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#if NCCL_VERSION_CODE != 23102
#error "Transport plugin requires NCCL 2.31.2"
#endif
static_assert(NCCL_UNIQUE_ID_BYTES == PIH_TRANSPORT_NCCL_ID_BYTES_V1);

struct pih_transport_communicator_v1 final {
  ncclComm_t comm{};
  uint32_t world{};
  uint32_t rank{};
  int32_t device{-1};
  CUcontext cuda_context{};
  enum State : uint32_t { kInitializing, kReady, kFinalizing, kFinalized, kQuarantined } state{kInitializing};
  bool destroy_attempted{};
  bool abort_attempted{};
};

namespace {
struct Provider final {
  const pih_host_api_v1* host{};
  uint32_t phase{};
  bool retiring{};
  std::mutex mutex;
  std::vector<std::unique_ptr<pih_transport_communicator_v1>> handles;
};
Provider provider;

pih_status_v1 Status(uint32_t code, const char* message = "") noexcept {
  pih_status_v1 result{};
  result.struct_size = sizeof(result);
  result.abi_version = PIH_STATUS_ABI_VERSION_V1;
  result.code = code;
  std::strncpy(result.message, message, sizeof(result.message) - 1);
  return result;
}
template<class Operation>
pih_status_v1 Guard(Operation operation) noexcept {
  try { return operation(); }
  catch (const std::bad_alloc&) { return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "transport_allocation_failed"); }
  catch (...) { return Status(PIH_STATUS_INTERNAL_V1, "transport_operation_failed"); }
}
pih_status_v1 Nccl(ncclResult_t result, const char* operation) noexcept {
  if (result == ncclSuccess || result == ncclInProgress) return Status(PIH_STATUS_OK_V1);
  return Status(PIH_STATUS_INTERNAL_V1, operation);
}
pih_transport_communicator_v1* Find(pih_transport_communicator_v1* handle) noexcept {
  for (auto& owned : provider.handles)
    if (owned.get() == handle) return owned.get();
  return nullptr;
}
pih_status_v1 Device(const pih_transport_communicator_v1& owner) noexcept {
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess)
    return Status(PIH_STATUS_INTERNAL_V1, "transport_cuda_device_unavailable");
  if (current != owner.device)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_cuda_device_changed");
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context ||
      context != owner.cuda_context)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_cuda_context_changed");
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 Stream(const pih_transport_communicator_v1& owner,
                     uintptr_t stream) noexcept {
  CUcontext context = nullptr;
  if (!stream || cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context) != CUDA_SUCCESS ||
      context != owner.cuda_context)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_stream_context_invalid");
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 Identity(const pih_transport_communicator_v1& owner) noexcept {
  const auto device = Device(owner);
  if (!pih_status_is_ok_v1(&device)) return device;
  int world = 0, rank = -1, ordinal = -1;
  if (ncclCommCount(owner.comm, &world) != ncclSuccess ||
      ncclCommUserRank(owner.comm, &rank) != ncclSuccess ||
      ncclCommCuDevice(owner.comm, &ordinal) != ncclSuccess)
    return Status(PIH_STATUS_INTERNAL_V1, "transport_identity_query_failed");
  if (world != static_cast<int>(owner.world) ||
      rank != static_cast<int>(owner.rank) || ordinal != owner.device)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_identity_changed");
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 Async(const pih_transport_communicator_v1& owner,
                   uint32_t* transition) noexcept {
  const auto device = Device(owner);
  if (!pih_status_is_ok_v1(&device)) return device;
  ncclResult_t asynchronous = ncclSuccess;
  const auto result = ncclCommGetAsyncError(owner.comm, &asynchronous);
  if (result != ncclSuccess || (asynchronous != ncclSuccess && asynchronous != ncclInProgress))
    return Status(PIH_STATUS_INTERNAL_V1, "transport_async_failure");
  *transition = asynchronous == ncclSuccess ? PIH_TRANSPORT_READY_V1 : PIH_TRANSPORT_PENDING_V1;
  return Status(PIH_STATUS_OK_V1);
}
uint32_t ElementBytes(uint32_t datatype) noexcept {
  if (datatype == PIH_TRANSPORT_BF16_V1) return 2;
  if (datatype == PIH_TRANSPORT_FP32_V1) return 4;
  return 0;
}
ncclDataType_t NcclType(uint32_t datatype) noexcept {
  return datatype == PIH_TRANSPORT_BF16_V1 ? ncclBfloat16 : ncclFloat32;
}
pih_status_v1 Buffer(const pih_transport_communicator_v1& owner,
                     uintptr_t address, uint64_t bytes, uint32_t datatype) noexcept {
  const auto element = ElementBytes(datatype);
  if (!element || !address || address % element || !bytes || bytes % element ||
      bytes / element > 4096ULL * 1048576 ||
      bytes > UINTPTR_MAX - address || bytes > SIZE_MAX)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_buffer_shape_invalid");
  const auto identity = Identity(owner);
  if (!pih_status_is_ok_v1(&identity)) return identity;
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(address)) != cudaSuccess ||
      attributes.type != cudaMemoryTypeDevice || attributes.device != owner.device)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_buffer_device_invalid");
  CUdeviceptr base = 0;
  size_t allocation = 0;
  if (cuMemGetAddressRange(&base, &allocation, static_cast<CUdeviceptr>(address)) != CUDA_SUCCESS ||
      address < base || address - base > allocation ||
      bytes > allocation - (address - base))
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_buffer_extent_invalid");
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 Start(void* context, const uint8_t* id, uint32_t world,
    uint32_t rank, int32_t device, pih_transport_communicator_v1** output) noexcept {
  if (output) *output = nullptr;
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !id || !output ||
        (world != 2 && world != 4 && world != 8) || rank >= world || device < 0)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_bootstrap_invalid");
    std::lock_guard lock(provider.mutex);
    if (provider.phase != 4 || provider.retiring)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_not_ready");
    if (provider.handles.size() >= 8)
      return Status(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "transport_handle_limit");
    provider.handles.reserve(provider.handles.size() + 1);
    auto owner = std::make_unique<pih_transport_communicator_v1>();
    int version = 0, current = -1;
    if (ncclGetVersion(&version) != ncclSuccess || version != NCCL_VERSION_CODE ||
        cudaGetDevice(&current) != cudaSuccess || current != device)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_version_or_device_invalid");
    if (cuCtxGetCurrent(&owner->cuda_context) != CUDA_SUCCESS || !owner->cuda_context)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_cuda_context_missing");
    ncclUniqueId unique{};
    std::memcpy(unique.internal, id, sizeof(unique.internal));
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    ncclComm_t comm = nullptr;
    const auto result = ncclCommInitRankConfig(&comm, static_cast<int>(world), unique,
                                               static_cast<int>(rank), &config);
    volatile char* wipe = unique.internal;
    for (size_t i = 0; i < sizeof(unique.internal); ++i) wipe[i] = 0;
    if (!comm) return Status(PIH_STATUS_INTERNAL_V1, "transport_init_returned_null");
    owner->comm = comm;
    owner->world = world;
    owner->rank = rank;
    owner->device = device;
    if (result != ncclSuccess && result != ncclInProgress)
      owner->state = owner->kQuarantined;
    *output = owner.get();
    provider.handles.push_back(std::move(owner));
    return result == ncclSuccess || result == ncclInProgress
        ? Status(PIH_STATUS_OK_V1)
        : Status(PIH_STATUS_INTERNAL_V1, "transport_init_failed_abort_required");
  });
}
pih_status_v1 Poll(void* context, pih_transport_communicator_v1* handle,
                   uint32_t* transition) noexcept {
  if (transition) *transition = PIH_TRANSPORT_PENDING_V1;
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !transition)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_poll_invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner || (owner->state != owner->kInitializing && owner->state != owner->kFinalizing))
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_transition_absent");
    const auto device = Device(*owner);
    if (!pih_status_is_ok_v1(&device)) { owner->state = owner->kQuarantined; return device; }
    const auto status = Async(*owner, transition);
    if (!pih_status_is_ok_v1(&status)) { owner->state = owner->kQuarantined; return status; }
    if (*transition == PIH_TRANSPORT_READY_V1) {
      if (owner->state == owner->kInitializing) {
        const auto identity = Identity(*owner);
        if (!pih_status_is_ok_v1(&identity)) { owner->state = owner->kQuarantined; return identity; }
        owner->state = owner->kReady;
      } else owner->state = owner->kFinalized;
    }
    return Status(PIH_STATUS_OK_V1);
  });
}
pih_status_v1 ValidateBuffer(void* context, pih_transport_communicator_v1* handle,
    uintptr_t address, uint64_t bytes, uint32_t datatype, uint32_t world,
    uint32_t rank) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_context_invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner || owner->state != owner->kReady || owner->world != world || owner->rank != rank)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_not_ready_or_rank_drifted");
    uint32_t transition = 0;
    const auto asynchronous = Async(*owner, &transition);
    if (!pih_status_is_ok_v1(&asynchronous)) { owner->state = owner->kQuarantined; return asynchronous; }
    if (transition != PIH_TRANSPORT_READY_V1)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_collective_pending");
    return Buffer(*owner, address, bytes, datatype);
  });
}
pih_status_v1 AllReduce(void* context, pih_transport_communicator_v1* handle,
    uintptr_t address, uint64_t bytes, uint32_t datatype, uintptr_t stream,
    uint32_t* transition) noexcept {
  if (transition) *transition = PIH_TRANSPORT_PENDING_V1;
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !transition || !stream)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_reduce_invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner || owner->state != owner->kReady)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_not_ready");
    auto status = Async(*owner, transition);
    if (!pih_status_is_ok_v1(&status)) { owner->state = owner->kQuarantined; return status; }
    if (*transition != PIH_TRANSPORT_READY_V1)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_collective_pending");
    status = Buffer(*owner, address, bytes, datatype);
    if (!pih_status_is_ok_v1(&status)) return status;
    status = Stream(*owner, stream);
    if (!pih_status_is_ok_v1(&status)) return status;
    const auto result = ncclAllReduce(reinterpret_cast<void*>(address),
        reinterpret_cast<void*>(address), static_cast<size_t>(bytes / ElementBytes(datatype)),
        NcclType(datatype), ncclSum, owner->comm, reinterpret_cast<cudaStream_t>(stream));
    status = Nccl(result, "transport_reduce_failed");
    if (!pih_status_is_ok_v1(&status)) { owner->state = owner->kQuarantined; return status; }
    *transition = result == ncclSuccess ? PIH_TRANSPORT_READY_V1 : PIH_TRANSPORT_PENDING_V1;
    return status;
  });
}
pih_status_v1 AllGather(void* context, pih_transport_communicator_v1* handle,
    uintptr_t input, uint64_t input_bytes, uintptr_t output, uint64_t output_bytes,
    uint32_t datatype, uintptr_t stream, uint32_t* transition) noexcept {
  if (transition) *transition = PIH_TRANSPORT_PENDING_V1;
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !transition || !stream || !input_bytes ||
        output_bytes / input_bytes == 0 || output_bytes / input_bytes > 8)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_gather_invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner || owner->state != owner->kReady)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_not_ready");
    if (output_bytes % input_bytes || output_bytes / input_bytes != owner->world)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_gather_shape_invalid");
    auto status = Async(*owner, transition);
    if (!pih_status_is_ok_v1(&status)) { owner->state = owner->kQuarantined; return status; }
    if (*transition != PIH_TRANSPORT_READY_V1)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_collective_pending");
    status = Buffer(*owner, input, input_bytes, datatype);
    if (!pih_status_is_ok_v1(&status)) return status;
    status = Buffer(*owner, output, output_bytes, datatype);
    if (!pih_status_is_ok_v1(&status)) return status;
    if (input < output + output_bytes && output < input + input_bytes)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_gather_alias_invalid");
    status = Stream(*owner, stream);
    if (!pih_status_is_ok_v1(&status)) return status;
    const auto result = ncclAllGather(reinterpret_cast<const void*>(input),
        reinterpret_cast<void*>(output), static_cast<size_t>(input_bytes / ElementBytes(datatype)),
        NcclType(datatype), owner->comm, reinterpret_cast<cudaStream_t>(stream));
    status = Nccl(result, "transport_gather_failed");
    if (!pih_status_is_ok_v1(&status)) { owner->state = owner->kQuarantined; return status; }
    *transition = result == ncclSuccess ? PIH_TRANSPORT_READY_V1 : PIH_TRANSPORT_PENDING_V1;
    return status;
  });
}
pih_status_v1 PollCollective(void* context, pih_transport_communicator_v1* handle,
    uint32_t* transition) noexcept {
  if (transition) *transition = PIH_TRANSPORT_PENDING_V1;
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !transition)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_poll_invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner || owner->state != owner->kReady)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_not_ready");
    const auto device = Device(*owner);
    if (!pih_status_is_ok_v1(&device)) { owner->state = owner->kQuarantined; return device; }
    const auto status = Async(*owner, transition);
    if (!pih_status_is_ok_v1(&status)) owner->state = owner->kQuarantined;
    return status;
  });
}
pih_status_v1 BeginFinalize(void* context, pih_transport_communicator_v1* handle) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider) return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_context_invalid");
    std::lock_guard lock(provider.mutex);
    auto* owner = Find(handle);
    if (!owner || owner->state != owner->kReady)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_not_ready");
    const auto device = Device(*owner);
    if (!pih_status_is_ok_v1(&device)) { owner->state = owner->kQuarantined; return device; }
    uint32_t transition = 0;
    const auto asynchronous = Async(*owner, &transition);
    if (!pih_status_is_ok_v1(&asynchronous)) { owner->state = owner->kQuarantined; return asynchronous; }
    if (transition != PIH_TRANSPORT_READY_V1)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_enqueue_pending");
    owner->state = owner->kQuarantined;
    const auto result = ncclCommFinalize(owner->comm);
    if (result != ncclSuccess && result != ncclInProgress)
      return Status(PIH_STATUS_INTERNAL_V1, "transport_finalize_failed");
    owner->state = owner->kFinalizing;
    return Status(PIH_STATUS_OK_V1);
  });
}
pih_status_v1 Release(void* context, pih_transport_communicator_v1** handle) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !handle || !*handle)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_handle_invalid");
    std::lock_guard lock(provider.mutex);
    auto item = std::find_if(provider.handles.begin(), provider.handles.end(),
        [&](const auto& owned) { return owned.get() == *handle; });
    if (item == provider.handles.end() || (*item)->state != (*item)->kFinalized ||
        (*item)->destroy_attempted)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_not_finalized");
    const auto device = Device(**item);
    if (!pih_status_is_ok_v1(&device)) { (*item)->state = (*item)->kQuarantined; return device; }
    (*item)->destroy_attempted = true;
    (*item)->state = (*item)->kQuarantined;
    if (ncclCommDestroy((*item)->comm) != ncclSuccess)
      return Status(PIH_STATUS_INTERNAL_V1, "transport_destroy_failed");
    provider.handles.erase(item);
    *handle = nullptr;
    return Status(PIH_STATUS_OK_V1);
  });
}
pih_status_v1 Abort(void* context, pih_transport_communicator_v1** handle) noexcept {
  return Guard([&]() -> pih_status_v1 {
    if (context != &provider || !handle || !*handle)
      return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_handle_invalid");
    std::lock_guard lock(provider.mutex);
    auto item = std::find_if(provider.handles.begin(), provider.handles.end(),
        [&](const auto& owned) { return owned.get() == *handle; });
    if (item == provider.handles.end() || (*item)->abort_attempted || (*item)->destroy_attempted)
      return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_abort_unavailable");
    const auto device = Device(**item);
    if (!pih_status_is_ok_v1(&device)) { (*item)->state = (*item)->kQuarantined; return device; }
    (*item)->abort_attempted = true;
    (*item)->state = (*item)->kQuarantined;
    if (ncclCommAbort((*item)->comm) != ncclSuccess)
      return Status(PIH_STATUS_INTERNAL_V1, "transport_abort_failed");
    provider.handles.erase(item);
    *handle = nullptr;
    return Status(PIH_STATUS_OK_V1);
  });
}

const pih_transport_collective_api_v1 api{
    sizeof(api), PIH_TRANSPORT_COLLECTIVE_ABI_V1, &provider,
    Start, Poll, ValidateBuffer, AllReduce, AllGather, PollCollective,
    BeginFinalize, Release, Abort};

pih_status_v1 Advance(void* context, uint32_t expected) noexcept {
  if (context != &provider || provider.phase != expected)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_lifecycle_order_invalid");
  ++provider.phase;
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 Register(void* context) noexcept {
  if (context != &provider || provider.phase != 0 || !provider.host)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_registration_invalid");
  const pih_capability_v1 capability{sizeof(capability), PIH_CAPABILITY_ABI_VERSION_V1,
      "transport.collective.v1", "pih.transport.collective.v1", &api,
      PIH_CAPABILITY_THREADING_SERIALIZED_V1, PIH_CAPABILITY_SCOPE_PROCESS_V1,
      PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1};
  const auto status = provider.host->register_capability(provider.host->context, &capability);
  if (!pih_status_is_valid_v1(&status))
    return Status(PIH_STATUS_INTERNAL_V1, "transport_registration_status_invalid");
  return pih_status_is_ok_v1(&status) ? Advance(context, 0) : status;
}
pih_status_v1 Configure(void* context) noexcept { return Advance(context, 1); }
pih_status_v1 StartPlugin(void* context) noexcept { return Advance(context, 2); }
pih_status_v1 Ready(void* context) noexcept { return Advance(context, 3); }
pih_status_v1 Drain(void* context) noexcept {
  if (context != &provider)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_drain_order_invalid");
  std::lock_guard lock(provider.mutex);
  if (provider.phase != 4)
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_drain_order_invalid");
  // Once retirement starts, existing owners may finalize/abort but no new
  // communicator can extend the activation. Keep the gate closed on retries.
  provider.retiring = true;
  if (!provider.handles.empty())
    return Status(PIH_STATUS_UNAVAILABLE_V1, "transport_live_communicators");
  return Advance(context, 4);
}
pih_status_v1 Stop(void* context) noexcept {
  if (context != &provider || (provider.phase != 3 && provider.phase != 5))
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_stop_order_invalid");
  std::lock_guard lock(provider.mutex);
  if (!provider.handles.empty())
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_live_communicators");
  provider.phase = 6;
  return Status(PIH_STATUS_OK_V1);
}
pih_status_v1 Dispose(void* context) noexcept {
  if (context != &provider || (provider.phase != 1 && provider.phase != 2 && provider.phase != 6))
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_dispose_order_invalid");
  std::lock_guard lock(provider.mutex);
  if (!provider.handles.empty())
    return Status(PIH_STATUS_FAILED_PRECONDITION_V1, "transport_live_communicators");
  provider.host = nullptr;
  provider.phase = 7;
  return Status(PIH_STATUS_OK_V1);
}
}  // namespace

extern "C" PIH_PLUGIN_EXPORT pih_status_v1 pih_plugin_entry_v1(
    const pih_host_api_v1* host, pih_plugin_api_v1* plugin) noexcept {
  if (!pih_host_api_is_valid_v1(host) || !pih_plugin_api_accepts_v1(plugin) ||
      provider.host || provider.phase)
    return Status(PIH_STATUS_INVALID_ARGUMENT_V1, "transport_plugin_entry_invalid");
  provider.host = host;
  plugin->plugin_id = "pih.transport.nccl";
  plugin->plugin_version = "1.0.0";
  plugin->context = &provider;
  plugin->lifecycle = {sizeof(pih_plugin_lifecycle_v1),
      PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1,
      Register, Configure, StartPlugin, Ready, Drain, Stop, Dispose};
  return Status(PIH_STATUS_OK_V1);
}
