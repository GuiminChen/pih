#include "worker_artifacts.h"
#include "worker_runtime.h"
#include "worker_communicator.h"
#include "rank_channels.h"
#include "worker/worker_bootstrap.h"
#include "pih/plugin_sdk/capability.h"
#include "kernel_binding.h"
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#if !defined(PIH_V41_COMPILED_SM)
#error "V4.1 worker must be bound to its CUDA build architecture"
#elif PIH_V41_COMPILED_SM != 103
#error "V4.1 rank worker requires SM103"
#endif

namespace {
using namespace pih;
using namespace pih::deepseek_v41;
[[noreturn]] void FailStop(const char* stage) noexcept {
  // Only fixed stage labels; never expose bootstrap or model material.
  std::fprintf(stderr, "pih-v41-rank-worker failed at %s\n", stage);
  std::fflush(stderr);
  std::_Exit(1);
}
void Require(const Status& status) { if (!status.ok()) throw std::runtime_error("rank_operation_failed"); }
template<class T> T Take(Result<T> result) {
  if (!result.ok()) throw std::runtime_error("rank_admission_failed");
  return std::move(*result);
}
void Yield() {
  const timespec delay{0, 1000000};
  const int status = ::clock_nanosleep(CLOCK_MONOTONIC, 0, &delay, nullptr);
  if (status && status != EINTR) throw std::runtime_error("rank_poll_yield_failed");
}
WorkerRuntime::Clock::time_point Deadline(std::uint64_t ns) {
  return WorkerRuntime::Clock::time_point(std::chrono::duration_cast<WorkerRuntime::Clock::duration>(
      std::chrono::nanoseconds(static_cast<std::int64_t>(ns))));
}
template<class T> const T& Api(const pih::worker::WorkerPluginStack& plugins, const char* contract) {
  const auto* value = static_cast<const T*>(plugins.Resolve(contract, PIH_CAPABILITY_SCOPE_PROCESS_V1,
      PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
  if (!value) throw std::runtime_error("rank_backend_api_missing");
  return *value;
}
}

int main(int argc, char**) {
  const char* stage = "bootstrap";
  try {
    // No standalone CLI switches or fallback configuration. This binary is
    // entered by the admitted supervisor with sealed startup data at FD 3.
    if (argc != 1) throw std::invalid_argument("rank_worker_takes_no_arguments");
    auto bootstrap = Take(SealedWorkerBootstrap::Read(3));
    ::close(3);
    stage = "architecture";
    if (bootstrap.sm_major * 10U + bootstrap.sm_minor != PIH_V41_COMPILED_SM)
      throw std::runtime_error("rank_compiled_architecture_mismatch");
    stage = "launch_authority";
    int death_signal = 0;
    if (::getppid() != static_cast<pid_t>(bootstrap.supervisor_pid) ||
        ::geteuid() != bootstrap.supervisor_uid || ::prctl(PR_GET_PDEATHSIG, &death_signal) ||
        death_signal != SIGKILL || ::prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1)
      throw std::runtime_error("rank_launch_authority_invalid");
    const auto startup = Deadline(bootstrap.startup_ns), sequence = Deadline(bootstrap.sequence_ns);
    if (WorkerRuntime::Clock::now() >= startup) throw std::runtime_error("rank_startup_expired");

    stage = "channels";
    const std::array<std::string_view, 4> names{bootstrap.endpoints[0], bootstrap.endpoints[1],
        bootstrap.endpoints[2], bootstrap.endpoints[3]};
    auto channels = Take(RankChannels::Connect(names, bootstrap.rank, bootstrap.world,
        static_cast<std::int32_t>(bootstrap.supervisor_pid), bootstrap.supervisor_uid, startup));
    while (!Take(channels->Poll())) Yield();
    auto endpoints = Take(channels->Views());

    stage = "artifacts";
    auto artifacts = Take(WorkerArtifacts::Open(bootstrap));
    // Reserve the complete payload budget before any backend allocation.
    auto device_budget = bootstrap.device_budget;
    const auto prefill = Take(ExpertWorkspaceBytes(bootstrap.prompt_tokens));
    const auto decode = Take(ExpertWorkspaceBytes(1));
    for (const auto bytes : {artifacts->weights().catalog().device_bytes(), prefill, decode}) {
      if (bytes > device_budget) throw std::runtime_error("rank_device_budget_exceeded");
      device_budget -= bytes;
    }
    stage = "memory-plan";
    auto plan = Take(InferenceMemoryPlan::Create(artifacts->config(), bootstrap.prompt_tokens,
        bootstrap.maximum_positions, bootstrap.world, bootstrap.rank, device_budget,
        bootstrap.host_budget - bootstrap.staging_bytes));

    stage = "plugins";
    pih::worker::WorkerPluginStack plugins;
    plugins.Start(std::string(artifacts->plugin_lock_json()), bootstrap.plugin_lock,
        bootstrap.plugin_lock_sha256, bootstrap.identity.epoch);
    const auto& device = Api<pih_nvidia_cuda_api_v1>(plugins, "pih.device.cuda-runtime.v1");
    const auto& memory = Api<pih_nvidia_cuda_memory_api_v1>(plugins, "pih.device.cuda-memory.v1");
    const auto& resources = Api<pih_nvidia_cuda_resources_api_v1>(plugins, "pih.device.cuda-resources.v1");
    const auto& async = Api<pih_nvidia_cuda_async_api_v1>(plugins, "pih.device.cuda-async.v1");
    const auto& collective = Api<pih_transport_collective_api_v1>(
        plugins, "pih.transport.collective.v1");
    if (device.struct_size != sizeof(device) ||
        device.contract_version != PIH_NVIDIA_CUDA_ABI_VERSION_V1 ||
        !device.context || !device.prepare_device)
      throw std::runtime_error("rank_cuda_device_api_invalid");
    if (memory.struct_size != sizeof(memory) ||
        memory.contract_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
        !memory.context || !memory.allocate_device || !memory.deallocate_device ||
        !memory.allocate_pinned_host || !memory.deallocate_pinned_host || !memory.copy_h2d)
      throw std::runtime_error("rank_cuda_memory_api_invalid");
    if (resources.struct_size != sizeof(resources) ||
        resources.contract_version != PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1 ||
        !resources.context || !resources.retain_primary_context || !resources.bind_runtime ||
        !resources.create_nonblocking_stream || !resources.create_disable_timing_event ||
        !resources.destroy_event || !resources.destroy_stream || !resources.release_primary_context)
      throw std::runtime_error("rank_cuda_resources_api_invalid");
    if (async.struct_size != sizeof(async) ||
        async.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
        !async.context || !async.activate_context || !async.validate_pinned_host ||
        !async.copy_async || !async.memset_async || !async.record_event ||
        !async.query_event || !async.synchronize_stream || !async.require_clean_last_error)
      throw std::runtime_error("rank_cuda_async_api_invalid");
    Require(BindCollectiveTransport(collective));
    const auto* kernels = static_cast<const pih_v41_sm103_kernels_api_v1*>(plugins.Resolve(
        "pih.deepseek-v41-sm103-kernels.v1", PIH_CAPABILITY_SCOPE_ACTIVATION_V1,
        PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1));
    if (!kernels) throw std::runtime_error("rank_sm103_kernels_missing");
    Require(BindSm103Kernels(*kernels));
    const auto prepared = device.prepare_device(device.context, static_cast<int32_t>(bootstrap.device), 10, 3);
    if (!pih_status_is_ok_v1(&prepared)) throw std::runtime_error("rank_b300_prepare_failed");
    const auto admitted_device = kernels->admit_device(static_cast<int32_t>(bootstrap.device));
    if (!pih_status_is_ok_v1(&admitted_device)) throw std::runtime_error("rank_b300_identity_mismatch");

    stage = "runtime";
    {
      WorkerRuntime runtime(artifacts->config(), artifacts->weights(), plan, device, resources, memory, async,
          static_cast<std::int32_t>(bootstrap.device), artifacts->hashes(), endpoints.requests, endpoints.receipts,
          endpoints.commands, endpoints.notices);
      try {
        // Keep runtime/artifacts/channels alive until process termination on any
        // active-stage failure. An outer catch alone runs their destructors first,
        // which can invalidate buffers still borrowed by CUDA/NCCL operations.
        Require(runtime.Start(bootstrap.nccl_id, bootstrap.sm_major, bootstrap.sm_minor,
          {bootstrap.device_budget, bootstrap.host_budget, bootstrap.staging_bytes}, bootstrap.identity,
          bootstrap.sampling, startup, sequence, std::chrono::milliseconds(bootstrap.retirement_ms)));
        volatile std::byte* id = bootstrap.nccl_id.data();
        for (std::size_t i = 0; i < bootstrap.nccl_id.size(); ++i) id[i] = std::byte{};
        while (Take(runtime.Poll()) != WorkerRuntimeState::kReleased) Yield();
      } catch (...) {
        FailStop(stage);
      }
    }
    stage = "plugin-shutdown";
    UnbindSm103Kernels();
    UnbindCollectiveTransport();
    plugins.Shutdown();
    return 0;
  } catch (...) {
    // Never print bootstrap material, paths, prompts or provider-controlled
    // diagnostics here. Process exit is the fault boundary; the supervisor
    // owns epoch failure, peer cancellation and cgroup reconciliation.
    FailStop(stage);
  }
}
