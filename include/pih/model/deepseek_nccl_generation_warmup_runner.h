#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_nccl_boundary_bootstrap.h"
#include "pih/model/deepseek_nccl_endpoint_warmup_runner.h"

namespace pih {

struct DeepSeekNcclGenerationWarmupEndpointResources final {
  std::uint64_t first_operation_ordinal = 0;
  std::uint64_t buffer_owner_id = 0;
  std::uint64_t buffer_generation = 0;
  void* device_buffer = nullptr;
  std::uint64_t buffer_capacity_bytes = 0;
  DriverStreamHandle stream = 0;
  std::array<DriverEventHandle, 2> completion_events{};
  std::uint64_t submit_ns = 0;
  std::uint64_t deadline_ns = 0;
  DeepSeekNcclWarmupPayloadOperations* payload = nullptr;
  CompletionEventDriver* event_driver = nullptr;
  CompletionEvidenceProvider* evidence = nullptr;
  DeepSeekBoundaryPreparationClock* clock = nullptr;
};

class DeepSeekNcclGenerationWarmupRunner final
    : public DeepSeekNcclGenerationWarmup {
 public:
  static Result<DeepSeekNcclGenerationWarmupRunner> Create(
      std::vector<DeepSeekNcclGenerationWarmupEndpointResources> endpoints);

  Result<std::vector<DeepSeekNcclWarmupReceipt>> run(
      DeepSeekNcclCommunicatorGeneration& generation,
      const DeepSeekNcclEndpointManifestPlan& manifests,
      std::uint32_t maximum_wire_tokens) override;

 private:
  explicit DeepSeekNcclGenerationWarmupRunner(
      std::vector<DeepSeekNcclGenerationWarmupEndpointResources> endpoints)
      noexcept : endpoints_(std::move(endpoints)) {}

  std::vector<DeepSeekNcclGenerationWarmupEndpointResources> endpoints_;
  bool consumed_ = false;
};

}  // namespace pih
