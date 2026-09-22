#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pih/model/deepseek_attention_runtime_resources.h"
#include "pih/model/deepseek_expert_lane_owner.h"

namespace pih {

struct CudaRuntimeResourceIdentity;
class CudaRuntimeResourceDriver;
class RegisteredPinnedAllocator;
class DeepSeekH2dRuntime;
class Allocator;
class MemoryCopier;
class DeepSeekExpertHostSource;
class DeepSeekExpertSlotTable;
struct DeepSeekExpertComputeArena;
class DeepSeekFixedStateBankOperations;
class DeepSeekEndpointSequenceOperations;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkEmbedOperations;
class DeepSeekDsparkHeadOperations;
class DeepSeekDsparkPrefillStageOperations;
class DeepSeekDsparkDecodeStageOperations;
class DeepSeekDsparkMoeStageOperations;
#endif
class DeepSeekAttentionProjectionOperations;
class DeepSeekAttentionOutputProjectionOperations;
class DeepSeekMhcSequenceOperations;
class DeepSeekRopeTableOperations;
class DeepSeekSharedExpertOperations;

class NvidiaDeepSeekRankRuntimeView : public DeepSeekRankRuntimeOwner {
 public:
  virtual Result<std::unique_ptr<RegisteredPinnedAllocator>>
  CreateSharedPinnedHostAllocator() = 0;
  virtual RegisteredPinnedAllocator* host_spill_pinned_allocator()
      noexcept = 0;
  virtual Result<std::unique_ptr<Allocator>> CreateDeviceAllocator() = 0;
  virtual Allocator* host_spill_device_allocator() noexcept = 0;
  virtual Result<std::unique_ptr<MemoryCopier>> CreateDeviceMemoryCopier() = 0;
  virtual const CudaRuntimeResourceIdentity& identity() const noexcept = 0;
  virtual CudaRuntimeResourceDriver& resource_driver() noexcept = 0;
  virtual DeepSeekH2dRuntime& h2d() noexcept = 0;
  virtual DeepSeekAttentionRuntimeOperations attention_operations()
      noexcept = 0;
  virtual DeepSeekFixedStateBankOperations& fixed_state_operations()
      noexcept = 0;
  virtual DeepSeekEndpointSequenceOperations& endpoint_operations()
      noexcept = 0;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  virtual DeepSeekDsparkEmbedOperations* dspark_embed_operations()
      noexcept = 0;
  virtual DeepSeekDsparkHeadOperations* dspark_head_operations()
      noexcept = 0;
  virtual DeepSeekDsparkPrefillStageOperations* dspark_prefill_operations()
      noexcept = 0;
  virtual DeepSeekDsparkDecodeStageOperations* dspark_decode_operations()
      noexcept = 0;
  virtual DeepSeekDsparkMoeStageOperations* dspark_moe_operations()
      noexcept = 0;
#endif
  virtual DeepSeekAttentionProjectionOperations&
  attention_projection_operations() noexcept = 0;
  virtual DeepSeekAttentionOutputProjectionOperations&
  attention_output_projection_operations() noexcept = 0;
  virtual DeepSeekMhcSequenceOperations& mhc_operations() noexcept = 0;
  virtual DeepSeekRopeTableOperations& rope_table_operations() noexcept = 0;
  virtual Result<std::unique_ptr<DeepSeekSharedExpertOperations>>
  CreateSharedExpertOperations() = 0;
  virtual Result<std::unique_ptr<DeepSeekExpertKernelLaneOwner>>
  CreateExpertKernelLane(
      RegisteredPinnedAllocator& pinned_allocator,
      DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
      std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
      std::uintptr_t accumulator_f32
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      , std::uintptr_t dspark_source_hidden_bf16
#endif
      ) = 0;
  virtual Result<std::unique_ptr<DeepSeekExpertTransferLaneOwner>>
  CreateExpertTransferLane(
      DeepSeekExpertHostSource& source,
      std::vector<std::uintptr_t> slot_bases,
      std::uint32_t transfer_reservation_window) = 0;
};

}  // namespace pih
