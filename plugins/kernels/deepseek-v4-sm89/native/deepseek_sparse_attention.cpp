#include "pih/backend/cuda/deepseek_sparse_attention.h"

namespace pih {

Status validate_deepseek_sparse_attention_launch(
    const DeepSeekSparseAttentionLaunch& launch) {
  const auto blocks = static_cast<std::uint64_t>(launch.query_count) *
                      launch.head_count;
  const auto paged = launch.compressed_kv_bf16 != 0 ||
                     launch.page_slots_u32 != 0 ||
                     launch.compressed_slot_count != 0 ||
                     launch.logical_page_count != 0 ||
                     launch.physical_page_count != 0;
  const auto paged_valid = !paged ||
      (launch.compressed_kv_bf16 != 0 && launch.page_slots_u32 != 0 &&
       launch.recent_physical_offset >= 0 &&
       launch.compressed_physical_offset >= 0 &&
       launch.compressed_slot_count != 0 &&
       launch.logical_page_count ==
           (launch.compressed_slot_count + 63U) / 64U &&
       launch.logical_page_count <= 4096 &&
       launch.physical_page_count != 0 &&
       launch.physical_page_count <= 4096);
  if (launch.query_bf16 == 0 || launch.latent_kv_bf16 == 0 ||
      launch.attention_sink_f32 == 0 || launch.indices_i32 == 0 ||
      launch.output_bf16 == 0 || launch.error_flag_u32 == 0 ||
      launch.stream == 0 || launch.query_count == 0 ||
      launch.query_count > 4096 || launch.head_count == 0 ||
      launch.head_count > 64 || launch.kv_count == 0 ||
      launch.kv_count > 1048704 || launch.index_count == 0 ||
      launch.index_count > 8320 || blocks > 0x7FFFFFFFULL || !paged_valid) {
    return Status::InvalidArgument(
        "DeepSeek sparse attention launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
