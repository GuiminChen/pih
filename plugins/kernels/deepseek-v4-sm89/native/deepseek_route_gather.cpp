#include "pih/backend/cuda/deepseek_route_gather.h"

namespace pih {

Status validate_deepseek_route_gather_launch(
    const DeepSeekRouteGatherLaunch& launch) {
  if (launch.source_hidden_bf16 == 0 || launch.token_indices_u32 == 0 ||
      launch.route_hidden_bf16 == 0 || launch.error_flag == 0 ||
      launch.stream == 0 || launch.route_count == 0 ||
      launch.packed_token_count == 0 ||
      launch.route_count > launch.packed_token_count) {
    return Status::InvalidArgument("DeepSeek route gather launch is invalid");
  }
  return Status::Ok();
}

}  // namespace pih
