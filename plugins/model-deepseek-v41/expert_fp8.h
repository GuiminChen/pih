#pragma once
#include "linear_fp8.h"
#include "linear_fp4.h"

namespace pih::deepseek_v41 {
struct ExpertActivationLaunch final {
  // BF16 gate/up/output [rows,2304]; optional FP32 route weight [rows].
  EngramDeviceRegion gate, up, route_weights, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t rows = 0;
};
Status ValidateExpertActivation(const ExpertActivationLaunch& launch);
Status LaunchExpertActivation(const ExpertActivationLaunch& launch);
struct SharedExpertLaunch final {
  Fp8LinearLaunch gate, up;
  ExpertActivationLaunch activation;
  Fp8LinearLaunch down;
};
Status ValidateSharedExpert(const SharedExpertLaunch& launch);
Status LaunchSharedExpert(const SharedExpertLaunch& launch);
struct RoutedExpertLaunch final {
  Fp4LinearLaunch gate, up;
  // Required FP32 route weight per gathered token, before down projection.
  ExpertActivationLaunch activation;
  Fp4LinearLaunch down;
};
Status ValidateRoutedExpert(const RoutedExpertLaunch& launch);
Status LaunchRoutedExpert(const RoutedExpertLaunch& launch);
struct ExpertMergeLaunch final {
  // FP32 routed sum AFTER global reduction, BF16 replicated shared output,
  // BF16 final output: all [tokens,5120]. Shared output is added exactly once.
  EngramDeviceRegion routed, shared, output, error_flag;
  std::uintptr_t stream = 0;
  std::uint32_t tokens = 0;
};
Status ValidateExpertMerge(const ExpertMergeLaunch& launch);
Status LaunchExpertMerge(const ExpertMergeLaunch& launch);
struct ExpertSharedMergeLaunch final {
  SharedExpertLaunch shared;
  ExpertMergeLaunch merge;
};
Status ValidateExpertSharedMerge(const ExpertSharedMergeLaunch& launch);
// Requires globally reduced routed input already ordered on the stream.
Status LaunchExpertSharedMerge(const ExpertSharedMergeLaunch& launch);
}  // namespace pih::deepseek_v41
