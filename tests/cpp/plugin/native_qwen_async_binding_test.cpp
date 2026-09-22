#include <gtest/gtest.h>
#include "../../../plugins/model-qwen3/async_binding.h"
#include "../../../plugins/model-qwen3/execution_binding.h"
#include "pih/model/nvidia_qwen3_bf16_kv_scrub_driver.h"

namespace {
using namespace pih;
using namespace pih::qwen_plugin;
struct Provider {
  unsigned records{}, queries{}, probes{}, copies{}, clears{};
  uint32_t result{PIH_CUDA_EVENT_PENDING_V1}, copy_kind{};
  uint64_t clear_bytes{32};
  bool failed{}, fail_after_copy{}, copy_failure{}, clear_failure{};
};
pih_status_v1 Ok() { return ExecutionBindingStatus(PIH_STATUS_OK_V1, ""); }
pih_nvidia_cuda_async_api_v1 Api(Provider& p) {
  pih_nvidia_cuda_async_api_v1 api{};
  api.struct_size = sizeof(api); api.contract_version = PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1; api.context = &p;
  api.activate_context = +[](void*, uintptr_t) { return Ok(); };
  api.validate_pinned_host = +[](void*, uintptr_t, uintptr_t) { return Ok(); };
  api.copy_async = +[](void* context, uintptr_t owner, uintptr_t destination,
                      uintptr_t source, uint64_t bytes, uint32_t kind, uintptr_t stream) {
    EXPECT_EQ(owner, 100U); EXPECT_EQ(destination, 4096U); EXPECT_EQ(source, 8192U);
    EXPECT_EQ(bytes, 32U); EXPECT_EQ(stream, 102U);
    auto& p = *static_cast<Provider*>(context); ++p.copies; p.copy_kind = kind;
    if (p.fail_after_copy) p.failed = true;
    return ExecutionBindingStatus(p.copy_failure ? PIH_STATUS_UNAVAILABLE_V1 : PIH_STATUS_OK_V1, "");
  };
  api.memset_async = +[](void* context, uintptr_t owner, uintptr_t destination,
                        uint32_t value, uint64_t bytes, uintptr_t stream) {
    EXPECT_EQ(owner, 100U); EXPECT_EQ(destination, 4096U); EXPECT_EQ(value, 0U);
    EXPECT_EQ(stream, 102U);
    auto& p = *static_cast<Provider*>(context); ++p.clears;
    EXPECT_EQ(bytes, p.clear_bytes);
    return ExecutionBindingStatus(p.clear_failure ? PIH_STATUS_INTERNAL_V1 : PIH_STATUS_OK_V1, "");
  };
  api.synchronize_stream = +[](void*, uintptr_t, uintptr_t) { return Ok(); };
  api.record_event = +[](void* context, uintptr_t owner, uintptr_t event, uintptr_t stream) {
    EXPECT_EQ(owner, 100U); EXPECT_EQ(event, 101U); EXPECT_EQ(stream, 102U);
    ++static_cast<Provider*>(context)->records; return Ok();
  };
  api.query_event = +[](void* context, uintptr_t owner, uintptr_t event, uint32_t* result) {
    EXPECT_EQ(owner, 100U); EXPECT_EQ(event, 101U);
    auto& p = *static_cast<Provider*>(context); ++p.queries; *result = p.result; return Ok();
  };
  api.require_clean_last_error = +[](void* context, uintptr_t owner) {
    EXPECT_EQ(owner, 100U);
    auto& p = *static_cast<Provider*>(context); ++p.probes;
    return ExecutionBindingStatus(p.failed ? PIH_STATUS_INTERNAL_V1 : PIH_STATUS_OK_V1, "");
  };
  return api;
}
TEST(NativeQwenAsyncBinding, PreservesPendingAndCompletedEventStates) {
  Provider p; auto api = Api(p); CapabilityEventDriver driver(api, 100);
  EXPECT_TRUE(driver.record(101, 102).ok());
  auto pending = driver.query(101); ASSERT_TRUE(pending.ok());
  EXPECT_EQ(*pending, CudaEventQueryResult::kNotReady);
  p.result = PIH_CUDA_EVENT_COMPLETE_V1;
  auto complete = driver.query(101); ASSERT_TRUE(complete.ok());
  EXPECT_EQ(*complete, CudaEventQueryResult::kSuccess);
  p.result = 99; EXPECT_FALSE(driver.query(101).ok());
  EXPECT_EQ(p.records, 1U); EXPECT_EQ(p.queries, 3U);
}
TEST(NativeQwenAsyncBinding, RequiresRealErrorProbe) {
  Provider p; auto api = Api(p); CapabilityEventDriver driver(api, 100);
  EXPECT_TRUE(driver.require_clean_last_error().ok());
  p.failed = true; EXPECT_FALSE(driver.require_clean_last_error().ok());
  EXPECT_EQ(p.probes, 2U);
  api.require_clean_last_error = nullptr;
  EXPECT_FALSE(ValidAsyncApi(api));
  EXPECT_FALSE(driver.record(101, 102).ok());
  EXPECT_FALSE(driver.require_clean_last_error().ok());
  EXPECT_EQ(p.records, 0U); EXPECT_EQ(p.probes, 2U);
}
TEST(NativeQwenAsyncBinding, MapsAllCopyDirectionsAndChecksBothErrorFrontiers) {
  Provider p; auto api = Api(p); CapabilityTypedCopyDriver driver(api, 100);
  EXPECT_EQ(driver.context_identity(), 100U);
  for (auto kind : {CudaCopyKind::kHostToDevice, CudaCopyKind::kDeviceToHost, CudaCopyKind::kDeviceToDevice}) {
    EXPECT_TRUE(driver.copy(kind, 4096, 8192, 32, 102).ok());
    const auto expected = kind == CudaCopyKind::kHostToDevice ? PIH_CUDA_COPY_H2D_V1 :
        kind == CudaCopyKind::kDeviceToHost ? PIH_CUDA_COPY_D2H_V1 : PIH_CUDA_COPY_D2D_V1;
    EXPECT_EQ(p.copy_kind, expected);
  }
  EXPECT_EQ(p.copies, 3U); EXPECT_EQ(p.probes, 6U);
}
TEST(NativeQwenAsyncBinding, DirtyThreadPreventsSubmission) {
  Provider p; p.failed = true; auto api = Api(p); CapabilityTypedCopyDriver driver(api, 100);
  EXPECT_FALSE(driver.copy(CudaCopyKind::kHostToDevice, 4096, 8192, 32, 102).ok());
  EXPECT_EQ(p.copies, 0U); EXPECT_EQ(p.probes, 1U);
}
TEST(NativeQwenAsyncBinding, PostCopyErrorCannotBecomeSuccess) {
  Provider p; p.fail_after_copy = true; auto api = Api(p); CapabilityTypedCopyDriver driver(api, 100);
  EXPECT_FALSE(driver.copy(CudaCopyKind::kHostToDevice, 4096, 8192, 32, 102).ok());
  EXPECT_EQ(p.copies, 1U); EXPECT_EQ(p.probes, 2U);
}
TEST(NativeQwenAsyncBinding, CopyFailureIsPreservedAfterCleanProbe) {
  Provider p; p.copy_failure = true; auto api = Api(p); CapabilityTypedCopyDriver driver(api, 100);
  EXPECT_EQ(driver.copy(CudaCopyKind::kHostToDevice, 4096, 8192, 32, 102).code(), StatusCode::kUnavailable);
  EXPECT_EQ(p.probes, 2U);
}
TEST(NativeQwenAsyncBinding, InvalidShapeNeverReachesProvider) {
  Provider p; auto api = Api(p); CapabilityTypedCopyDriver driver(api, 100);
  EXPECT_FALSE(driver.copy(static_cast<CudaCopyKind>(99), 4096, 8192, 32, 102).ok());
  EXPECT_FALSE(driver.copy(CudaCopyKind::kHostToDevice, 0, 8192, 32, 102).ok());
  EXPECT_FALSE(driver.copy(CudaCopyKind::kHostToDevice, 4096, 8192, 0, 102).ok());
  EXPECT_FALSE(driver.copy(CudaCopyKind::kHostToDevice, 4096, 8192, 32, 0).ok());
  EXPECT_FALSE(driver.copy(CudaCopyKind::kHostToDevice, UINTPTR_MAX - 8, 8192, 32, 102).ok());
  EXPECT_FALSE(driver.copy(CudaCopyKind::kHostToDevice, 4096, UINTPTR_MAX - 8, 32, 102).ok());
  EXPECT_EQ(p.copies, 0U); EXPECT_EQ(p.probes, 0U);
}
TEST(NativeQwenAsyncBinding, KvScrubWaitsForCompletionThroughProvider) {
  Provider p; p.result = PIH_CUDA_EVENT_COMPLETE_V1; auto api = Api(p);
  auto driver = NvidiaQwenBf16KvScrubDriver::Create(api, 100, 1000000);
  ASSERT_TRUE(driver.ok());
  EXPECT_TRUE(driver->clear_and_wait(4096, 32, 102, 101).ok());
  EXPECT_EQ(p.clears, 1U); EXPECT_EQ(p.records, 1U); EXPECT_EQ(p.queries, 1U);
  EXPECT_EQ(p.probes, 2U);
}
TEST(NativeQwenAsyncBinding, FailedKvClearDoesNotRecordCompletion) {
  Provider p; p.clear_failure = true; auto api = Api(p);
  auto driver = NvidiaQwenBf16KvScrubDriver::Create(api, 100, 1000000);
  ASSERT_TRUE(driver.ok());
  EXPECT_FALSE(driver->clear_and_wait(4096, 32, 102, 101).ok());
  EXPECT_EQ(p.clears, 1U); EXPECT_EQ(p.records, 0U); EXPECT_EQ(p.queries, 0U);
}
TEST(NativeQwenAsyncBinding, PendingKvScrubCannotReportSuccess) {
  Provider p; auto api = Api(p);
  auto driver = NvidiaQwenBf16KvScrubDriver::Create(api, 100, 1);
  ASSERT_TRUE(driver.ok());
  EXPECT_EQ(driver->clear_and_wait(4096, 32, 102, 101).code(), StatusCode::kUnavailable);
  EXPECT_GE(p.queries, 1U);
}
TEST(NativeQwenAsyncBinding, KvScrubRejectsInvalidIdentityAndOverflow) {
  Provider p; auto api = Api(p);
  EXPECT_FALSE(NvidiaQwenBf16KvScrubDriver::Create(api, 0, 1).ok());
  EXPECT_FALSE(NvidiaQwenBf16KvScrubDriver::Create(api, 100, 0).ok());
  auto driver = NvidiaQwenBf16KvScrubDriver::Create(api, 100, 1000000);
  ASSERT_TRUE(driver.ok());
  EXPECT_FALSE(driver->clear_and_wait(UINTPTR_MAX - 8, 32, 102, 101).ok());
  EXPECT_EQ(p.clears, 0U); EXPECT_EQ(p.probes, 0U);
}
TensorView ErrorTarget(std::int32_t rank = 0, std::int64_t bytes = 4) {
  const std::array<std::int64_t, 1> shape{bytes};
  return TensorView::Create(reinterpret_cast<void*>(4096), DType::kUInt8,
      shape, {}, Device::Create(DeviceType::kCuda, rank).value(), 7).value();
}
TEST(NativeQwenAsyncBinding, ErrorClearZerosExactlyFourBytesThroughProvider) {
  Provider p; p.clear_bytes = 4; auto api = Api(p);
  CapabilityErrorClearDriver driver(api, 100, 0);
  EXPECT_TRUE(driver.clear_u32_async(ErrorTarget(), 0, 102).ok());
  EXPECT_EQ(p.clears, 1U); EXPECT_EQ(p.probes, 2U);
}
TEST(NativeQwenAsyncBinding, ErrorClearRejectsForeignRankAndWrongShape) {
  Provider p; auto api = Api(p); CapabilityErrorClearDriver driver(api, 100, 0);
  EXPECT_FALSE(driver.clear_u32_async(ErrorTarget(), 1, 102).ok());
  EXPECT_FALSE(driver.clear_u32_async(ErrorTarget(1), 0, 102).ok());
  EXPECT_FALSE(driver.clear_u32_async(ErrorTarget(0, 8), 0, 102).ok());
  EXPECT_FALSE(driver.clear_u32_async(ErrorTarget(), 0, 0).ok());
  EXPECT_EQ(p.clears, 0U); EXPECT_EQ(p.probes, 0U);
}
TEST(NativeQwenAsyncBinding, ErrorClearCannotMaskSubmissionFailure) {
  Provider p; p.clear_bytes = 4; p.clear_failure = true; auto api = Api(p);
  CapabilityErrorClearDriver driver(api, 100, 0);
  EXPECT_EQ(driver.clear_u32_async(ErrorTarget(), 0, 102).code(), StatusCode::kInternal);
  EXPECT_EQ(p.clears, 1U); EXPECT_EQ(p.probes, 2U);
}
TEST(NativeQwenAsyncBinding, ErrorClearRejectsDirtyThreadBeforeSubmission) {
  Provider p; p.failed = true; auto api = Api(p);
  CapabilityErrorClearDriver driver(api, 100, 0);
  EXPECT_FALSE(driver.clear_u32_async(ErrorTarget(), 0, 102).ok());
  EXPECT_EQ(p.clears, 0U); EXPECT_EQ(p.probes, 1U);
}
}  // namespace
