#include <vector>

#include <gtest/gtest.h>

#include "../../../plugins/model-qwen3/resource_binding.h"
#include "../../../plugins/model-qwen3/execution_binding.h"

namespace {
using namespace pih;
using namespace pih::qwen_plugin;
struct Provider {
  unsigned retained{}, created{}, fail_at{};
  bool zero{}, bad_status{}, fail_retirement{};
  std::vector<std::uintptr_t> retired;
};
pih_status_v1 Ok() { return ExecutionBindingStatus(PIH_STATUS_OK_V1, ""); }
pih_status_v1 Retain(void* context, int32_t ordinal, uint32_t flags, uintptr_t* handle) {
  auto& p = *static_cast<Provider*>(context);
  ++p.retained;
  EXPECT_EQ(ordinal, 0);
  EXPECT_EQ(flags, PIH_CUDA_CONTEXT_SCHED_YIELD_V1);
  if (p.bad_status) return {};
  *handle = 100;
  return Ok();
}
pih_status_v1 Bind(void*, int32_t ordinal, uintptr_t context) {
  EXPECT_EQ(ordinal, 0);
  EXPECT_EQ(context, 100U);
  return Ok();
}
pih_status_v1 Create(void* context, uintptr_t owner, uintptr_t* handle) {
  auto& p = *static_cast<Provider*>(context);
  EXPECT_EQ(owner, 100U);
  ++p.created;
  if (p.fail_at == p.created)
    return ExecutionBindingStatus(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "resource_limit");
  *handle = p.zero ? 0 : 100 + p.created;
  return Ok();
}
pih_status_v1 Destroy(void* context, uintptr_t handle) {
  auto& p = *static_cast<Provider*>(context);
  p.retired.push_back(handle);
  return p.fail_retirement ? ExecutionBindingStatus(PIH_STATUS_INTERNAL_V1, "release_failed") : Ok();
}
pih_status_v1 Release(void* context, int32_t ordinal, uintptr_t handle) {
  EXPECT_EQ(ordinal, 0);
  return Destroy(context, handle);
}
pih_nvidia_cuda_resources_api_v1 Api(Provider& p) {
  return {sizeof(pih_nvidia_cuda_resources_api_v1), PIH_NVIDIA_CUDA_RESOURCES_ABI_VERSION_V1,
          &p, Retain, Bind, Create, Create, Destroy, Destroy, Release};
}
TEST(NativeQwenResourceBinding, RetiresResourcesBeforeTheirContext) {
  Provider p;
  auto api = Api(p);
  CapabilityResourceDriver driver(api);
  {
    auto resources = CudaRuntimeResources::Create(0, 0, 1, PIH_CUDA_CONTEXT_SCHED_YIELD_V1, driver);
    ASSERT_TRUE(resources.ok());
    EXPECT_EQ(resources->identity().context, 100U);
    EXPECT_EQ(resources->identity().stream, 101U);
    EXPECT_EQ(resources->identity().event, 105U);
    EXPECT_TRUE(p.retired.empty());
  }
  EXPECT_EQ(p.retired, (std::vector<std::uintptr_t>{108,107,106,105,104,103,102,101,100}));
}
TEST(NativeQwenResourceBinding, PartialCreationFailureRetiresOnlyCreatedResources) {
  Provider p;
  p.fail_at = 3;
  auto api = Api(p);
  CapabilityResourceDriver driver(api);
  auto resources = CudaRuntimeResources::Create(0, 0, 1, PIH_CUDA_CONTEXT_SCHED_YIELD_V1, driver);
  ASSERT_FALSE(resources.ok());
  EXPECT_EQ(resources.status().code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(p.retired, (std::vector<std::uintptr_t>{102,101,100}));
}
TEST(NativeQwenResourceBinding, NullHandleDoesNotBecomeAResource) {
  Provider p;
  p.zero = true;
  auto api = Api(p);
  CapabilityResourceDriver driver(api);
  auto resources = CudaRuntimeResources::Create(0, 0, 1, PIH_CUDA_CONTEXT_SCHED_YIELD_V1, driver);
  EXPECT_FALSE(resources.ok());
  EXPECT_EQ(p.retired, (std::vector<std::uintptr_t>{100}));
}
TEST(NativeQwenResourceBinding, RejectsIncompleteApiBeforeCallingProvider) {
  Provider p;
  auto api = Api(p);
  api.destroy_event = nullptr;
  EXPECT_FALSE(ValidResourceApi(api));
  CapabilityResourceDriver driver(api);
  EXPECT_FALSE(driver.retain_primary_context(0, PIH_CUDA_CONTEXT_SCHED_YIELD_V1).ok());
  EXPECT_EQ(p.retained, 0U);
}
TEST(NativeQwenResourceBinding, RejectsMalformedStatus) {
  Provider p;
  p.bad_status = true;
  auto api = Api(p);
  CapabilityResourceDriver driver(api);
  auto result = driver.retain_primary_context(0, PIH_CUDA_CONTEXT_SCHED_YIELD_V1);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInternal);
}
TEST(NativeQwenResourceBinding, FailedRetirementFailStopsInsteadOfUnloadingProvider) {
  Provider p;
  p.fail_retirement = true;
  auto api = Api(p);
  CapabilityResourceDriver driver(api);
  EXPECT_DEATH(driver.destroy_event(105), "");
}
}  // namespace
