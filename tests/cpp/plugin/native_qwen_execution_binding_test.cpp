#include <stdexcept>

#include <gtest/gtest.h>

#include "../../../plugins/model-qwen3/execution_binding.h"

namespace {
using namespace pih::qwen_plugin;

struct Provider {
  unsigned calls{};
  unsigned mode{};
  pih_execution_capacity_request_v1 request{};
};

pih_status_v1 Compile(void* context,
                      const pih_execution_capacity_request_v1* request,
                      pih_execution_capacity_v1* capacity) {
  auto& provider = *static_cast<Provider*>(context);
  ++provider.calls;
  provider.request = *request;
  if (provider.mode == 1) return {};
  if (provider.mode == 2)
    return ExecutionBindingStatus(PIH_STATUS_RESOURCE_EXHAUSTED_V1, "capacity_busy");
  if (provider.mode == 3) throw std::runtime_error("provider failed");
  *capacity = {sizeof(*capacity), PIH_EXECUTION_DEFAULT_ABI_VERSION_V1,
               kMaximumStepTokens, kMaximumBatchSequences, kMaximumStepTokens};
  if (provider.mode == 4) --capacity->max_pipeline_tokens;
  if (provider.mode == 5) --capacity->maximum_sequences;
  if (provider.mode == 6) --capacity->expert_tokens;
  if (provider.mode == 7) --capacity->struct_size;
  if (provider.mode == 8) ++capacity->abi_version;
  return ExecutionBindingStatus(PIH_STATUS_OK_V1, "");
}

pih_execution_default_api_v1 Api(Provider& provider) {
  return {sizeof(pih_execution_default_api_v1),
          PIH_EXECUTION_DEFAULT_ABI_VERSION_V1, &provider, Compile};
}

TEST(NativeQwenExecutionBinding, CompilesExactSharedEngineCeilings) {
  Provider provider;
  auto api = Api(provider);
  const auto status = CompileExecutionCapacity(&api);
  ASSERT_TRUE(pih_status_is_ok_v1(&status));
  EXPECT_EQ(provider.calls, 1U);
  EXPECT_EQ(provider.request.struct_size, sizeof(provider.request));
  EXPECT_EQ(provider.request.abi_version, PIH_EXECUTION_DEFAULT_ABI_VERSION_V1);
  EXPECT_EQ(provider.request.maximum_prefill_chunk_tokens, kMaximumStepTokens);
  EXPECT_EQ(provider.request.maximum_decode_sequences, kMaximumBatchSequences);
  EXPECT_EQ(provider.request.maximum_verify_sequences, kMaximumBatchSequences);
  EXPECT_EQ(provider.request.speculative_tokens_per_sequence, 0U);
}

TEST(NativeQwenExecutionBinding, RejectsMalformedTableWithoutCallingProvider) {
  Provider provider;
  EXPECT_FALSE(ValidExecutionApi(nullptr));
  EXPECT_EQ(CompileExecutionCapacity(nullptr).code, PIH_STATUS_FAILED_PRECONDITION_V1);
  for (unsigned mode = 0; mode < 4; ++mode) {
    auto api = Api(provider);
    if (mode == 0) --api.struct_size;
    if (mode == 1) ++api.contract_version;
    if (mode == 2) api.context = nullptr;
    if (mode == 3) api.compile_capacity = nullptr;
    EXPECT_FALSE(ValidExecutionApi(&api));
    EXPECT_EQ(CompileExecutionCapacity(&api).code, PIH_STATUS_FAILED_PRECONDITION_V1);
  }
  EXPECT_EQ(provider.calls, 0U);
}

TEST(NativeQwenExecutionBinding, RejectsBadStatusAndContainsProviderException) {
  Provider provider;
  auto api = Api(provider);
  for (unsigned mode : {1U, 3U}) {
    provider.mode = mode;
    EXPECT_EQ(CompileExecutionCapacity(&api).code, PIH_STATUS_INTERNAL_V1);
  }
}

TEST(NativeQwenExecutionBinding, PreservesProviderFailure) {
  Provider provider;
  provider.mode = 2;
  auto api = Api(provider);
  const auto status = CompileExecutionCapacity(&api);
  EXPECT_EQ(status.code, PIH_STATUS_RESOURCE_EXHAUSTED_V1);
  EXPECT_STREQ(status.message, "capacity_busy");
}

TEST(NativeQwenExecutionBinding, RejectsMismatchedCapacityAndResultAbi) {
  Provider provider;
  auto api = Api(provider);
  for (unsigned mode = 4; mode <= 8; ++mode) {
    provider.mode = mode;
    EXPECT_EQ(CompileExecutionCapacity(&api).code, PIH_STATUS_FAILED_PRECONDITION_V1);
  }
}
}  // namespace
