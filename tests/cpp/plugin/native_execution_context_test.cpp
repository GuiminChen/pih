#include <gtest/gtest.h>
#include <array>

// Test the real capability/lifecycle callbacks in a dedicated executable.
#include "../../../plugins/execution-default/entrypoint.cpp"

TEST(NativeExecutionContext, ForeignStateCannotInvokeLifecycle) {
  State foreign;
  foreign.phase = 4;
  for (auto operation : std::array{&Register, &Configure, &Start, &Ready,
                                   &Drain, &Stop, &Dispose}) {
    const auto status = operation(&foreign);
    EXPECT_TRUE(pih_status_is_valid_v1(&status));
    EXPECT_EQ(status.code, PIH_STATUS_INVALID_ARGUMENT_V1);
    EXPECT_EQ(foreign.phase, 4U);
  }
}

TEST(NativeExecutionContext, NullStateCannotInvokeLifecycle) {
  for (auto operation : std::array{&Register, &Configure, &Start, &Ready,
                                   &Drain, &Stop, &Dispose}) {
    EXPECT_EQ(operation(nullptr).code, PIH_STATUS_INVALID_ARGUMENT_V1);
  }
}

TEST(NativeExecutionContext, ForeignReadyStateCannotCompileCapacity) {
  State foreign;
  foreign.phase = 4;
  const pih_execution_capacity_request_v1 request{
      sizeof(request), PIH_EXECUTION_DEFAULT_ABI_VERSION_V1, 16, 1, 1, 0};
  pih_execution_capacity_v1 capacity{
      sizeof(capacity), PIH_EXECUTION_DEFAULT_ABI_VERSION_V1, 99, 99, 99};
  const auto status = CompileCapacity(&foreign, &request, &capacity);
  EXPECT_EQ(status.code, PIH_STATUS_INVALID_ARGUMENT_V1);
  EXPECT_FALSE(foreign.capacity_compiled);
  EXPECT_EQ(capacity.max_pipeline_tokens, 0U);
  EXPECT_EQ(capacity.maximum_sequences, 0U);
  EXPECT_EQ(capacity.expert_tokens, 0U);
}
