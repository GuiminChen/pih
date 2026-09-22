#include "pih/backend/cuda/completion_frontier.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

CudaCompletionFrontier frontier() {
  return CudaCompletionFrontier::Create(
             {11, 0, 19, CudaCompletionPhase::kAttention, 23}, 29, 100, 200)
      .value();
}

TEST(CudaCompletionFrontierTest, AuthorizesOnlyCleanGenerationBoundSuccess) {
  auto value = frontier();
  EXPECT_EQ(value.observe(29, CudaEventQueryResult::kNotReady, true, 0, false)
                .code(),
            StatusCode::kUnavailable);
  EXPECT_FALSE(value.publication_authorized());
  EXPECT_TRUE(
      value.observe(29, CudaEventQueryResult::kSuccess, true, 0, false).ok());
  EXPECT_TRUE(value.completed());
  EXPECT_TRUE(value.publication_authorized());
  EXPECT_FALSE(
      value.observe(29, CudaEventQueryResult::kSuccess, true, 0, false).ok());
}

TEST(CudaCompletionFrontierTest, StaleEventPoisonsAndFirstErrorWins) {
  auto value = frontier();
  EXPECT_FALSE(
      value.observe(28, CudaEventQueryResult::kSuccess, true, 0, false).ok());
  EXPECT_TRUE(value.poisoned());
  EXPECT_TRUE(value.draining());
  EXPECT_EQ(value.first_failure(),
            CudaFrontierFailure::kEventGenerationMismatch);
  EXPECT_FALSE(value.observe(29, CudaEventQueryResult::kError, true, 7, false)
                   .ok());
  EXPECT_EQ(value.first_failure(),
            CudaFrontierFailure::kEventGenerationMismatch);
  EXPECT_EQ(value.first_device_error_code(), 0);
}

TEST(CudaCompletionFrontierTest, DeviceErrorAndPoisonBlockPublication) {
  auto device_error = frontier();
  EXPECT_FALSE(device_error
                   .observe(29, CudaEventQueryResult::kSuccess, true, 6, false)
                   .ok());
  EXPECT_EQ(device_error.first_failure(),
            CudaFrontierFailure::kDeviceInvariant);
  EXPECT_EQ(device_error.first_device_error_code(), 6);
  EXPECT_FALSE(device_error.publication_authorized());

  auto poisoned = frontier();
  EXPECT_FALSE(poisoned
                   .observe(29, CudaEventQueryResult::kSuccess, true, 0, true)
                   .ok());
  EXPECT_EQ(poisoned.first_failure(), CudaFrontierFailure::kEnginePoisoned);
}

TEST(CudaCompletionFrontierTest, DeadlineEqualityPoisonsWithoutCompletion) {
  auto value = frontier();
  EXPECT_EQ(value.expire(199).code(), StatusCode::kUnavailable);
  EXPECT_FALSE(value.poisoned());
  EXPECT_FALSE(value.expire(200).ok());
  EXPECT_EQ(value.first_failure(), CudaFrontierFailure::kDeadlineExpired);
  EXPECT_FALSE(value.completed());
  EXPECT_FALSE(value.publication_authorized());
}

TEST(CudaCompletionFrontierTest, RejectsMalformedIdentityAndClockRegression) {
  EXPECT_FALSE(CudaCompletionFrontier::Create(
                   {0, 0, 1, CudaCompletionPhase::kDecode, 1}, 1, 10, 20)
                   .ok());
  EXPECT_FALSE(CudaCompletionFrontier::Create(
                   {1, 0, 1, static_cast<CudaCompletionPhase>(255), 1}, 1, 10,
                   20)
                   .ok());
  auto value = frontier();
  EXPECT_FALSE(value.expire(99).ok());
  EXPECT_EQ(value.first_failure(), CudaFrontierFailure::kInvalidObservation);
}

}  // namespace
}  // namespace pih
