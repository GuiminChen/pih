#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "pih/core/result.h"
#include "pih/core/status.h"

namespace pih {
namespace {

TEST(StatusTest, OkHasNoMessage) {
  const Status status = Status::Ok();
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kOk);
  EXPECT_TRUE(status.message().empty());
}

TEST(StatusTest, ErrorPreservesCodeAndMessage) {
  const Status status = Status::InvalidArgument("alignment must be nonzero");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "alignment must be nonzero");
}

TEST(StatusTest, ErrorMessageIsBounded) {
  const Status status = Status::Internal(std::string(5000, 'x'));
  EXPECT_EQ(status.message().size(), Status::kMaxMessageBytes);
}

TEST(StatusTest, DeadlineExceededIsTerminalAndBounded) {
  const auto status = Status::DeadlineExceeded("startup expired");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kDeadlineExceeded);
  EXPECT_EQ(status.message(), "startup expired");
}

TEST(ResultTest, HoldsMoveOnlyValue) {
  Result<std::unique_ptr<int>> result(std::make_unique<int>(7));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(**result, 7);
}

TEST(ResultTest, HoldsErrorWithoutValue) {
  Result<int> result(Status::ResourceExhausted("overflow"));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
  EXPECT_THROW(static_cast<void>(result.value()), std::logic_error);
}

TEST(ResultTest, RejectsOkStatusWithoutValue) {
  EXPECT_THROW(static_cast<void>(Result<int>(Status::Ok())), std::invalid_argument);
}

}  // namespace
}  // namespace pih
