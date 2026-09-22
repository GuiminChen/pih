#include "pih/model/nvidia_qwen3_int4_execution_driver.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(NvidiaQwenInt4ExecutionDriverTest,
     RejectsInvalidOwnerAndWorkspaceBeforeCudaContextAccess) {
  EXPECT_FALSE(NvidiaQwenInt4ExecutionDriver::Create(nullptr,0,0,-1).ok());
  EXPECT_FALSE(NvidiaQwenInt4ExecutionDriver::Create(nullptr,1,1,0).ok());
  std::byte workspace{};
  EXPECT_FALSE(NvidiaQwenInt4ExecutionDriver::Create(&workspace,1,2,0).ok());
}

}  // namespace
}  // namespace pih
