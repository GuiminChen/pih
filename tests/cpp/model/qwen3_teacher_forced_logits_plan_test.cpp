#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_logits_plan.h"

namespace pih {
namespace {

TensorView tf_view(std::uintptr_t address, DType dtype,
                   std::span<const std::int64_t> shape) {
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape, {},
      Device::Create(DeviceType::kCuda, 0).value(), 7).value();
}

ResolvedKernelFunction gather_function() {
  const std::string cubin(64, 'd');
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kSampleHidden, cubin).value();
  return {std::string(qwen_bf16_packed_kernel_symbol(
              QwenBf16PackedPrimitive::kSampleHidden).value()),
          std::string(manifest.logical_id()), cubin,
          std::string(manifest.parameter_abi_sha256()), 81};
}

class Kernel final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override {
    calls.push_back("gather"); return result;
  }
  std::vector<std::string> calls;
  Status result = Status::Ok();
};

class Linear final : public QwenBf16LinearExecutionDriver {
 public:
  Status execute(const QwenBf16LinearBinding& binding,
                 DriverStreamHandle) override {
    rows = binding.input().dim(0); calls.push_back("lm_head"); return result;
  }
  std::vector<std::string> calls;
  std::int64_t rows = 0;
  Status result = Status::Ok();
};

class Int4Linear final : public QwenInt4LmHeadExecutionDriver {
 public:
  Status execute(const QwenInt4LmHeadBinding& binding,
                 DriverStreamHandle) override {
    rows = binding.input().dim(0); ++calls; return Status::Ok();
  }
  int calls = 0;
  std::int64_t rows = 0;
};

QwenTeacherForcedLogitsPlan make_plan() {
  const std::array<std::int64_t, 2> source_shape{10, 1024};
  const std::array<std::int64_t, 1> row_shape{4};
  const std::array<std::int64_t, 2> gathered_shape{4, 1024};
  const std::array<std::int64_t, 2> weight_shape{151936, 1024};
  const std::array<std::int64_t, 2> logits_shape{4, 151936};
  const std::array<std::int64_t, 1> error_shape{4};
  return QwenTeacherForcedLogitsPlan::Create(
      gather_function(), tf_view(0x10000, DType::kBFloat16, source_shape),
      tf_view(0x20000, DType::kUInt32, row_shape),
      tf_view(0x30000, DType::kBFloat16, gathered_shape),
      tf_view(0x40000, DType::kBFloat16, weight_shape),
      tf_view(0x14000000, DType::kFloat32, logits_shape),
      tf_view(0x15000000, DType::kUInt8, error_shape), 0).value();
}

TEST(QwenTeacherForcedLogitsPlanTest, SubmitsGatherBeforeMultiRowLmHead) {
  auto plan = make_plan(); Kernel kernel; Linear linear;
  ASSERT_TRUE(plan.submit(kernel, linear, 91).ok());
  EXPECT_EQ(kernel.calls, (std::vector<std::string>{"gather"}));
  EXPECT_EQ(linear.calls, (std::vector<std::string>{"lm_head"}));
  EXPECT_EQ(linear.rows, 4);
  EXPECT_FALSE(plan.submit(kernel, linear, 91).ok());
}

TEST(QwenTeacherForcedLogitsPlanTest, StopsBeforeLmHeadWhenGatherFails) {
  auto plan = make_plan(); Kernel kernel; Linear linear;
  kernel.result = Status::Internal("injected");
  EXPECT_FALSE(plan.submit(kernel, linear, 91).ok());
  EXPECT_TRUE(linear.calls.empty());
}

TEST(QwenTeacherForcedLogitsPlanTest, ReusesIdenticalRowsForInt4Bf16Head) {
  auto plan = make_plan(); Kernel kernel; Int4Linear linear;
  ASSERT_TRUE(plan.submit_int4(kernel, linear, 92).ok());
  EXPECT_EQ(kernel.calls, (std::vector<std::string>{"gather"}));
  EXPECT_EQ(linear.calls, 1);
  EXPECT_EQ(linear.rows, 4);
}

}  // namespace
}  // namespace pih
