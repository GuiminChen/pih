#include "pih/backend/cuda/deepseek_expert_cuda_backend.h"
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace pih { namespace {
class FakeCudaOps final : public DeepSeekExpertCudaOperations {
 public:
  Status validate_host_staging(const DeepSeekExpertHostStaging&) override {
    ++validation_calls; return validation_status;
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return add("zero"); }
  Status copy_h2d_async(std::uintptr_t, const void*, std::size_t, std::uintptr_t) override { return add("h2d"); }
  Status copy_d2h_async(void*, std::uintptr_t, std::size_t, std::uintptr_t) override { return add("d2h"); }
  Status gather(DeepSeekRouteGatherLaunch v) override { gather_launch=v; return add("gather"); }
  Status quantize(DeepSeekFp8ActivationQuantLaunch v) override { quant_launches.push_back(v); return add("quant"); }
  Status gemm(DeepSeekFp4GemmLaunch v) override { gemm_launches.push_back(v); return add("gemm"); }
  Status swiglu(DeepSeekExpertSwiGluLaunch v) override { swiglu_launch=v; return add("swiglu"); }
  Status accumulate(DeepSeekExpertAccumulateLaunch v) override { accumulate_launch=v; return add("accumulate"); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return add("event"); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override { return query; }
  Status add(const char* name) { calls.emplace_back(name); return fail_at==calls.size() ? Status::Internal("injected failure") : Status::Ok(); }
  std::vector<std::string> calls;
  std::vector<DeepSeekFp8ActivationQuantLaunch> quant_launches;
  std::vector<DeepSeekFp4GemmLaunch> gemm_launches;
  DeepSeekRouteGatherLaunch gather_launch{};
  DeepSeekExpertSwiGluLaunch swiglu_launch{};
  DeepSeekExpertAccumulateLaunch accumulate_launch{};
  DeepSeekExpertAsyncStatus query=DeepSeekExpertAsyncStatus::kInProgress;
  std::size_t fail_at=0;
  int validation_calls=0;
  Status validation_status=Status::Ok();
};

DeepSeekExpertComputeSubmission submission(DeepSeekExpertComputeArena arena, DeepSeekExpertRoute* routes) {
  DeepSeekExpertComputeSubmission v{};
  v.context_identity=77;
  v.bundle.w1={{1001,10},{1002,10}};
  v.bundle.w2={{2001,10},{2002,10}};
  v.bundle.w3={{3001,10},{3002,10}};
  v.arena=arena; v.routes=routes; v.route_count=2; v.packed_token_count=4;
  v.source_hidden_bf16=4001; v.accumulator_f32=4002; v.stream=4003;
  return v;
}

DeepSeekExpertComputeArena arena() {
  return {{10,1},{10,1},{20,1},{30,1},{40,1},{50,1},{60,1},{70,1},{80,4}};
}

TEST(DeepSeekExpertCudaBackendTest, EnqueuesCanonicalExpertPipeline) {
  FakeCudaOps ops; float weights[4]{}; std::uint32_t indices[4]{}; std::uint32_t host_error=9;
  auto backend=DeepSeekExpertCudaBackend::Create(ops,{weights,indices,&host_error,4},88,77);
  ASSERT_TRUE(backend.ok());
  DeepSeekExpertRoute routes[2]{{3,0,0,0.25F},{3,2,1,0.75F}};
  ASSERT_TRUE(backend->submit(submission(arena(),routes)).ok());
  EXPECT_EQ(ops.calls,(std::vector<std::string>{"zero","h2d","h2d","gather","quant","gemm","gemm","swiglu","quant","gemm","accumulate","d2h","event"}));
  EXPECT_FLOAT_EQ(weights[0],0.25F); EXPECT_EQ(indices[1],2U); EXPECT_EQ(host_error,0U);
  ASSERT_EQ(ops.quant_launches.size(),2U); EXPECT_EQ(ops.quant_launches[0].logical_k,4096U); EXPECT_EQ(ops.quant_launches[1].logical_k,2048U);
  ASSERT_EQ(ops.gemm_launches.size(),3U); EXPECT_EQ(ops.gemm_launches[0].packed_weight,1001U); EXPECT_EQ(ops.gemm_launches[1].packed_weight,3001U); EXPECT_EQ(ops.gemm_launches[2].packed_weight,2001U);
  EXPECT_EQ(ops.accumulate_launch.token_indices,70U);
}

TEST(DeepSeekExpertCudaBackendTest, PollsEventAndSurfacesDeviceError) {
  FakeCudaOps ops; float weights[1]{}; std::uint32_t indices[1]{}; std::uint32_t host_error{};
  auto backend=DeepSeekExpertCudaBackend::Create(ops,{weights,indices,&host_error,1},88,77); ASSERT_TRUE(backend.ok());
  DeepSeekExpertRoute route{3,0,0,1.0F}; auto v=submission(arena(),&route); v.route_count=1;
  ASSERT_TRUE(backend->submit(v).ok()); EXPECT_EQ(*backend->poll(),DeepSeekExpertAsyncStatus::kInProgress);
  ops.query=DeepSeekExpertAsyncStatus::kSuccess; host_error=1; EXPECT_EQ(*backend->poll(),DeepSeekExpertAsyncStatus::kError);
}

TEST(DeepSeekExpertCudaBackendTest, RejectsContextMismatchAndPoisonsOnEnqueueFailure) {
  FakeCudaOps ops; float weights[1]{}; std::uint32_t indices[1]{}; std::uint32_t host_error{};
  auto backend=DeepSeekExpertCudaBackend::Create(ops,{weights,indices,&host_error,1},88,77); ASSERT_TRUE(backend.ok());
  DeepSeekExpertRoute route{3,0,0,1.0F}; auto v=submission(arena(),&route); v.route_count=1; v.context_identity=78;
  EXPECT_FALSE(backend->submit(v).ok()); v.context_identity=77; ops.fail_at=4;
  EXPECT_FALSE(backend->submit(v).ok()); EXPECT_FALSE(backend->submit(v).ok());
}

TEST(DeepSeekExpertCudaBackendTest, RejectsUnverifiedPinnedStagingAtCreation) {
  FakeCudaOps ops;
  ops.validation_status=Status::FailedPrecondition("not pinned");
  float weights[1]{}; std::uint32_t indices[1]{}; std::uint32_t host_error{};
  auto backend=DeepSeekExpertCudaBackend::Create(
      ops,{weights,indices,&host_error,1},88,77);
  EXPECT_FALSE(backend.ok());
  EXPECT_EQ(ops.validation_calls,1);
}
} }  // namespace pih
