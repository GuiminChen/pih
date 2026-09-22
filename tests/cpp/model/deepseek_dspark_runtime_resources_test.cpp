#include "pih/model/deepseek_dspark_runtime_resources.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class Pinned final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("DSpark runtime");
    return Allocation{data, bytes, alignment, 9, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override { _aligned_free(value.data); }
};
class Embed final : public DeepSeekDsparkEmbedOperations {
 public:
  Status validate_host_error(std::uint32_t* p) override { return p ? Status::Ok() : Status::InvalidArgument("null"); }
  Status zero_u32_async(std::uintptr_t,std::uintptr_t) override{return Status::Ok();}
  Status quant(DeepSeekFp8ActivationQuantLaunch) override{return Status::Ok();}
  Status gemm(DeepSeekFp8GemmLaunch) override{return Status::Ok();}
  Status rms(DeepSeekRmsNormLaunch) override{return Status::Ok();}
  Status draft_init(DeepSeekDsparkDraftInitLaunch) override{return Status::Ok();}
  Status copy_error_d2h_async(std::uint32_t*,std::uintptr_t,std::uintptr_t) override{return Status::Ok();}
};
class Head final : public DeepSeekDsparkHeadOperations {
 public:
  Status validate_host_error(std::uint32_t* p) override { return p ? Status::Ok() : Status::InvalidArgument("null"); }
  Status zero_u32_async(std::uintptr_t,std::uintptr_t) override{return Status::Ok();}
  Status hc_head(DeepSeekHcHeadLaunch) override{return Status::Ok();}
  Status rms_norm(DeepSeekRmsNormLaunch) override{return Status::Ok();}
  Status lm_head(DeepSeekLmHeadLaunch) override{return Status::Ok();}
  Status markov(DeepSeekDsparkMarkovLaunch) override{return Status::Ok();}
  Status argmax(DeepSeekArgmaxLaunch) override{return Status::Ok();}
  Status confidence(DeepSeekDsparkConfidenceLaunch) override{return Status::Ok();}
  Status copy_error_d2h_async(std::uint32_t*,std::uintptr_t,std::uintptr_t) override{return Status::Ok();}
};
class Prefill final : public DeepSeekDsparkPrefillStageOperations {
 public:
  Status validate_host_error(std::uint32_t* p) override {
    return p ? Status::Ok() : Status::InvalidArgument("null");
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override {
    return Status::Ok();
  }
  Status gemm(DeepSeekFp8GemmLaunch) override { return Status::Ok(); }
  Status rms(DeepSeekRmsNormLaunch) override { return Status::Ok(); }
  Status rotary(DeepSeekRotaryLaunch) override { return Status::Ok(); }
  Status kv_fp8_simulate(DeepSeekKvFp8SimulateLaunch) override {
    return Status::Ok();
  }
  Status recent_store(DeepSeekDsparkRecentStoreLaunch) override {
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  Status synchronize_stream(std::uintptr_t) override { return Status::Ok(); }
};
class Decode final : public DeepSeekDsparkDecodeStageOperations {
 public:
  Status validate_host_error(std::uint32_t* p) override {
    return p ? Status::Ok() : Status::InvalidArgument("null");
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status positions(DeepSeekDsparkPositionLaunch) override {
    return Status::Ok();
  }
  Status mhc_pre(DeepSeekMhcPreLaunch) override { return Status::Ok(); }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override {
    return Status::Ok();
  }
  Status gemm(DeepSeekFp8GemmLaunch) override { return Status::Ok(); }
  Status rms(DeepSeekRmsNormLaunch) override { return Status::Ok(); }
  Status head_rms(DeepSeekHeadRmsLaunch) override { return Status::Ok(); }
  Status rotary(DeepSeekRotaryLaunch) override { return Status::Ok(); }
  Status kv_simulate(DeepSeekKvFp8SimulateLaunch) override {
    return Status::Ok();
  }
  Status recent_store(DeepSeekDsparkRecentStoreLaunch) override {
    return Status::Ok();
  }
  Status attention(DeepSeekDsparkAttentionLaunch) override {
    return Status::Ok();
  }
  Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch) override {
    return Status::Ok();
  }
  Status mhc_post(DeepSeekMhcPostLaunch) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  Status synchronize_stream(std::uintptr_t) override { return Status::Ok(); }
};
class Moe final : public DeepSeekDsparkMoeStageOperations {
 public:
  Status validate_host_error(std::uint32_t* p) override {
    return p ? Status::Ok() : Status::InvalidArgument("null");
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status zero_bytes_async(std::uintptr_t, std::uint64_t,
                          std::uintptr_t) override { return Status::Ok(); }
  Status mhc_pre(DeepSeekMhcPreLaunch) override { return Status::Ok(); }
  Status router_gemm(DeepSeekRouterBf16GemmLaunch) override {
    return Status::Ok();
  }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override {
    return Status::Ok();
  }
  Status fp4_gemm(DeepSeekFp4GemmLaunch) override { return Status::Ok(); }
  Status shared_swiglu(DeepSeekSharedExpertSwiGluLaunch) override {
    return Status::Ok();
  }
  Status finalize(DeepSeekExpertFinalizeLaunch) override {
    return Status::Ok();
  }
  Status mhc_post(DeepSeekMhcPostLaunch) override { return Status::Ok(); }
  Status synchronize_stream(std::uintptr_t) override { return Status::Ok(); }
};

TEST(DeepSeekDsparkRuntimeResourcesTest, OwnsCoordinatorsOnlyOnDsparkRank) {
  Pinned pinned; Embed embed; Head head; Prefill prefill; Decode decode;
  Moe moe;
  DeepSeekStagePlan last{1,{20,42},false,true,true};
  auto owned=DeepSeekDsparkRuntimeResources::Allocate(
      last,&embed,&head,&prefill,&decode,&moe,pinned);
  ASSERT_TRUE(owned.ok()) << owned.status().message();
  EXPECT_NE(owned->embed_coordinator(),nullptr);
  EXPECT_NE(owned->head_executor(),nullptr);
  EXPECT_NE(owned->prefill_operation(DeepSeekDsparkStageId::kMtp0),nullptr);
  EXPECT_NE(owned->prefill_operation(DeepSeekDsparkStageId::kMtp1),nullptr);
  EXPECT_NE(owned->prefill_operation(DeepSeekDsparkStageId::kMtp2),nullptr);
  EXPECT_NE(owned->decode_attention_operation(
                DeepSeekDsparkStageId::kMtp0), nullptr);
  EXPECT_NE(owned->decode_attention_operation(
                DeepSeekDsparkStageId::kMtp1), nullptr);
  EXPECT_NE(owned->decode_attention_operation(
                DeepSeekDsparkStageId::kMtp2), nullptr);
  EXPECT_NE(owned->moe_operation(DeepSeekDsparkStageId::kMtp0), nullptr);
  EXPECT_NE(owned->moe_operation(DeepSeekDsparkStageId::kMtp1), nullptr);
  EXPECT_NE(owned->moe_operation(DeepSeekDsparkStageId::kMtp2), nullptr);
  EXPECT_EQ(owned->router_host_scores().size(), 5U * 256U);
  EXPECT_EQ(owned->router_host_bias().size(), 256U);
  DeepSeekStagePlan first{0,{0,19},true,false,false};
  auto empty=DeepSeekDsparkRuntimeResources::Allocate(
      first,nullptr,nullptr,nullptr,nullptr,nullptr,pinned);
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(empty->embed_coordinator(),nullptr);
  EXPECT_EQ(empty->head_executor(),nullptr);
  EXPECT_EQ(empty->prefill_operation(DeepSeekDsparkStageId::kMtp0),nullptr);
  EXPECT_EQ(empty->decode_attention_operation(
                DeepSeekDsparkStageId::kMtp0), nullptr);
  EXPECT_EQ(empty->moe_operation(DeepSeekDsparkStageId::kMtp0), nullptr);
  EXPECT_TRUE(empty->router_host_scores().empty());
}

} }  // namespace pih
