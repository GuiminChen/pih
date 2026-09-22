#include "pih/model/qwen3_int4_gemm_plan.h"

#include <array>

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(QwenInt4GemmPlanTest, FreezesFiveOfficialCanonicalFamilies) {
  struct Expected{QwenInt4LinearShapeFamily family;std::uint64_t n,k,packed,scales;};
  constexpr std::array expected{
      Expected{QwenInt4LinearShapeFamily::kQProj,2048,1024,1'048'576,32'768},
      Expected{QwenInt4LinearShapeFamily::kKvProj,1024,1024,524'288,16'384},
      Expected{QwenInt4LinearShapeFamily::kOProj,1024,2048,1'048'576,32'768},
      Expected{QwenInt4LinearShapeFamily::kGateUpProj,3072,1024,1'572'864,49'152},
      Expected{QwenInt4LinearShapeFamily::kDownProj,1024,3072,1'572'864,49'152}};
  for(const auto& item:expected){auto plan=QwenInt4GemmPlan::Create(item.family,7);ASSERT_TRUE(plan.ok());EXPECT_EQ(plan->rows(),7);EXPECT_EQ(plan->output_features(),item.n);EXPECT_EQ(plan->input_features(),item.k);EXPECT_EQ(plan->packed_bytes(),item.packed);EXPECT_EQ(plan->scale_bytes(),item.scales);EXPECT_EQ(plan->input_bytes(),7*item.k*2);EXPECT_EQ(plan->output_bytes(),7*item.n*2);}
}

TEST(QwenInt4GemmPlanTest, FreezesTypedKernelAbiAndPointerSpans) {
  auto plan=QwenInt4GemmPlan::Create(QwenInt4LinearShapeFamily::kQProj,1).value();
  auto signature=plan.signature(std::string(64,'a'));
  ASSERT_TRUE(signature.ok());
  EXPECT_EQ(signature->logical_id(),"qwen.linear.w4a16.compatibility.v1");
  EXPECT_EQ(signature->parameter_count(),8U); EXPECT_EQ(signature->total_device_parameter_bytes(),64U);
  EXPECT_EQ(signature->parameter(0).role,"input"); EXPECT_EQ(signature->parameter(7).role,"k");
  auto contracts=plan.pointer_contracts(0,0); ASSERT_TRUE(contracts.ok()); ASSERT_EQ(contracts->size(),5U);
  EXPECT_EQ((*contracts)[0].required_bytes(),2048U); EXPECT_EQ((*contracts)[1].required_bytes(),1'048'576U); EXPECT_EQ((*contracts)[4].required_bytes(),4U);
  EXPECT_EQ(plan.kernel_symbol(),"pih_qwen_w4a16_gemm_compat_v1");
}

TEST(QwenInt4GemmPlanTest, RejectsUnknownVariantFamilyAndUnboundedRows) {
  EXPECT_FALSE(QwenInt4GemmPlan::Create(QwenInt4LinearShapeFamily::kQProj,0).ok());
  EXPECT_FALSE(QwenInt4GemmPlan::Create(QwenInt4LinearShapeFamily::kQProj,4097).ok());
  EXPECT_FALSE(QwenInt4GemmPlan::Create(static_cast<QwenInt4LinearShapeFamily>(255),1).ok());
  EXPECT_FALSE(QwenInt4GemmPlan::Create(QwenInt4LinearShapeFamily::kQProj,1,static_cast<QwenInt4KernelVariant>(2)).ok());
}

} }  // namespace pih
