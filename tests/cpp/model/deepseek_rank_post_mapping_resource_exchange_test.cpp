#include "pih/model/deepseek_rank_post_mapping_resource_exchange.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekRankPostMappingResourceExchangeTest, PublishesFrozenAbiName) {
  EXPECT_EQ(kDeepSeekRankPostMappingResourceExchangeAbi,
            "pih_deepseek_rank_post_mapping_resource_exchange_v1");
}

}  // namespace
}  // namespace pih
