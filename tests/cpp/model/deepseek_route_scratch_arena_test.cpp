#include "pih/model/deepseek_route_scratch_arena.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekRouteScratchArenaTest, ReusesMaximumRouteBacking) {
  auto arena=DeepSeekRouteScratchArena::Create(4096);
  ASSERT_TRUE(arena.ok());
  auto maximum=arena->routes(4096); ASSERT_TRUE(maximum.ok());
  auto* backing=maximum->data();
  auto small=arena->routes(1); ASSERT_TRUE(small.ok());
  EXPECT_EQ(small->data(),backing);
  EXPECT_EQ(small->size(),6U);
  EXPECT_EQ(arena->maximum_token_count(),4096U);
  EXPECT_FALSE(arena->routes(4097).ok());
}

} }  // namespace pih
