#include "../../../plugins/surface-openai-http/http_request_framing.h"
#include <gtest/gtest.h>
#include <string>

namespace pih::surface_openai_http {
TEST(NativeDiscoveryFraming, ExplicitOptInAndEmptyBody) {
  const std::string wire = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
  EXPECT_EQ(ParseRequest(wire).error, ParseError::kEndpointNotFound);
  const auto parsed = ParseRequest(wire, true);
  ASSERT_TRUE(parsed.ok());
  EXPECT_EQ(parsed.request.endpoint, Endpoint::kHealth);
  EXPECT_TRUE(parsed.request.body.empty());
}
TEST(NativeDiscoveryFraming, RoutesHaveExactMethods) {
  for (const auto path : {"/health", "/readyz", "/v1/models"}) {
    const std::string tail = std::string(path) + " HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n";
    EXPECT_TRUE(ParseRequest("GET " + tail, true).ok());
    EXPECT_EQ(ParseRequest("POST " + tail, true).error, ParseError::kMethodNotAllowed);
  }
}
TEST(NativeDiscoveryFraming, RejectsMalformedOrAmbiguousHeaders) {
  for (const auto headers : {"", "Host: a\r\nHost: b\r\n",
       "Host: a\r\nTransfer-Encoding: chunked\r\n", "Host: a\r\nContent-Length: 1\r\n",
       "Host: a\r\nContent-Length: 00\r\n", "Host: a\r\n X-Folded: a\r\n"}) {
    EXPECT_FALSE(ParseRequest(std::string("GET /readyz HTTP/1.1\r\n") + headers + "\r\n", true).ok());
  }
}
TEST(NativeDiscoveryFraming, RejectsTrailingPayloadAndPipelining) {
  const std::string wire = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
  EXPECT_EQ(ParseRequest(wire + "x", true).error, ParseError::kBodyLengthMismatch);
  EXPECT_EQ(ParseRequest(wire + wire, true).error, ParseError::kBodyLengthMismatch);
}
TEST(NativeDiscoveryFraming, InferenceStillRequiresPayloadHeaders) {
  const std::string wire = "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n";
  EXPECT_EQ(ParseRequest(wire, true).error, ParseError::kMissingContentLength);
}
TEST(NativeDiscoveryFraming, MethodErrorRetainsEndpointForAllowHeader) {
  const auto discovery = ParseRequest("POST /readyz HTTP/1.1\r\nHost: a\r\n\r\n", true);
  EXPECT_EQ(discovery.error, ParseError::kMethodNotAllowed);
  EXPECT_EQ(discovery.request.endpoint, Endpoint::kReady);
  const auto inference = ParseRequest("GET /v1/chat/completions HTTP/1.1\r\nHost: a\r\n\r\n", true);
  EXPECT_EQ(inference.error, ParseError::kMethodNotAllowed);
  EXPECT_EQ(inference.request.endpoint, Endpoint::kChatCompletions);
}
TEST(NativeDiscoveryFraming, MapsClientFailuresToSpecificHttpStatus) {
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kEndpointNotFound), 404U);
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kMethodNotAllowed), 405U);
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kMissingContentLength), 411U);
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kBodyTooLarge), 413U);
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kUnsupportedMediaType), 415U);
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kHeaderTooLarge), 431U);
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kHeadersTooLarge), 431U);
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kUnsupportedVersion), 505U);
  EXPECT_EQ(ParseErrorHttpStatus(ParseError::kDuplicateHeader), 400U);
}
TEST(NativeDiscoveryFraming, ExposesDeclaredSizeBeforePayloadArrival) {
  const std::string wire = "POST /v1/completions HTTP/1.1\r\nHost: a\r\n"
      "Content-Type: application/json\r\nContent-Length: 1048577\r\n\r\n";
  const auto parsed = ParseRequest(wire, true);
  EXPECT_EQ(parsed.error, ParseError::kBodyLengthMismatch);
  EXPECT_EQ(parsed.expected_wire_size, wire.size() + 1048577);
}
TEST(NativeDiscoveryFraming, RejectsExpectWithoutWaitingForBody) {
  for (const auto expectation : {"100-continue", "custom-extension", ""}) {
    const std::string wire = std::string("POST /v1/completions HTTP/1.1\r\nHost: a\r\n") +
        "Content-Type: application/json\r\nContent-Length: 50\r\neXpEcT: " +
        expectation + "\r\n\r\n";
    const auto result = ParseRequest(wire, true);
    EXPECT_EQ(result.error, ParseError::kExpectationUnsupported);
    EXPECT_EQ(result.expected_wire_size, 0U);
    EXPECT_EQ(ParseErrorHttpStatus(result.error), 417U);
  }
}
}  // namespace pih::surface_openai_http
