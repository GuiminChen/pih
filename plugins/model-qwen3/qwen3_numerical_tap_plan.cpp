#include "pih/model/qwen3_numerical_tap_plan.h"

#include <algorithm>
#include <array>
#include <string_view>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t index = 0; index < wire.size(); ++index) {
    wire[index] = static_cast<std::byte>(value >> (index * 8U));
  }
  return digest.update(wire);
}

struct TapShape final {
  DType dtype;
  std::uint64_t width;
  bool single_row;
  bool model_layer;
};

Result<TapShape> shape_for(QwenNumericalTapPoint point) {
  switch (point) {
    case QwenNumericalTapPoint::kLayerHidden:
    case QwenNumericalTapPoint::kKeyAfterNorm:
    case QwenNumericalTapPoint::kKeyAfterRope:
    case QwenNumericalTapPoint::kKvKey:
    case QwenNumericalTapPoint::kKvValue:
      return TapShape{DType::kBFloat16, 8 * 128, false, true};
    case QwenNumericalTapPoint::kQueryAfterNorm:
    case QwenNumericalTapPoint::kQueryAfterRope:
    case QwenNumericalTapPoint::kPrefillAttention:
      return TapShape{DType::kBFloat16, 16 * 128, false, true};
    case QwenNumericalTapPoint::kFinalNorm:
      return TapShape{DType::kBFloat16, 1024, false, false};
    case QwenNumericalTapPoint::kLogits:
      return TapShape{DType::kFloat32, 151936, true, false};
  }
  return Status::InvalidArgument("Qwen numerical tap point is unknown");
}

}  // namespace

Result<Sha256Digest> QwenNumericalTapPlan::semantic_digest() const {
  Sha256 digest;
  constexpr std::string_view domain = "pih.qwen_numerical_tap_plan.v1";
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = update_u64(digest, captures_.size());
  if (status.ok()) status = update_u64(digest, arena_bytes_);
  for (const auto& capture : captures_) {
    if (status.ok()) {
      status = update_u64(digest, static_cast<std::uint8_t>(capture.request.point));
    }
    if (status.ok()) status = update_u64(digest, capture.request.layer);
    if (status.ok()) status = update_u64(digest, capture.request.position);
    if (status.ok()) status = update_u64(digest, capture.request.rows);
    if (status.ok()) {
      status = update_u64(digest, static_cast<std::uint8_t>(capture.dtype));
    }
    if (status.ok()) status = update_u64(digest, capture.element_count);
    if (status.ok()) status = update_u64(digest, capture.offset_bytes);
    if (status.ok()) status = update_u64(digest, capture.size_bytes);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

Result<QwenNumericalTapPlan> QwenNumericalTapPlan::Create(
    std::span<const QwenNumericalTapRequest> requests,
    std::uint64_t arena_capacity_bytes) {
  if (requests.empty()) {
    return Status::InvalidArgument("Qwen numerical tap plan cannot be empty");
  }
  if (requests.size() > kMaximumCaptureCount) {
    return Status::ResourceExhausted("Qwen numerical tap count exceeds bound");
  }
  if (arena_capacity_bytes > kMaximumArenaBytes) {
    return Status::ResourceExhausted("Qwen numerical tap arena exceeds bound");
  }

  std::vector<QwenNumericalTapCapture> captures;
  captures.reserve(requests.size());
  std::uint64_t next = 0;
  for (const auto& request : requests) {
    if (std::find(requests.begin(),
                  requests.begin() + static_cast<std::ptrdiff_t>(captures.size()),
                  request) != requests.begin() +
                                  static_cast<std::ptrdiff_t>(captures.size())) {
      return Status::InvalidArgument("Qwen numerical tap request is duplicated");
    }
    auto shape = shape_for(request.point);
    if (!shape.ok()) return shape.status();
    if (request.rows == 0 || request.position >= 40960 ||
        request.rows > 40960 - request.position) {
      return Status::InvalidArgument("Qwen numerical tap token range is invalid");
    }
    if ((shape->model_layer && request.layer >= 28) ||
        (!shape->model_layer && request.layer != 28)) {
      return Status::InvalidArgument("Qwen numerical tap layer is invalid");
    }
    const bool kv = request.point == QwenNumericalTapPoint::kKvKey ||
                    request.point == QwenNumericalTapPoint::kKvValue;
    if ((shape->single_row || kv) && request.rows != 1) {
      return Status::InvalidArgument("Qwen numerical tap requires one row");
    }
    auto elements = checked_mul_u64(request.rows, shape->width);
    if (!elements.ok()) return elements.status();
    auto element_bytes = dtype_size(shape->dtype);
    if (!element_bytes.ok()) return element_bytes.status();
    auto bytes = checked_mul_u64(*elements, *element_bytes);
    if (!bytes.ok()) return bytes.status();
    auto offset = checked_align_up_u64(next, kAlignment);
    if (!offset.ok()) return offset.status();
    auto end = checked_add_u64(*offset, *bytes);
    if (!end.ok()) return end.status();
    if (*end > arena_capacity_bytes || *end > kMaximumArenaBytes) {
      return Status::ResourceExhausted(
          "Qwen numerical tap snapshots exceed preallocated arena");
    }
    captures.push_back({request, shape->dtype, *elements, *offset, *bytes});
    next = *end;
  }
  auto arena_bytes = checked_align_up_u64(next, kAlignment);
  if (!arena_bytes.ok()) return arena_bytes.status();
  if (*arena_bytes > arena_capacity_bytes) {
    return Status::ResourceExhausted(
        "Qwen numerical tap alignment exceeds preallocated arena");
  }
  return QwenNumericalTapPlan(std::move(captures), *arena_bytes);
}

Status validate_qwen_bf16_tap_coverage(const QwenNumericalTapPlan& plan) {
  const auto captures = plan.captures();
  auto has = [&](QwenNumericalTapPoint point, std::uint32_t layer,
                 std::uint32_t position) {
    return std::ranges::any_of(captures, [&](const auto& capture) {
      return capture.request.point == point && capture.request.layer == layer &&
             capture.request.position == position;
    });
  };
  for (const std::uint32_t layer : {0U, 13U, 27U}) {
    if (!has(QwenNumericalTapPoint::kLayerHidden, layer, 0)) {
      return Status::FailedPrecondition(
          "Qwen BF16 baseline lacks required hidden-state tap");
    }
  }
  for (std::uint32_t layer = 0; layer < 28; ++layer) {
    for (const auto point : {QwenNumericalTapPoint::kQueryAfterNorm,
                             QwenNumericalTapPoint::kKeyAfterNorm,
                             QwenNumericalTapPoint::kQueryAfterRope,
                             QwenNumericalTapPoint::kKeyAfterRope,
                             QwenNumericalTapPoint::kPrefillAttention}) {
      if (!has(point, layer, 0)) {
        return Status::FailedPrecondition(
            "Qwen BF16 baseline lacks required per-layer attention tap");
      }
    }
    for (const std::uint32_t position : {2U, 17U, 129U, 4097U}) {
      if (!has(QwenNumericalTapPoint::kKvKey, layer, position) ||
          !has(QwenNumericalTapPoint::kKvValue, layer, position)) {
        return Status::FailedPrecondition(
            "Qwen BF16 baseline lacks required KV decode tap");
      }
    }
  }
  if (!has(QwenNumericalTapPoint::kFinalNorm, 28, 0) ||
      !has(QwenNumericalTapPoint::kLogits, 28, 0)) {
    return Status::FailedPrecondition(
        "Qwen BF16 baseline lacks final norm or full logits tap");
  }
  return Status::Ok();
}

}  // namespace pih
