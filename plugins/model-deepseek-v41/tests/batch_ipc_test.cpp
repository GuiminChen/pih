#include "../supervisor_batch_execution.h"
#include <cstdlib>
using namespace pih::deepseek_v41::batch_ipc;
static void Require(bool value) { if (!value) std::abort(); }
static std::string Wire(Header header, std::string_view body = {}) {
  std::string bytes(reinterpret_cast<const char*>(&header), sizeof(header)); bytes.append(body); return bytes;
}
int main() {
  const auto first = Wire({kMagic, 5, 11, 3, 0}, "one");
  const auto second = Wire({kMagic, 5, 12, 3, 0}, "two");
  const auto complete = Wire({kMagic, 6, 11, 0, 0});
  Decoder decoder;
  for (std::size_t index = 0; index < first.size(); ++index) {
    Require(decoder.Feed(std::string_view(first).substr(index,1)).ok());
    auto frame = decoder.Take(); Require(frame.ok());
    if (index + 1 < first.size()) Require(!*frame);
    else Require((*frame)->header.id == 11 && (*frame)->body == "one");
  }
  Require(decoder.Feed(second + complete).ok());
  auto frame = decoder.Take(); Require(frame.ok() && *frame && (*frame)->header.id == 12 && (*frame)->body == "two");
  frame = decoder.Take(); Require(frame.ok() && *frame && (*frame)->header.id == 11 && (*frame)->header.kind == 6);
  frame = decoder.Take(); Require(frame.ok() && !*frame);
  for (Header invalid : {Header{0,5,1,0,0}, Header{kMagic,0,1,0,0}, Header{kMagic,10,1,0,0},
                         Header{kMagic,5,1,0,7}, Header{kMagic,5,1,kMaximumFrame+1,0}}) {
    Decoder rejected; Require(rejected.Feed(Wire(invalid)).ok()); Require(!rejected.Take().ok());
  }
  for (unsigned kind : {8U,9U}) {
    Decoder admitted; Require(admitted.Feed(Wire({kMagic,kind,11,0,0})).ok());
    auto control = admitted.Take(); Require(control.ok() && *control && (*control)->header.kind == kind);
  }
  Decoder bounded;
  Require(!bounded.Feed(std::string(kMaximumFrame + sizeof(Header) + 16385, 'x')).ok());
  const std::string maximum(kMaximumFrame, 'x');
  Decoder full; Require(full.Feed(Wire({kMagic,1,55,kMaximumFrame,0}, maximum)).ok());
  frame = full.Take(); Require(frame.ok() && *frame && (*frame)->body.size() == kMaximumFrame);
}
