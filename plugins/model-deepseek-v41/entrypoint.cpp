#include "supervisor_execution.h"
#include "pih/contracts/text_inference_v2.h"
#include "pih/plugin_sdk/abi.h"
#include "../common/text_json.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unicode/unistr.h>

namespace {
using namespace pih;
using namespace pih::deepseek_v41;
constexpr std::string_view model_id = "deepseek-ai/DeepSeek-V4.1-Flash";
struct State {
  const pih_host_api_v1* host{};
  uint32_t phase{};
  bool failed{};
  std::mutex mutex;
  std::unique_ptr<SupervisorDeployment> deployment;
  std::unique_ptr<SupervisorExecution> active;
} state;
pih_status_v1 Convert(const Status& s) {
  pih_status_v1 out{}; out.struct_size = sizeof(out); out.abi_version = PIH_STATUS_ABI_VERSION_V1;
  switch(s.code()) {
    case StatusCode::kOk: out.code = PIH_STATUS_OK_V1; break;
    case StatusCode::kInvalidArgument: out.code = PIH_STATUS_INVALID_ARGUMENT_V1; break;
    case StatusCode::kResourceExhausted: out.code = PIH_STATUS_RESOURCE_EXHAUSTED_V1; break;
    case StatusCode::kFailedPrecondition: out.code = PIH_STATUS_FAILED_PRECONDITION_V1; break;
    case StatusCode::kUnavailable: out.code = PIH_STATUS_UNAVAILABLE_V1; break;
    case StatusCode::kDeadlineExceeded: out.code = PIH_STATUS_DEADLINE_EXCEEDED_V1; break;
    default: out.code = PIH_STATUS_INTERNAL_V1; break;
  }
  auto message = s.message(); std::memcpy(out.message, message.data(), std::min(message.size(), sizeof(out.message)-1)); return out;
}
template<class F> pih_status_v1 Guard(F f) noexcept {
  try { return Convert(f()); }
  catch (const std::bad_alloc&) { return Convert(Status::ResourceExhausted("V4.1 allocation failed")); }
  catch (const std::exception&) { return Convert(Status::InvalidArgument("Invalid V4.1 text request or configuration")); }
  catch (...) { return Convert(Status::Internal("V4.1 plugin exception")); }
}
const JsonValue& Field(const JsonValue& object, const char* key) {
  const auto* value = object.at(key); if (!value) throw std::invalid_argument("missing field"); return *value;
}
JsonValue Parse(const char* bytes, uint64_t size, size_t maximum) {
  if (!bytes || !size || size > maximum) throw std::invalid_argument("invalid JSON extent");
  auto parsed = JsonValue::Parse({bytes, static_cast<size_t>(size)}, {maximum, 16, 8192, maximum});
  if (!parsed.ok() || !parsed->is_object()) throw std::invalid_argument("invalid JSON object"); return std::move(*parsed);
}
uint32_t Integer(const JsonValue& x, uint32_t minimum, uint32_t maximum) {
  if (!x.is_integer() || x.integer() < minimum || static_cast<uint64_t>(x.integer()) > maximum)
    throw std::invalid_argument("integer outside request bounds"); return static_cast<uint32_t>(x.integer());
}
float Number(const JsonValue& x) {
  if (!x.is_number() || !std::isfinite(x.number()) || !std::isfinite(static_cast<float>(x.number())))
    throw std::invalid_argument("sampling number invalid"); return static_cast<float>(x.number());
}
// Native V4.1 chat-mode text framing. Tools, images, thinking and assistant
// prefills are explicitly excluded from this initial text endpoint.
std::string RenderChat(const JsonValue& messages) {
  if (!messages.is_array() || messages.array().empty() || messages.array().size() > 256)
    throw std::invalid_argument("invalid messages");
  std::string out = "<｜begin▁of▁sentence｜>";
  const auto& array = messages.array();
  for (size_t i=0; i<array.size(); ++i) {
    const auto& m = array[i];
    if (!m.is_object() || m.object().size()!=2) throw std::invalid_argument("unsupported message fields");
    const auto& role = Field(m,"role").string(); const auto& text = Field(m,"content").string();
    plugin_text::RequireUtf8(text);
    if (role == "user") out += "<｜User｜>";
    else if (role == "system") out += "<｜System｜>";
    else if (role != "assistant" || i == 0 || Field(array[i-1],"role").string() == "assistant" ||
        (i == 1 && Field(array[0],"role").string() == "system"))
      throw std::invalid_argument("unsupported message role or ordering");
    out += text;
    if (role == "assistant") out += "<｜end▁of▁sentence｜>";
    if (role == "user" || (role == "system" && i>0)) {
      if (i+1 == array.size() || Field(array[i+1],"role").string() == "assistant")
        out += "<｜Assistant｜></think>";
    }
    if (out.size() > (1U<<20)) throw std::invalid_argument("rendered prompt too large");
  }
  const auto& last = Field(array.back(),"role").string();
  if (last == "assistant" || (array.size()==1 && last=="system")) throw std::invalid_argument("chat requires a generation header");
  return out;
}
struct Output {
  const pih_text_output_sink_v2& sink;
  bool streaming, chat;
  std::string identity, pending, text;
  size_t emitted_bytes{};
  static constexpr size_t kMaximumTextBytes = 8U << 20;
  bool Chunk(std::string_view part, const char* finish = nullptr, std::string_view usage = {}) {
    const auto json = identity + "\"object\":" + plugin_text::QuoteText(chat ? "chat.completion.chunk":"text_completion") +
        ",\"choices\":[{\"index\":0,\"finish_reason\":" + (finish ? plugin_text::QuoteText(finish):"null") +
        (chat ? ",\"delta\":{\"content\":" + plugin_text::QuoteText(part) + "}" : ",\"text\":" + plugin_text::QuoteText(part)) +
        "}]" + std::string(usage) + "}";
    return sink.write(sink.context,json.data(),json.size()) == 1;
  }
  Status Feed(std::string_view bytes, bool final = false) {
    if (pending.size() > kMaximumTextBytes - emitted_bytes ||
        bytes.size() > kMaximumTextBytes - emitted_bytes - pending.size())
      return Status::ResourceExhausted("V4.1 response exceeds byte limit");
    pending.append(bytes);
    size_t end = pending.size();
    if (!final && end) {
      size_t start = end-1; while(start && (static_cast<unsigned char>(pending[start])&0xc0)==0x80) --start;
      const auto lead = static_cast<unsigned char>(pending[start]);
      const size_t need = lead>=0xc2 && lead<=0xdf?2:lead>=0xe0&&lead<=0xef?3:lead>=0xf0&&lead<=0xf4?4:1;
      if(end-start<need) end=start;
    }
    std::string decoded;
    icu::UnicodeString::fromUTF8(icu::StringPiece(pending.data(),static_cast<int32_t>(end))).toUTF8String(decoded);
    pending.erase(0,end);
    if (decoded.size() > kMaximumTextBytes - emitted_bytes)
      return Status::ResourceExhausted("V4.1 response exceeds byte limit");
    emitted_bytes += decoded.size();
    if (decoded.empty()) return Status::Ok();
    if (streaming) return Chunk(decoded) ? Status::Ok() : Status::Unavailable("V4.1 client output cancelled");
    text += decoded; return Status::Ok();
  }
  static Result<size_t> Write(void* context, std::string_view bytes) {
    auto& x=*static_cast<Output*>(context);
    if (x.sink.cancelled(x.sink.context)) return Status::Unavailable("V4.1 client output cancelled");
    auto status=x.Feed(bytes); if (!status.ok()) return status;
    return bytes.size();
  }
  static bool Cancelled(void* context) {
    auto& x=*static_cast<Output*>(context); return x.sink.cancelled(x.sink.context)!=0;
  }
};
pih_status_v1 Load(void* context, const char* bytes, uint64_t size) noexcept {
  return Guard([&]() -> Status {
    if(context!=&state) return Status::InvalidArgument("invalid engine context");
    std::lock_guard lock(state.mutex);
    if(state.phase!=4 || state.deployment || state.active || state.failed) return Status::FailedPrecondition("engine not loadable");
    auto root=Parse(bytes,size,16384);
    if(root.object().size()!=2) throw std::invalid_argument("unexpected configuration fields");
    auto digest=Sha256Digest::ParseHex(Field(root,"supervisor_configuration_sha256").string());
    if(!digest.ok()) return digest.status();
    auto config=SupervisorConfig::Load(Field(root,"supervisor_configuration").string(),*digest);
    if(!config.ok()) return config.status();
    if(config->identity.epoch != state.host->activation_epoch)
      return Status::FailedPrecondition("Supervisor epoch differs from plugin activation");
    for(const auto& rank:config->placements)
      if(rank.sm_major!=10 || rank.sm_minor!=3) return Status::InvalidArgument("V4.1 service requires SM103 placements");
    if(config->sampling.logprobs || config->sampling.top_count) return Status::InvalidArgument("V4.1 HTTP logprobs not enabled");
    auto process=AdmitSupervisorProcess(); if(!process.ok()) return process;
    auto deployment=SupervisorDeployment::Create(std::move(*config)); if(!deployment.ok()) return deployment.status();
    state.deployment=std::move(*deployment); return Status::Ok();
  });
}
pih_status_v1 Complete(void* context,uint32_t chat,const char* bytes,uint64_t size,
    char* response,uint64_t capacity,uint64_t* written,const pih_text_output_sink_v2* sink) noexcept {
  if(written) *written=0;
  return Guard([&]() -> Status {
    if(context!=&state || chat>1 || !response || !written || capacity<1024 || !sink ||
        sink->struct_size!=sizeof(*sink) || sink->contract_version!=PIH_TEXT_INFERENCE_ABI_V2 ||
        !sink->context || !sink->start || !sink->write || !sink->cancelled)
      return Status::InvalidArgument("invalid text contract");
    std::unique_lock lock(state.mutex,std::try_to_lock);
    if(!lock.owns_lock()) return Status::Unavailable("V4.1 engine busy");
    if(state.phase!=4 || state.failed || !state.deployment || state.active) return Status::FailedPrecondition("V4.1 engine not ready");
    auto body=Parse(bytes,size,1U<<20);
    for(const auto& [key,unused]:body.object())
      if(key!="model" && key!=(chat?"messages":"prompt") && key!="stream" && key!="max_tokens" &&
         key!="temperature" && key!="top_p" && key!="top_k" && key!="seed" && key!="stop" && key!="n")
        throw std::invalid_argument("unsupported request field");
    if(Field(body,"model").string()!=model_id) throw std::invalid_argument("model identity mismatch");
    const auto& config=state.deployment->configuration();
    auto sampling=config.sampling; auto stopping=config.stopping;
    if(auto* x=body.at("max_tokens")) stopping.maximum=Integer(*x,1,config.stopping.maximum);
    if(auto* x=body.at("temperature")) sampling.temperature=Number(*x);
    if(auto* x=body.at("top_p")) sampling.top_p=Number(*x);
    if(auto* x=body.at("top_k")) sampling.top_k=Integer(*x,0,129280);
    if(auto* x=body.at("n")) (void)Integer(*x,1,1);
    if(auto* x=body.at("seed")) { if(!x->is_integer() || x->integer()<0) throw std::invalid_argument("seed invalid"); sampling.seed=x->integer(); }
    bool streaming=false;
    if(auto* x=body.at("stream")) { if(!x->is_boolean()) throw std::invalid_argument("stream invalid"); streaming=x->boolean(); }
    if(auto* x=body.at("stop")) {
      auto append=[&](const JsonValue& pattern) {
        if(!pattern.is_string() || stopping.pattern_count==stopping.patterns.size()) throw std::invalid_argument("stop limit");
        stopping.patterns[stopping.pattern_count++]=pattern.string();
      };
      if(x->is_array()) { for(const auto& p:x->array()) append(p); } else append(*x);
    }
    const auto prompt=chat?RenderChat(Field(body,"messages")):Field(body,"prompt").string();
    auto admitted=state.deployment->Admit(prompt,sampling,std::move(stopping));
    if(!admitted.ok()) return admitted.status();
    const auto generation=(*admitted)->ledger().identity().sequence_generation;
    const auto created=std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    Output output{*sink,streaming,chat!=0,
      "{\"id\":\"pih-v41-"+std::to_string(generation)+"\",\"created\":"+std::to_string(created)+
      ",\"model\":"+plugin_text::QuoteText(model_id)+","};
    if(sink->cancelled(sink->context) || sink->start(sink->context,streaming?1:0)!=1)
      return Status::Unavailable("V4.1 request cancelled before submission");
    state.active=std::make_unique<SupervisorExecution>(std::move(*admitted));
    auto generated=state.active->Run(config,&output,Output::Write,Output::Cancelled);
    if(!state.active->retired()) {
      state.failed=true; return Status::Internal("V4.1 retirement unresolved; provider unload forbidden");
    }
    state.active.reset();
    if(!generated.ok()) {
      if(generated.status().code()!=StatusCode::kUnavailable && generated.status().code()!=StatusCode::kDeadlineExceeded)
        state.failed=true;
      return generated.status();
    }
    if(sink->cancelled(sink->context)) return Status::Unavailable("V4.1 final output cancelled");
    auto final_output=output.Feed({},true); if(!final_output.ok()) return final_output;
    const auto* finish=generated->finish==TokenFinish::kLength?"length":"stop";
    const auto usage=",\"usage\":{\"prompt_tokens\":"+std::to_string(generated->prompt_tokens)+
      ",\"completion_tokens\":"+std::to_string(generated->completion_tokens)+
      ",\"total_tokens\":"+std::to_string(generated->prompt_tokens+generated->completion_tokens)+"}";
    if(streaming) return output.Chunk("",finish,usage)?Status::Ok():Status::Unavailable("V4.1 final chunk cancelled");
    const auto json=output.identity+"\"object\":"+plugin_text::QuoteText(chat?"chat.completion":"text_completion")+
      ",\"choices\":[{\"index\":0,\"finish_reason\":"+plugin_text::QuoteText(finish)+
      (chat?",\"message\":{\"role\":\"assistant\",\"content\":"+plugin_text::QuoteText(output.text)+"}":
            ",\"text\":"+plugin_text::QuoteText(output.text))+"}]"+usage+"}";
    if(json.size()>capacity) return Status::ResourceExhausted("V4.1 response buffer too small");
    std::memcpy(response,json.data(),json.size()); *written=json.size(); return Status::Ok();
  });
}
pih_status_v1 Close(void* context) noexcept {
  return Guard([&]() -> Status {
    if(context!=&state) return Status::InvalidArgument("invalid engine context");
    std::unique_lock lock(state.mutex,std::try_to_lock);
    if(!lock.owns_lock()) return Status::Unavailable("V4.1 request active");
    if(state.active && !state.active->retired()) return Status::Internal("V4.1 retirement unresolved");
    state.active.reset(); state.deployment.reset(); return Status::Ok();
  });
}
pih_text_inference_api_v2 inference{sizeof(inference),PIH_TEXT_INFERENCE_ABI_V2,&state,"deepseek-ai/DeepSeek-V4.1-Flash",Load,Complete,Close};
pih_status_v1 Advance(void* c,uint32_t expected) noexcept {
  if(c!=&state || state.phase!=expected) return Convert(Status::FailedPrecondition("plugin lifecycle order invalid"));
  ++state.phase; return Convert(Status::Ok());
}
pih_status_v1 Register(void* c) noexcept {
  if(c!=&state || !state.host || state.phase) return Convert(Status::FailedPrecondition("plugin registration invalid"));
  const pih_capability_v1 capability{sizeof(capability),PIH_CAPABILITY_ABI_VERSION_V1,
      "inference.text.v2","pih.inference.text.v2",&inference,PIH_CAPABILITY_THREADING_SERIALIZED_V1,
      PIH_CAPABILITY_SCOPE_ACTIVATION_V1,PIH_CAPABILITY_CARDINALITY_EXACTLY_ONE_V1};
  const auto status=state.host->register_capability(state.host->context,&capability);
  if(!pih_status_is_valid_v1(&status)) return Convert(Status::Internal("host registration status invalid"));
  return pih_status_is_ok_v1(&status)?Advance(c,0):status;
}
pih_status_v1 Configure(void* c) noexcept { return Advance(c,1); }
pih_status_v1 Start(void* c) noexcept { return Advance(c,2); }
pih_status_v1 Ready(void* c) noexcept { return Advance(c,3); }
pih_status_v1 Drain(void* c) noexcept {
  if(c!=&state || state.phase!=4) return Convert(Status::FailedPrecondition("drain order invalid"));
  auto status=Close(c); return pih_status_is_ok_v1(&status)?Advance(c,4):status;
}
pih_status_v1 Stop(void* c) noexcept {
  if(c==&state && state.phase==3) { auto status=Close(c); if(pih_status_is_ok_v1(&status)) state.phase=6; return status; }
  return Advance(c,5);
}
pih_status_v1 Dispose(void* c) noexcept {
  if(c!=&state || (state.phase!=1 && state.phase!=2 && state.phase!=6) || state.active || state.deployment)
    return Convert(Status::FailedPrecondition("dispose order or resource custody invalid"));
  state.host=nullptr; state.phase=7; return Convert(Status::Ok());
}
}
extern "C" PIH_PLUGIN_EXPORT pih_status_v1 pih_plugin_entry_v1(const pih_host_api_v1* host,pih_plugin_api_v1* plugin) noexcept {
  if(!pih_host_api_is_valid_v1(host) || !pih_plugin_api_accepts_v1(plugin) || state.host || state.phase)
    return Convert(Status::InvalidArgument("invalid V4.1 plugin entry"));
  state.host=host; plugin->plugin_id="pih.model.deepseek-v41"; plugin->plugin_version="1.0.0"; plugin->context=&state;
  plugin->lifecycle={sizeof(pih_plugin_lifecycle_v1),PIH_PLUGIN_LIFECYCLE_ABI_VERSION_V1,Register,Configure,Start,Ready,Drain,Stop,Dispose};
  return Convert(Status::Ok());
}
