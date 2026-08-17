// vllm.cpp ORIGINAL — see include/vllm/config/weight_residency.h for the schema,
// the precedence rule, why this is not part of the mirrored offload config, and
// why the latch throws. Row `ENG-RESIDENCY-CONFIG`, issue #1110.
#include "vllm/config/weight_residency.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {
namespace {

// The tree's existing environment polarity, transcribed from
// `EnvOn` (src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:60-65) rather
// than re-invented: an environment-only run must resolve byte-for-byte the way it
// did before this file existed.
bool EnvTruth(const char* value) {
  return !(std::strcmp(value, "") == 0 || std::strcmp(value, "0") == 0 ||
           std::strcmp(value, "false") == 0 || std::strcmp(value, "off") == 0);
}

struct Global {
  std::mutex mu;
  WeightResidencyConfig config;
  // ATOMIC, not a bool under `mu`, because `ResolveExpertStreamRequested` marks
  // this on EVERY call and that function sits on the per-expert-slice decode path
  // (`KqExpertSlice` -> `Qwen35ExpertStream::Get`). A process-wide mutex there
  // would serialise the lane this row exists to make configurable, and a row whose
  // whole claim is "changes no kernel, no allocation and no perf axis" must not
  // quietly add a lock to a hot loop. Relaxed ordering is enough: the flag only has
  // to be observed as true by a LATER install, and that install takes `mu` and
  // therefore synchronises with nothing weaker than it needs.
  std::atomic<bool> latched{false};
};

Global& State() {
  static Global g;
  return g;
}

const nlohmann::json* ExtObject(const nlohmann::json& parent, const char* key) {
  auto it = parent.find(key);
  if (it == parent.end() || it->is_null()) return nullptr;
  if (!it->is_object()) {
    throw std::invalid_argument(std::string("offload config: \"vllm_cpp.") +
                                key + "\" must be a JSON object");
  }
  return &*it;
}

// Refuse a key nobody reads. `parse_offload_config_json` ignores what it does not
// know, which is exactly what lets this extension live in the same document — and
// exactly what would make a misspelling silently disable the tier holding a
// 370 GiB model in 119 GB. So the extension enumerates its own keys.
void RejectUnknownKeys(const nlohmann::json& obj, const char* path,
                       const std::vector<std::string>& known) {
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    bool ok = false;
    for (const std::string& k : known) {
      if (it.key() == k) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      std::string msg = std::string("offload config: unknown key \"") + path +
                        "." + it.key() + "\" (expected one of:";
      for (const std::string& k : known) msg += " " + k;
      msg += ")";
      throw std::invalid_argument(msg);
    }
  }
}

std::optional<bool> ExtBool(const nlohmann::json& obj, const char* key,
                            const char* path) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return std::nullopt;
  if (!it->is_boolean()) {
    throw std::invalid_argument(std::string("offload config: \"") + path + "." +
                                key + "\" must be a boolean");
  }
  return it->get<bool>();
}

std::optional<int64_t> ExtPositiveInt(const nlohmann::json& obj, const char* key,
                                      const char* path) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return std::nullopt;
  if (!it->is_number_integer()) {
    throw std::invalid_argument(std::string("offload config: \"") + path + "." +
                                key + "\" must be an integer");
  }
  const int64_t v = it->get<int64_t>();
  if (v <= 0) {
    // The environment readers TOLERATE a zero or negative value and fall back to
    // the default, because they parse with atol and have no way to report. A
    // config document is parsed at startup where a message still reaches the
    // operator, so it is refused instead: a zero slot count that silently became
    // 64 is a cache the operator does not have.
    throw std::invalid_argument(std::string("offload config: \"") + path + "." +
                                key + "\" must be positive (got " +
                                std::to_string(v) + ")");
  }
  return v;
}

}  // namespace

bool WeightResidencyConfig::empty() const {
  return !mmap.has_value() && !prefault.has_value() &&
         !expert_stream.has_value() && !expert_stream_slots.has_value() &&
         !expert_stream_slot_bytes.has_value();
}

bool WeightResidencyConfig::operator==(
    const WeightResidencyConfig& other) const {
  return mmap == other.mmap && prefault == other.prefault &&
         expert_stream == other.expert_stream &&
         expert_stream_slots == other.expert_stream_slots &&
         expert_stream_slot_bytes == other.expert_stream_slot_bytes;
}

std::string WeightResidencyConfig::Describe() const {
  if (empty()) return "";
  std::string out;
  const auto add_bool = [&out](const char* name, std::optional<bool> v) {
    if (!v.has_value()) return;
    if (!out.empty()) out += " ";
    out += name;
    out += *v ? "=on" : "=off";
  };
  const auto add_int = [&out](const char* name, std::optional<int64_t> v) {
    if (!v.has_value()) return;
    if (!out.empty()) out += " ";
    out += name;
    out += "=";
    out += std::to_string(*v);
  };
  add_bool("mmap", mmap);
  add_bool("prefault", prefault);
  add_bool("expert_stream", expert_stream);
  add_int("expert_stream_slots", expert_stream_slots);
  add_int("expert_stream_slot_bytes", expert_stream_slot_bytes);
  return out;
}

std::string WeightResidencyConfig::DescribeEnvOverrides() const {
  // Presence only — no Resolve* call, so this cannot latch anything. Reading the
  // variable's VALUE here would also be misleading: the resolvers apply the
  // tree's tolerant integer parsing, under which a garbage count falls THROUGH to
  // the config and is therefore not an override at all.
  struct Pair {
    bool set_by_operator;
    const char* env_name;
    const char* field;
  };
  const Pair pairs[] = {
      {mmap.has_value(), "VT_GGUF_MMAP", "mmap"},
      {prefault.has_value(), "VT_GGUF_PREFAULT", "prefault"},
      {expert_stream.has_value(), "VT_MOE_EXPERT_STREAM", "expert_stream"},
      {expert_stream_slots.has_value(), "VT_MOE_EXPERT_STREAM_SLOTS",
       "expert_stream_slots"},
      {expert_stream_slot_bytes.has_value(), "VT_MOE_EXPERT_STREAM_SLOT_BYTES",
       "expert_stream_slot_bytes"},
  };
  std::string out;
  for (const Pair& p : pairs) {
    if (!p.set_by_operator) continue;
    if (std::getenv(p.env_name) == nullptr) continue;
    if (!out.empty()) out += ", ";
    out += p.env_name;
    out += " (";
    out += p.field;
    out += ")";
  }
  return out;
}

WeightResidencyConfig parse_weight_residency_extension_json(
    const std::string& json_text) {
  WeightResidencyConfig cfg;

  // An empty or blank document is the inert default, exactly as for the mirrored
  // parser this shares a flag with.
  const auto first = json_text.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return cfg;

  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("offload config: malformed JSON: ") +
                                e.what());
  }
  if (!doc.is_object()) {
    throw std::invalid_argument("offload config: must be a JSON object");
  }

  const nlohmann::json* ext = ExtObject(doc, "vllm_cpp");
  if (ext == nullptr) return cfg;
  RejectUnknownKeys(*ext, "vllm_cpp", {"mmap", "expert_stream"});

  if (const nlohmann::json* m = ExtObject(*ext, "mmap")) {
    RejectUnknownKeys(*m, "vllm_cpp.mmap", {"enabled", "prefault"});
    cfg.mmap = ExtBool(*m, "enabled", "vllm_cpp.mmap");
    cfg.prefault = ExtBool(*m, "prefault", "vllm_cpp.mmap");
  }
  if (const nlohmann::json* s = ExtObject(*ext, "expert_stream")) {
    RejectUnknownKeys(*s, "vllm_cpp.expert_stream",
                      {"enabled", "slots", "slot_bytes"});
    cfg.expert_stream = ExtBool(*s, "enabled", "vllm_cpp.expert_stream");
    cfg.expert_stream_slots =
        ExtPositiveInt(*s, "slots", "vllm_cpp.expert_stream");
    cfg.expert_stream_slot_bytes =
        ExtPositiveInt(*s, "slot_bytes", "vllm_cpp.expert_stream");
  }
  return cfg;
}

void SetWeightResidencyConfig(const WeightResidencyConfig& config) {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  if (!g.latched.load(std::memory_order_relaxed)) {
    g.config = config;
    return;
  }
  // An EMPTY install after a latch is a NO-OP, not an overwrite. A second engine
  // in the same process carries no residency config of its own, and letting it
  // clear the first one's would change what a store built LATER reads: the expert
  // slot store is constructed lazily, on the first slice taken, which can be long
  // after a second engine loaded. So the stored value survives.
  if (config.empty()) return;
  if (config != g.config) {
    // Refusing beats being ignored. The knobs this config feeds are read through
    // function-local statics, so by the time a decision is latched the process's
    // answer is fixed; accepting the call would record a configuration the engine
    // is not running.
    throw std::logic_error(
        "weight residency config installed AFTER a residency decision was "
        "already latched (" +
        (g.config.empty() ? std::string("environment/default")
                          : g.config.Describe()) +
        "); the knobs it feeds are read once per process, so this config "
        "would be silently ignored. Install it before any weight I/O");
  }
}

const WeightResidencyConfig& ActiveWeightResidencyConfig() {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  return g.config;
}

bool WeightResidencyLatched() {
  return State().latched.load(std::memory_order_relaxed);
}

void ResetWeightResidencyConfigForTesting() {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  g.config = WeightResidencyConfig{};
  g.latched.store(false, std::memory_order_relaxed);
}

bool ResolveResidencyBool(const char* env_name, std::optional<bool> configured,
                          bool builtin_default) {
  State().latched.store(true, std::memory_order_relaxed);
  const char* v = std::getenv(env_name);
  if (v != nullptr) return EnvTruth(v);
  if (configured.has_value()) return *configured;
  return builtin_default;
}

int64_t ResolveResidencyCount(const char* env_name,
                              std::optional<int64_t> configured,
                              int64_t builtin_default) {
  State().latched.store(true, std::memory_order_relaxed);
  const char* v = std::getenv(env_name);
  if (v != nullptr && *v != '\0') {
    // atol, and a non-positive result ignored, is what the existing readers do
    // (qwen3_5.cpp's store constructor). Kept verbatim so an environment-only run
    // resolves exactly as before.
    const long long parsed = std::atoll(v);
    if (parsed > 0) return static_cast<int64_t>(parsed);
  }
  if (configured.has_value()) return *configured;
  return builtin_default;
}

bool ResolveGgufMmap(bool builtin_default) {
  return ResolveResidencyBool("VT_GGUF_MMAP",
                              ActiveWeightResidencyConfig().mmap,
                              builtin_default);
}

namespace {
std::atomic<uint64_t>& PrefaultedSpans() {
  static std::atomic<uint64_t> n{0};
  return n;
}
}  // namespace

uint64_t GgufPrefaultedSpanCount() {
  return PrefaultedSpans().load(std::memory_order_relaxed);
}

void ResetGgufPrefaultedSpanCountForTesting() {
  PrefaultedSpans().store(0, std::memory_order_relaxed);
}

void NoteGgufPrefaultedSpan() {
  PrefaultedSpans().fetch_add(1, std::memory_order_relaxed);
}

namespace {
ExpertStreamGeometry& BuiltGeometry() {
  static ExpertStreamGeometry g;
  return g;
}
}  // namespace

ExpertStreamGeometry BuiltExpertStreamGeometry() {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  return BuiltGeometry();
}

void NoteExpertStreamGeometry(int64_t slots, int64_t slot_bytes) {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  BuiltGeometry().slots = slots;
  BuiltGeometry().slot_bytes = slot_bytes;
}

bool ResolveGgufPrefault() {
  return ResolveResidencyBool("VT_GGUF_PREFAULT",
                              ActiveWeightResidencyConfig().prefault,
                              /*builtin_default=*/true);
}

bool ExpertStreamRequestedFrom(const char* env_value,
                               std::optional<bool> configured) {
  // THE FIRST-CHARACTER RULE, transcribed rather than normalised. The site this
  // replaces read `v != nullptr && v[0] != '0' && v[0] != '\0'`, so
  // `VT_MOE_EXPERT_STREAM=false` is ON, and docs/ENVIRONMENT.md says so. It is
  // not routed through ResolveResidencyBool for exactly that reason: that helper
  // applies the tree's WHOLE-VALUE polarity and would silently flip this one.
  if (env_value != nullptr) return env_value[0] != '0' && env_value[0] != '\0';
  return configured.value_or(false);
}

bool ResolveExpertStreamRequested() {
  static const bool on = [] {
    return ExpertStreamRequestedFrom(
        std::getenv("VT_MOE_EXPERT_STREAM"),
        ActiveWeightResidencyConfig().expert_stream);
  }();
  // Mark the latch even on a cached call. Whether the value came from this call
  // or an earlier one, the process's answer is fixed from here, and that is the
  // fact SetWeightResidencyConfig has to refuse a late install against.
  //
  // THIS IS WHY THE FLAG IS ATOMIC. This function is reached once per expert slice
  // through `KqExpertSlice`, so taking the process-wide mutex here would put a lock
  // in the decode loop of the lane the row is about. A relaxed store costs nothing
  // measurable and carries the only guarantee the install needs.
  State().latched.store(true, std::memory_order_relaxed);
  return on;
}

int64_t ResolveExpertStreamSlots() {
  return ResolveResidencyCount("VT_MOE_EXPERT_STREAM_SLOTS",
                               ActiveWeightResidencyConfig().expert_stream_slots,
                               /*builtin_default=*/64);
}

int64_t ResolveExpertStreamSlotBytes(int64_t computed_default) {
  return ResolveResidencyCount(
      "VT_MOE_EXPERT_STREAM_SLOT_BYTES",
      ActiveWeightResidencyConfig().expert_stream_slot_bytes, computed_default);
}

}  // namespace vllm
