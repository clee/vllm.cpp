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
// `EnvOn` (src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:61-66) rather
// than re-invented: an environment-only run must resolve byte-for-byte the way it
// did before this file existed.
bool EnvTruth(const char* value) {
  return !(std::strcmp(value, "") == 0 || std::strcmp(value, "0") == 0 ||
           std::strcmp(value, "false") == 0 || std::strcmp(value, "off") == 0);
}

// The COUNT variable's value, if and only if it would beat a configured field.
// atoll, and anything non-positive ignored, is what the existing readers do
// (qwen3_5.cpp's store constructor), kept verbatim so an environment-only run
// resolves exactly as before — which means an empty, garbage or non-positive value
// is NOT an override at all: it falls through to the config.
//
// ONE rule with two callers: the resolver below, and `DescribeEnvOverrides`, which
// used to announce mere presence and therefore reported `SLOTS=banana` as an
// override the resolver then ignored (#1122 L7).
std::optional<int64_t> EnvCountThatWins(const char* env_name) {
  const char* v = std::getenv(env_name);
  if (v == nullptr || *v == '\0') return std::nullopt;
  const long long parsed = std::atoll(v);
  if (parsed <= 0) return std::nullopt;
  return static_cast<int64_t>(parsed);
}

struct Global {
  std::mutex mu;
  WeightResidencyConfig config;

  // TWO LATCHES, NOT ONE, and mmap and prefault are in NEITHER. A single flag set
  // by every resolver refused a legal second load: `GgufLoadPolicy::FromEnv()` runs
  // per load and this change removed the prefault site's static, so those two knobs
  // freeze nothing, yet the coarse flag let an ordinary first load block a second
  // engine's whole document (#1122 M1, measured through `vllm_engine_load`). What
  // genuinely freezes is the streaming ANSWER — a function-local static in
  // `ResolveExpertStreamRequested` — and the slot store's GEOMETRY, because the
  // store is built once per process and its reservation cannot be resized.
  //
  // ATOMIC, not a bool under `mu`, because `ResolveExpertStreamRequested` marks the
  // first one on EVERY call and that function sits on the per-expert-slice decode
  // path (`KqExpertSlice` -> `Qwen35ExpertStream::Get`). A process-wide mutex there
  // would serialise the lane this row exists to make configurable, and a row whose
  // whole claim is "changes no kernel, no allocation and no perf axis" must not
  // quietly add a lock to a hot loop.
  //
  // RELAXED is enough, and the reason is NOT a synchronises-with edge. A relaxed
  // store made outside `mu` is not ordered by a later acquire of `mu`, so the
  // earlier claim to that effect was wrong (#1122 L2). Two facts carry it instead.
  // No config state is published through these flags — they carry one bit, "a
  // decision was taken", and the config itself is read and written under `mu`. And
  // each flag is monotonic: set to true, never cleared except by the test reset, so
  // no ordering between two of them can be observed.
  //
  // WHAT IS NOT GUARANTEED, stated because it is a real window rather than a
  // theoretical one now that a second engine may install while a first decodes: an
  // install running concurrently with the very first expert slice of another engine
  // may not observe that latch and may accept the config. Release/acquire would not
  // close it either — no ordering makes a concurrent store visible — only holding
  // `mu` across the decode path would, and that is the lock this row must not add.
  // The consequence is the same as arriving a microsecond earlier, which is legal:
  // whoever got there first decides. Every production path installs before the model
  // exists, so the window needs a caller that loads a second engine at the instant
  // the first one starts streaming.
  std::atomic<bool> latched_expert_stream{false};
  std::atomic<bool> latched_geometry{false};

  bool Latched(ResidencyLatch knob) const {
    switch (knob) {
      case ResidencyLatch::kExpertStream:
        return latched_expert_stream.load(std::memory_order_relaxed);
      case ResidencyLatch::kExpertStreamGeometry:
        return latched_geometry.load(std::memory_order_relaxed);
    }
    return false;
  }

  bool AnyLatched() const {
    return Latched(ResidencyLatch::kExpertStream) ||
           Latched(ResidencyLatch::kExpertStreamGeometry);
  }
};

Global& State() {
  static Global g;
  return g;
}

// The key's path in the DOCUMENT, which is the only path an operator can act on.
// `parent_path` is empty at the top level; a hardcoded prefix here is what made
// `{"vllm_cpp": 5}` report `"vllm_cpp.vllm_cpp" must be a JSON object`.
std::string Dotted(const char* parent_path, const std::string& key) {
  if (parent_path == nullptr || *parent_path == '\0') return key;
  return std::string(parent_path) + "." + key;
}

const nlohmann::json* ExtObject(const nlohmann::json& parent, const char* key,
                                const char* parent_path) {
  auto it = parent.find(key);
  if (it == parent.end() || it->is_null()) return nullptr;
  if (!it->is_object()) {
    throw std::invalid_argument(std::string("offload config: \"") +
                                Dotted(parent_path, key) +
                                "\" must be a JSON object");
  }
  return &*it;
}

// Refuse a key nobody reads, at EVERY level of the document including the top.
// `parse_offload_config_json` ignores what it does not know, which is what lets
// this extension live in the same document — and what made `{"vllm-cpp":{...}}`
// parse to an empty config and start a server running this tier at its DEFAULTS —
// prefault ON and streaming OFF, the two the 370 GiB case exists to change — so the
// typo is discovered as an out-of-memory kill rather than as an error. The hyphen is the
// likeliest spelling of all, because every flag around it is hyphenated.
//
// Refusing is the MIRROR-FAITHFUL polarity rather than a local invention: upstream
// has no `--offload-config` flag at all at the pin (no such string anywhere in the
// tree), and vLLM builds its config dataclasses with the `@config` decorator, whose
// body sets `ConfigDict(extra="forbid")` under the comment "Extra fields are
// forbidden by default" (vllm/config/utils.py:68-69 @ 555967922). `OffloadConfig`
// carries that decorator (offload.py:80) and so does `KVTransferConfig`
// (kv_transfer.py:22-23), so `--kv-transfer-config`, the JSON-document flag next
// door, refuses an unknown key. The mirrored `parse_offload_config_json` stays untouched;
// this parser reads the same string at both production entry points, so enumerating
// here closes the document without editing the transcription.
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
      std::string msg = std::string("offload config: unknown key \"") +
                        Dotted(path, it.key()) + "\" (expected one of:";
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
  // A variable is reported when it would WIN, not when it is merely set. For a
  // boolean any value wins, so presence is exact. For a COUNT the tolerant parse the
  // existing readers use means an empty, garbage or non-positive value falls THROUGH
  // to the config: `VT_MOE_EXPERT_STREAM_SLOTS=banana` overrides nothing, and
  // announcing it as an override sent the operator after a line the resolver ignores
  // (#1122 L7). `EnvCountThatWins` is the same predicate the resolver uses, so the
  // two cannot drift.
  //
  // No Resolve* call and no latch is marked, which matters for the ORDERING this
  // header is careful about: the line prints at install time, ahead of the weight
  // load that takes the decisions.
  struct Pair {
    bool set_by_operator;
    bool env_wins;
    const char* env_name;
    const char* field;
  };
  const auto bool_set = [](const char* name) {
    return std::getenv(name) != nullptr;
  };
  const auto count_wins = [](const char* name) {
    return EnvCountThatWins(name).has_value();
  };
  const Pair pairs[] = {
      {mmap.has_value(), bool_set("VT_GGUF_MMAP"), "VT_GGUF_MMAP", "mmap"},
      {prefault.has_value(), bool_set("VT_GGUF_PREFAULT"), "VT_GGUF_PREFAULT",
       "prefault"},
      {expert_stream.has_value(), bool_set("VT_MOE_EXPERT_STREAM"),
       "VT_MOE_EXPERT_STREAM", "expert_stream"},
      {expert_stream_slots.has_value(),
       count_wins("VT_MOE_EXPERT_STREAM_SLOTS"), "VT_MOE_EXPERT_STREAM_SLOTS",
       "expert_stream_slots"},
      {expert_stream_slot_bytes.has_value(),
       count_wins("VT_MOE_EXPERT_STREAM_SLOT_BYTES"),
       "VT_MOE_EXPERT_STREAM_SLOT_BYTES", "expert_stream_slot_bytes"},
  };
  std::string out;
  for (const Pair& p : pairs) {
    if (!p.set_by_operator || !p.env_wins) continue;
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

  // THE DOCUMENT'S OWN KEYS FIRST. These four are every key `--offload-config`
  // accepts: the three the mirrored parser reads by name
  // (src/vllm/config/offload.cpp, `offload_backend`/`uva`/`prefetch`) and this
  // extension. A fifth spelling is a typo, and a typo here is the failure the
  // refusal exists for — `{"vllm-cpp":{...}}` used to disable the whole tier in
  // silence. Keeping the mirrored names in this list is what lets one parser close
  // the document while `parse_offload_config_json` stays a byte-faithful
  // transcription; the two run on the same string at both entry points.
  RejectUnknownKeys(doc, "",
                    {"offload_backend", "uva", "prefetch", "vllm_cpp"});

  const nlohmann::json* ext = ExtObject(doc, "vllm_cpp", "");
  if (ext == nullptr) return cfg;
  RejectUnknownKeys(*ext, "vllm_cpp", {"mmap", "expert_stream"});

  if (const nlohmann::json* m = ExtObject(*ext, "mmap", "vllm_cpp")) {
    RejectUnknownKeys(*m, "vllm_cpp.mmap", {"enabled", "prefault"});
    cfg.mmap = ExtBool(*m, "enabled", "vllm_cpp.mmap");
    cfg.prefault = ExtBool(*m, "prefault", "vllm_cpp.mmap");
  }
  if (const nlohmann::json* s = ExtObject(*ext, "expert_stream", "vllm_cpp")) {
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

namespace {

// Which fields of an incoming config a taken decision has already frozen. `mmap`
// and `prefault` are in no branch here, and that is the whole point of the
// function: they resolve per load, so a later engine may still set them.
std::string FrozenFields(const Global& g, const WeightResidencyConfig& in) {
  std::string out;
  const auto note = [&out](const char* field) {
    if (!out.empty()) out += ", ";
    out += field;
  };
  if (g.Latched(ResidencyLatch::kExpertStream) &&
      in.expert_stream != g.config.expert_stream) {
    note("expert_stream");
  }
  if (g.Latched(ResidencyLatch::kExpertStreamGeometry)) {
    if (in.expert_stream_slots != g.config.expert_stream_slots) {
      note("expert_stream_slots");
    }
    if (in.expert_stream_slot_bytes != g.config.expert_stream_slot_bytes) {
      note("expert_stream_slot_bytes");
    }
  }
  return out;
}

}  // namespace

void SetWeightResidencyConfig(const WeightResidencyConfig& config) {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  if (config.empty()) {
    // An EMPTY install after a latch is a NO-OP, not an overwrite. A second engine
    // in the same process carries no residency config of its own, and letting it
    // clear the first one's would change what a store built LATER reads: the expert
    // slot store is constructed lazily, on the first slice taken, which can be long
    // after a second engine loaded. So the stored value survives.
    if (g.AnyLatched()) return;
    g.config = config;
    return;
  }
  // REFUSE ONLY WHAT CANNOT BE HONOURED. An equal re-install changes no field and
  // passes; so does a document that touches only knobs nothing has frozen, which is
  // the ordinary two-engine load. A field a taken decision has fixed is refused,
  // because recording it would publish a configuration the engine is not running —
  // the invisible-fallback shape this tree refuses everywhere else.
  const std::string frozen = FrozenFields(g, config);
  if (!frozen.empty()) {
    throw std::logic_error(
        "weight residency config: " + frozen +
        " cannot be changed after this process already latched that decision (" +
        (g.config.empty() ? std::string("environment/default")
                          : g.config.Describe()) +
        "). The streaming answer is cached on first read and the slot store is "
        "built once, so accepting this would record a configuration the engine "
        "is not running. Install it before any weight I/O. `mmap` and "
        "`prefault` are NOT latched and can still be set by a later engine");
  }
  g.config = config;
}

// BY VALUE, and copied under the lock. Returning a reference and then releasing
// the mutex gave the caller an unsynchronised read behind a lock that looked like
// it covered one (#1122 L3). The copy is five optionals; the callers are per load,
// per prefaulted span (against megabytes of pages) and once per store.
WeightResidencyConfig ActiveWeightResidencyConfig() {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  return g.config;
}

bool WeightResidencyLatched() { return State().AnyLatched(); }

bool WeightResidencyLatched(ResidencyLatch knob) {
  return State().Latched(knob);
}

void ResetWeightResidencyConfigForTesting() {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  g.config = WeightResidencyConfig{};
  g.latched_expert_stream.store(false, std::memory_order_relaxed);
  g.latched_geometry.store(false, std::memory_order_relaxed);
}

// NEITHER of these two helpers marks a latch. A latch is a property of the SITE
// that caches the answer, not of reading a variable, and marking it here is what
// made an ordinary `GgufLoadPolicy::FromEnv()` refuse a second engine's document.
bool ResolveResidencyBool(const char* env_name, std::optional<bool> configured,
                          bool builtin_default) {
  const char* v = std::getenv(env_name);
  if (v != nullptr) return EnvTruth(v);
  if (configured.has_value()) return *configured;
  return builtin_default;
}

int64_t ResolveResidencyCount(const char* env_name,
                              std::optional<int64_t> configured,
                              int64_t builtin_default) {
  if (const std::optional<int64_t> winner = EnvCountThatWins(env_name)) {
    return *winner;
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
  // THE GEOMETRY LATCH, and this is the moment it happens: the store now holds a
  // `slots x slot_bytes` reservation for the life of the process and cannot be
  // resized. Reading the two sizes freezes nothing; building the store does. Marked
  // here rather than in the resolvers so a run that never streams never freezes
  // anything.
  g.latched_geometry.store(true, std::memory_order_relaxed);
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
  // fact SetWeightResidencyConfig has to refuse a late CHANGE against.
  //
  // THIS IS WHY THE FLAG IS ATOMIC. This function is reached once per expert slice
  // through `KqExpertSlice`, so taking the process-wide mutex here would put a lock
  // in the decode loop of the lane the row is about. A relaxed store costs nothing
  // measurable and carries the only guarantee the install needs.
  State().latched_expert_stream.store(true, std::memory_order_relaxed);
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
