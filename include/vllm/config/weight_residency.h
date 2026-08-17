// vllm.cpp ORIGINAL — the host-RAM -> DISK weight-residency tier as a config
// surface. Row `ENG-RESIDENCY-CONFIG`, spec
// .agents/specs/weight-residency-config.md, issue #1110.
//
// WHY THIS IS NOT IN offload.h. `include/vllm/config/offload.h` is a
// transcription of upstream `vllm/config/offload.py` @ 555967922, cited
// line-for-line. Upstream offloads weights device -> host RAM and stops there:
// `OffloadBackend` is `Literal["auto", "uva", "prefetch"]` (offload.py:12),
// `offloader/uva.py:21` is a CPU-blanket UVA offloader and
// `offloader/prefetch.py:557-560` is cpu-only. NOTHING upstream reads a weight
// off a file at inference time, so there is nothing to mirror for this tier and
// adding a field to `OffloadConfig` would break a mirror to describe behaviour
// vLLM does not have. This file is therefore vllm.cpp-original by construction,
// and it names itself so in its JSON: the knobs live under the `vllm_cpp` key of
// the SAME document `--offload-config` already carries, which keeps one
// user-facing flag for one user-facing concept while leaving the mirrored fields
// byte-identical. `parse_offload_config_json` looks its three keys up by name
// and never enumerates the document, so a `vllm_cpp` sibling is invisible to it.
//
// THE SCHEMA:
//
//   {"vllm_cpp": {"mmap":          {"enabled": bool, "prefault": bool},
//                 "expert_stream": {"enabled": bool, "slots": int,
//                                   "slot_bytes": int}}}
//
// Every field is optional and an absent field means UNCHANGED, so an absent
// `vllm_cpp` key is byte-identical to the engine before this row existed.
//
// PRECEDENCE, and it is deliberate: **environment variable > JSON config >
// built-in default**. The environment keeps winning because several of these
// variables exist so that a benchmark arm is switchable without restarting the
// server with a new config, and a measurement in flight depends on that. The
// config is the DOCUMENTED surface; the environment is the OVERRIDE. An override
// that could not turn a configured knob back OFF would not be one, so `VT_X=0`
// beats a config `true` as well.
//
// WHAT IS DELIBERATELY ABSENT: `VT_MOE_EXPERT_STREAM_STATS_EVERY`. Every knob
// here changes what memory the process reserves or where a weight lives, which
// is a deployment decision and what a config document is for. That one changes
// only how often a diagnostic line reaches stderr: it moves no byte, reserves
// nothing and changes no number. It is the instrument, not the configuration,
// and it is the thing an operator flips while watching a run. Recorded so a
// later reader sees a decision rather than an omission.
//
// THE LATCH, which is the one real hazard here. Every knob below is read through
// a function-local static in the code that consumes it
// (`Qwen35ExpertStreamRequested`, `PrefaultBorrowedSpan`'s `enabled`, the
// `Qwen35ExpertStream` constructor's one-shot store), so the FIRST read decides
// the process's answer forever. A config installed after that first read would
// not merely be late, it would be SILENTLY ignored — the invisible-fallback
// shape this tree refuses everywhere else. So `Resolve*` records that a decision
// was latched and `SetWeightResidencyConfig` THROWS when a non-empty config
// arrives afterwards. Install at `LoadedEngine::FromModelDir`, in the same block
// that installs the weight offloader, which is already before any weight I/O.
#ifndef VLLM_CONFIG_WEIGHT_RESIDENCY_H_
#define VLLM_CONFIG_WEIGHT_RESIDENCY_H_

#include <cstdint>
#include <optional>
#include <string>

namespace vllm {

// The `vllm_cpp` extension object of an `--offload-config` document. Each
// optional is "the operator said so"; an empty optional is "unchanged".
struct WeightResidencyConfig {
  // `mmap.enabled` -> VT_GGUF_MMAP. Keep the GGUF file mmap-resident and borrow
  // weight bytes out of the mapping instead of copying them into owned buffers.
  // This is what makes a checkpoint larger than host RAM loadable at all.
  std::optional<bool> mmap;

  // `mmap.prefault` -> VT_GGUF_PREFAULT. Fault the borrowed pages in at load,
  // off the timed prefill. Set it FALSE for a model larger than memory: the
  // prefault would read the whole tower to populate a page cache that cannot
  // hold it.
  std::optional<bool> prefault;

  // `expert_stream.enabled` -> VT_MOE_EXPERT_STREAM. Serve routed expert slices
  // from a bounded host slot cache instead of faulting them out of the mapping
  // in router order.
  std::optional<bool> expert_stream;

  // `expert_stream.slots` -> VT_MOE_EXPERT_STREAM_SLOTS. How many expert slices
  // stay resident. Must be positive.
  std::optional<int64_t> expert_stream_slots;

  // `expert_stream.slot_bytes` -> VT_MOE_EXPERT_STREAM_SLOT_BYTES. Bytes per
  // slot, fixed for the process's life. Must be positive. This IS a user
  // surface, however internal it looks: the engine refuses an oversized slice
  // BY NAME and tells the operator to raise exactly this value, and a dynamic
  // (UD) quant whose `down_proj` outweighs its gate/up pair is precisely the
  // case where the computed default is wrong.
  std::optional<int64_t> expert_stream_slot_bytes;

  // True when the operator set nothing, i.e. the byte-identical default path.
  bool empty() const;

  // One line naming every field the operator set, for the install log. Returns
  // an empty string for an empty config.
  std::string Describe() const;

  // The environment variables that are SET and that shadow a field this config
  // sets, as `VT_NAME (field)` pairs; empty when nothing is shadowed.
  //
  // This is the whole mitigation for the env-wins precedence, and it is the one
  // way that precedence hurts: a document silently shadowed by a variable
  // somebody exported weeks ago. It deliberately reports PRESENCE rather than a
  // resolved value, because resolving here would latch every knob at install time
  // and move the latch ahead of the weight load — the exact ordering this header
  // is careful about. Presence is also the fact the operator is missing; the
  // value they can read off their own shell.
  std::string DescribeEnvOverrides() const;

  // Field-by-field equality. Used by the install to allow a repeated install of
  // an EQUAL config after a latch (see the header note) while still refusing a
  // different one.
  bool operator==(const WeightResidencyConfig& other) const;
  bool operator!=(const WeightResidencyConfig& other) const {
    return !(*this == other);
  }
};

// Parse the `vllm_cpp` extension out of the same JSON document
// `--offload-config` carries. An empty/blank document or an absent `vllm_cpp`
// key yields an empty (inert) config.
//
// Throws std::invalid_argument on a malformed document, a non-object document, a
// `vllm_cpp` that is not an object, an UNKNOWN key inside `vllm_cpp` or inside
// either sub-object, a field of the wrong type, or a non-positive `slots` /
// `slot_bytes`.
//
// The unknown-key refusal is the load-bearing half. `parse_offload_config_json`
// ignores a key it does not know, which is what lets this extension exist at
// all — and it is also what would make `{"vllm-cpp":{...}}` or
// `{"vllm_cpp":{"mmapp":{...}}}` silently do nothing. A typo that quietly
// disables the tier keeping a 370 GiB model in 119 GB is worse than a startup
// error, so it is an error. Same polarity as the mirrored parser refusing an
// unknown `offload_backend`.
WeightResidencyConfig parse_weight_residency_extension_json(
    const std::string& json_text);

// Install the process-global config. Call BEFORE any weight I/O.
//
// Throws std::logic_error when a NON-EMPTY config that differs from the
// installed one arrives after any Resolve* call has already latched a decision,
// because the latched knobs cannot be changed and honouring the call would be a
// lie. An empty config, or a re-install of an equal one, is always accepted: the
// first is the no-op every default load performs, and the second is what lets a
// process load two engines with the same configuration.
void SetWeightResidencyConfig(const WeightResidencyConfig& config);

// The installed config. Empty until something installs one.
const WeightResidencyConfig& ActiveWeightResidencyConfig();

// True once any Resolve* call has read the config, i.e. once a residency
// decision has been latched somewhere in the process.
bool WeightResidencyLatched();

// Drop the installed config AND the latch. Tests only: the latch is
// process-wide, so a suite with more than one case needs to be able to clear it.
void ResetWeightResidencyConfigForTesting();

// env var (if set) > `configured` (if set) > `builtin_default`.
//
// The env var's value is read with the tree's existing polarity: "", "0",
// "false" and "off" are FALSE, anything else is TRUE. Marks the config latched.
bool ResolveResidencyBool(const char* env_name, std::optional<bool> configured,
                          bool builtin_default);

// The integer form, same precedence. A non-positive or unparseable ENV value is
// ignored and falls through to the config, then to `builtin_default` — the
// tolerant parsing the existing knobs already document, kept byte-for-byte so
// this row changes no behaviour for an environment-only run. A non-positive
// CONFIG value cannot reach here: the parser refuses it at startup, where the
// operator can still see the message.
int64_t ResolveResidencyCount(const char* env_name,
                              std::optional<int64_t> configured,
                              int64_t builtin_default);

// ── The five knobs, one named resolver each ───────────────────────────────────
//
// Each one owns its environment NAME and its exact historical POLARITY, and each
// is the SOLE reader of its variable after this row. Two reasons this is five
// functions and not one call at each site with a string literal.
//
// First, the polarities are NOT the same and one of them is deliberately odd.
// `VT_GGUF_MMAP` and `VT_GGUF_PREFAULT` compare the whole value against "", "0",
// "false" and "off" (the tree's `EnvOn`,
// gguf_keep_quant.cpp:60-65). `VT_MOE_EXPERT_STREAM` examines only the FIRST
// CHARACTER — `v[0] != '0' && v[0] != '\0'` — so `VT_MOE_EXPERT_STREAM=false`
// reads as ON, which docs/ENVIRONMENT.md states explicitly. That is documented
// behaviour, not an accident, so it is preserved rather than tidied: a row whose
// subject is "where does this value come from" must not also change what an
// existing value means.
//
// Second, an environment-only run has to resolve BYTE-FOR-BYTE as it did before,
// and the cheapest way to hold that is for each variable to keep having exactly
// one reader with its own transcribed rule.

// `VT_GGUF_MMAP` > `vllm_cpp.mmap.enabled` > `builtin_default`. Not latching:
// `GgufLoadPolicy::FromEnv()` is called per load and always has been.
bool ResolveGgufMmap(bool builtin_default);

// How many borrowed spans have actually been prefaulted in this process, and the
// reset that lets a test A/B it.
//
// THIS EXISTS BECAUSE THE DECISION WAS OTHERWISE UNOBSERVABLE. Measured: with the
// prefault site mutated to never consult its resolver at all, the whole GGUF suite
// stayed green — `test_gguf_keep_quant` 39/39, `test_gguf_qwen36_loader` 6/6,
// `test_gguf_expert_span` 11/11 — because the only prefault case asserts BYTE
// TRANSPARENCY, which holds whether the prefault runs or not. So a config key
// wired to a site nothing watches would have landed looking tested. The counter is
// the cheapest instrument that distinguishes "prefaulted" from "skipped": the
// prefault reads pages and changes no byte, so there is nothing else to see.
uint64_t GgufPrefaultedSpanCount();
void ResetGgufPrefaultedSpanCountForTesting();

// The slot geometry the expert-stream store was actually BUILT with; both zero
// until something builds one. Same reason as the counter above, and the same
// measurement behind it: with the slot-count site mutated to a hardcoded 64, so
// that `VT_MOE_EXPERT_STREAM_SLOTS=8000` would have silently stopped working,
// `test_expert_stream_mixed_slot` and `test_gguf_expert_span` both stayed green.
// No test in this tree had ever exercised a non-default slot count — the gap
// predates this row, since the old inline `getenv` was equally unobserved — and a
// row that turns the knob into a config key is the wrong place to leave it.
struct ExpertStreamGeometry {
  int64_t slots = 0;
  int64_t slot_bytes = 0;
};
ExpertStreamGeometry BuiltExpertStreamGeometry();

// Called by the store's constructor with the values it resolved. Nothing but the
// accessor above reads it.
void NoteExpertStreamGeometry(int64_t slots, int64_t slot_bytes);

// `VT_GGUF_PREFAULT` > `vllm_cpp.mmap.prefault` > ON.
//
// NOT LATCHING, and that is a change: the site used to cache the answer in a
// function-local static. Dropping the cache costs one `getenv` per prefaulted
// span — set against reading megabytes of pages, which is what the function then
// does — and it buys two things. A config installed at load is honoured even if
// some earlier caller had already asked, and the existing A/B case
// (tests/vllm/test_gguf_keep_quant.cpp, "L7 load-time prefault is
// byte-transparent") stops being silently vacuous: under the static its second
// `setenv` could not affect anything, so both of its arms ran the same way.
bool ResolveGgufPrefault();

// Called by the prefault site after it has actually faulted a span in. Feeds
// GgufPrefaultedSpanCount above; nothing else reads it.
void NoteGgufPrefaultedSpan();

// `VT_MOE_EXPERT_STREAM` > `vllm_cpp.expert_stream.enabled` > OFF, with the
// FIRST-CHARACTER environment rule described above.
//
// LATCHING, deliberately: the answer decides whether an ~18 GiB slot store is
// built and whether the default-on grouped-MoE path is disabled, and those two
// must not be able to disagree with each other later in the same process. That
// latch is exactly why `SetWeightResidencyConfig` refuses a late install.
bool ResolveExpertStreamRequested();

// The pure decision behind `ResolveExpertStreamRequested`, with the environment
// value passed in (`nullptr` == unset). Exposed because the wrapper LATCHES and
// can therefore be exercised exactly once per process, which would leave the
// first-character rule and the env-beats-config direction untested — the two
// things most likely to be got wrong here. Takes no lock and touches no global.
bool ExpertStreamRequestedFrom(const char* env_value,
                               std::optional<bool> configured);

// `VT_MOE_EXPERT_STREAM_SLOTS` > `vllm_cpp.expert_stream.slots` > 64.
int64_t ResolveExpertStreamSlots();

// `VT_MOE_EXPERT_STREAM_SLOT_BYTES` > `vllm_cpp.expert_stream.slot_bytes` >
// `computed_default` (the largest gate/up/down slice the caller is about to
// take).
int64_t ResolveExpertStreamSlotBytes(int64_t computed_default);

}  // namespace vllm

#endif  // VLLM_CONFIG_WEIGHT_RESIDENCY_H_
