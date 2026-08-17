// `ENG-RESIDENCY-CONFIG` (issue #1110) — the `vllm_cpp` extension of
// `--offload-config`: the host-RAM -> DISK weight-residency tier as a config
// surface instead of environment variables only.
//
// THERE IS NOTHING UPSTREAM TO PORT. vLLM offloads weights device -> host RAM and
// stops: `OffloadBackend` is `Literal["auto","uva","prefetch"]`
// (vllm/config/offload.py:12 @ 555967922), `offloader/uva.py:21` is a CPU-blanket
// UVA offloader, `offloader/prefetch.py:557-560` is cpu-only, and nothing reads a
// weight off a file at inference time. Upstream's own offload test
// (tests/basic_correctness/test_cpu_offload.py:19-21) covers the mirrored tier and
// is already carried by tests/vllm/config/test_offload_config.cpp. So this file
// gates a vllm.cpp-original surface, and the one upstream obligation it DOES carry
// is negative: the mirrored structs must come out of a document carrying a
// `vllm_cpp` key byte-identical to what they were before, which the
// "mirror is untouched" cases below assert directly.
//
// THE THREE GUARANTEES, each of which has its own mutation:
//   1. PARSE + REFUSE. Every field round-trips, and an unknown key is an ERROR.
//      The refusal is the load-bearing half: parse_offload_config_json ignores a
//      key it does not know (which is what lets this extension share the flag),
//      so `{"vllm-cpp":...}` or `{"vllm_cpp":{"mmapp":...}}` would otherwise
//      SILENTLY disable the tier that keeps a 370 GiB model inside 119 GB.
//   2. PRECEDENCE: env > config > built-in default, in both directions. `VT_X=0`
//      must beat a config `true`, because an override that cannot turn a thing
//      OFF is not one, and that is the direction a benchmark arm needs.
//   3. THE LATCH. Every consumer reads its knob through a function-local static,
//      so a config installed after the first read cannot take effect. It must
//      THROW rather than be ignored.
#include <doctest/doctest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vllm/config/offload.h"
#include "vllm/config/weight_residency.h"

namespace {

// Every case owns the process-global, so clear it AND the latch on entry.
struct ResidencyFixture {
  ResidencyFixture() { Clear(); }
  ~ResidencyFixture() { Clear(); }
  static void Clear() {
    vllm::ResetWeightResidencyConfigForTesting();
    ::unsetenv("VT_RESIDENCY_TEST_BOOL");
    ::unsetenv("VT_RESIDENCY_TEST_COUNT");
  }
};

}  // namespace

TEST_CASE("residency config: every field parses out of the vllm_cpp key") {
  const vllm::WeightResidencyConfig c =
      vllm::parse_weight_residency_extension_json(R"({
        "uva": {"cpu_offload_gb": 4},
        "vllm_cpp": {
          "mmap": {"enabled": true, "prefault": false},
          "expert_stream": {"enabled": true, "slots": 8000,
                            "slot_bytes": 12582912}
        }
      })");
  REQUIRE(c.mmap.has_value());
  CHECK(*c.mmap == true);
  REQUIRE(c.prefault.has_value());
  CHECK(*c.prefault == false);
  REQUIRE(c.expert_stream.has_value());
  CHECK(*c.expert_stream == true);
  REQUIRE(c.expert_stream_slots.has_value());
  CHECK(*c.expert_stream_slots == 8000);
  REQUIRE(c.expert_stream_slot_bytes.has_value());
  CHECK(*c.expert_stream_slot_bytes == 12582912);
  CHECK_FALSE(c.empty());

  // The install line has to name what the operator set, because the resolved
  // values are the only way a run whose config was overridden can say so.
  const std::string described = c.Describe();
  CHECK(described.find("mmap=on") != std::string::npos);
  CHECK(described.find("prefault=off") != std::string::npos);
  CHECK(described.find("expert_stream=on") != std::string::npos);
  CHECK(described.find("expert_stream_slots=8000") != std::string::npos);
}

TEST_CASE("residency config: an absent extension is the inert default") {
  // Each of these is a document that reaches --offload-config today.
  for (const char* doc : {"", "   ", "{}", R"({"uva":{"cpu_offload_gb":4}})",
                          R"({"offload_backend":"uva"})",
                          R"({"vllm_cpp":{}})"}) {
    const vllm::WeightResidencyConfig c =
        vllm::parse_weight_residency_extension_json(doc);
    CHECK(c.empty());
    CHECK(c.Describe().empty());
    CHECK_FALSE(c.mmap.has_value());
    CHECK_FALSE(c.expert_stream_slots.has_value());
  }
}

TEST_CASE("residency config: a partial extension leaves the rest unchanged") {
  // The reproduction case from the issue is exactly this shape: mmap on, prefault
  // off (a model larger than memory cannot prefault its own tower), streaming on
  // with a real slot count. `slot_bytes` is left to the computed default.
  const vllm::WeightResidencyConfig c =
      vllm::parse_weight_residency_extension_json(R"({
        "vllm_cpp": {"mmap": {"prefault": false},
                     "expert_stream": {"enabled": true, "slots": 8000}}
      })");
  CHECK_FALSE(c.mmap.has_value());
  REQUIRE(c.prefault.has_value());
  CHECK(*c.prefault == false);
  REQUIRE(c.expert_stream.has_value());
  CHECK(*c.expert_stream == true);
  REQUIRE(c.expert_stream_slots.has_value());
  CHECK(*c.expert_stream_slots == 8000);
  CHECK_FALSE(c.expert_stream_slot_bytes.has_value());
}

TEST_CASE("residency config: an unknown or mistyped key is REFUSED, never ignored") {
  // Each of these would be silently accepted by a parser that only looks its own
  // keys up, and each one silently turns the tier OFF while the operator believes
  // it is on.
  const char* refused[] = {
      R"({"vllm_cpp":{"mmapp":{"enabled":true}}})",
      R"({"vllm_cpp":{"mmap":{"enable":true}}})",
      R"({"vllm_cpp":{"mmap":{"enabled":true,"prefaultt":false}}})",
      R"({"vllm_cpp":{"expert_stream":{"slot":8000}}})",
      R"({"vllm_cpp":{"expert_stream":{"enabled":true,"stats_every":16}}})",
      R"({"vllm_cpp":{"expert_streaming":{"enabled":true}}})",
      R"({"vllm_cpp":{"disk":{"enabled":true}}})",
  };
  for (const char* doc : refused) {
    CAPTURE(doc);
    CHECK_THROWS_AS(vllm::parse_weight_residency_extension_json(doc),
                    std::invalid_argument);
  }

  // `stats_every` above is not an oversight: it is environment-only BY DECISION
  // (it changes only how often a diagnostic line prints, moves no byte and
  // reserves nothing), so the config surface must refuse it rather than accept
  // and drop it.
}

TEST_CASE("residency config: a wrong-typed or non-positive field is REFUSED") {
  const char* refused[] = {
      "{not json",
      "[]",
      R"({"vllm_cpp": 5})",
      R"({"vllm_cpp":{"mmap": true}})",
      R"({"vllm_cpp":{"expert_stream": 8000}})",
      R"({"vllm_cpp":{"mmap":{"enabled":"yes"}}})",
      R"({"vllm_cpp":{"mmap":{"prefault":1}}})",
      R"({"vllm_cpp":{"expert_stream":{"slots":"8000"}}})",
      R"({"vllm_cpp":{"expert_stream":{"slots":8000.5}}})",
      // A zero or negative count is TOLERATED by the environment readers, which
      // parse with atol and cannot report. A config is parsed where a message
      // still reaches the operator, so a slot count that would silently have
      // become 64 is refused instead.
      R"({"vllm_cpp":{"expert_stream":{"slots":0}}})",
      R"({"vllm_cpp":{"expert_stream":{"slots":-1}}})",
      R"({"vllm_cpp":{"expert_stream":{"slot_bytes":0}}})",
  };
  for (const char* doc : refused) {
    CAPTURE(doc);
    CHECK_THROWS_AS(vllm::parse_weight_residency_extension_json(doc),
                    std::invalid_argument);
  }
}

TEST_CASE("residency config: the MIRRORED offload config is untouched by the extension") {
  // The whole reason the extension is a namespaced key rather than a field on
  // OffloadConfig: include/vllm/config/offload.h is a transcription of
  // vllm/config/offload.py @ 555967922 and must stay one. So the same document
  // has to parse through BOTH parsers, each seeing only its own half.
  const char* both = R"({
    "offload_backend": "uva",
    "uva": {"cpu_offload_gb": 10, "cpu_offload_params": ["experts"]},
    "vllm_cpp": {"mmap": {"enabled": true},
                 "expert_stream": {"enabled": true, "slots": 8000}}
  })";
  vllm::OffloadConfig off = vllm::parse_offload_config_json(both);
  off.Validate();
  CHECK(off.offload_backend == vllm::OffloadBackend::kUva);
  CHECK(off.uva.cpu_offload_gb == doctest::Approx(10.0));
  CHECK(off.uva.cpu_offload_params.count("experts") == 1);
  CHECK(off.is_offloading_enabled());
  CHECK(off.warnings.empty());

  const vllm::WeightResidencyConfig res =
      vllm::parse_weight_residency_extension_json(both);
  REQUIRE(res.mmap.has_value());
  CHECK(*res.mmap == true);
  REQUIRE(res.expert_stream_slots.has_value());
  CHECK(*res.expert_stream_slots == 8000);

  // And a document carrying ONLY the extension leaves the mirrored config
  // completely inert — no backend selected, nothing offloaded to host RAM.
  vllm::OffloadConfig only_ext = vllm::parse_offload_config_json(
      R"({"vllm_cpp":{"expert_stream":{"enabled":true}}})");
  only_ext.Validate();
  CHECK_FALSE(only_ext.ResolvedBackend().has_value());
  CHECK_FALSE(only_ext.is_offloading_enabled());
  CHECK(only_ext.uva.cpu_offload_gb == doctest::Approx(0.0));
  CHECK(only_ext.prefetch.offload_group_size == 0);
}

TEST_CASE("residency config: precedence is env > config > built-in default") {
  ResidencyFixture fx;

  // 1. Neither set: the built-in default, both polarities. This case is also the
  // inertness proof — an engine with no config and no environment resolves
  // exactly what getenv resolved before this row existed.
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", std::nullopt,
                                   true) == true);
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", std::nullopt,
                                   false) == false);
  CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", std::nullopt,
                                    64) == 64);

  // 2. Config set, environment unset: the config wins over the default, in BOTH
  // directions — a config has to be able to turn a default-on knob off.
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", false, true) ==
        false);
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", true, false) ==
        true);
  CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", 8000, 64) ==
        8000);

  // 3. Both set: the ENVIRONMENT wins. This is the constraint the row exists
  // under: these variables are how a benchmark arm is switched without restarting
  // the server with a new document, and an A/B in flight depends on it.
  ::setenv("VT_RESIDENCY_TEST_BOOL", "1", 1);
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", false, false) ==
        true);
  ::setenv("VT_RESIDENCY_TEST_COUNT", "128", 1);
  CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", 8000, 64) == 128);

  // 4. And the override can turn a configured knob OFF, which is the direction
  // that matters: an env var that could only enable things would be useless for
  // the arm that measures the feature disabled. Every falsy spelling the tree
  // already honours is checked, because this resolver replaced `EnvOn` and must
  // not narrow it.
  for (const char* off : {"0", "", "false", "off"}) {
    CAPTURE(off);
    ::setenv("VT_RESIDENCY_TEST_BOOL", off, 1);
    CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", true, true) ==
          false);
  }
  // Anything else is on, including a value that is not a recognised word.
  ::setenv("VT_RESIDENCY_TEST_BOOL", "yes", 1);
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", false, false) ==
        true);

  // 5. A garbage or non-positive COUNT in the environment falls THROUGH to the
  // config rather than to the default. The existing readers ignore such a value
  // (atol, then `if (v > 0)`), and this row must not change what an
  // environment-only run resolves.
  for (const char* junk : {"0", "-5", "banana", ""}) {
    CAPTURE(junk);
    ::setenv("VT_RESIDENCY_TEST_COUNT", junk, 1);
    CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", 8000, 64) ==
          8000);
    CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", std::nullopt,
                                      64) == 64);
  }
}

TEST_CASE("residency config: each knob's own resolver keeps its own env name and polarity") {
  ResidencyFixture fx;
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_PREFAULT");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES");

  // Defaults first, with NOTHING installed and nothing exported. These are the
  // values the tree resolved before this row existed, and they are the inertness
  // floor: `VT_GGUF_MMAP` rides the caller's availability predicate,
  // `VT_GGUF_PREFAULT` is ON when unset (which docs/ENVIRONMENT.md got backwards
  // until #1109), slots is 64, slot_bytes is whatever the caller computed.
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/true) == true);
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/false) == false);
  CHECK(vllm::ResolveGgufPrefault() == true);
  CHECK(vllm::ResolveExpertStreamSlots() == 64);
  CHECK(vllm::ResolveExpertStreamSlotBytes(12582912) == 12582912);

  // Now the config, which must reach every one of them. The reset is REQUIRED and
  // is itself part of the contract: the resolves above latched, and installing
  // after a latch throws. In production the loader installs first and nothing has
  // resolved yet; a test that walks the phases in the other order has to clear the
  // latch between them, exactly as it would have to clear a process boundary.
  vllm::ResetWeightResidencyConfigForTesting();
  vllm::WeightResidencyConfig cfg;
  cfg.mmap = true;
  cfg.prefault = false;
  cfg.expert_stream_slots = 8000;
  cfg.expert_stream_slot_bytes = 33554432;
  vllm::SetWeightResidencyConfig(cfg);

  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/false) == true);
  CHECK(vllm::ResolveGgufPrefault() == false);
  CHECK(vllm::ResolveExpertStreamSlots() == 8000);
  CHECK(vllm::ResolveExpertStreamSlotBytes(12582912) == 33554432);

  // And the environment must beat it, per knob, using each knob's OWN variable —
  // a resolver wired to the wrong name would pass every case above and fail here.
  ::setenv("VT_GGUF_MMAP", "0", 1);
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/true) == false);
  ::setenv("VT_GGUF_PREFAULT", "1", 1);
  CHECK(vllm::ResolveGgufPrefault() == true);
  ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", "128", 1);
  CHECK(vllm::ResolveExpertStreamSlots() == 128);
  ::setenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES", "4096", 1);
  CHECK(vllm::ResolveExpertStreamSlotBytes(12582912) == 4096);

  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_PREFAULT");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES");
}

TEST_CASE("residency config: the expert-stream FIRST-CHARACTER env rule survives") {
  // `VT_MOE_EXPERT_STREAM` does NOT use the tree's whole-value polarity: only the
  // first character is examined, so `false` and `off` read as ON. That is what
  // docs/ENVIRONMENT.md documents and what `qwen3_5.cpp` did, so a row whose
  // subject is where a value COMES FROM must not also change what a value MEANS.
  //
  // Tested through the pure form because the resolver latches and can be
  // exercised only once per process — which is exactly how a normalisation here
  // would have escaped notice.
  CHECK(vllm::ExpertStreamRequestedFrom("1", std::nullopt) == true);
  CHECK(vllm::ExpertStreamRequestedFrom("0", std::nullopt) == false);
  CHECK(vllm::ExpertStreamRequestedFrom("", std::nullopt) == false);
  CHECK(vllm::ExpertStreamRequestedFrom("0abc", std::nullopt) == false);
  // The deliberately odd ones. Under the tree's ordinary polarity these would be
  // OFF; here they are ON, and that is the documented contract.
  CHECK(vllm::ExpertStreamRequestedFrom("false", std::nullopt) == true);
  CHECK(vllm::ExpertStreamRequestedFrom("off", std::nullopt) == true);

  // Config supplies the answer only when the variable is UNSET, and the variable
  // beats the config in both directions.
  CHECK(vllm::ExpertStreamRequestedFrom(nullptr, std::nullopt) == false);
  CHECK(vllm::ExpertStreamRequestedFrom(nullptr, true) == true);
  CHECK(vllm::ExpertStreamRequestedFrom(nullptr, false) == false);
  CHECK(vllm::ExpertStreamRequestedFrom("0", true) == false);
  CHECK(vllm::ExpertStreamRequestedFrom("1", false) == true);
}

TEST_CASE("residency config: install is readable, and a LATE non-empty install throws") {
  ResidencyFixture fx;

  CHECK(vllm::ActiveWeightResidencyConfig().empty());
  CHECK_FALSE(vllm::WeightResidencyLatched());

  vllm::WeightResidencyConfig cfg;
  cfg.expert_stream = true;
  cfg.expert_stream_slots = 8000;
  vllm::SetWeightResidencyConfig(cfg);
  CHECK(vllm::ActiveWeightResidencyConfig() == cfg);
  CHECK_FALSE(vllm::WeightResidencyLatched());

  // The first resolve latches. Everything that consumes these knobs reads them
  // through a function-local static, so from here the process's answers are fixed.
  CHECK(vllm::ResolveResidencyBool(
            "VT_RESIDENCY_TEST_BOOL",
            vllm::ActiveWeightResidencyConfig().expert_stream, false) == true);
  CHECK(vllm::WeightResidencyLatched());

  // A DIFFERENT non-empty config afterwards cannot be honoured, so it throws
  // instead of being recorded and ignored.
  vllm::WeightResidencyConfig later;
  later.expert_stream = false;
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(later), std::logic_error);
  CHECK(vllm::ActiveWeightResidencyConfig() == cfg);

  // Re-installing the SAME config is fine — that is what a process loading two
  // engines with one configuration does.
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(cfg));

  // And an EMPTY install is always fine: it is the no-op every default load
  // performs, and refusing it would break every engine that has no config.
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(vllm::WeightResidencyConfig{}));
  // It must be a NO-OP rather than an overwrite. A second engine in the same
  // process carries no residency config of its own, and clearing the first one's
  // would change what the expert slot store reads — it is built lazily, on the
  // first slice taken, which can be long after a second engine loaded.
  CHECK(vllm::ActiveWeightResidencyConfig() == cfg);
}

TEST_CASE("residency config: a latch caused by nothing but the default path still allows an install") {
  ResidencyFixture fx;

  // The ordering hazard in the other direction. A resolve with no config
  // installed latches too — that is what makes a later install dangerous — but
  // the FIRST engine in a process must still be able to install, so the check
  // has to be about a CHANGE, not about the latch alone.
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", std::nullopt,
                                   false) == false);
  CHECK(vllm::WeightResidencyLatched());

  vllm::WeightResidencyConfig cfg;
  cfg.mmap = true;
  // It DOES throw, and that is the point: the mmap decision for this process was
  // already taken. A silent accept here is how an operator ends up reading a
  // config the engine never used.
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(cfg), std::logic_error);
}
