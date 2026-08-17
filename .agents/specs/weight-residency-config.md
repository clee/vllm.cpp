# The disk-residency tier as a config surface (`ENG-RESIDENCY-CONFIG`)

Make the host-RAM→disk weight-residency tier reachable from the server's JSON
config surface instead of only from `VT_*` environment variables. Issue
[#1110](https://github.com/mudler/vllm.cpp/issues/1110). Also closes
[#1109](https://github.com/mudler/vllm.cpp/issues/1109), the documented default
of `VT_GGUF_PREFAULT` being the opposite of the code's, because this row writes
that default into a resolver and a config key and cannot ship a document that
contradicts its own new code.

**Verdict up front.** vllm.cpp offloads weights at two tiers and only one is
configurable. The device→host tier takes
`--offload-config '{"uva":{"cpu_offload_gb":N,"cpu_offload_params":["experts"]}}'`
and reaches the loader through `server_main.cpp` → `EngineParams` →
`LoadedEngine::FromModelDir`. The host→disk tier — the one that makes
`Qwen3.8-2.4T-A95B UD-Q1_0` (370 GiB) serve on a 119 GB box, because ~330 GiB of
experts stay borrowed in the mapping — is `VT_GGUF_MMAP`, `VT_GGUF_PREFAULT`,
`VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS`,
`VT_MOE_EXPERT_STREAM_SLOT_BYTES`, and nothing else. A user reaches for
`--offload-config` because its first tier is already expert-aware
(`cpu_offload_params: ["experts"]` targets expert weights by name segment), gets
the GPU tier, and finds nothing for the tier the big-model case needs.

Three findings shape the change:

- **The mirror is not negotiable and does not need to be broken.**
  `include/vllm/config/offload.h` is a transcription of upstream
  `vllm/config/offload.py` @ `555967922`, cited line-for-line
  (`UVAOffloadConfig:16-44`, `PrefetchOffloadConfig:48-76`, `OffloadConfig:80-93`,
  `validate_offload_config:96-136`). Upstream has no disk tier, so there is
  nothing to mirror and the extension is vllm.cpp-original by construction. It
  therefore lives in its own struct in its own file, under its own namespaced
  JSON key, and `OffloadConfig` gains no field.
- **An unknown top-level key is already accepted, so option 2 of the issue is
  workable.** `parse_offload_config_json` (`src/vllm/config/offload.cpp:234-283`)
  looks up `offload_backend`, `uva` and `prefetch` by name and never enumerates
  the document's keys; `validate_offload_config`'s mirror
  (`OffloadConfig::Validate`) reads fields, not keys. A `vllm_cpp` sibling is
  invisible to both. This is what makes the namespaced key possible rather than a
  sibling flag, and it is also the one hazard the extension parser has to close
  itself: a *misspelled* extension key would be silently ignored by both parsers,
  so the extension parser refuses an unknown key inside `vllm_cpp` by name.
- **Every one of these knobs latches in a function-local static, and two of them
  latch during weight load.** `Qwen35ExpertStreamRequested()`
  (`qwen3_5.cpp:5146-5154`) and `PrefaultBorrowedSpan`'s `enabled`
  (`qwen3_5_gguf_weights.cpp:37-44`) both compute once, on first call, and cache
  forever. A config installed after the first call is not merely late — it is
  *silently* ignored, which is the invisible-fallback shape this tree refuses
  everywhere else. So the resolver records that a decision was latched and
  `SetWeightResidencyConfig` **throws** when a non-empty config arrives after
  one, rather than being quietly discarded. The install site is the first
  statement block of `LoadedEngine::FromModelDir`, beside the weight-offloader
  install, which is already documented as being before any weight I/O.

## Scope

| Field | Content |
|---|---|
| Row ID | `ENG-RESIDENCY-CONFIG` (engine-matrix, KV cache and memory). Issue [#1110](https://github.com/mudler/vllm.cpp/issues/1110); fixes [#1109](https://github.com/mudler/vllm.cpp/issues/1109) in flow |
| In | A vllm.cpp-original `WeightResidencyConfig` under the `vllm_cpp` key of the existing `--offload-config` document; its parser, with unknown-key and wrong-type refusal; a process-global install/resolve seam with a defined config-vs-env precedence and a late-install refusal; the three call sites that resolve these knobs today (`GgufLoadPolicy::FromEnv` for `mmap`, `PrefaultBorrowedSpan` for `prefault`, `Qwen35ExpertStreamRequested` + the `Qwen35ExpertStream` constructor for the streaming lane); the flag→`EngineParams`→install chain through both production entry points (`server_main.cpp` and the C ABI's `offload_config`); `docs/USAGE.md` and `docs/ENVIRONMENT.md` |
| Out | Any change to `OffloadConfig`, `UVAOffloadConfig`, `PrefetchOffloadConfig` or their validator — the mirror stays byte-faithful. Any change to what the knobs *do*: this row moves where their value comes from and nothing else. A new flag. `VT_MOE_EXPERT_STREAM_STATS_EVERY` (see below). `VT_GGUF_KEEP_QUANT`, `VT_CPU_REF`, `VT_GGUF_KEEP_F16` and the rest of the load-transform family — they are a different tier and a different row |
| Supported modes | `{"vllm_cpp":{"mmap":{"enabled":bool,"prefault":bool},"expert_stream":{"enabled":bool,"slots":int,"slot_bytes":int}}}`. Every field is optional and every absent field means "unchanged", so an absent `vllm_cpp` key is byte-identical to today |
| Dispatch behavior | Resolved once, at first read, from **env var if set, else config if set, else the built-in default**. Nothing is resolved when neither is set, so the default engine path is byte-identical |
| Regimes served | A checkpoint larger than host RAM on a single box: the mmap-borrowed weight tower plus the bounded expert slot cache. CPU keep-quant expert towers today; a device platform serves the slice device-resident and is unaffected |

## Upstream chain

Pin `555967922` (`.agents/upstream-sync.md`), verified in the local checkout.

**Upstream implements the mirrored half and nothing else.** `vllm/config/offload.py`
defines `OffloadBackend = Literal["auto", "uva", "prefetch"]` (`:12`) and the two
sub-configs; there is no third arm and no disk tier. `create_offloader`
(`vllm/model_executor/offloader/base.py:139-162`) selects prefetch, uva, or
`NoopOffloader`. `offloader/uva.py:21` is a CPU-blanket UVA offloader and
`offloader/prefetch.py:557-560` is cpu-only. Neither reads a file at inference
time.

So there is nothing to mirror for this tier, and this row must not invent
anything inside the mirrored structs. What it does mirror is the **shape** of the
surface next door: `--kv-transfer-config` and `--offload-config` both take a JSON
object parsed and validated at startup, before the multi-GB load, and refuse a
typo rather than defaulting it (`server_main.cpp:1096-1109`). The extension
follows that contract exactly, including the refuse-rather-than-default rule,
which is why an unknown key inside `vllm_cpp` is an error.

The knobs themselves are ports and already carry their anchors: the load-time
prefault mirrors llama.cpp's mmap prefetch under `use_mmap`
(`src/llama-mmap.cpp:451` @ `237ad9b96`), recorded at
`qwen3_5_gguf_weights.cpp:27-36`; the expert slot cache is the surpass-track
`ENG-EXPERT-STREAM` lane, whose absence upstream is recorded in the engine
matrix. This row adds no upstream behavior, so it inherits their anchors rather
than claiming new ones.

## Our baseline

Measured on this tree at `281e6a120`.

| Knob | Read at | Latches | Default |
|---|---|---|---|
| `VT_GGUF_MMAP` | `gguf_keep_quant.cpp:248` (`GgufLoadPolicy::FromEnv`) | no — `FromEnv()` is called per load | `p.keep_quant`, i.e. on wherever the device can execute the quantized GEMM; forced off by `VT_CPU_REF` |
| `VT_GGUF_PREFAULT` | `qwen3_5_gguf_weights.cpp:39` (`PrefaultBorrowedSpan`) | **yes**, function-local static | **on** — unset reads as enabled — and `docs/ENVIRONMENT.md:50` says off, which is [#1109](https://github.com/mudler/vllm.cpp/issues/1109) |
| `VT_MOE_EXPERT_STREAM` | `qwen3_5.cpp:5148` (`Qwen35ExpertStreamRequested`) | **yes**, function-local static | off |
| `VT_MOE_EXPERT_STREAM_SLOTS` | `qwen3_5.cpp:5365` (`Qwen35ExpertStream` ctor) | effectively — the store is a process-lifetime singleton built once | 64 |
| `VT_MOE_EXPERT_STREAM_SLOT_BYTES` | `qwen3_5.cpp:5359` (same ctor) | same | the largest of the first MoE layer's gate/up/down slices |
| `VT_MOE_EXPERT_STREAM_STATS_EVERY` | `qwen3_5.cpp:5370` (same ctor) | same | 16 |

The config side already exists and is wired: `--offload-config`
(`server_main.cpp:579`) → `parse_offload_config_json` + `Validate`
(`:1101-1108`) → `EngineParams::offload_config`
(`include/vllm/entrypoints/model_loader.h:155`) → `LoadedEngine::FromModelDir`
(`model_loader.cpp:1256-1288`), which installs the offloader **before any weight
I/O** and says so. The C ABI carries the same string
(`include/vllm.h:436`, `src/capi/vllm_c.cpp:638-645`). So the plumbing this row
needs is one field wide, and the install point is already chosen and already
documented for exactly this ordering reason.

The row's own gap: `EngineParams` has nowhere to put the extension, and the three
resolve sites have no input but `getenv`.

## Port map

| Piece | Where |
|---|---|
| `WeightResidencyConfig` + the resolve/install seam | new `include/vllm/config/weight_residency.h`, `src/vllm/config/weight_residency.cpp` |
| Extension parser | `parse_weight_residency_extension_json` in the same pair, reading the `vllm_cpp` key of the SAME document `--offload-config` carries |
| Engine field | `EngineParams::weight_residency` in `include/vllm/entrypoints/model_loader.h` |
| Install (production call site) | `LoadedEngine::FromModelDir`, `src/vllm/entrypoints/model_loader.cpp`, in the block that already installs the weight offloader |
| Server flag | `src/vllm/entrypoints/openai/server_main.cpp`, inside the existing `if (!args.offload_config.empty())` block |
| C ABI | `src/capi/vllm_c.cpp`, inside the existing `offload_config` block |
| `mmap` resolve | `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp`, `GgufLoadPolicy::FromEnv` |
| `prefault` resolve | `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`, `PrefaultBorrowedSpan` |
| `expert_stream`, `slots`, `slot_bytes` resolve | `src/vllm/model_executor/models/qwen3_5.cpp`, `Qwen35ExpertStreamRequested` and the `Qwen35ExpertStream` constructor |
| Docs | `docs/USAGE.md` (the streaming section gains the config form), `docs/ENVIRONMENT.md` (precedence note + the #1109 default correction) |
| Named resolvers | `ResolveGgufMmap`, `ResolveGgufPrefault`, `ResolveExpertStreamRequested` (+ its pure `ExpertStreamRequestedFrom`), `ResolveExpertStreamSlots`, `ResolveExpertStreamSlotBytes` — one per knob, each the sole reader of its variable |

**Five knobs, five named resolvers, and the polarities are not the same.** Each
knob gets one function in the new header that owns its environment NAME and its
exact historical rule, and each becomes the sole reader of its variable. That is
not decoration. `VT_GGUF_MMAP` and `VT_GGUF_PREFAULT` compare the whole value
against `""`, `"0"`, `"false"` and `"off"` (the tree's `EnvOn`,
`gguf_keep_quant.cpp:60-65`). `VT_MOE_EXPERT_STREAM` examines only the **first
character** — `v[0] != '0' && v[0] != '\0'` — so `VT_MOE_EXPERT_STREAM=false`
reads as ON, and `docs/ENVIRONMENT.md` states that explicitly. Routing all five
through one generic helper would silently normalise the odd one, and a row whose
subject is *where a value comes from* must not also change *what a value means*.
So the odd rule is transcribed, and it is additionally exposed in a pure form
(`ExpertStreamRequestedFrom(env_value, configured)`) because its wrapper latches
and can be exercised only once per process — which is exactly how a
normalisation there would have escaped a test.

**The prefault resolve stops latching, deliberately.** `PrefaultBorrowedSpan`
cached its answer in a function-local static. Dropping the cache costs one
`getenv` per prefaulted span, against the megabytes of pages the function then
reads, and it removes two defects: a config installed at load could be ignored by
whichever caller asked first, and the existing A/B case
(`tests/vllm/test_gguf_keep_quant.cpp`, "L7 load-time prefault is
byte-transparent") was **silently vacuous** — its second `setenv` could not
change an already-latched value, so both arms ran identically. The
expert-stream resolve keeps its latch, because that answer decides whether an
~18 GiB slot store is built and whether the grouped-MoE path is disabled, and
those two must not be able to disagree later in one process.

**Precedence, stated once and pinned by a test: environment variable > JSON
config > built-in default.** The environment keeps winning because several of
these variables exist so a benchmark arm is switchable without restarting with a
new config, and an A/B in flight depends on that. The config is the *documented*
surface; the environment is the *override*. Any other polarity would break a
running measurement to make a document tidier.

**`VT_MOE_EXPERT_STREAM_STATS_EVERY` stays environment-only, deliberately.**
Every other knob here changes what memory the process reserves or where a weight
lives — a deployment decision, which is what a config file is for. This one
changes only how often a diagnostic line is printed to stderr. It moves no byte,
reserves nothing, and changes no number; it is the instrument, not the
configuration. Putting it in the config surface would invite it into deployment
manifests where it means nothing, and it is exactly the kind of thing an operator
flips while staring at a run. Recorded here so a later reader sees a decision
rather than an omission.

**`slot_bytes` is IN, and that is not obvious.** It looks like an internal sizing
detail, but the code refuses a slice that exceeds it **by name** and tells the
operator to raise it (`qwen3_5.cpp:5204-5207`), and a dynamic (UD) quant is
precisely the case where the computed default is wrong. A knob a documented error
message tells you to change is a user surface.

## Tests to port

There is nothing to port: upstream has no disk tier, so there is no upstream test
for this surface. Upstream's `tests/basic_correctness/test_cpu_offload.py:19-21`
covers the mirrored tier only, and `test_offload_config.cpp` already carries it.

New tests, each red before its implementation:

- `tests/vllm/config/test_weight_residency_config.cpp`
  - the parser: every field, an absent `vllm_cpp` key, an empty document, a
    `vllm_cpp` that is not an object, an unknown key inside `vllm_cpp`, an
    unknown key inside `mmap`/`expert_stream`, a wrong-typed field, a
    non-positive `slots`/`slot_bytes`;
  - the mirror is untouched: the SAME document that carries a `vllm_cpp` key
    still parses through `parse_offload_config_json` to a byte-identical
    `OffloadConfig`, and a document with only `vllm_cpp` leaves the offload
    config inert;
  - **precedence**: env-set + config-set ⇒ env; env-unset + config-set ⇒ config;
    both unset ⇒ built-in default; env set to `0` beats a config `true` (an
    override has to be able to turn a thing OFF, which is the direction a
    benchmark arm usually needs);
  - **the latch**: a non-empty install after a resolve throws; an empty install
    after a resolve does not (it is the no-op the default path performs); a
    re-install of the same config does not.
- `tests/vllm/entrypoints/test_weight_residency_reach.cpp` — reachability
  through `LoadedEngine::FromModelDir` on a nonexistent model directory: the
  install happens before the load fails, so the process-global carries the
  parsed values afterwards.
- `tests/vllm/entrypoints/openai/test_serve_residency_config.cpp` — reachability
  through the REAL `VllmServerMain`, re-exec'ing the test binary as
  `test_serve_recipe_args.cpp` does, asserting the install line on the child's
  stderr for a `--offload-config` document carrying only a `vllm_cpp` key. This
  is the test the reachability mutation deletes the call site under.

## Gates

Correctness only. This row moves the *source* of a value; it changes no kernel,
no dtype, no allocation and no token, so it has no throughput axis of its own and
claims none.

1. Full gate: `cmake --build build -j 8 && ctest -j 6`, exit 0, no case-count
   regression against the pre-change baseline on the same tree.
2. The three new suites green with non-zero case counts, each red first.
3. Every guarantee mutation-proven: for each added test, delete or invert the
   behavior it names, rebuild, require its suite red with a non-zero case count,
   restore by byte copy and verify by sha256. A mutation that fails to compile is
   INVALID, not a pass, and the compile status is recorded beside every row.
4. The reachability mutation: delete the install call site in
   `LoadedEngine::FromModelDir` in a scratch copy and require the server-level
   suite red.
5. `scripts/agent-preflight.sh --staged` green, including `check-env-doc`
   (the environment table is edited here) and `check-agent-record`.
6. Inertness: with no `vllm_cpp` key and no environment variable set, the
   resolvers return exactly what `getenv` returned before. Pinned by the
   both-unset precedence case rather than asserted.

## Dependencies

- `ENG-WEIGHT-OFFLOAD` ([#797](https://github.com/mudler/vllm.cpp/issues/797))
  owns `OffloadConfig` and its mirror. This row must not touch it, and the
  "mirror is untouched" test case is what keeps that honest. If a future vLLM
  release adds a disk arm upstream, the reconciliation is that row's: the
  extension is then superseded by a mirrored field and this row's spec records
  the migration.
- `ENG-EXPERT-STREAM` ([#912](https://github.com/mudler/vllm.cpp/issues/912))
  owns the streaming mechanism, and
  [`specs/expert-streaming.md:193`](expert-streaming.md) already anticipated
  routing through `offload_config` for exactly this reason. This row does that
  and changes nothing about the mechanism.
- No hardware dependency for the gate. The measured big-model reproduction is a
  GB10 job and is owed (see `## Owed`).

## Work breakdown

One wave. The change is one field wide at every hop, and splitting it would land
a parser nothing reaches.

| Step | Content |
|---|---|
| W1a | The struct, the parser, the install/resolve seam, and the config suite — red first |
| W1b | The three resolve sites, switched from `getenv` to the resolver |
| W1c | `EngineParams`, `server_main.cpp`, the C ABI, the install site, and both reachability suites |
| W1d | `docs/USAGE.md`, `docs/ENVIRONMENT.md` (including the #1109 correction), the records |

## Risks and decisions

- **A namespaced key inside a mirrored flag can read as "vLLM takes this".** It
  does not. The key is literally `vllm_cpp`, which is the cheapest possible
  signal that the contents are not upstream, and the docs say so in the same
  sentence they introduce it. The alternative — a second flag — costs a user two
  flags for one concept and still needs the same disclaimer.
- **Env-wins precedence can surprise a user whose config is being overridden by
  an exported variable they forgot.** Accepted, and mitigated where it is
  cheapest to see: the install line prints the RESOLVED values, so a run whose
  config was overridden says so on stderr at startup. The opposite polarity was
  rejected because it breaks a measurement in flight.
- **The late-install throw could fire in a process that loads twice.** A second
  `FromModelDir` in one process with a *different* non-empty residency config
  cannot be honored — the knobs latched during the first load — so throwing is
  the correct answer, not a defect. Re-installing an EQUAL config is allowed
  precisely so the ordinary two-engine test binaries do not trip on it.
- **`GgufLoadPolicy::FromEnv` keeps its name while no longer being env-only.**
  Renaming it touches 30 call sites across tests and four model loaders for no
  behavior change, which is a bigger diff than this row's whole subject. The
  function's doc comment states that it resolves env-over-config-over-default,
  and the resolver it calls is named for what it does. Recorded as debt, not
  hidden.

## Owed

- **The measured big-model reproduction.** The headline case
  (`Qwen3.8-2.4T-A95B UD-Q1_0`, 370 GiB on a 119 GB GB10) has to be re-run once
  through the config surface to prove the JSON form reaches the same state the
  environment form did, with the startup line and the `[expert-stream]`
  statistics as the evidence. It was NOT run for this row: dgx.casa is
  unreachable at the SSH layer (TCP/22 accepts, then times out during banner
  exchange), and everything else here is CPU-local. Owned by
  `ENG-RESIDENCY-CONFIG`, issue
  [#1110](https://github.com/mudler/vllm.cpp/issues/1110).
- **`GgufLoadPolicy::FromEnv`'s name.** See Risks. Owned by this row.
- **A positive on-versus-off observation of the prefault.** The tree has never had
  one. The only prefault case asserts byte-transparency, which holds whether the
  prefault runs or not, and under the function-local static its second arm could
  not differ from its first anyway. This row removes the static, so the case's two
  arms now genuinely differ, and it adds a resolver-level test of the decision —
  but nothing yet observes the *residency effect* (an `mincore()`-style check over
  the borrowed pages, as `test_load_direct_upload` already does for the release
  path). Owned by `ENG-RESIDENCY-CONFIG`, issue
  [#1110](https://github.com/mudler/vllm.cpp/issues/1110).

## Now

`ACTIVE`. The config surface exists, parses, refuses a typo, defines its
precedence, and is reachable from both production entry points — the server flag
and the C ABI — with the reachability proven by deleting the install call site
and watching the server-level suite go red. The mmap, prefault, expert-stream,
slots and slot-bytes knobs all resolve through it. `stats_every` stays
environment-only by decision, recorded in Port map. `docs/ENVIRONMENT.md`'s
`VT_GGUF_PREFAULT` default is corrected here, closing #1109.

What keeps it `ACTIVE` rather than `DONE` is the one thing above under `## Owed`:
nobody has yet driven the 370 GiB checkpoint through the JSON form on the box
that can hold it.
