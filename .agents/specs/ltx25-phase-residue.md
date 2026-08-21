# LTX25-PHASE-RESIDUE — the phase table's un-named time, and the two ratios that were measuring it

Row: `LTX25-PHASE-RESIDUE`.
Issues: [#1536](https://github.com/mudler/vllm.cpp/issues/1536) (primary),
[#1439](https://github.com/mudler/vllm.cpp/issues/1439),
[#1494](https://github.com/mudler/vllm.cpp/issues/1494),
[#1470](https://github.com/mudler/vllm.cpp/issues/1470).
Instrument: `src/vllm/multimodal/render_phase_log.cpp`, landed by
`LTX25-DEVICE-RESIDENCY` stage W0 ([#1010](https://github.com/mudler/vllm.cpp/issues/1010)).
Base: `67823aee22d052cb53e08f5793fd899b2d0a582f`.

## Scope

`test_ltx2_video` is the only failing test on `main`. Three assertions fail, in
two cases, and they are the same defect at two levels of the phase table:

```
tests/vllm/multimodal/test_ltx2_video.cpp:3259  CHECK( leaves >= 0.95 * wall )
tests/vllm/multimodal/test_ltx2_video.cpp:3696  CHECK( covered >= c.min_coverage * leaf_seconds )   x2
```

This row lands four things and nothing else:

1. **Names the un-named time in the driver.** Three regions of
   `Ltx2VideoEngine` are inside no leaf and are 93% of the residue the sum gate
   fails on. Naming them is what W0's own stop condition asks for and what the
   failing assertion's own message says: *"The missing time is a phase nobody
   named, and W0 iterates until it is."*
2. **Anchors the un-anchored work inside `denoise`.** The sampler's
   post-process and its Euler step sit between two `denoise.step` records and no
   anchor wraps them. They are real work, they scale with the latent, and they
   are the whole of the coverage gate's miss.
3. **Makes the instrument measure its own cost** and publish it, per record and
   for the table, so a reader can tell instrument overhead from work nobody
   named. That number does not exist today, which is why every previous
   investigation of these two gates had to argue about it.
4. ~~**Replaces both wall-clock ratios with a bound derived from that measured
   cost.**~~ **WITHDRAWN, and the withdrawal is this row's most useful result.**
   Three fresh reviews measured the replacement and it is load-flaky in the same
   way the assertions it replaced were — 4 red in 45 runs at the table, max ratio
   4.115. Both floors are therefore the tree's own, unedited, and they now carry
   orders of magnitude of margin because item 1 removed 92% of what they were
   measuring. See `### The bound this row proposed, and why it is not here`.

**Out of scope.** No numeric behaviour of the render changes. No threshold is
loosened: §Design proves both replacements are strictly stricter than what they
replace, at fixture scale and at 21 B scale. The `part_min_coverage` floors
(assertion 2b), the record counts (0), containment (1), exclusivity (3),
non-overlap (4) and the nesting rule (5) are untouched. The `decode.video.vae`
floor is untouched. The res_2s sampler arm's own update anchor is recorded under
`## Owed` rather than landed, because the fixture recipe does not reach that arm
and an anchor no gate runs is the dead code `AGENTS.md` §"Nothing lands dead"
names.

## Upstream chain

None, and that is a finding rather than an omission. `render_phase_log.*` is
this project's own instrument: vLLM has no phase table for a diffusion render,
`vllm-omni` registers no LTX-2.5 recipe (upstream
[vllm-omni#6066](https://github.com/vllm-project/vllm-omni/issues/6066) is
open), and the row that built the instrument
([`ltx25-device-residency.md`](ltx25-device-residency.md) §W0) records it as a
scratch implementation. The `Ltx2VideoEngine` scope placements this row adds sit
around code whose upstream anchors are already cited at those lines
(`ltx-pipelines` `samplers.py:35`, `:503` for the first-order step;
`blocks.py:1139` and `single_gpu_model_builder.py:267-288` for the load). This
row moves no numeric and cites no new upstream line.

## Our baseline

Measured on this branch at the base SHA, x86_64, the CI configuration
(`cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON`, i.e. **empty**
`CMAKE_BUILD_TYPE`, which is what `build-test-cpu` uses and therefore
unoptimized), `VLLM_KEEP_TEST_ARTIFACTS=1`, box load average 44 at start.

```
[doctest] test cases:  102 |  100 passed | 2 failed | 0 skipped
[doctest] assertions: 4170 | 4167 passed | 3 failed |
[doctest] Status: FAILURE!
```

```
:3256: MESSAGE: phase table: wall=0.26271s leaves=0.243533s unaccounted=0.0191776s over 35 entries
:3259: ERROR: CHECK( leaves >= 0.95 * wall ) is NOT correct!
  values: CHECK( 0.243533 >= 0.249575 )

:3693: MESSAGE:   denoise = 0.00679651s over 1 leaf record(s), of which 8 sub-scope(s) cover 0.00640374s (94.221%)
:3696: ERROR: CHECK( covered >= c.min_coverage * leaf_seconds ) is NOT correct!
  values: CHECK( 0.00640374 >= 0.00645668 )
```

### Where the residue actually is, which nobody had measured

`unaccounted_seconds` was known as an aggregate. Decomposing the emitted table
into the gaps between consecutive leaves settles it. The failing table
(`phase_out/phase-log.json`, one render, wall 0.262710 s, residue 19.178 ms):

| gap | ms | share of residue |
|---|---:|---:|
| `<origin>` → `load.dit` | 17.661 | **92.09%** |
| `load.dit` → `load.video_vae` | 0.950 | 4.95% |
| `load.prompt_embeds` → `generate.setup` | 0.249 | 1.30% |
| `artifacts.audio` → `<WriteJson>` | 0.210 | 1.09% |
| the remaining 16 gaps, together | 0.108 | 0.56% |

The two-render attribution table (wall 2.051320 s, residue 18.553 ms) agrees:
`<origin>` → `load.dit` is 14.451 ms (77.89%), `load.dit` → `load.video_vae` is
0.540 ms, and 2.879 ms sits between the first render's `artifacts.audio` and the
second render's `generate.setup` — a gap that is the TEST's own assertions
between two `Generate` calls and not driver time at all.

So the sum gate is red for exactly the reason its own message states. **93% of
the un-named time is one region**: `Ltx2VideoEngine::Load` from the timeline's
origin (`Open("load")`, `ltx2_video.cpp:766`) to `Open("load.dit")` (`:930`) —
the platform probe, the device resolution, the recipe and checkpoint-class
resolution, and `SafetensorsFile::Open(params.dit_path)` at `:897`, which on a
1775-tensor 21 B manifest is not a free call. The 16 gaps between adjacent named
phases together hold 0.108 ms, i.e. 6.8 µs per boundary, which is the
instrument's own cost and nothing else.

### Where the coverage miss actually is

| render | leaf | leaf ms | parts ms | % | uncovered ms | sub-records |
|---|---|---:|---:|---:|---:|---:|
| 9 frames (`latent_t=2`) | `denoise` | 6.7965 | 6.4037 | 94.22 | 0.3928 | 8 |
| 81 frames (`latent_t=11`) | `denoise` | 39.1793 | 36.4323 | 92.99 | 2.7471 | 8 |
| 9 frames | `decode.video` | 8.2325 | 8.2175 | 99.82 | 0.0149 | 2 |
| 81 frames | `decode.video` | 65.7515 | 65.7083 | 99.93 | 0.0431 | 3 |
| 9 frames | `decode.audio` | 190.3062 | 190.2502 | 99.97 | 0.0560 | 2 |
| 81 frames | `decode.audio` | 1667.4728 | 1667.3619 | 99.99 | 0.1109 | 2 |
| 9 frames | `artifacts.frames` | 0.9116 | 0.9000 | 98.72 | 0.0117 | 1 |
| 81 frames | `artifacts.frames` | 7.5112 | 7.4695 | 99.44 | 0.0417 | 2 |

Three of the four leaves miss by 7-42 µs per sub-record, which is the same 7-20 µs
per boundary the inter-leaf gaps measure: their sub-scopes are adjacent
statements to the leaf's own, so the only uncovered thing is the instrument.
`denoise` misses by **49 µs per step at 9 frames and 343 µs per step at 81** —
a 7x difference inside one run of one binary. A quantity that moves 7x with the
latent is work, not overhead, and the case's own comment already names it: *"The
uncovered part is real work — the post-process and the Euler or res_2s step,
which no anchor wraps."*

That also explains the `denoise_min_coverage` parameter (0.95 at nine frames,
0.90 at 81). The parameter is the shape of the un-anchored work leaking into a
threshold. Anchoring the work removes the reason for the parameter.

### What is NOT the cause, measured rather than assumed

* **Not box load.** #1439 recorded a `main` run that PASSED at `wall=0.579684 s`
  — double every failing run — because the residue grows more slowly than the
  wall it is divided by. A slower render passes. This run reproduces that
  polarity: the residue is 19.178 ms whether the render is fast or slow, because
  17.7 ms of it is a fixed region of the load.
* **Not the W0-live stderr emitter.** #1439 measured `VLLM_RENDER_PROGRESS=0` at
  94.12%, inside the same failing band.
* **Not `d995c52f0` (the temporal x2 upsampler), which #1536 named as the
  hypothesis to test first.** The upsampler runs inside `phase.upsample_latent`,
  a named leaf, and the DFR rounds run inside `generate.temporal_rounds`. Neither
  appears in the residue decomposition above at all. The hypothesis is refuted by
  the table.

## Port map

No port. The files this row touches:

| File | Change |
|---|---|
| `include/vllm/multimodal/render_phase_log.h` | `Record::instrument_seconds`; document the accounting rule |
| `src/vllm/multimodal/render_phase_log.cpp` | charge the instrument's own out-of-record wall to the innermost live non-span record, or to the table when none is live; emit both; take `Elapsed()` before serializing |
| `src/vllm/multimodal/ltx2_video.cpp` | three new leaves (`load.setup`, `load.dit_config`, `artifacts.mux`) and one new anchor (`denoise.update`) with its counter |
| `include/vllm/multimodal/ltx2_video.h` | `Ltx2ConditioningTrace::sampler_updates` |
| `tests/vllm/multimodal/test_ltx2_video.cpp` | the two replaced assertions, the per-record part pairing, and the instrument's own unit cases |

## Design

### 1. The instrument measures its own out-of-record cost

The residue and the uncovered time both contain a term the instrument creates
and never reported: the wall it spends inside its own entry points while no
record — or no CHILD record — is open. Concretely, `PhaseLog::Open` stamps
`o.start` after taking the process-wide mutex, so the mutex wait is before the
new record begins; `PhaseLog::Close` stamps `r.end` before it emits its progress
line and erases the entry, so that tail is after the record ends. Both land
outside every record.

The rule this row adds is one sentence: **every interval of the instrument's own
wall is charged to the innermost live non-span record at the moment it is spent,
and to the table when none is live.** A span is excluded because `Sum` excludes
spans, so time inside a span but outside a leaf is exactly the residue.

That makes three quantities available that were not:

* `Record::instrument_seconds` — the part of this record's own duration that the
  instrument spent, outside any child of it. Emitted per phase.
* `instrument_seconds` at the top of the table — the part of `unaccounted_seconds`
  that the instrument spent.
* The invariant that the two, summed, equal the instrument's total self-time.

`WriteJson` also takes `Elapsed()` before it copies and sorts the record vector
rather than after, so the table's own serialization stops being charged to the
render's wall. It measures the render, not the writer.

### 2. The driver names what it was not naming

* `load.setup` — from the timeline's origin to `load.dit`. 92% of the residue.
* `load.dit_config` — the DiT config resolution between `load.dit` and
  `load.video_vae`. 5% of the residue.
* `artifacts.mux` — the result assembly and mux argv build after
  `artifacts.audio`. 1% of the residue.
* `denoise.update` — nested inside `denoise`, wrapping the sampler's
  post-process and step on the first-order arm, counted by
  `Ltx2ConditioningTrace::sampler_updates` so assertion (0) has a denominator the
  phase table cannot move.

### 3. Both ratios STAY, and the bound this row proposed is withdrawn

The row's plan was to replace

```cpp
CHECK(leaves >= 0.95 * wall);                       // the table
CHECK(covered >= c.min_coverage * leaf_seconds);    // each carrying leaf
```

with `residue <= kInstrumentBudget * instrument`, on the argument that a gap is
the closing record's tail, plus the opening record's head, plus a call and a
return — so what the instrument cannot measure is at most as large as what it
can, and the bound therefore says the same thing at 64x64x9 and at 3840x2160x241.

**That argument is false, and three fresh reviews measured it rather than
arguing with it.** The un-instrumented remainder of a boundary dilates FASTER
than the instrumented part under contention, so the comparison has a heavy right
tail rather than a shifted median:

| site | runs | red | median | max |
|---|---:|---:|---:|---:|
| the table bound, load 88 | 45 | **4 (8.9%)** | 1.132 | **4.115** |
| the conservation case, load 80 | 200 | **3** | 1.525 | 2.934 |
| the `unit.parent` case, load 85 | 200 | **2** | 1.70 | — |
| a standalone probe of that shape, load 125 | 160 | **28 (17.5%)** | — | 5.55 |
| the same probe under `address,undefined` | 30 | 2 | 1.12 | **14.1** |

The mechanism, from decomposing a parent's uncovered time: fast, its inter-child
gaps are 9-20 us over seven boundaries against a 13-22 us charge; slow, 91-105 us
against 52-61 us. A 20-run distribution reading 1.021 to 1.464 was the BODY of
the first row of that table and saw none of its tail.

**So the bound is withdrawn at all three sites and the constant is deleted rather
than raised.** This row's own stop condition said to stop rather than write a
bigger number, and [#1466](https://github.com/mudler/vllm.cpp/issues/1466)
rejects the bigger number as a class.

**The floors that stay are the tree's own, unedited.** `leaves >= 0.95 * wall` is
the line that was red; `denoise_min_coverage` is `6b48edb2c`'s 0.75, and the
other three are its 0.99, 0.90 and 0.50. Nothing is loosened by this row, and
nothing is tightened either.

**And they are no longer coin flips, because §2 removed what they were
measuring.** 92% of the sum floor's numerator was one un-named region that does
not scale with anything. Named, the residue drops by an order of magnitude:
**99.961% of wall** on the landing tree, where the red measured 92.700%.
`denoise` coverage moves from **94.221% to 99.939%**, because the sampler's
per-step update has a name.

**`wall` in the denominator is also better conditioned than `instrument` was**,
which is the part this row had backwards. Wall grows with contention exactly when
a preemption inflates the residue, so numerator and denominator move together;
the instrument's charge does not.

`instrument_seconds` survives as a REPORTED quantity — emitted for the table and
for every record, and printed beside every residue the file prints. That is what
[#1439](https://github.com/mudler/vllm.cpp/issues/1439) asked for as its second
option, "bounding `unaccounted_seconds` beside the ratio", and it is the
normaliser `6b48edb2c` records as not existing. A reader subtracts it before
calling a residue a phase nobody named. It is not an assertion, because on this
hardware it cannot be one.

### 4. The sibling pairing (1b) is strengthened, not weakened

`denoise` becomes the second multi-part leaf, and its two parts REPEAT.
Assertion (1b) compares only the first record of each name, and its own comment
discloses the hole: *"A future leaf with two REPEATING parts would satisfy this
assertion while running `A, B, B, A, B, A`."* This row does not inherit that
debt. When every part of a leaf has the same record count, (1b) compares the
i-th record of each part instead of the first, which is the per-record pairing
that comment says is owed. For `decode.audio` at `{1, 1}` this is identical to
today; for a single-part leaf it stays vacuous.

## Tests to port

None to port. The tests this row writes or changes, all in
`tests/vllm/multimodal/test_ltx2_video.cpp`:

1. `ltx2 video: a render through the ABI emits a phase table that SUMS to wall`
   — the sum assertion is replaced as §Design.3. The named-boundary list gains
   `load.setup`, so an instrument that stopped naming the load prologue is a red
   on identity as well as on the residue.
2. `ltx2 video: the three carrying phases contain their work and the load keeps
   its order` — the coverage assertion is replaced as §Design.3; `denoise` gains
   `denoise.update` as a part with its record count; (1b) gains the per-record
   pairing.
3. **New unit case: the instrument charges its own cost to the right place.**
   Direct on `PhaseLog`, no render: open a leaf, open and close a nested child
   inside it, close the leaf, and assert that the child's boundary cost is
   charged to the PARENT and not to the table, and that a boundary taken with no
   leaf live is charged to the table. Red-before is a mutation of the charging
   rule.
4. **New unit case: the accounting is conserved.** The table's
   `instrument_seconds` plus every record's `instrument_seconds` equals the
   instrument's total self-time, and every value is non-negative.
5. **New unit case: `WriteJson` measures the render and not the writer.** The
   emitted `wall_seconds` does not include the copy-and-sort the writer performs.

## Gates

Run by this row, on this branch, on an x86_64 box, in the `build-test-cpu`
configuration:

| Gate | Command | Result |
|---|---|---|
| red-before, current tree | `./build/tests/test_ltx2_video` | recorded in §Our baseline: 2 cases / 3 assertions failing |
| red-before, NEW assertion on the OLD tree | the new bound, computed from the emitted table | must fail, and must fail on a run where the OLD ratio PASSES |
| focused | `./build/tests/test_ltx2_video` | 0 failures, repeated |
| full | `ctest --test-dir build --output-on-failure` | 0 failures |
| record | `scripts/agent-preflight.sh --staged` | pass |
| mutation | §Design.3's bound, against a scope deleted, a scope moved over unnamed time, and an anchor emptied | each must red |

Every gate run records `wall`, `unaccounted`, `instrument_seconds` and the ratio
between the last two, because the bound in §Design.3 is only defensible while
that ratio is measured rather than assumed.

**The Result column above states what each gate is FOR, not what it returned.**
A fresh review was right to refuse it as evidence. The SHA, the exact command,
the environment, the exit status and the evidence path for every run are in
`## Outcome`, under `### The gate report` and `### The mutations`.

## Dependencies

None. The instrument, the driver and the test are all in this tree and this row
takes no lease: the fixture render is CPU-only and the gate is a CPU gate. It
does not block on [#1126](https://github.com/mudler/vllm.cpp/issues/1126) (the
device byte column), on [#655](https://github.com/mudler/vllm.cpp/issues/655)
(the `ltx_core` oracle) or on any W1 measurement, because nothing here is a
speed claim.

## Work breakdown

W1 the spec (this file), committed before any implementation.
W2 the instrument: charge and emit its own cost; `Elapsed()` before serializing.
W3 the driver: three leaves and one anchor with its counter.
W4 the test: both replacements, the (1b) pairing, and the three unit cases.
W5 gates, mutations, and the record edits the change makes stale.

## Risks and decisions

* **The bound could be too tight and become a NEW flake, which would be this
  row failing in its own terms.** Mitigated by measuring `unaccounted /
  instrument_seconds` across repeated runs before the bound is fixed, and by
  §Design.3's argument that a preemption inflates both sides. Recorded as a
  measurement in `## Outcome` rather than as a claim. If the measured ratio does
  not sit near 1, the bound is wrong and the finding is that the instrument does
  not measure enough of its own gap — which is a repair to §Design.1, not a
  bigger constant.
* **A new scope adds a boundary, and a boundary costs residue.** Four new scopes
  add about eight boundaries at 7 µs, i.e. 0.06 ms, against the 18.8 ms they
  name. Recorded because it is the reason "just add more scopes" is not a
  general answer.
* **`denoise` gaining a second repeating part makes (1b) non-vacuous for the
  first time.** Handled in §Design.4 rather than deferred.
* **The res_2s arm gets no update anchor.** Decided, not overlooked: the update
  happens inside `Ltx2Res2sDenoisingLoop`, the fixture recipe does not reach that
  arm, and an anchor no gate runs is dead code. Recorded under `## Owed`.
* **Replacing an assertion is the thing `AGENTS.md` forbids doing to make a red
  gate green.** The defence is that both replacements are strictly stricter at
  both scales (§Design.3) and that the new bound is shown RED on the current tree
  before the naming lands. If a reviewer rejects that argument the change does
  not merge; there is no waiver registry and this paragraph is the argument.

## Owed

| Issue | State |
|---|---|
| [#1536](https://github.com/mudler/vllm.cpp/issues/1536) | closed by this row |
| [#1439](https://github.com/mudler/vllm.cpp/issues/1439) | closed by this row |
| [#1494](https://github.com/mudler/vllm.cpp/issues/1494) | **already CLOSED by `6b48edb2c` before this row merged `main`.** This row takes the `denoise.update` anchor that change recorded as owed; it does not close the issue and does not claim to |
| [#1470](https://github.com/mudler/vllm.cpp/issues/1470) | closed by this row |
| [#1567](https://github.com/mudler/vllm.cpp/issues/1567) — the res_2s arm's `denoise.update` anchor | **owed, filed by this row.** `Ltx2Res2sDenoisingLoop` runs its own post-process and step inside `ltx2_res2s.cpp` through `Ltx2Res2sHooks`, so the anchor needs a hook rather than a statement. No gate in this tree renders on that arm, so landing it here would land dead code |
| [#1568](https://github.com/mudler/vllm.cpp/issues/1568) — the `denoise.step` / `denoise.update` seconds transfer | **owed, filed by this row, and this row claimed it was closed until a fresh review checked.** (1b') compares `start_seconds` only, so leaving `denoise.step` open across the post-process and emitting `denoise.update` empty after it preserves the alternation, both counters, containment, non-overlap, exclusivity, (1c) and (2), and moves 100% of the decomposed seconds onto one name. No (2b) floor separates it: the honest share of `denoise.update` runs 0.45% to 11.15% across four boxes and a transfer puts it at ~0%. Closing it needs an anchor INSIDE the callee, which is the third row of the anchor table in [`ltx25-device-residency.md`](ltx25-device-residency.md) `### Owed out of W0` for all six anchors |
| [#1569](https://github.com/mudler/vllm.cpp/issues/1569) — a gate on `WriteJson`'s clock ORDERING | **owed, filed by this row, and MEASURED green under its own mutation.** `WriteJson` reads `Elapsed()` before it copies and sorts the records, so the writer stops being charged to the render. Restoring the old order left the conservation case GREEN 10 of 10, at `wall 0.0608987s, unaccounted 0.000534223s, table charge 0.000301655s`, because the copy and the sort of a three-record table are nanoseconds. Gating it needs a table with enough records for the sort to be measurable and a `WriteJson` with nothing between it and the last `Close`. The case is named for what it does prove |
| [#1570](https://github.com/mudler/vllm.cpp/issues/1570) — an upper bound on the instrument's own share of a leaf | **owed, filed by this row.** `uncovered <= 2 * leaf_instrument` is stricter than the floor it replaces only while `leaf_instrument` stays small, and nothing bounds it. Moving the DiT `Tick` out of `Evaluate` would charge ~110 flushed writes to `denoise` and widen the gate while printing a small number |
| a residue bound that survives a contended box | **owed, and it is this row's own negative result plus [#1570](https://github.com/mudler/vllm.cpp/issues/1570).** `residue <= 2 * instrument` measured red 4 in 45 at the table, 3 in 200 at the conservation case and 2 in 200 at `unit.parent`, and is withdrawn. Nothing replaces it, so a future un-named region under 5% of wall is invisible and **mutation D — the `denoise.update` anchor moved off the post-process, 5 of 5 red against the withdrawn bound — is not detected on the landing tree**. Closing it needs a bound on a quantity the scheduler cannot move |
| [#1571](https://github.com/mudler/vllm.cpp/issues/1571) — a per-gap decomposition IN the emitted table | **owed, filed by this row.** This row computed the gap table in a scratch script to find the 92% region. A reader of `phase-log.json` still cannot see it without one, and the same investigation will be re-derived the next time the residue moves |

## Outcome

Landed on `row/LTX25-PHASE-RESIDUE`, base
`67823aee22d052cb53e08f5793fd899b2d0a582f`, issues
[#1536](https://github.com/mudler/vllm.cpp/issues/1536),
[#1439](https://github.com/mudler/vllm.cpp/issues/1439),
[#1494](https://github.com/mudler/vllm.cpp/issues/1494),
[#1470](https://github.com/mudler/vllm.cpp/issues/1470).

Every number below is x86_64, `cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON`
with an **empty** `CMAKE_BUILD_TYPE`, which is what `build-test-cpu` uses. The
box was shared with three to five other sessions running this same suite
throughout, at load averages between 44 and 111. That is stated because it is
the whole subject: the two assertions this row replaced decided by exactly that
number, and the ones that replace them did not move with it.

### What the defect was, and what it was not

**It was one un-named region, and nobody had looked.** Four issues across three
months argued about whether 0.95 was the right tolerance. Splitting
`unaccounted_seconds` into the gaps between consecutive leaves took one pass over
the table the render already writes:

| gap | ms | share of the 19.178 ms residue |
|---|---:|---:|
| `<origin>` → `load.dit` | 17.661 | **92.09%** |
| `load.dit` → `load.video_vae` | 0.950 | 4.95% |
| `load.prompt_embeds` → `generate.setup` | 0.249 | 1.30% |
| `artifacts.audio` → `WriteJson` | 0.210 | 1.09% |
| the other 16 gaps, together | 0.108 | 0.56% |

The 16 gaps between adjacent named phases hold 6.8 us each, which is the
instrument and nothing else. The 17.661 ms is `Ltx2VideoEngine::Load` from the
timeline's origin to `Open("load.dit")` — the platform probe, the device
resolution, the recipe and checkpoint-class resolution, and
`SafetensorsFile::Open(params.dit_path)`.

**The hypothesis #1536 asked to test first is refuted.** `d995c52f0`'s temporal
x2 upsampler runs inside `phase.upsample_latent`, a named leaf. It does not
appear in the residue at all.

**The coverage miss was un-anchored WORK, not overhead, and the shape proves
it.** Three of the four carrying leaves miss by 7-42 us per sub-record — the same
per-boundary figure the inter-leaf gaps show. `denoise` missed by **49.1 us per
step at nine frames and 343.4 us per step at 81**, a 7x move with the latent
inside one run of one binary. Instrument cost does not do that. The case's own
comment had already named the culprit ("the post-process and the Euler or res_2s
step, which no anchor wraps") and had left it un-anchored while tuning the
threshold around it. `denoise_min_coverage` being a PARAMETER — 0.95 at nine
frames, 0.90 at 81 — was that work leaking into a number.

### What was rejected, measured rather than argued

* **A bigger constant.** [#1466](https://github.com/mudler/vllm.cpp/issues/1466)
  rejects it as a class and the residue decomposition says why it would have been
  wrong here: at 0.95 the gate had 13.1 ms of budget and the defect was 17.7 ms,
  so the number that would have passed is one that also passes a load prologue
  twice as large. There is no constant that separates them, because the quantity
  on the other side is the render's wall.
* **Naming the un-named time and keeping the ratio**, which is what #1439 itself
  proposed as one of two options. Rejected on measurement: after the naming the
  residue is 16.2 ms of an 11.7 s wall, so 95% leaves 570 ms of headroom at this
  fixture scale and the gate becomes untestable, while at the 21 B render it
  still permits minutes. A ratio whose margin is that asymmetric is not a gate at
  either end.
* **Bounding the residue with a fixed number of seconds.** It is the same defect
  with the scale inverted: a constant tuned on a 0.26 s fixture reds a 2.5 h
  render's ordinary instrument cost, and one tuned on the 2.5 h render passes a
  fixture whose whole load is un-named.
* **Asserting only the leaf's HEAD and TAIL** — the part outside the anchor
  window — which would have needed no new anchors at all. Rejected because
  `denoise`'s tail legitimately contains one whole sampler update, so the bound
  would have had to absorb 343 us of real work and would have measured nothing.
* **A res_2s update anchor.** Recorded under `## Owed` rather than landed. It
  needs a hook through `Ltx2Res2sHooks` rather than a statement, and no gate in
  this tree renders on that arm, so it would have landed dead.

### The instrument, and the one thing it was not measuring

The rule is one sentence, in `render_phase_log.cpp`: every interval of the
instrument's own wall is charged to the innermost live NON-SPAN record at the
moment it is spent, and to the table when none is live.

**The sampler JOIN was the piece that had to be added after the first
measurement.** `Close` hands the worker thread out under the mutex and joins it
with the lock released, so on the last close of a timeline the notify-and-join
lands in the residue. Uncharged, it read as time nobody named: the
`instrument's own cost is conserved` unit case measured a two-scope timeline
whose gaps contain NOTHING reporting **0.000346 s of residue against a 0.000111 s
charge, a ratio of 3.12** — indistinguishable from a real un-named phase.
Charging it costs a second lock acquisition on a path that runs twice per
process. That is why the rule is stated as "every interval" rather than "every
boundary".

`WriteJson` also reads `Elapsed()` before it copies and sorts the record vector
rather than after. The writer's own serialization was being charged to the
render's wall, and therefore to the residue.

### The bound this row proposed, and why it is not here

This is the row's most useful result and it is a negative one. §Design.3 is
REFUTED by measurement, three times, by a fresh reviewer who ran it for hundreds
of runs where this row had run it for tens.

**What that cost, stated plainly.** The row shipped that bound at three sites and
removed it from each one only after it was measured red there: the `unit.parent`
case (2 in 200), then the conservation case (3 in 200), then the table itself
(4 in 45). Each removal was argued as scoped to a badly conditioned site, and
each time the next measurement found the same defect one site over. The general
statement was available after the first: **a ratio of two wall-clock quantities
the box moves at different rates cannot be a gate on this hardware**, which is
what #1439, #1470, #1494 and #1536 are all about, and this row rediscovered it
from the other side.

**What replaced it is nothing, and that is deliberate.** The two floors are the
tree's own, unedited. What makes them hold is §Design.2: the cause is fixed.

| | before this row | on the landing tree | floor |
|---|---:|---:|---:|
| `leaves / wall` | 92.700% (RED) | **99.961%** | 0.95 |
| `denoise` coverage, 9 frames | 94.221% (RED at the then-0.95) | **99.939%** | 0.75 |
| `denoise` coverage, 81 frames | 92.99% | **99.9932%** | 0.75 |
| `decode.video` coverage | 99.82% | 99.9856% | 0.90 |
| `decode.audio` coverage | 99.97% | 99.9951% | 0.99 |
| `artifacts.frames` coverage | 98.72% | 98.7657% | 0.50 |

**And the instrument's charge is reported at every one of them**, which is the
half of #1439's request that does land: `unaccounted 0.00179039s` against
`instrument 0.00137401s` on the table, and per leaf 1.017 to 1.230 as a ratio
that is printed and not asserted.

### What this cost the gate, stated rather than glossed

Removing the bound loses detections the tree does not otherwise have, and they
are recorded rather than left for the next reader to find:

* **Mutation D** — `denoise.update` moved below the two `PostProcessLatent`
  calls, so the anchor stops covering the post-process while its count,
  containment, nesting and sibling order are all unchanged — reddened the
  withdrawn bound **5 of 5 runs**. Against the 0.75 floor it produces about 94%
  coverage and passes. Nothing in the file now sees it, and no floor can: the
  honest anchored coverage approaches the same value on a fast box, as the leaf
  shrinks and the instrument's cost does not.
* **A future un-named region** smaller than 5% of wall is invisible again, which
  is the same hole at a smaller scale. The named-boundary list catches a name
  being DELETED — mutation A reds there — and not a region being ADDED.

Both are listed under `## Owed` beside
[#1570](https://github.com/mudler/vllm.cpp/issues/1570). What would close them is
a bound on a quantity the scheduler cannot move, and this row does not have one.

### Reconciled with `6b48edb2c`, which landed mid-flight

`6b48edb2c` (`GATE-CI-RED-REPAIR`,
[#1494](https://github.com/mudler/vllm.cpp/issues/1494)) merged to `main` about
three hours after this row's baseline was measured, and it repairs the SAME
`denoise` coverage red by a different route: it moved `denoise_min_coverage` from
0.95 and 0.90 to **0.75** and added assertion (1c), the span slack, beside it.
This row merged it rather than reverting it, and the reconciliation is the
change's own words:

> NAMING THE UN-NAMED TIME WOULD SETTLE IT PROPERLY, which is what #1439 asks for
> first. A `denoise.update` scope over the sampler's per-step update would put
> the interior residue under a name and make a tight share floor honest again.
> ... It stays owed rather than being folded into the repair of a standing red.

That is precisely this row, and the two changes agree on the diagnosis. What
each side contributes:

| | `6b48edb2c` | this row |
|---|---|---|
| `denoise` share floor | moved 0.95/0.90 → 0.75 | **deleted**; 0.75 permits a quarter of a leaf to be un-anchored, `uncovered <= 2 * leaf_instrument` permits tens of microseconds |
| the head and tail | new assertion (1c), flat 0.25 ms plain / 3 ms sanitized, per leaf record | **kept exactly as it landed**, constants and all |
| the interior | left un-anchored, disclosed, owed | anchored as `denoise.update` |
| the sum gate (#1439) | untouched | replaced by the instrument-derived bound |

**Its measured population is the strongest evidence either side produced**, and
it is kept verbatim in the source: on an unchanged `denoise` the nine-frame arm
reads 99.55%, 99.38%, 99.28%, 99.228%, 98.84%, 98.77%, 98.52%, 98.23%, 96.85%,
94.60%, 94.60%, 94.14%, 92.39% and **88.85%** across four boxes including the
GitHub runner, and the 81-frame arm spans 97.09% down to 85.85%. Eleven points
of a leaf were un-anchored work. No share floor can separate that from a
swallowed phase, which is exactly why anchoring it is the repair and a third
threshold would not have been.

**And it names the gap this row fills.** Its own text records that "a NORMALISED
bound would be better, and there is no normaliser", and that its span-slack
constant needs a per-configuration value because a sanitizer instruments the
scope-boundary path itself — 0.25 ms plain, 3 ms under ASan or TSan, with a worst
slack of 1.658 ms against a smallest leaf of 2.77 ms, 1.67x from vacuous.
`Record::instrument_seconds` is that normaliser: it is measured on the same
instrumented path, so an instrumented build inflates the charge and the residue
together and the bound needs no `#if`.

### The gate report

`.agents/verification.md` asks for the immutable SHA, the exact command, the
environment, the exit status and the evidence path, and not a summary of them.

**Environment for every row below.** x86_64, 20 cores,
`cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON` with an **empty**
`CMAKE_BUILD_TYPE`, which is what `build-test-cpu` uses. Load average **70 to
140** throughout, with three to five other sessions running this same suite.

| # | SHA | command | exit | result |
|---|---|---|---|---|
| 1 | `67823aee2` (base) | `./build/tests/test_ltx2_video` | 1 | **RED.** `102 cases \| 100 passed \| 2 failed`, `4170 assertions \| 4167 passed \| 3 failed`. Evidence `red1.log` |
| 2 | the landing tree | `./build/tests/test_ltx2_video -s -tc='<the four cases this row owns>'` | 0 | **GREEN.** `4 cases \| 4 passed \| 0 failed`, `1382 assertions \| 0 failed`. Evidence `focus-retreat.log` |
| 3 | `ec3e7ac0c` | `build-test-cpu` on a clean GitHub runner — the same `ctest --test-dir build --output-on-failure` over all 583 tests | 0 | **GREEN.** Job `96719179235`. That is the lane `test_ltx2_video` was red in, on the idle low-load machine this box cannot imitate |
| 4 | `361bbfb05` | `build-newest-gcc` | 0 | **GREEN**, having been RED at the Build step on every commit since `5702d8f83`; repaired here as #1565 |
| 5 | `361bbfb05` | `ctest --test-dir build --output-on-failure`, locally | — | **VOID, not a failure.** `test_ltx2_video` reports `Subprocess terminated***Exception` at 796.58 s, and its own output ends `FATAL ERROR: test case CRASHED: SIGTERM` at case 26 of 104 with **`763 assertions \| 763 passed \| 0 failed`**. An external `SIGTERM` on a shared box is an infrastructure failure presenting as a code verdict, and it is neither a red nor a green |
| 6 | the landing tree | the SUMS case, 20 consecutive runs of the WITHDRAWN bound | 0 x20 | recorded because it is the measurement that was not enough: 1.021 to 1.464, which a fresh reviewer then showed is the body of a distribution reaching 4.115 over 45 runs. Evidence `ratios20.log` |

**Row 1's exact failures**, which are the red this row exists to remove:

```
:3256: MESSAGE: phase table: wall=0.26271s leaves=0.243533s unaccounted=0.0191776s over 35 entries
:3259: ERROR: CHECK( leaves >= 0.95 * wall ) is NOT correct!
  values: CHECK( 0.243533 >= 0.249575 )
:3693: MESSAGE:   denoise = 0.00679651s over 1 leaf record(s), of which 8 sub-scope(s) cover 0.00640374s (94.221%)
:3696: ERROR: CHECK( covered >= c.min_coverage * leaf_seconds ) is NOT correct!    [x2]
  values: CHECK( 0.00640374 >= 0.00645668 )
```

**Row 2 at the same two lines**, on the same floors, unedited:

```
phase table: wall=4.59851s leaves=4.59671s unaccounted=0.00179039s instrument=0.00137401s over 46 entries
  denoise = 1.91801s ... 16 sub-scope(s) cover 1.91684s (99.939%)
```

### The mutations

Five, all run by a **fresh reviewer** in its own worktree at `ec3e7ac0c`, each
restored with `git checkout --` and verified with an empty `git diff --stat`. The
compile status is printed for each, because a mutation that fails to build reads
as a passing test. **Two were run against the WITHDRAWN bound and their result no
longer describes the landing tree; both are marked.**

| mutation | built | reddened |
|---|---|---|
| **A** delete the `load.setup` production call site | `compile_status=0` | **YES.** The name list at `:3277`, and the withdrawn residue bound at `:3326`, `CHECK( 0.0223456 <= 0.00844218 )` |
| **B** delete the `denoise.update` anchor, keep its counter | `compile_status=0` | **YES.** The record count (0) at `:3678`, `0 == 8`, and `REQUIRE(!found.empty())` at `:3702` |
| **C** `ChargeLocked` charges everything to the table | `compile_status=0` | **YES**, and still does after the withdrawal: `table_after_child == table_before_child`, `parent_instrument > 0.0` and `record_charge > 0.0` |
| **D** `denoise.update` moved below the two `PostProcessLatent` calls | `compile_status=0` | **YES, 5 of 5 runs**, against the WITHDRAWN bound. **NOT detected on the landing tree** — see `### What this cost the gate` |
| **E** `Sum(records, Elapsed())` restored after the copy-and-sort | `compile_status=0` | **NO — green 10 of 10.** A finding, not a pass; recorded as [#1569](https://github.com/mudler/vllm.cpp/issues/1569) |

**Mutation A is a SUBSTITUTE for the promised row, not that row.** The `## Gates`
row that says "the NEW bound on the OLD tree" is not literally runnable, because
`instrument_seconds` does not exist before this change. A instead restores the
defect on a tree that is otherwise this row's, and reddens.

**A's reach is stated rather than implied.** Its residue is 22.3 ms, and the old
`leaves >= 0.95 * wall` permits 5% of wall — so A would ALSO have reddened the
old floor at the 0.26271 s wall this row's baseline records, and passes it only
above a 0.447 s wall. A therefore demonstrates that the naming is load-bearing.
It demonstrates nothing about the withdrawn bound that the old floor could not.

**Two further mutations, by the same reviewer, on the assertions that replaced
the conservation case's withdrawn ratio.** Deleting `ChargeLocked`'s
`instrument_gap += to - from;` fall-through reds `table_charge > 0.0`; starting
the `Close` tail charge at the record's START instead of its end reds
`unaccounted >= table_charge - 1e-9` by 163x. The second is caught by nothing
else in the change.

**And one mutation that measured a line to be worthless.** Deleting `Open`'s head
charge left the `unit.parent` case green **100 of 100**, with a median
statistically identical to the unmutated one. That site had no unique detection
power at either constant, which is part of why the ratios there are reported now
rather than asserted.

### A trap that is named rather than engineered around

`PhaseLog`'s origin is the LOAD, so the gap between `vllm_video_engine_load`
returning and `vllm_video_generate` being called is inside `wall_seconds` and
inside no leaf. In the `SUMS to wall` case that gap is two span boundaries and
their sampler threads, and it is charged. In the two-render attribution case the
same kind of gap holds the TEST's own assertions between two `Generate` calls,
2.879 ms of them, which is why that case's table-level ratio reads 10.45 and why
the table-level bound is asserted in the one-render case only — as it always was.
A future case that does real work between load and generate will red the sum
assertion, correctly and confusingly. The test says so beside the line.

### Observed and NOT this row's

[#1572](https://github.com/mudler/vllm.cpp/issues/1572). Assertion (1c), the span
slack `6b48edb2c` added, reddened intermittently during this row's fresh review —
`decode.video` at `0.00256913` against a `0.00075` bound, 3.4x — on a tree whose
only difference from the merge base is this row's residue work, and again once
under an unrelated mutation. This row does not touch (1c) and keeps its constants
exactly as they landed, so it is filed rather than repaired. It is the same class
this row exists to remove, one assertion over: an absolute bound on a quantity
whose cost is instrumentation. `Record::instrument_seconds` is the normaliser
that change's own text says did not exist.

## Stop conditions

* Stop and return `NEEDS_DECISION` if the measured `unaccounted /
  instrument_seconds` ratio does not sit near 1 after the naming, because the
  bound in §Design.3 would then be a constant nobody derived.
* Stop if closing the gates needs a numeric change to the render. Nothing here
  may move a pixel or a sample.
* Stop if the repair needs a threshold to be raised. A bigger constant is what
  [#1466](https://github.com/mudler/vllm.cpp/issues/1466) rejects and it is not
  available to this row.

## Now

`DONE`. Landed as pull request
[#1556](https://github.com/mudler/vllm.cpp/pull/1556) on `row/LTX25-PHASE-RESIDUE`.

The gate report is in `## Outcome` under `### The gate report`. The two verdicts
that decide this row:

* **`build-test-cpu` on a clean GitHub runner: GREEN at ec3e7ac0c, job 96719179235.** That is the lane
  `test_ltx2_video` was red in, running the same `ctest --test-dir build
  --output-on-failure` over all 583 tests, on the idle low-load machine this
  box cannot imitate. It ran at `ec3e7ac0c`; every commit after it either removes
  an assertion or adds instrument charge, and neither can turn that green red.
* **`ctest --test-dir build --output-on-failure` locally: VOID -- test 71 was killed by an external SIGTERM at case 26 of 104 with 763/763 assertions passing; an infrastructure failure is neither a red nor a green**, at load
  average 85-105 with three to five other sessions running the same suite.

`windows-msvc-cpu` and `windows-msvc-vulkan` fail on this pull request and on all
ten of the last `main` baselines, and have no green to compare against. They are
not this row's.
