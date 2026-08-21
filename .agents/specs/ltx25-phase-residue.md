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
4. **Replaces both wall-clock ratios with a bound derived from that measured
   cost.** This is the part that closes the class rather than one member.

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

### 3. Both ratios are replaced by a bound derived from §1

Today:

```cpp
CHECK(leaves >= 0.95 * wall);                       // :3259
CHECK(covered >= c.min_coverage * leaf_seconds);    // :3696
```

After:

```cpp
CHECK(unaccounted <= kInstrumentBudget * instrument_seconds);
CHECK(leaf_seconds - covered <= kInstrumentBudget * leaf_instrument_seconds);
```

**The derivation.** A gap between two adjacent records is the closing record's
tail plus the opening record's head plus whatever the caller ran between them.
The instrument measures the first two. `kInstrumentBudget = 2` states that the
part it cannot measure — the `Close` return, the caller's own statements, and the
`Open` call — is at most as large as the part it can. Nothing in the constant is
a share of the render, so the assertion says the same thing at 64x64x9 and at
3840x2160x241. The number is checked against measurement in §Gates: the ratio
`unaccounted / instrument_seconds` is reported for every gate run and the bound
is only defensible while that ratio sits near 1.

**Why this is stricter and not looser, which is the rule this row must not
break.** At the fixture scale of the failing run, `0.95 * wall` permits 13.1 ms
of un-named time. The new bound permits about twice the measured instrument cost
of ~0.3 ms. At the 21 B render the instrument exists for — `ltx25-decode-speed.md`
records rungs measured in hours — `0.95 * wall` permits **minutes** of time
nobody named, while the new bound still permits milliseconds. The replacement is
five orders of magnitude tighter at the scale the tolerance was originally
argued for. The same argument holds for the coverage bound: 0.95 of a
`decode.audio` leaf permits 83 ms of that leaf to be un-anchored at fixture
scale and 0.90 of `denoise` at 81 frames permits 3.9 ms.

**It is an empirical claim and not a theorem, which a fresh review was right to
say.** `uncovered <= 2 * leaf_instrument` beats `covered >= 0.75 * leaf_seconds`
only while `2 * leaf_instrument < 0.25 * leaf_seconds`, and nothing bounds
`leaf_instrument` from above. `Tick` charges to the innermost live record, so
moving the DiT tick out of `Evaluate` would charge about 110 flushed writes to
`denoise` and buy a budget larger than the deleted floor. The ratio is emitted as
a `MESSAGE` rather than asserted, so a change that made the instrument ten times
more expensive would widen both gates and print a small number. That is a real
direction of drift, and it is recorded under `## Owed` rather than papered over.

**Why it is not load-flaky, which is the property the four issues are about —
and the argument this row first gave for it, which is WRONG.**

The old ratio compares a residue that a preemption inflates against a wall that
the same preemption inflates only if it lands inside a leaf; the residue is 0.5%
of the timeline, so 99.5% of preemptions help the ratio and 0.5% destroy it, and
that is the coin flip #1439 and #1494 measured. That half stands.

This spec then argued that the new bound is immune because *"a preemption inside
a gap lands inside an instrument entry point with overwhelming probability — the
un-instrumented part of an adjacent-scope gap is a call and a return — so it
inflates both sides of the comparison together."* **A fresh review measured that
and it is false.** Decomposing a bare micro-timeline's uncovered time: fast, its
inter-child gaps are 9-20 us over seven boundaries against a 13-22 us charge;
slow, 91-105 us against 52-61 us. The un-instrumented part dilates FASTER. The
same review reddened the unit case 2 times in 200 at load 85 and 28 in 160 at
load 125.

**What actually conditions the render-level bounds is different, and it is
measured rather than argued.** A render's boundaries carry a `Tick` and a
`/proc/self/statm` read INSIDE the instrumented region, so the measured part of
each gap dominates the part that is not, and the ratio stays near 1: 20 of 20
runs of the table bound at 1.021 to 1.464, and the eight leaf measurements at
1.02 to 1.46 by two independent measurers. Eight bare scopes carry neither, which
is why the unit case reports the ratio and no longer asserts it. The claim this
row makes is therefore empirical and scoped to the render, not a property of the
comparison in general, and `## Outcome` carries the distribution that supports
it.

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

### The measured ratios, which are what makes the bound defensible

`kInstrumentBudget = 2` says the part of a gap the instrument cannot measure is
at most as large as the part it can. That is only worth anything as a
distribution, so here is one, at `37f7f9aca` on the loaded box described above.

**The table bound, 20 consecutive runs of the SUMS case at load average 94-102,
20 of 20 green and every run reporting `cases=1`:**

| min | median | max | bound |
|---:|---:|---:|---:|
| 1.021 | 1.065 | **1.464** | 2 |

**The four carrying leaves, both geometries, one run:**

| leaf | 9 frames | 81 frames |
|---|---:|---:|
| `denoise` | 1.109 | 1.068 |
| `decode.video` | 1.220 | 1.163 |
| `decode.audio` | 1.056 | 1.071 |
| `artifacts.frames` | 1.257 | 1.196 |

A fresh review measured the same eight quantities independently at `ec3e7ac0c`,
before the `Open`-progress-line charge landed, and read 1.02 to 1.46.

**And the one place the ratio does NOT behave, which the same review found and
which is why the unit case no longer asserts it.** The `unit.parent`
micro-timeline reddened 2 of 200 consecutive runs at load 85, and a standalone
probe of the same shape reddened 28 of 160 at load 125, reaching 5.55, and 14.1
under `address,undefined`. Decomposing that parent's uncovered time explains it:
fast, its inter-child gaps are 9-20 us over seven boundaries against a 13-22 us
charge; slow, 91-105 us against 52-61 us. **The un-instrumented part of a
boundary dilates faster than the instrumented part when the box slows**, which is
the opposite of §Design.3's claim that a preemption inflates both sides together.

That claim is therefore wrong as stated, and what saves the render-level bounds
is not it: a render's boundaries carry a `Tick` and a `/proc/self/statm` read
INSIDE the instrumented region, so the measured part dominates and the ratio
stays near 1. Eight bare scopes carry neither. The gates keep the bound because
their measured distribution supports it over 20 runs and two independent
measurers; the unit case reports the ratio and does not assert it, because at
that scale the quantity is not well conditioned. The stop condition this row set
itself — that a ratio far from 1 makes the bound a constant nobody derived — is
met at the render and is NOT met at the micro scale, and both halves are written
down here rather than only the convenient one.

### What is stricter, stated as the numbers at both ends

| | old gate permits | new gate permits |
|---|---|---|
| the fixture, wall 0.26 s | 13.1 ms un-named | ~0.6 ms, twice the measured charge |
| a 2.5 h render | **7.5 minutes** un-named | milliseconds, twice the measured charge |
| `decode.audio` at 0.99 | 83 ms un-anchored at fixture scale | ~0.13 ms |
| `denoise` at 0.90, 81 frames | 3.9 ms un-anchored | ~0.5 ms |

The replacement is tighter at fixture scale and five orders of magnitude tighter
at the scale the tolerance was originally argued for. That is the whole defence
for replacing an assertion rather than editing its constant, and it is why this
change is not the thing `AGENTS.md` forbids.

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
The `## Gates` table above states intentions, which is what a fresh review
correctly rejected as evidence. This is the record.

**Environment for every row below.** x86_64, 20 cores,
`cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON` with an **empty**
`CMAKE_BUILD_TYPE`, which is what `build-test-cpu` uses. Load average **94 to
140** throughout, with three to five other sessions running this same suite. That
is stated because it is the subject: the two assertions this row replaced decided
by that number.

| # | SHA | command | exit | result |
|---|---|---|---|---|
| 1 | `67823aee2` (base) | `./build/tests/test_ltx2_video` | 1 | **RED.** `102 cases \| 100 passed \| 2 failed`, `4170 assertions \| 4167 passed \| 3 failed`. Evidence `red1.log` |
| 2 | `37f7f9aca` | `./build/tests/test_ltx2_video -s -tc='<the four cases this row owns>'` | 0 | **GREEN.** `4 cases \| 4 passed \| 0 failed`, `1381 assertions \| 0 failed`. Evidence `focus-repaired.log` |
| 3 | `37f7f9aca` | the SUMS case, 20 consecutive runs | 0 x20 | **GREEN 20/20**, every run reporting `cases=1` so the filter is not silently empty. Table ratio min **1.021**, median **1.065**, max **1.464** against the bound of 2, at load 94-102. Evidence `ratios20.log` |
| 4 | `37f7f9aca` | `ctest --test-dir build --output-on-failure` | see `## Now` | the full gate |

**Row 1's exact failures**, which are the red this row exists to remove:

```
:3256: MESSAGE: phase table: wall=0.26271s leaves=0.243533s unaccounted=0.0191776s over 35 entries
:3259: ERROR: CHECK( leaves >= 0.95 * wall ) is NOT correct!
  values: CHECK( 0.243533 >= 0.249575 )
:3693: MESSAGE:   denoise = 0.00679651s over 1 leaf record(s), of which 8 sub-scope(s) cover 0.00640374s (94.221%)
:3696: ERROR: CHECK( covered >= c.min_coverage * leaf_seconds ) is NOT correct!    [x2]
  values: CHECK( 0.00640374 >= 0.00645668 )
```

**What row 2 reports where row 1 failed.** `denoise` coverage moves from
**94.221% to 99.9685%** at nine frames and to 99.9872% at 81, because the
sampler's per-step update now has a name. The four carrying leaves read, as
`uncovered / leaf_instrument`:

| leaf | 9 frames | 81 frames |
|---|---:|---:|
| `denoise` | 1.109 | 1.068 |
| `decode.video` | 1.220 | 1.163 |
| `decode.audio` | 1.056 | 1.071 |
| `artifacts.frames` | 1.257 | 1.196 |

A fresh review measured the same eight quantities independently at
`ec3e7ac0c`, before the `Open`-progress-line charge landed, and read 1.02 to
1.46. The bound is 2.

### The mutations

Five, all run by a **fresh reviewer** on `ec3e7ac0c` in its own worktree, each
one restored with `git checkout --` and verified with an empty `git diff --stat`.
The compile status is printed for each, because a mutation that fails to build
reads as a passing test.

| mutation | built | reddened |
|---|---|---|
| **A** delete the `load.setup` production call site | `compile_status=0` | **YES.** The name list at `:3277` **and** the new residue bound at `:3326`, `CHECK( 0.0223456 <= 0.00844218 )` |
| **B** delete the `denoise.update` anchor, keep its counter | `compile_status=0` | **YES.** The record count (0) at `:3678`, `0 == 8`, and `REQUIRE(!found.empty())` at `:3702` |
| **C** `ChargeLocked` charges everything to the table | `compile_status=0` | **YES.** Four assertions: `:4978`, `:5004`, `:5020`, `:5086` |
| **D** `denoise.update` moved below the two `PostProcessLatent` calls | `compile_status=0` | **YES, on one of the two renders.** The coverage bound at `:4122`, ratio 2.40 on one and 1.28 on the other |
| **E** `Sum(records, Elapsed())` restored after the copy-and-sort | `compile_status=0` | **NO — green 10 of 10.** Recorded under `## Owed`, not explained away |

**Mutation A is the red-before evidence for the replacement**, and it is the row
that `## Gates` promised and did not have. It reddens the NEW bound by 2.6x on a
tree where the naming is gone and everything else is this row's, which is the
same defect the old ratio was firing on — and it does so at a wall where the old
ratio's 5% would have passed.

**Mutation D is honest about its reach.** It reds on one render of two. The
anchor stops covering the post-process, the uncovered part grows by exactly that
work, and on the render where the leaf is larger the growth is still inside the
budget. That is a partial detection, not a full one, and it is the same shape the
`part_min_coverage` note already records for a half-covered anchor.

**Mutation E is a finding, not a pass.** It is F2 above: `WriteJson`'s clock
ordering is a reading of the source and no assertion sees it.

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

`ACTIVE`. Spec committed; implementation follows in the same pull request, in
commit order.
