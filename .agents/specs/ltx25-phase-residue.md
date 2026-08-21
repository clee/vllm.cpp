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

**Why it is not load-flaky, which is the property the four issues are about.**
The old ratio compares a residue that a preemption inflates against a wall that
the same preemption inflates only if it lands inside a leaf; the residue is 0.5%
of the timeline, so 99.5% of preemptions help the ratio and 0.5% destroy it, and
that is the coin flip #1439 and #1494 measured. The new bound compares the gap
against the instrument's own measurement OF THAT GAP. A preemption inside a gap
lands inside an instrument entry point with overwhelming probability — the
un-instrumented part of an adjacent-scope gap is a call and a return — so it
inflates both sides of the comparison together.

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
| [#1494](https://github.com/mudler/vllm.cpp/issues/1494) | closed by this row |
| [#1470](https://github.com/mudler/vllm.cpp/issues/1470) | closed by this row |
| the res_2s arm's `denoise.update` anchor | **owed, no issue yet.** `Ltx2Res2sDenoisingLoop` runs its own post-process and step inside `ltx2_res2s.cpp` through `Ltx2Res2sHooks`, so the anchor needs a hook rather than a statement. No gate in this tree renders on that arm, so landing it here would land dead code |
| a per-gap decomposition IN the emitted table | **owed, no issue yet.** This row computed the gap table in a scratch script to find the 92% region. A reader of `phase-log.json` still cannot see it without one, and the same investigation will be re-derived the next time the residue moves |

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
at most as large as the part it can. Measured, on the loaded box described above:

| where | uncovered / charged |
|---|---:|
| the table, one-render case | **1.32** |
| `denoise`, 9 frames | 1.08 |
| `denoise`, 81 frames | 1.10 |
| `decode.audio`, 9 frames | 1.08 |
| `decode.audio`, 81 frames | 1.08 |
| `decode.video`, 9 frames | 1.26 |
| `decode.video`, 81 frames | 1.22 |
| `artifacts.frames`, 9 frames | 1.41 |
| `artifacts.frames`, 81 frames | 1.25 |
| the `unit.parent` unit case | 1.21 |

The largest is 1.41 and most sit near 1.1. The spec's stop condition was that a
ratio far from 1 would mean the instrument does not measure enough of its own
gap and the bound would be a constant nobody derived; it is not, and the join
charge above is the one place where it WAS and was repaired rather than absorbed
into a larger factor.

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
