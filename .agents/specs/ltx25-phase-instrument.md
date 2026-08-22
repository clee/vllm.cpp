# `LTX25-PHASE-INSTRUMENT` — the phase table measures its own cost, and says where its residue is

Issues: [#1668](https://github.com/mudler/vllm.cpp/issues/1668) (item 4 of four),
[#1569](https://github.com/mudler/vllm.cpp/issues/1569),
[#1571](https://github.com/mudler/vllm.cpp/issues/1571).
Record this row implements from:
[`ltx25-phase-residue.md`](ltx25-phase-residue.md).
Related and deliberately NOT closed here:
[#1439](https://github.com/mudler/vllm.cpp/issues/1439),
[#1567](https://github.com/mudler/vllm.cpp/issues/1567),
[#1568](https://github.com/mudler/vllm.cpp/issues/1568),
[#1570](https://github.com/mudler/vllm.cpp/issues/1570).

This row has **no matrix row and therefore no lifecycle state**, for the reason
`ltx25-phase-residue.md` records for itself: the phase log is an instrument
inside the LTX-2.5 driver, and no matrix in this tree keys instruments. Its state
is the issues it closes and the tests that hold it.

## Scope

[`ltx25-phase-residue.md`](ltx25-phase-residue.md) is a RECORD of work that was
measured, gate-run and reviewed three times, and then closed unmerged. This row
lands the part of it that is about the INSTRUMENT rather than about the LTX-2.5
driver, and it lands the two gaps that record filed against itself.

IN SCOPE:

1. **`Record::instrument_seconds` and `PhaseLog::Instrument()`** — the
   instrument charging its own out-of-record wall to the innermost live non-span
   record, and to the table when none is live. #1668 item 4.
2. **`WriteJson` reads its clock before it serialises**, plus the gate that
   holds it. #1569, and #1668 item 4's second half.
3. **The residue decomposed into the gaps between adjacent leaves**, in the
   emitted file. #1571.

OUT OF SCOPE, and each is named because each was tempting:

- **The three driver anchors** — `load.dit_config`, `artifacts.mux`,
  `denoise.update` and `Ltx2ConditioningTrace::sampler_updates`. They are #1668
  items 1 to 3, they touch `ltx2_video.cpp` and the `Carrying` table in
  `test_ltx2_video.cpp`, and #1568 and #1570 both need `denoise.update` to exist
  before they can be closed. They are one unit and this is not it. They land in
  the follow-on row, which is why #1668 stays open here.
- **Widening either floor.** `leaves >= 0.95 * wall` and
  `covered >= min_coverage * leaf_seconds` are untouched by this row, in both
  their form and their constants.
- **Any bound with `instrument_seconds` in a denominator.** See `## Design` 4.

## Our baseline

Everything measured here is already measured, in
[`ltx25-phase-residue.md`](ltx25-phase-residue.md) `## Our baseline`, and is not
re-derived. What that record establishes and this row builds on:

- 92% of a 19.178 ms residue on the 64x64x9 fixture is ONE gap, the load's
  prologue. The sixteen gaps between adjacent named phases hold 6.8 us each.
- The residue does NOT scale with wall — about 1 ms across walls of 0.8 s to
  4.6 s, and 0.82 to 86 ms across 10 s to 120 s. A share-based bound is therefore
  worst at the SMALLEST wall.
- `residue <= 2 * instrument` was measured across three fresh reviews and
  WITHDRAWN. It is not re-proposed. `## Design` 4 states what replaced it.

What is NEW here is one measurement, and it is the one #1569 asks for: what the
copy-and-sort inside `WriteJson` costs, and how far the two clock orderings are
apart once the table is large enough for that copy and that sort to exist. It is
in `## Evidence`.

## Design

### 1. The instrument charges its own wall

`PhaseLog::Open` stamps `o.start` AFTER taking the process-wide mutex, so the
mutex wait precedes the record. `PhaseLog::Close` stamps `r.end` BEFORE it emits
its progress line and erases the entry, so that tail follows the record. Both
land outside every record, and until now nothing could tell them from a phase
nobody named.

The rule is one sentence: **every interval of the instrument's own wall is
charged to the innermost live non-span record at the moment it is spent, and to
the table when none is live.** Spans are excluded because `Sum` excludes spans,
so time inside a span but outside a leaf is exactly the residue; charging it to
the enclosing `load` or `generate` span would hide it in a number nothing adds
up.

Ported from `refs/pull/1556/head` = `b45ea3bbb`, which measured, gate-ran and
three-times-reviewed this mechanism. The port is behaviour-identical. What is
NOT ported is that branch's `load.setup` anchor, which `519303d15` already landed
on `main` as `load.open`, and its withdrawn bound, whose constant is deleted
rather than raised.

### 2. `WriteJson` reads its clock first

`Sum(records, Elapsed())` after `ByStart(Records())` charges the WRITER's copy
and sort to the RENDER's wall, and therefore to `unaccounted_seconds`. The clock
read moves to the first statement of the function.

### 3. The residue is decomposed in the file

The leaves `Sum` adds are non-overlapping — `Open` marks a leaf `nested`
whenever another leaf is live — and `ByStart` orders them. So the complement of
their union inside `[0, wall]` is exactly `wall - sum_leaf_seconds`. The emitter
writes that complement as `gaps`: one interval before each leaf, one after the
last, each carrying the two names it lies between.

**This is the row's best gate, and the reason is that it is not a measurement.**
The gaps add to `unaccounted_seconds` by construction. A gate over that sum is
arithmetic over numbers already in the file, so no box load can move its verdict.
Every other assertion this table has ever carried was a ratio of two wall-clock
quantities, and two of them spent three months being argued about.

### 4. What replaces the withdrawn bound, and what does not

**Nothing in this row puts `instrument_seconds` in a denominator.** That is the
single most important sentence here, and [`ltx25-phase-residue.md`](ltx25-phase-residue.md)
`## Design` 3 is the evidence: the un-instrumented remainder of a boundary
dilates FASTER than the instrumented part under contention, so a residue measured
against the instrument's own charge has a heavy right tail — 4 red in 45 runs at
load 88 with a maximum of 4.115, and 28 in 160 at load 125 reaching 5.55.

`instrument_seconds` is therefore emitted and REPORTED, never asserted against.
A reader subtracts it before calling a residue a phase nobody named. The two
floors keep `wall` and `leaf_seconds` in their denominators, which is the better
conditioning: those grow with contention exactly when a preemption inflates the
numerator.

### 5. The one new bound, and how it is derived

#1569 needs a gate, and a gate needs a comparison. The comparison is
`head < 0.5 * serialize`, where both quantities are measured in the same run:

- `head` is `wall_seconds` as the writer recorded it, minus the elapsed clock the
  test read immediately before calling the writer. Under the correct ordering it
  contains one function call and one uncontended mutex — the instrument's own
  resolution. Under the mutated ordering it contains one whole copy of the record
  vector and one whole `stable_sort` of it.
- `serialize` is that same copy and that same sort, performed by the test through
  the same public `Records()`, on the same data, on this box, in this run.

So the constant is not a tolerance. Under the correct order the head holds ZERO
copies and ZERO sorts; under the mutated order it holds exactly one of each and
is therefore at least `1.0 * serialize` **by the definition of the two
quantities**. Any constant strictly inside `(0, 1)` separates them. 0.5 is the
midpoint, and the measured separation is five orders of magnitude, not a factor
of two.

**The estimator is a MINIMUM over K probes, and that is what makes this not the
withdrawn bound wearing a new name.** Contention is one-sided: it can only make
a measured interval longer. The honest head is a floor near the clock's
resolution plus a preemption that sometimes lands in it; the mutated head has a
HARD floor of one serialization, present in every iteration. A minimum over K
strips the sporadic term from the honest side and cannot strip the deterministic
term from the defective side. The withdrawn bound compared two single
measurements of comparable magnitude and the tail decided it. This compares the
minima of two populations that differ by five orders of magnitude.

`serialize > 1e-5` guards the comparison from the other side. A table too cheap
to serialise cannot separate the two orderings at all, which is precisely why
#1569's three-record case stayed green 10 of 10 under its own mutation. A
precondition that fails loudly is the difference between a gate and a mute
switch.

## Dependencies

None. `LTX25-DEVICE-RESIDENCY` owns `render_phase_log.{h,cpp}`'s existence and
the `load.open` anchor; this row extends the instrument and renames nothing.

## Risks and decisions

**D1 — the instrument's cost is CHARGED, never subtracted globally.** A single
global subtraction is a number nobody can attribute. Charging each interval to
the innermost live non-span record keeps the attribution local and makes the
conservation invariant testable, which is what the unit cases assert.

**D2 — a new test executable rather than a block in `test_ltx2_video`.** Two of
the four cases need a table of thousands of records, which no render produces,
and `test_ltx2_video` costs a fixture build and has been measured at 30-36 GB of
anonymous resident set. Three other issues are editing that file concurrently.
The instrument's own cases go in `tests/vllm/multimodal/test_render_phase_log.cpp`.

**D3 — reachability stays in `test_ltx2_video`.** Every case in the new file
calls `PhaseLog` directly, which proves the class works and never that a render
reaches it — the exact failure `AGENTS.md` "Nothing lands dead" names. The
assertion that `vllm_video_generate`'s own table carries `instrument_seconds` and
a reconciling `gaps` is added to `a render through the ABI emits a phase table
that SUMS to wall`, and it is the only thing this row adds to that file.

**D4 — a negative gap is emitted rather than clamped.** It cannot arise while
the non-overlap invariant holds, so clamping would hide a broken instrument
inside a number that still adds up. The unit case asserts it is never negative.

**D5 — `#1668` is NOT closed by this row.** It owns four items and this row
lands one. Closing it on the strength of item 4 would lose items 1 to 3, which
is the failure #1668 was filed to prevent.

## Tests

`tests/vllm/multimodal/test_render_phase_log.cpp`, four cases:

| Case | What it holds | Shape |
|---|---|---|
| the instrument charges its own cost to the innermost LEAF | attribution: a child's boundary is the parent's cost, a boundary under a bare span is the table's, a span is not a leaf | "it moved", "it did not move at all", "it is positive" — no duration compared |
| the instrument's own cost is CONSERVED across the table and its records | every charge non-negative, no record charged past its own duration, the table's share no larger than the residue it is part of | inequalities between two numbers in the same file |
| the emitted table DECOMPOSES its residue into the gaps between leaves | N leaves give N+1 gaps, each names the two leaves it lies between, none is negative, and they SUM to `unaccounted_seconds` | an accounting identity, plus one lower bound on a `sleep` |
| the emitter reads its CLOCK before it serialises the table | #1569 | `## Design` 5 |

Plus, in `tests/vllm/multimodal/test_ltx2_video.cpp`, inside the existing ABI
render case: the emitted table carries `instrument_seconds`, carries `gaps`, and
those gaps reconcile to that render's own `unaccounted_seconds`. That is D3.

## Gates

`ctest --test-dir build -R 'test_render_phase_log|test_ltx2_video'`, plus
`scripts/agent-preflight.sh`.

`main` is RED on its own baseline at `019f66c1a` — `build-test-cpu`, both
`sanitize-cpu` arms and both `windows-msvc-*` — so inheritance is established by
FAILURE TEXT rather than by job name.

## Stop conditions

- Do not widen either floor to close a red. Name the phase, or leave the red and
  file the gap.
- Do not put `instrument_seconds` in a denominator without reading
  [`ltx25-phase-residue.md`](ltx25-phase-residue.md) `## Design` 3, and never
  accept a 20-run distribution as evidence about a quantity of that shape.
- Do not close #1668 from this row. D5.
- Do not close #1439. It asks for a bound on a quantity the scheduler cannot
  move; the gap decomposition gives a reader that quantity and asserts an
  identity over it, and neither is the budget #1439 asks for.

## Owed

| Issue | Owed |
|---|---|
| [#1668](https://github.com/mudler/vllm.cpp/issues/1668) | items 1 to 3, the three driver anchors and `sampler_updates`. Item 4 lands here |
| [#1570](https://github.com/mudler/vllm.cpp/issues/1570) | the bound on `instrument_seconds / duration_seconds` per record. It needs `instrument_seconds`, which this row lands, and it is worth setting on the anchors rather than on this row's synthetic scopes — a bound on `unit.child` measures nothing anybody ships |
| [#1568](https://github.com/mudler/vllm.cpp/issues/1568) | the `denoise.step` / `denoise.update` seconds transfer. `denoise.update` does not exist yet |
| [#1567](https://github.com/mudler/vllm.cpp/issues/1567) | the res_2s arm's anchor. No gate in this tree renders on that arm |
| [#1439](https://github.com/mudler/vllm.cpp/issues/1439) | NOT closed. See `## Stop conditions` |

## Evidence

Filled by the run that lands this row. See `## Outcome`.

## Now

Spec committed. Implementation follows in the same pull request, and the commit
order proves the spec came first.
