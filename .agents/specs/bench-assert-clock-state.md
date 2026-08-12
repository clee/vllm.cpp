# BENCH-ASSERT-CLOCK-STATE — a ratio may not be quoted without the clock it was measured at

Issue: [#543](https://github.com/mudler/vllm.cpp/issues/543) (the defect —
per-call attribution is not reproducible across box states),
[#545](https://github.com/mudler/vllm.cpp/issues/545) (the reboot blocker that
makes cross-boot comparison the *normal* case rather than the exception)
Row: `BENCH-ASSERT-CLOCK-STATE`
Prior art: [#375](https://github.com/mudler/vllm.cpp/issues/375) and
[#520](https://github.com/mudler/vllm.cpp/issues/520) — the same class. An
environment variable nobody recorded silently repriced every number, and the
harness could not tell because it never wrote the variable down.

## The defect

On `dgx.casa` (GB10, driver `580.159.03`) the SM clock differs **between boots**
and is not throttling — `clocks_throttle_reasons.active = 0x0`, persistence
`Enabled`:

| boot | SM clock over the captured window | our ms/step |
|---|---|---|
| `f6bbbfc6` | n=61, min 2398 / **med 2470** / max 2489 | **82.1664** |
| `2fca2b02` | n=50, **flat 2190** (`clocks.max.sm` 3003, applications 2418) | **88.1000** |

A **12.79%** median-clock delta produced **+7.22%** step time. The control is
what settles it: `marlin::Marlin`, 129 calls/step, byte-identical invocation,
**no source change** between `a170c81c` and `4064558d0`, moved
**45.2845 → 49.6544 ms/step = +9.65%**.

That control drift is **larger than either deficit it was used to rank** —
`in_proj` +2.97%, `out_proj`/`o_proj` +6.28%. Those two are therefore **NOT
ESTABLISHED**: they were never taken against a clock control. The same effect
explains a same-binary same-arm swing of 382.60 → 357.59 us/call (−6.5%) across
a reboot, and two probes disagreeing ~6% uniformly eight minutes apart *within
one boot* (a 2398 MHz entry against a 1781 MHz one).

Nothing in the tree records any of this. `grep -rn 'clocks\.' tools/ scripts/`
returns one hit, in prose, inside `.agents/benchmark-record.md`. The three
harness scripts that touch `nvidia-smi` capture `-q -d
PERFORMANCE,TEMPERATURE,POWER` into an **unparsed text blob** that no summary
reads, or query `--query-compute-apps` for idleness. No manifest carries a clock
value, a boot id, or a throttle state, so no ratio in this repository can be
attributed to the clock it was measured at.

## Scope

**In scope.**

1. **One helper**, `tools/bench/gpu_clock_state.py`: sample `nvidia-smi`, build
   the per-leg record, validate it fail-closed, and compare two arms. Other
   harnesses import it; it duplicates nothing and defines no framework. It is
   standard-library-only, like `serve_low_common.py`, so its logic runs in CPU
   CI with no GPU and no `nvidia-smi`.
2. **Recording** in the leg-producing harness, `scripts/dgx-online-serving.sh`,
   as a background sampler across the timed bench loop — the same shape as the
   memory sampler that already runs there, written to
   `clocks/<model>/<engine>/r<N>.{samples.jsonl,summary.json}` beside
   `memory/…`.
3. **Asserting** in the ratio-producing surface, `tools/bench/
   online_gate_summary.py`, through the existing `reasons` seam: a violated
   clock contract makes a leg *not binding-eligible*, which is this harness's
   spelling of NOT ESTABLISHED. Every ratio additionally carries a `clock`
   block naming both arms' medians, their offset, and the estimated timing
   effect, so a reader can size the clock against the effect without leaving
   the row.
4. **The operational fix** in `.agents/benchmarking.md`, including the
   shared-host hazard.
5. **A note where existing numbers are cited** that they predate clock
   assertion.

**Out of scope, deliberately.**

- **Editing any recorded number.** AGENTS.md: never delete evidence. The past
  figures stay exactly as they are and gain a note; nothing is restated.
- **Re-measuring.** The GPU is held by another session, `$HOME/gpu.lock` is
  taken, and #545 means the box does not survive a four-leg chain. This row
  makes the *next* measurement attributable; it takes none.
- **Changing the clock.** `nvidia-smi -lgc` is documented here and executed by
  nobody in this row — it is a shared-host mutation (§Operational fix).
- **The trace/per-kernel harnesses** (`finalize_*_trace.py`,
  `summarize_torch_kernels.py`, `gdn_packed_component.py`). They import the same
  helper and the same sampler CLI is what they would call, but wiring each one
  is a separate change with its own fixtures. Recorded as **owed** below rather
  than quietly skipped — and it is the trace path, not the online gate, that
  produced the two retracted findings.

## Design

### What is recorded

Per leg, `clocks/<model>/<engine>/r<N>.summary.json`:

| field | source |
|---|---|
| `boot_id` | `/proc/sys/kernel/random/boot_id` |
| `sm_clock_mhz` | `{n, min, median, max, spread_pct}` over the window |
| `clocks_max_sm_mhz` | `clocks.max.sm` |
| `clocks_applications_graphics_mhz` | `clocks.applications.graphics` |
| `throttle_reasons_active` | sorted union of `clocks_throttle_reasons.active` |
| `persistence_mode` | `persistence_mode` |
| `driver_version`, `gpu_name` | `driver_version`, `name` |
| `idle_samples_excluded` | count of samples with `utilization.gpu == 0` |

`spread_pct` is `(max − min) / median × 100` over the retained samples. The raw
per-sample rows stay in `r<N>.samples.jsonl`, because a summary that cannot be
recomputed from its own evidence is a claim, not a record.

**Idle samples are excluded from the statistics and counted, not dropped.** A
clock read while the GPU is doing nothing did not price any work, and the timed
window necessarily contains the harness's own gaps between concurrency points.
Excluding them silently would be a lie; the count is in the record, and a leg
that is *entirely* idle has `n == 0` and fails validation.

### What is asserted

| assertion | value | how justified |
|---|---|---|
| both arms of a ratio share `boot_id` | exact | cross-boot comparison is what produced the retracted findings; there is no threshold that makes it safe |
| within-run spread | `≤ 5.0%` | see below |
| cross-arm median offset | `≤ 1.0%` | see below |
| `throttle_reasons_active` carries no non-benign bit | mask | a throttled window is not the window the number claims |
| `persistence_mode == Enabled` | exact | already true on the box; its absence changes idle clock behavior |

**Within-run spread, 5.0%.** The admissible band is bounded on both sides by the
data above. It must *accept* the only clean window we have —
`(2489 − 2398) / 2470 = 3.68%` — because a threshold that voids our one good
measurement is useless. It must *reject* the within-boot disagreement that a
2398 MHz entry and a 1781 MHz one represent, which is ~26% however it is
normalized. 5.0 sits just above the clean observation, with ~1.3 points of
headroom so a marginally noisier but still healthy window is not spuriously
voided, and roughly five times below the failure it exists to catch.

**Cross-arm median offset, 1.0%.** The one cross-boot event we have repriced a
byte-identical kernel by **+9.65%** for a **12.79%** median-clock offset: a
transfer of **0.754** percentage points of kernel time per point of clock
(step-level transfer was lower, 0.565; the larger is used). That coefficient is
`n = 1` and is recorded as such — it is used only to *report* an estimated
effect, never as a gate term. At a 1.0% offset the estimate is ≈0.75% of kernel
time, comfortably under the ~2.97% smallest deficit this harness has been used
to rank, so a pair inside the threshold cannot have had its ranking inverted by
clocks.

**The override.** `--allow-cross-boot` exists because #545 makes same-boot
capture of a four-leg chain unreliable, and a gate nobody can satisfy is a gate
everybody routes around. It does not make the comparison clean: it converts the
refusal into a recorded caveat (`cross_boot_override: true` plus the boot ids
and the offset in the ratio's `clock` block and in the report). The spread and
offset assertions still apply — the override waives *identity*, not *state*.

### Why this seam

`online_gate_summary._memory_for_leg` already reads an optional per-leg evidence
artifact and turns every defect into a `reason` that clears
`binding_eligible`. `_clock_for_leg` is the same function shape against the same
seam. Nothing new is invented, no gate is weakened, and a missing clock record
voids a leg exactly the way a missing memory summary already does.

That has a consequence worth stating rather than discovering: **re-summarizing
an existing evidence tree now yields NOT ESTABLISHED**, because no tree on disk
carries a clock record. That is the correct answer — those trees genuinely
cannot be attributed to a clock — and it is why the note in §Records exists
instead of a retroactive edit.

## Risks

- **The gate is unpassable if the spread rule is too tight.** Mitigated by
  excluding idle samples and by choosing the threshold from the clean window
  rather than from theory. If a real leg still exceeds 5%, that is a
  *measurement finding to record*, not a threshold to widen quietly.
- **`nvidia-smi` field names drift.** Driver 580 accepts
  `clocks_throttle_reasons.active`; newer drivers prefer
  `clocks_event_reasons.active`. The sampler queries the throttle spelling, and
  a query failure is a refusal to sample, never a default.
- **The sampler perturbs the measurement.** One `nvidia-smi` per second against
  a 128-token × six-concurrency leg, launched identically on both arms, so it
  cancels in the ratio — the same argument `start_server` already makes for the
  memory sampler, and it is recorded in the same place.

## Tests

`tests/tools/test_gpu_clock_state.py`, synthetic manifests only, no GPU:

- parsing a real `--query-gpu` CSV line, including the `nounits` form
- summary statistics, idle exclusion, and the `n == 0` refusal
- fail-closed validation: missing field, non-finite value, wrong type,
  negative clock, empty throttle list
- **the cross-boot refusal**, and that `--allow-cross-boot` records a caveat
  rather than silently passing
- **the over-spread refusal**, at the threshold, either side of it
- the cross-arm offset refusal and the reported estimated effect
- the throttle mask: benign bits accepted, each non-benign bit refused
- wiring: `online_gate_summary` voids a leg with a missing, cross-boot, or
  over-spread clock record, matched against **that call site's own** message —
  #520 established that an unanchored `assertRaises` stays green on a gutted
  check

Regression surface: `tests/tools` in full — **233** tests on the base SHA
`8b00f79f2`, **295** here; 0 removed, proven by a sorted test-name diff. A
changed count is RED even when it prints `OK`.

Twelve mutations, applied one at a time with `count == 1` anchors and restored
byte-for-byte by sha256: the cross-boot refusal, each threshold widened to
1000%, the throttle mask opened to every bit, idle samples counted as busy, the
straddled-boot fold, leg reasons and arm reasons dropped from the aggregate,
each of the two ratio-level `clock_established` terms, the missing-record
reason, and the stream/summary reconciliation. **Three survived the first
round** — both ratio-level terms and the missing-record reason. That is #520's
lesson repeating: the leg-level reason already voided every gate assertion, so
each ratio-level site was dominated and a `gate_pass` assertion let the two
ratio families mask each other's removal. Two cases now assert the two families'
`binding_eligible` SEPARATELY on a pair whose arms are individually clean, and a
third pins the reason text that names the offending arm, which is the only thing
the reader has. Twelve of twelve RED.

## Gates

- `python3 -m unittest discover -s tests/tools -t .` — full, serial. `pytest`
  mis-collects this tree (16 false failures on clean `main`); do not use it.
- `scripts/agent-preflight.sh --staged`, then again on committed HEAD.
- `bash -n scripts/dgx-online-serving.sh`.
- **No benchmark gate.** The GPU is held, the box is at 99% disk, and #545 is
  open. Recorded `PENDING` with the exact handoff rather than waived.

## Stop conditions

- Stop before running any GPU work or changing any clock: `$HOME/gpu.lock` is
  held by another session and `-lgc` would corrupt their in-flight measurement.
- Stop before editing a recorded number to agree with the new contract. A past
  figure that cannot be attributed gains a note, never an edit.
- Stop if satisfying the contract requires weakening an existing eligibility
  reason. The clock rule is additive or it is wrong.
- Stop if the helper starts to need a plugin surface, a registry, or a second
  file. One module, imported.

## Evidence

- Live read-only `ssh dgx.casa` sample, 2026-08-12, used to fix the fixture
  format verbatim: `0, NVIDIA GB10, 580.159.03, 2190 MHz, 3003 MHz, 2418 MHz,
  0x0000000000000000, Enabled, P0`, boot id
  `13dc5579-455c-45c8-8e4d-d09c457fa826`. That is a **third** boot id, and the
  clock is again the degraded 2190 — the defect is live, not historical.
- The two-boot table and the `marlin::Marlin` control are #543's, already
  commented there.

## Owed

- The trace/per-kernel harnesses do not yet call the helper (§Scope). Until they
  do, a per-call `us/call` figure carries no clock attribution — which is
  precisely where #543's retracted findings came from.
- The `0.754` transfer coefficient is `n = 1`. A second cross-boot pair, once
  #545 allows one, either confirms it or replaces it.

## Now

`ACTIVE` — recorded, asserted, and documented; no measurement is taken and none
is restated. The first attributable grid is `PENDING` on `$HOME/gpu.lock` and
on #545.
