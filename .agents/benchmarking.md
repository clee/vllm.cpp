# Task guide — measuring performance

How to produce a number worth believing. The rules are in
[`AGENTS.md`](../AGENTS.md); this is the method.

## The denominator

vLLM is the bar, quant-matched, in its **production** configuration. Never
benchmark against `--enforce-eager` and call it parity. llama.cpp may appear
only as an explicitly labelled secondary comparison.

Both sides run the pinned oracle on identical model artifacts, prompts, token
counts, batching, concurrency, and sampling. If the two sides differ in any of
those, the ratio means nothing.

Prove the oracle actually *runs* the model before trusting it as a
denominator — constructing a config proves nothing.

## Getting a clean measurement

One GPU job at a time. Take the box lock before any measurement, stop competing
services, and never run two large models at once — unified-memory boxes reboot
rather than swap.

Calibrate the noise band from repeated identical legs *before* interpreting a
delta. Discard cold legs for a named cause, never because they are
inconvenient. Use paired, order-alternated A/B legs and a majority rule; a
single pair is an anecdote.

Prefer an instrument that is immune to page-cache effects (GPU-active time per
step) over wall clock when the host is doing heavy I/O.

## The clock is part of the measurement

**The SM clock differs between boots and does not announce it.** On `dgx.casa`
one boot ran the timed window at a median 2470 MHz and the next at a flat
2190 — a 12.79% delta, with `clocks_throttle_reasons.active = 0x0` and
persistence `Enabled` throughout, so nothing looked wrong. It repriced a
byte-identical `marlin::Marlin` with no source change by **+9.65%**, which is
larger than either deficit that comparison was being used to rank (#543). Two
probes eight minutes apart *inside one boot* disagreed by ~6% uniformly.

So a number is quotable only with the clock it was taken at. Every leg records
the SM clock across the measured window (min/median/max and n), `clocks.max.sm`,
`clocks.applications.graphics`, the active throttle reasons, persistence mode,
and the **boot id** — `tools/bench/gpu_clock_state.py` is the one helper that
samples, folds, and asserts it, and other harnesses import it rather than
rolling their own.

Two arms on **different boots are not comparable**. The summary refuses that
pair outright; `--allow-cross-boot` waives *identity*, never *state*, and stamps
a recorded caveat rather than passing silently. Within a run the SM-clock spread
must stay at or below **5%**, and the two arms' medians within **1%** of each
other — the first accepts the one clean window we have (3.68%) and rejects the
within-boot disagreement, the second keeps the clock's estimated contribution
under the smallest deficit anyone ranks. The argument for both numbers is in
[`specs/bench-assert-clock-state.md`](specs/bench-assert-clock-state.md).

**Pin the clocks before measuring, under the lock.** Passwordless `sudo` is
available on `dgx`, and `-lgc` is supported:

```sh
sudo nvidia-smi -lgc 2100        # pin, before the first leg
sudo nvidia-smi -rgc             # release, after the last one
```

Pinning is a **shared-host mutation**. Never run `-lgc` or `-rgc` while another
session holds `$HOME/gpu.lock` — it silently reprices their in-flight
measurement, which is the very defect this section exists for. Take the lock,
pin, measure, reset, release. It is a **pre-measurement step**, not a standing
configuration: leaving the box pinned makes every later run inherit a state
nobody recorded, which is where this started.

Figures recorded before 2026-08-12 predate clock assertion. They are not
withdrawn and are not restated — they simply carry no clock attribution, so a
delta smaller than ~10% between two of them is not established by them alone.

Budget the disk before the run. A production RelWithDebInfo CUDA build tree is
about **169 GiB** — the build contract claimed ~3 GiB until 2026-08-10, a 56x
underestimate on the one number that decides whether a grid fits. A full disk
does not fail loudly: it voids the binding through memory-return tolerance while
still emitting plausible ratios. Leave real headroom, and delete the tree once
the evidence directory is captured (evidence is tens of MiB).

Two ratio sets that disagree may be two different HARNESSES rather than a
regression. Compare their absolute numbers before believing either; ratios are
scale-invariant and hide an order-of-magnitude mismatch completely. If the
change between the readings is provably inert (a byte-identical refactor),
suspect the measurement, not the code.

## Reading a profile

**A whole-run kernel ranking is a trap.** It sums prefill and decode, so the
top-percentage kernel is frequently one-time prefill work that no decode step
touches. Use a decode-only window or diff two sequence lengths. A `Max` far
above the `Median` means you are looking at a mixture, not a hot loop.

Profile the entire step, not only the kernels. Several of the largest wins here
were host-side waste, not slow math.

Before accepting a gap as "GPU-bound", trace both implementations with the same
tool on the same workload and compare what actually ran.

## Recording it

Record the exact build and run recipe, revisions, model hashes, environment,
clock and contention state, raw output, and the same-binary A/B. Reproduce on an
idle box before acceptance. "Clock state" is the concrete list in §The clock is
part of the measurement, boot id included, not a prose adjective.

Record every required axis — throughput, latency, memory — as both values and
ratios. An axis below floor is an open gap, not a rounding error.

Never record a ceiling. An apparent same-architecture limit is an unresolved
implementation difference; name the next traceable hypothesis instead.

Accepted and pending results go in [`benchmark-record.md`](benchmark-record.md)
and `docs/BENCHMARKS.md`. Method specific to one lever stays in
[`parity-lever-protocol.md`](parity-lever-protocol.md).
