# VT-CONV1D-TIME-BLOCK — what limits the vocoder's parallel decomposition, measured

Row `VT-CONV1D-TIME-BLOCK`. Issue
[#1664](https://github.com/mudler/vllm.cpp/issues/1664). Lane
[#672](https://github.com/mudler/vllm.cpp/issues/672); the owed item is
[#1334](https://github.com/mudler/vllm.cpp/issues/1334) §18.9, *"the vocoder's
PARALLEL DECOMPOSITION is now the lever"*.

## Now

`ACTIVE`. The ablation is taken (§2a) and it refuted the candidate the row was
written to test. Two changes follow from it, §3a and §3b; the gates and the
mutation evidence are §6, and the `thor:gpu0` A/B/C is §2b.

## 0. Scope

**In scope.** The CPU provider of `vt::Conv1d`
(`src/vt/cpu/cpu_conv1d_general.cpp`), its parallel decomposition, and the
instrument that decides what limits it.

**Out of scope, deliberately.**

- `vt::ConvTranspose1d`. §18.5 prices it at **~6 % of the chain's wall** at
  `-O3`, and its scatter form is input-driven: a time block of INPUT positions
  writes an output range that overlaps its neighbour's by `dilation * (kernel -
  1)`, so two blocks would contribute to one cell and the per-cell order would
  have to be argued rather than inherited. That is a different row.
- The **arithmetic**. Nothing here changes what is computed. §3 states the
  bit-identity argument and §6 states how it is gated.
- The **device arm**. `VLLM_CPP_VOCODER_DEVICE` still defaults to `kCPU`
  (`vocoder1d.cpp:98`) and this row does not move that default.
- `vt::cpu::ParallelForRows` itself. §4 argues why the decomposition belongs in
  the kernel's index mapping rather than in the shared pool primitive, and what
  that buys in blast radius.

## 1. The gap

`vocoder.decode_window` is **122.169 s of a 449.969 s run** at 20 s / 30 steps
on `thor:gpu0`, about **27 %**, and it is the largest term no row owns.

§18.8b measured the window at **2.16x per core** and **1.365x on 14 threads**,
which is a scaling of **6.76x (before) and 4.27x (after) of a possible 14x**.
The faster kernel scales worse. §18.8b attributed that to a shared-bandwidth
limit and **named the attribution as an inference**: *"no bandwidth counter was
read, and none is available on this worker."*

`Conv1dKernel` (`cpu_conv1d_general.cpp:152`) partitions
`rows = batch * out_channels` and nothing else. Every thread therefore sweeps
the whole input tensor for its own slice of output channels, and the reuse
available across output channels is not taken. At 344 latent frames the b3
residual activation is **67.6 MiB** against **96** output channels; the b1
residual is **33.8 MiB** against **384**.

## 2. What was not established, and the ablation for each

`ncu` is refused on this fleet — `ERR_NVGPUCTRPERM` even from the root `rc`
worker, `NVreg_RestrictProfilingToAdminUsers` on the host driver module — and
Thor is aarch64. So each candidate gets an ablation, and **a counter that is
unavailable is reported as unavailable rather than replaced by an estimate**.

| # | Candidate | The ablation that separates it |
|---|---|---|
| 1 | Activation residency | `--control`: the SAME geometry with the time axis cut into declared pieces, all output channels of a piece before the next. Identical arithmetic, identical code path, identical position tile; only the footprint one piece touches changes. Flat row ⇒ not the limit. A knee ⇒ the limit is AT a cache level and the knee names it |
| 2 | `Threadpool::Barrier` / dispatch | `--dispatch`: one empty `ParallelForRows` priced at each thread count, against the **62** dispatches the window makes (31 convolutions x 2 streams). Converts the worry into arithmetic. The residency control is BIASED AGAINST ITSELF here — 128 positions per piece is 172x more dispatches at the b3 geometry — so a control that still wins has refuted this twice |
| 3 | Work-partition granularity | Every row prints the chunk grid `ParallelForRows` will build, transcribed from `cpu_threadpool.cpp:413-458`. `conv_out` has ONE output row and therefore runs inline on the caller at every thread count; that is visible rather than suspected |
| 4 | The threads are not running | `getrusage` user+system CPU over the wall clock of the same window. Near 1.0 ⇒ not dispatched. Near the thread count while the speedup does not follow ⇒ dispatched and stalled |
| 5 | The CPU clock falls as cores light up | Not measurable from inside the process. The job script samples `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` around every leg and reports min/median/max, plus the governor. **A 14-core leg at a lower clock than a 1-core leg is a scaling loss that has nothing to do with the code** |

The instrument is `tools/bench/conv1d_scaling_probe.cpp`, registered as
`vllm_conv1d_scaling_probe` and as no test. The window's own curve comes from
the existing `vllm_music3_vocoder_conv_ab`, which drives `VocoderDecode` itself.

**The two instruments are never multiplied together.** §18.9 records a **1.80x
disagreement** between this kernel bench and the e2e bucket at 344 latents
(54.091 s against 97.4463 s through the same call), and §18.4 prices the
kernel-to-window gap at a further factor of ~2. A ratio from the probe is a
ratio ON the probe.

## 2a. THE ANSWER — measured, and it is not the convolution

**`rc` job `706f15ef-8add-4e8e-a976-954af66e90f5` on `thor:gpu0`**, worker
`rc-worker-m4d7t`, `Linux 6.8.12-1021-tegra` aarch64, 14 cores, boot id
`fabedc13-97a1-4cb9-909f-217a425d3f70`, `--max-runtime 150m`, 2026-08-22. One
lease, one boot, no `ssh`, no file mutex. Release `-O3`, CPU only, built inside
the lease from `1745d9017`. Box facts that the rest of this section rests on,
read rather than assumed: **L1d 64 KiB 4-way and L2 1 MiB 8-way, both PRIVATE
per core, and NO shared last-level cache in `sysfs` at all**; governor
`schedutil`, `scaling_max_freq` 2 601 000 kHz.

### The window does not scale, and the op does

| threads | `vocoder.decode_window`, 20 latents | speedup | `vt::Conv1d`, one of each geometry, 86 latents | speedup |
|---|---|---|---|---|
| 1 | 9.6374 s | 1.00x | 4.24978 s | 1.00x |
| 2 | 6.2616 s | 1.54x | 2.20365 s | 1.93x |
| 4 | 4.6068 s | 2.09x | 1.08414 s | 3.92x |
| 8 | 3.7876 s | 2.54x | 0.53646 s | 7.92x |
| 14 | 3.4247 s | **2.81x** | 0.34177 s | **12.44x** |

**The convolution scales 12.44x of 14 while the window it sits in scales
2.81x.** `user/wall` on the op is 13.91, 13.82, 13.52 and 13.45 at the four
heaviest geometries, so the pool is running and productive. Every arm printed
one fingerprint per length across every thread count, so nothing here trades
correctness for the ratio.

### Each candidate, and what killed it

| # | Candidate | Verdict | The measurement |
|---|---|---|---|
| 1 | Activation residency | **NOT the limit** | The residency sweep is nearly flat where it matters: at 14 threads `b2_res_conv1` reads 1.008x/1.021x/0.997x/0.989x across a 165x footprint range and `b3_res_conv1` reads 0.977x/1.019x/1.132x/1.067x. The largest reading anywhere at 14 threads is `b1_res_conv1` at **1.307x**. Real, and worth about 6 % of a window the convolution is 20 % of |
| 2 | Barrier / dispatch | **DEAD BY ARITHMETIC** | One empty `ParallelForRows` costs 11.845 µs at 14 threads. The window makes **62**. That is **0.734 ms against 3.4247 s — 0.02 %** |
| 3 | Granularity | **NOT the limit, with one real exception** | The chunk grid is 48-55 chunks on 14 threads at every heavy geometry. The exception is `conv_out`: ONE output row, so `chunks = 1` and `user/wall = 1.00` at every thread count. It is 0.2 % of the op total, so it is a defect and not the gap |
| 4 | The pool is not running | **DEAD** | `user/wall` 13.45-13.91 at 14 threads |
| 5 | The CPU clock falls as cores light up | **DEAD** | `scaling_cur_freq` sampled every 2 s across all 14 CPUs: max **2 601 000 kHz at every leg**, and the MEDIAN over 14 CPUs is 2 601 000 at 8 and at 14 threads — i.e. with the box full, the typical core is at the cap. (At 1 and 2 threads the median reads the 972 000 kHz idle floor, because 12 of the 14 samples are idle cores. That is the statistic reporting what it was asked, not a throttle) |

**No bandwidth counter was read and none is claimed.** `perf` is NOT INSTALLED
on the `thor:gpu0` worker and `perf_event_paranoid` is 2; `ncu` is refused on
this fleet. Candidate 1 is settled by ablation instead, and the ablation says
residency is worth 1.0-1.31x rather than the several-fold factor a
bandwidth-saturation story needs.

### So where does the window go? The split says it in one line

`vocoder.decode_window` split into leaves through `profile::Timer` on the
production path, 14 threads. **PROVENANCE, because the two halves of this
section come from different boxes:** the scaling curve and the five verdicts
above are `thor:gpu0` under the lease named at the top. The split below is the
AUTHORING host (x86-64, 20 cores, `uptime` load 18.5 — not an idle box) at 20
latents, because the instrument did not exist when the Thor job was submitted.
It is quoted as a SHARE and never as a duration, and §2b re-takes it on Thor.

| leaf | seconds | calls | share |
|---|---|---|---|
| **`vocoder.snake`** | 4.231 | 58 | **70.70 %** |
| `vocoder.conv1d` | 1.197 | 54 | 20.00 % |
| `vocoder.conv_transpose` | 0.489 | 8 | 8.17 % |
| `vocoder.pad` | 0.053 | 28 | 0.89 % |
| `vocoder.copy` | 0.006 | 32 | 0.10 % |
| `vocoder.residual_add` | 0.004 | 24 | 0.07 % |
| `vocoder.tanh` | 0.000 | 2 | 0.00 % |
| sum(leaf) | 5.982 | — | 99.95 % |

`vocoder1d::SnakeActivation` is a **serial** loop — no partition of any kind —
over every element of every activation in the chain, computing a `double`
`std::sin` per element. Amdahl's law over it predicts the window's whole curve:
solving `p/12.44 + (1 - p) = 3.4247/9.6374` gives **p = 0.70**, and the split
independently reads the parallel part at 0.70 of the T=1 window if the snake is
the serial term. Two instruments, one number.

**§18.8b's shared-bandwidth candidate is therefore refuted rather than
refined.** The scaling it measured was real, its reading of the cause was not,
and the reason it could not see this is stated plainly: it timed the WINDOW and
reasoned about the KERNEL, with nothing in between. The instrument that was
missing was a split, not a counter.

## 3. What the row changes, and why each is bit-identical BY CONSTRUCTION

The measurement moved the row's lever, so there are TWO changes and they are
not the same size. The larger one is the one §2a found; the smaller one is the
one the row was written to make, kept because it is measured and because it
reaches a shape the row partition could not.

### 3a. The activation function gets a partition — the row's lever

`vocoder1d::SnakeActivation` had none. It now partitions the CHANNEL axis
through `host_parallel::ForOutputRows`, the same seam every other host-reference
body in this tree uses.

**It needs none of the argument §3b needs.** Output element `(c, t)` is a
function of input element `(c, t)` and of `alpha[c]` alone. There is no
reduction, so there is no summation order, no accumulator, no reassociation and
no tolerance. The `double` intermediates, the `std::sin` and `kSnakeEps` are
UNTOUCHED — narrowing them is a numerics change an oracle owns and this row does
not make.

The size guard is the shared one, so a small activation still runs inline; and
`channels == 1` stays inline, which no shipped consumer geometry hits.

### 3b. The convolution gets a second axis

**Blocked over the time axis, and parallel over (time block, output row).**

The shipped nest is `for oc: for t0 tile: for ic: for k`. The proposed nest is
`for t block: for oc: for t0 tile: for ic: for k`, with the parallel grid over
the flattened `(block, oc)` pair rather than over `oc` alone.

Fix any single output cell and read what it receives, in order: the bias, then
`(ic=0,k=0)`, `(ic=0,k=1)`, ... — **the identical sequence of IEEE-754
additions of the identical products, in the identical order**, in the identical
f32 accumulator. Nothing splits a reduction, nothing accumulates atomically,
and no cell is touched by two units of work. This is §18.3's argument, applied
to a second scheduling change rather than restated for it.

Two conditions make the equality hold at the level of the code path as well as
the arithmetic, and both are gated:

1. **The block length is a multiple of `kConv1dPosTile`.** Position tiles then
   land on exactly the boundaries they land on today, so a tile that takes the
   constant-trip-count fast path today takes it after the change. §18.4 prices
   that path at up to 5x, so a change that silently moved a tile onto the
   chunked path would report a speed result about a different kernel.
2. **The trailing block carries the remainder.** `length` is not in general a
   multiple of the block, and the last block is short exactly as the last
   position tile is short today.

`conv_out` is the case that shows the decomposition is not only about cache: it
has **one** output row, so it is inline and serial at every thread count today,
and a `(block, oc)` grid gives it `ceil(length / block)` units of work.

## 4. The seam

`vt::cpu::ParallelForRows` **is** the seam and it is not modified. What changes
is the index space the kernel hands it: `rows` becomes `blocks * rows`, and the
body decodes the pair. That is the same relationship `cpu_paged_attn.cpp:211`
already has with the primitive (`total_q`, a flattened pair) and
`cpu_attn_relpos.cpp:89` (`hq * t`).

**This is a deliberate choice about blast radius, not an avoidance of the
seam.** A new pooled primitive would be inherited by every one of the primitive's
consumers — `cpu_layernorm.cpp` (5 sites), `cpu_quant_gemm.cpp` (3),
`cpu_paged_attn.cpp`, `cpu_attn_relpos.cpp`, `cpu_conv2d.cpp`,
`cpu_conv3d.cpp`, `cpu_conv1d_depthwise.cpp`, `cpu_quant_repack_arm.cpp`,
`cpu_ops.cpp`, `tenstorrent_ops.cpp` and `ltx2_video_vae.cpp` — and would owe a
gate on each. The blocking factor that is right for a convolution over a 67.6
MiB activation is not a property those consumers share, and inventing one
shared knob for them would be the parallel path this rule exists to prevent,
wearing a seam's name.

**Consumers this row does affect, and how each is gated** — every model that
reaches `vt::Conv1d` on the CPU device:

| Consumer | Reached from | Gate |
|---|---|---|
| MiniMax-Music3 vocoder | `vocoder1d::Conv1d` | `test_vocoder1d`, `test_minimax_music3_acoustic` |
| LTX-2.5 audio VAE | `ltx2_audio_vae.cpp` | `test_ltx2_audio_vae` — **13 audio arms at `5e-6`, the tightest tolerance any consumer holds** |
| IndexTTS-2.5 BigVGAN | `bigvgan.cpp` | `test_bigvgan` |
| MiniMax-H3 audio VAE | `minimax_h3_audio_vae.cpp` | `test_minimax_h3_audio_vae` |
| the op itself | `vt::Conv1d` | `test_ops_conv1d_general`, `test_host_parallel` |

## 5. Risks

- **R1 — a residency win that does not exist on the box that matters.** The
  authoring host is an x86 part with a very large last-level cache, where the
  whole activation is already resident and the control is flat BY
  CONSTRUCTION. Every ratio this row quotes is from `thor:gpu0`, and the
  authoring host is used for compile and correctness only.
- **R2 — the block length becomes a tuned constant nobody can re-derive.**
  Mitigated by the residency sweep: the block is chosen at the knee the sweep
  measures, and the sweep is a shipped executable.
- **R3 — the ratio is a code-path ratio.** Mitigated by §3's multiple-of-32
  condition and gated by a test that asserts it.
- **R4 — the answer is candidate 5.** If the 14-core clock is materially below
  the 1-core clock, part of §18.8b's 4.27x is thermal and is not addressable
  from this repository. It is then **reported as such** and the remaining
  attributable part is what this row acts on. Leases cannot pin clocks
  (`LGC_RC=4`, [#1354](https://github.com/mudler/vllm.cpp/issues/1354)), so
  sampling is the only instrument available.

## 6. Gates and evidence

### 6a. The suites — counts and `Status:` in full, because `assertions: 0` is a skip wearing a pass

Authoring host, x86-64 20 cores, Release `-O3`, CPU only, at the row's head.
Every binary's sha256 was taken before it ran.

| suite | result | rc |
|---|---|---|
| `test_ops_conv1d_general` | 14 cases, 19 617 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_host_parallel` | 11 cases, 968 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_vocoder1d` | 10 cases, 58 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_bigvgan` | 6 cases, 65 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_minimax_music3_acoustic` | 36 cases, 345 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_ltx2_vae` | 44 cases, 3 131 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_minimax_h3` | 79 cases, 57 395 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_indextts2_pipeline` | 8 cases, 433 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_indextts2_family` | 7 cases, 22 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_ops_conv1d_depthwise` | 5 cases, 1 184 assertions, 0 failed, `SUCCESS!` | 0 |

`test_ops_conv1d_general` prints four `[SKIP]` lines and they are read rather
than ignored: this host has no CUDA toolkit, so every CPU-vs-CUDA arm in that
file — including #1334's two cancellation arms — did NOT run. The device
provider is unchanged by this row and §7 carries the re-measure as owed.

### 6b. Red before green — every mutation with its compile rc, its hunk count and its binary sha256

Baseline binaries: `test_ops_conv1d_general` `0f886491b735fdea…`,
`test_host_parallel` `63a79608a7f270de…`.

| # | Mutation | compile_rc | hunks | binary sha256 | result |
|---|---|---|---|---|---|
| M1 | `Conv1dTimeBlock` drops the position-tile rounding | 0 | 1 | `08c97a3952bc6b57…` | **2 cases / 1 343 assertions FAILED**, rc 1 |
| M2b | the trailing time block is dropped (floor for ceil) | 0 | 1 | `463a2bab3160077c…` | **2 cases / 2 assertions FAILED**, rc 1 |
| M3 | the snake partition drops the last channel of every chunk | 0 | 1 | `82b2f989663a4dc6…` | **1 case / 20 assertions FAILED**, rc 1 |
| M4 | the snake body is never dispatched | 0 | 1 | `daa17366f828bc71…` | **1 case / 20 assertions FAILED**, rc 1 |
| M5b | the registered CPU `vt::Conv1d` provider refuses every call | 0 | 1 | per suite, all distinct | **every consumer suite red** — see 6c |
| — | restored | 0 | 0 | `0f886491b735fdea…`, `63a79608a7f270de…` | back to baseline, byte for byte |

**M2 is recorded as a defect in the INSTRUMENT, not omitted.** The first
attempt at M2 removed the clamp on the last block, which writes past the output
row. That is undefined behaviour rather than a wrong answer: the binary built
(`2eb599bc05546129…`) and the suite had not finished after 201 s, so it was
killed and reads `test_rc=137`. A hang is a detection nobody can grade, so M2b
replaces it with a mutation that computes FEWER cells and writes none out of
range, and that one reds cleanly on the two block-crossing cancellation cases.

### 6c. Reachability, measured rather than argued

A scheduling change is invisible to every arithmetic assertion BY DESIGN, so
two separate instruments carry the reachability claim.

**The op is reached from all four consumers' production paths.** M5b makes the
registered CPU provider refuse every call, and the suites red in exactly the
places that prove it: `test_ops_conv1d_general` 5 cases, `test_host_parallel` 3,
`test_vocoder1d` 3, `test_bigvgan` 3, `test_minimax_music3_acoustic` 6,
`test_ltx2_vae` 5. `test_indextts2_family` and `test_ops_conv1d_depthwise` stay
green, which is correct — neither reaches this op — and is the control that
says the red is a signal and not a broken build.

**The second axis is entered at the SHIPPED geometries.** The gate evaluates
`Conv1dTimeBlock` at MiniMax-Music3's own eight `vt::Conv1d` shapes at a
344-latent window and asserts `blocks > 1` on every one, plus the tile alignment
and the slice budget. Without it a future budget change could collapse the
decomposition back to one block and no arithmetic test would notice.

**And the snake's dispatch is mutated directly.** M4 replaces the
`ForOutputRows` call with one that runs the body over an empty range; the gate
reds. A green there would have meant the case was measuring a class.

## 7. Owed

Named here rather than left to a profile, because each is a real gap this row
declines to close.

- **The CUDA provider is not re-measured against either change.** The authoring
  host has no CUDA toolkit, so `test_ops_conv1d_general`'s four CPU-vs-CUDA arms
  printed `[SKIP]` and the `thor:gpu0` job was built CPU-only. Neither change
  touches `cuda_conv1d_general.cu`, and the `memcmp` argument is §18.3's — the
  per-cell order is unchanged on the host — but an argument is not a
  measurement. It needs a CUDA build, which §20.2 records as reachable inside a
  lease (`apt` installs `cuda-nvcc-13-0`).
- **`vt::ConvTranspose1d` keeps the row-only partition.** It was 8.17 % of the
  window before this row and is a larger share after it, and its scatter form
  makes a time block's output overlap its neighbour's by
  `dilation * (kernel - 1)`, so the per-cell order would have to be argued
  rather than inherited. Deliberately a different row.
- **The snake returns about 7x, not 14x**, on the authoring host. Why it stops
  there is not measured: candidates are the `std::sin` call's own throughput,
  the pool's chunk grid at 96 channels, and the pass being a pure stream over an
  activation that does not fit any cache here. Not chased, because after this
  row it is no longer the largest term.
- **`vocoder.pad` and `vocoder.copy` are unpartitioned**, at 0.89 % and 0.10 %.
  They are named so that the next reader does not re-derive that they are small.
- **The block byte budget was not swept per geometry ON `thor:gpu0`.** The
  residency instrument sweeps four footprints at each geometry and
  `kConv1dSliceBytes` was chosen at half of the measured 1 MiB private L2 rather
  than at a per-geometry optimum. A finer sweep may move it; the constant is
  derived from a stated budget, so moving it is a one-line change with a shipped
  instrument behind it.
- **The snake's arithmetic is untouched and unexamined.** It evaluates a
  `double` `std::sin` per element. Whether upstream's `Snake1d` needs that width
  is a numerics question an oracle owns, and narrowing it would re-gate four
  models — exactly the shape #1474 was.

## 8. Stop conditions

Stop and report rather than widen scope if: the ablation says the limit is
candidate 4 or 5, which this row cannot act on; a decomposition change cannot
be made without moving a per-cell summation order; the blast radius exceeds
`vt::Conv1d`'s CPU provider; or `thor:gpu0` is unavailable, in which case no
ratio is quoted at all rather than quoted from another box.

## 9. Outcome

Written when the row reaches `DONE`.
