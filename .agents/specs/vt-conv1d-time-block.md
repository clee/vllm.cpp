# VT-CONV1D-TIME-BLOCK — what limits the vocoder's parallel decomposition, measured

Row `VT-CONV1D-TIME-BLOCK`. Issue
[#1664](https://github.com/mudler/vllm.cpp/issues/1664). Lane
[#672](https://github.com/mudler/vllm.cpp/issues/672); the owed item is
[#1334](https://github.com/mudler/vllm.cpp/issues/1334) §18.9, *"the vocoder's
PARALLEL DECOMPOSITION is now the lever"*.

## Now

`ACTIVE`. The instrument is written and the measurement is the first gate; no
kernel change is committed until the ablation says which candidate the curve
actually has.

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

## 2. What is not established, and the ablation for each

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

## 3. The design, and why it is bit-identical BY CONSTRUCTION

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

**Correctness before speed, in this order.**

1. `test_ops_conv1d_general` and `test_host_parallel` — the two suites whose
   serial references are verbatim copies of the pre-op host loops, so they
   prove *the order did not move*, which is the property at risk. Both carry
   catastrophic-cancellation cases on the `ic` and the `k` axis, and both are
   run because an f32 accumulator on benign data cannot see a reordering.
2. Every consumer suite in §4's table.
3. `test cases:`, `assertions:` **and** `Status:` quoted for each, because
   `assertions: 0` is a skip wearing a pass.
4. A red-before test for the new geometry conditions, and a mutation that
   deletes the production call site to prove the gate is measuring a capability
   and not a class.

**Speed.** `thor:gpu0` under an `rc` lease, boot id recorded, every ratio inside
one boot, `uptime` on both sides, arms alternated, the loudest pair kept, and a
clock window sampled from `/sys` around every leg.

## 7. Owed

- Nothing yet. Items land here as the measurement closes them out.

## 8. Stop conditions

Stop and report rather than widen scope if: the ablation says the limit is
candidate 4 or 5, which this row cannot act on; a decomposition change cannot
be made without moving a per-cell summation order; the blast radius exceeds
`vt::Conv1d`'s CPU provider; or `thor:gpu0` is unavailable, in which case no
ratio is quoted at all rather than quoted from another box.

## 9. Outcome

Written when the row reaches `DONE`.
