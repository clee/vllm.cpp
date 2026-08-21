# LTX25-DIT-ATTN-FLASH — the LTX-2.5 DiT never opted into a fast attention op, and paid 47.84 s a forward for it

Row: `LTX25-DIT-ATTN-FLASH`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)), against the
model-matrix row `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`.
Issue: [#1549](https://github.com/mudler/vllm.cpp/issues/1549).

## Now

`ACTIVE`. The diagnosis is confirmed against the tree, the call sites are
identified, and the change is scoped to one production call site plus one
prerequisite kernel repair. The A/B is the confirming evidence for the whole
diagnosis and is not optional; it needs `dgx:gpu0`, which is the box the
denominator was measured on, and no other box is a valid denominator for it.

Every bare `file:line` anchor below is read at this row's **base, `6b48edb2c`**,
which is `origin/main` at the claim. The change itself moves some of them —
`ltx2_device.cpp:421` most obviously — and a reader who wants the line rather
than the symbol should check out that SHA. Anchors written `file::Symbol`
survive the move and are what `scripts/check-symbol-anchors.py` gates.

## 0. The one-sentence finding

`src/vllm/model_executor/models/ltx2_device.cpp:421` calls `vt::Attention` for
every DiT self-attention, that op is frozen on a kernel whose own header calls
itself "Correctness-grade (M0.9)", and nothing anywhere tells a caller that a
faster op exists.

## 1. The measurement this row starts from

One LTX-2.5 DiT forward at `768x448 / 49f` (2352 video tokens), full checkpoint,
bf16, on `dgx:gpu0` (GB10):

| quantity | value |
|---|---|
| mean per forward | 47.84 s |
| median | 47.91 s |
| n | 119 |
| spread | 5.8% |

The samples are the engine's own `last=` lines, emitted by
`src/vllm/multimodal/render_phase_log.cpp:388` from the tick at
`src/vllm/multimodal/ltx2_video.cpp:4048-4054`. That tick fires immediately
before the DiT call, so `last=` is the interval between the start of forward
N-1 and the start of forward N: one fused two-stream (video + audio) DiT
forward plus the small host glue between them. It is **not** a per-denoise-step
number — one guided denoiser evaluation is one to four forwards — and this row
never treats it as one.

A governor/estimator in the same harness has reported 1.00 s, 69.1 s, 162 s and
396.9 s for this one quantity. Only `last=` is used here, and every number below
is reduced from those lines.

## 2. Attribution, and why it is not a guess

`vt::Attention` dispatches to `AttentionKernel` at
`src/vt/cuda/cuda_ops.cu:1463`. The kernel's own header comment at `:1456-1459`
says what it is. Its shape:

- one block of `kBlock = 256` threads (`cuda_ops.cu:33`) per (query, head);
- for every key, a 256-wide shared-memory tree reduction, about 12
  `__syncthreads()` per key;
- no K/V tiling, so K and V are re-read from global memory once per
  (query, head).

For the LTX video stream at 2352 tokens and 32 heads: grid = 75,264 blocks, each
looping 2352 keys = **1.77e8 block-key iterations per call**, times **48
layers**.

The per-iteration cost is not assumed. `.agents/specs/multimodal-speed.md:24-26`
records nsys attributing **98.9%** of the Qwen3-VL tower forward to this same
kernel, on this same box, at **5.70 ns per block-key iteration**. Scaling that
to LTX's geometry:

    1.77e8 x 5.70 ns x 48 = 48.4 s   vs the measured 47.84 s

A 1% match. The kernel is the forward. Everything else in the DiT — 48 blocks of
GEMMs, norms, RoPE, gating, six attentions per block — fits in the remaining 1%.

This is a scaling argument across two models, not a direct nsys run on LTX. It
is offered as such. The A/B in §7 is what actually settles it: if the diagnosis
is right, replacing this one kernel moves the whole forward, and if it is wrong,
it will not.

## 3. Why this was missed, which is the part worth keeping

`kAttention` was deliberately frozen on the naive kernel so that text decode
stays byte-identical (`src/vt/cuda/cuda_ops.cu:3120-3122`). That decision is
correct and this row does not touch it.

The consequence is the problem. The fast kernels are **separate ops that each
caller must opt into by name**: `kAttentionDenseFast`, `kAttentionDenseFlash`,
`kAttentionDenseFa2`. There is no automatic selection, no shape-based routing,
and no fallback notice. A model that never opts in silently gets the
correctness-grade kernel: **correct output, roughly 500x the cost, no warning
anywhere**.

Nothing in this tree can detect that. The output is right, so every golden
passes. The op is registered, so no refusal fires. `GetOpProviderStats` counts a
selection of `kAttention` and reports it as a success, because it is one. The
only symptom is the wall clock, and a diffusion render has no reference wall
clock to be measured against.

The lesson generalises past LTX: **every remaining caller of `vt::Attention` on
a non-decode path is a candidate for the same defect**, and nobody will be told.
This row does not sweep them — that is a separate row with its own issue — but
it states the shape so that the next reader recognises it.

## 4. What changes

### 4.1 The production call site

`src/vllm/model_executor/models/ltx2_device.cpp:421`, the self-attention branch
of `AttentionDev`, moves from `vt::Attention` to `vt::AttentionDenseFlash`.

`vt::AttentionDenseFlash` is registered at `src/vt/cuda/cuda_ops.cu:3664-3669`
and is head_dim-generic up to 256 (`cuda_ops.cu:3336-3339`), so LTX's video
head_dim 128 and audio head_dim 64 are both served. Its contract requires a
square problem (`key.shape[0] == query.shape[0]`, `src/vt/ops.cpp:2988-2990`),
which the self-attention branch already satisfies: that branch is entered only
when `context == nullptr`, and `AttentionDev` sets `s = tq` in exactly that case
(`ltx2_device.cpp:355`).

The dispatch **rule** is unchanged. The branch is still selected by
`context == nullptr && a.bias == nullptr`, which is upstream's own
self-attention marker, and not by what the numbers happen to be. Only the op the
branch calls changes.

### 4.2 The host call site is deliberately NOT changed

`src/vllm/model_executor/models/ltx2.cpp:959` stays on `vt::Attention`. That arm
computes into `std::vector<float>` and hands host pointers to
`Tensor::Contiguous`, so it is CPU-only by construction. On CPU
`kAttentionDenseFlash` is registered to the **same** `AttentionKernel` as
`kAttention` (`src/vt/cpu/cpu_ops.cpp:3557-3562`), so the swap would be a
byte-identical no-op that buys nothing and moves the L2 parity reference off the
reference op. The reference arm stays on the reference kernel.

### 4.3 A prerequisite kernel repair, found while doing this

`LaunchAttentionDenseFlash` (`src/vt/cuda/cuda_ops.cu:3330-3352`) requests

    shmem = 2 * kFlashBc(64) * head_dim * sizeof(Tin)

bytes of dynamic shared memory and never opts in, so it is capped at the 48 KiB
every CUDA architecture guarantees without `cudaFuncSetAttribute`. At **f32 and
head_dim 128 that is 65,536 B and the launch fails.**

This matters here because LTX's f32 device arm is a supported arm — the forward
names it "the L2 parity arm" at `ltx2_device.cpp:1186-1188` — at head_dim 128.
Swapping the call site without this repair would convert a slow-but-correct f32
arm into a refusal.

The repair mirrors the pattern this tree already established for the same
problem in `cuda_paged_attn.cu:112-127`: query the ceiling rather than assume
it, through the cached `vt::cuda::DynamicSmemFits` /
`vt::cuda::GetDeviceCaps` seam (`src/vt/cuda/cuda_device_caps.h`). Where the
tile fits the opt-in ceiling, opt in. Where it does not, **fall back to
`AttentionDenseFastKernelCuda`**, which the flash kernel's own header
(`cuda_ops.cu:3232-3234`) states is bit-identical to it by construction: same
per-warp online-softmax recurrence, same lane grouping, same j-order, same f32
accumulation, with K/V read from global rather than shared. The fallback is
therefore numerically free and only gives up the tiling speedup.

This is the same shape as the existing `AttentionDenseFa2` -> `AttentionDenseFlash`
fallback (`cuda_ops.cu:3407`): the caller gets the best available kernel for its
shape rather than a hard refusal.

The bug is pre-existing and reaches every caller of `vt::AttentionDenseFlash` at
f32 with head_dim >= 96, not only LTX. It is filed as part of
[#1549](https://github.com/mudler/vllm.cpp/issues/1549) and fixed in the same
flow, per the AGENTS.md in-flow rule.

**This is not a hypothesis: the identical failure is already documented one file
over, and was fixed there and never back-ported.**
`src/vt/cuda/cuda_attention_cross.cu:88-96` — a kernel written for this very
model, and itself a structural port of `AttentionDenseFlashKernel` — says:

> LTX-2.5's video stream is head_dim 128, and at f32 a fixed 64-column tile would
> ask for `2 * 64 * 128 * 4` = 64 KiB of dynamic shared memory — over the 48 KiB
> a launch gets without opting in — so the kernel would fail to launch on exactly
> the real geometry while every reduced-dimension gate (head_dim 8 and 4) passed.

Same number, same arithmetic, same reason the existing gates cannot see it. That
author chose to **halve the tile** until it fits (`ChooseTileCols`). This row
chooses to **opt in** instead, and falls back only where the device's queried
ceiling refuses: GB10 reports 101,376 B, so the full 64-column tile fits at f32
head_dim 128 and the reuse is kept rather than halved. Where a device cannot, the
fallback is `AttentionDenseFast`, which is bit-identical rather than merely
narrower.

Two things follow that are worth stating. The first is that "every
reduced-dimension gate passed" is why this row's new `test_ops_attention` case
runs at the REAL head_dims (64, 128, 256) rather than the fixture's. The second
is that `vt::AttentionCross` on CUDA is already flash-tiled, so **the LTX DiT's
cross-attentions were never on the naive kernel** — the four cross-attentions per
block and the two self-attentions were on different kernels the whole time, and
only the self-attentions were slow. That makes §2's attribution tighter, not
looser.

### 4.4 The A/B knob

`VLLM_LTX2_DIT_FLASH_ATTN=0` restores `vt::Attention` at the swapped call site,
so both arms of §7 run from **one binary**. Default on, read fresh per call on a
path taken about 96 times per forward. This mirrors `VT_FA2_DENSE`
(`cuda_ops.cu:3383-3389`), which exists for exactly this reason and is
documented as "Same-binary A/B + RED knob".

The knob is documented in `docs/ENVIRONMENT.md`, which `scripts/check-env-doc.py`
requires for any `VT_*` / `VLLM_*` string literal under `src/`.

## 5. Numerics: what is bit-identical and what is not

State plainly, because the two halves have different answers.

**On CPU: byte-identical.** `kAttentionDenseFlash` and `kAttention` are the same
registered function pointer (`src/vt/cpu/cpu_ops.cpp:3551-3562`). Every existing
CPU golden case in `tests/vllm/models/test_ltx2_device.cpp` must stay
byte-unchanged, and if one moves, the change is wrong.

**On CUDA: NOT bit-identical, and this row does not pretend otherwise.**
`AttentionDenseFast` differs from `Attention` in how the head_dim partial sums
are grouped (`include/vt/ops.h:2966-2972`): the naive kernel reduces across a
256-thread block, the warp kernel across 32 lanes with `__shfl_xor`. The
arithmetic is the same f32 online softmax; the association order is not.
`AttentionDenseFlash` is then bit-identical to `AttentionDenseFast`
(`include/vt/ops.h:2976-2985`), so the whole difference is naive-vs-warp
regrouping.

The magnitude is therefore a floating-point reassociation of a length-`head_dim`
sum, not an algorithmic change. The binding gate is the existing CUDA
host-vs-device parity case `tests/vllm/models/test_ltx2_device.cpp:752`, which
holds the f32 device forward to the CPU f32 host forward at
`kDeviceRoundOff = 2e-5` and the bf16 arm at `kBf16RoundOff = 5e-3`. That case
already exercises exactly the swapped path and does not need to be invented.

The measured deviation is reported in §7 as a number, not as a claim. If it
exceeds the committed tolerance, this row reports that and stops rather than
widening the tolerance. Widening a gate to admit a change is the failure this
protocol exists to prevent.

## 6. Reachability

AGENTS.md "Nothing lands dead". The swap must be proved to be *entered* from a
production entry point, not merely to compile.

Instrument: `vt::EnableOpProviderCallStats(true)` plus
`vt::GetOpProviderStats(OpId, DeviceType)` (`include/vt/op_provider.h:155-186`),
which counts dispatches through `GetOp`. This is a positive signal and it works
on the **CPU backend**, so the proof does not need a GPU.

Entry point: `vllm::Ltx2DitForwardDevice`, the production device forward that
`src/vllm/multimodal/ltx2_video.cpp:4055-4059` calls. The test drives that
function, not a hand-constructed `vt::Attention` call.

The assertion is **two-sided**, because a one-sided count cannot tell a routed
call from an added one:

1. `kAttentionDenseFlash` selections > 0 after one forward, and
2. `kAttention` selections == 0 after that same forward.

Half (2) is what makes it a routing proof: LTX's cross-attentions use
`kAttentionCross`, so after the swap there is no remaining `kAttention` caller
anywhere in the DiT forward. If somebody adds a second, unswapped self-attention
path later, that half goes red.

Mutation: delete the `vt::AttentionDenseFlash` call at
`ltx2_device.cpp:421` in a scratch copy, restoring `vt::Attention`, and rerun the
focused gate. Both halves must go red. The result is recorded in `## Outcome`
with the exact diff applied and the exact failure text, and the tree is restored
byte-for-byte afterwards.

## 7. The A/B

Required, not optional: it is the confirming evidence for §2's whole attribution
chain.

**Method.** One binary. Two arms selected by `VLLM_LTX2_DIT_FLASH_ATTN`
(unset = flash, `0` = the naive kernel). Identical geometry
(`--width 768 --height 448 --frames 49`, 2352 tokens), identical seed, identical
prompt, identical checkpoint, one `rc` lease, one build.

**Denominator.** `dgx:gpu0` (GB10), which is the box the 47.84 s baseline was
measured on. A different box is not a valid denominator for this comparison and
this row will not substitute one.

**Statistic.** Per-forward **median** and **n**, reduced from the engine's own
`last=` lines exactly as `job/profile_forward.sh` already reduces them. No
governor output, no estimator output, no mean-of-a-skewed-tail.

**Prediction.** The precedent for the size of the win is
`.agents/specs/multimodal-speed.md` §7 and §16: the Qwen3-VL tower went from the
naive kernel at 56 ms/block to the warp kernel at 4.66 ms/block, **12.0x**, and
warp-to-flash then added 1.04x at 784 tokens and 1.82x at 1500 tokens (§14),
because the flash tiling pays off with context length and LTX runs at 2352. So
~48 s should become single-digit seconds.

**Stop condition.** If the measured speedup is far from that range, this row
reports the measurement and stops. It does not tune until the number looks
right.

### 7.1 What has been measured, 2026-08-21

Lease `6c724dfd-a5e8-4832-b4fb-d0fd7d6eb458` on `dgx:gpu0` (GB10, sm_121a),
source `30dce3a1d`, one binary built in-lease at
`-DVLLM_CPP_CUDA=ON` with cutlass-nvfp4, cutlass-fp8 and FlashAttention-2 all
`ENABLED for [121a]` — the same production feature set the recorded reference
build used. Artifacts under `/workspace/ltx25-attnflash/out/20260821T092516Z/`.

**Correctness first, and it passed before any speed number was read.**

| gate | result |
|---|---|
| `test_ops_attention` on CUDA | **10/10 cases, 88,439/88,439 assertions, SUCCESS** |
| `test_ltx2_device` on CUDA | **22/22 cases, 749/749 assertions, SUCCESS** |
| CUDA-vs-host f32 parity | video `8.9407e-08`, audio `4.47035e-08`, against the committed `2e-5` |
| CUDA-vs-CPU-backend bf16 | `0` on both streams |

The f32 assertion counts are the load-bearing ones: 88,439 on CUDA against 23 on
a CPU box, so the new head_dim 64/128/256 dense-flash case really ran rather than
printing its skip line. §4.3's launch failure is therefore repaired against a
device and not against a reading of the code.

The **numerics answer is now a number**: the swap is **not bit-identical on
CUDA**, and the measured deviation is **8.94e-08 video / 4.47e-08 audio**, which
is f32 round-off scale and sits **224x inside** the gate it was held to. The gate
was not widened. Caveat stated rather than left implicit: that case runs the
fixture's reduced dimensions, so it bounds the *arithmetic* change, not the
change at head_dim 128.

**Production reachability, on the real model, from the render's own log.**
`VT_OP_PROVIDER_STATS=1` makes each op announce itself once when it resolves.
In the full 21.00B render at `768x448/49f`:

- `op=21 device=1` (`kAttentionDenseFlash` on CUDA): **1** announce — present.
- `op=18 device=1` (`kAttention` on CUDA): **0** announces — the naive op is
  never resolved at all.
- `op=19 device=1` (`kAttentionCross` on CUDA): 1 announce.

That is the same two-sided claim §6's unit case makes, taken at full scale
through `vllm_video_engine_load` on `--device cuda` rather than on a fixture.

**The flash arm, per forward.** `768x448/49f` = 2352 video tokens, seed 20260820,
full checkpoint, bf16, reduced from the engine's own `last=` lines:

| n | median | mean | min | max | spread |
|---|---|---|---|---|---|
| 19 | **7.680 s** | 7.633 s | 7.109 s | 8.196 s | 14.2% |

Against the recorded **47.84 s** denominator that is **6.23x**, inside §7's
predicted single-digit seconds.

**The naive arm did not run, so there is NO same-binary A/B yet.** At forward 20
of the flash arm the `rc` worker was **lost** — `rc devices` then read
`dgx:gpu0 unhealthy (no contact)` for at least 19 minutes. GB10 shares host RAM
with the GPU and an unconstrained job has OOM-rebooted this box before, and
`job/ab.sh` as first written carried **no memory guard and no sample cap**, unlike
the sibling campaign's `runguard.py`. That is a defect in this row's harness, not
a finding about the change.

Until the naive arm is taken **on the same binary in the same lease**, the
honest statement is:

- the flash arm is **measured**: 7.680 s median, n=19;
- the **6.23x is a cross-run comparison** against a number produced by a
  different binary in a different lease, which is exactly the weaker form the
  same-binary rule exists to replace;
- so the A/B result is **PENDING**, not satisfied.

`job/ab.sh` now carries a sample cap (13 per arm), a `MemAvailable` floor
(12 GiB, the sibling campaign's), a per-arm memory trace, a build cache keyed on
the source SHA so a resumed run does not re-spend 18 minutes compiling, and it
runs the **naive arm first** — the previous order took the cheap arm first and
lost the box before the expensive one, which is how a two-arm measurement became
a one-arm one.

## 8. Gates

Exactly one result per gate. `PENDING` names the resource it waits on; it is
never a synonym for "probably fine".

| gate | where | result |
|---|---|---|
| CPU byte-identity | `test_ltx2_device`, `test_ltx2` | **PASS** — 22/22 and 652/652; 43/43 and 4581/4581; goldens unmoved |
| reachability, unit | `test_ltx2_device`, new case | **PASS** — flash 8 of 8, naive 0 |
| reachability mutation | in-place, restored and re-gated | **PASS** — both halves red, `CHECK( 0 == 8 )` and `CHECK( 8 == 0 )`, exit 1, compile rc 0 |
| reachability, production | the real render's own log, GB10 | **PASS** — `op=21 device=1` present, `op=18 device=1` absent (§7.1) |
| CUDA host-vs-device parity | `test_ltx2_device` on `dgx:gpu0` | **PASS** — f32 8.94e-08 / 4.47e-08 vs 2e-5; bf16 0 |
| f32 head_dim 128/256 launch | `test_ops_attention` on `dgx:gpu0` | **PASS** — 10/10, 88,439 assertions (23 on a CPU box, so the case really ran) |
| A/B, same binary, both arms | `dgx:gpu0` under an `rc` lease | **PENDING** — the flash arm is measured at 7.680 s median (n=19); the worker was lost before the naive arm, so no pair exists (§7.1) |
| full preflight | `scripts/agent-preflight.sh` | **PASS** — clean before the first edit, on the staged spec, on the staged implementation, and after the `origin/main` merge |
| `build-newest-gcc` | CI | **PASS for this change, red on the branch from a pre-existing break** — `main` at `483cd3198` fails the same job on `::getpid` in `test_qwen3_dflash2_gguf.cpp:547`, a file this change does not touch; owned by [#1565](https://github.com/mudler/vllm.cpp/issues/1565) |
| `windows-msvc-cpu` / `-vulkan` | CI | **PASS for this change, baseline-less lane** — a markdown-only control PR (#1295) fails the identical step; #503/#603 own it |

One gate below the bar is named rather than averaged away: the A/B. Everything
this row claims about SPEED rests on one arm, and the row does not read as
finished until the pair exists.

## 9. Stop conditions

- `NEEDS_DECISION` if the CUDA deviation exceeds the committed tolerances in
  `test_ltx2_device.cpp`. The tolerance is not widened to admit the change.
- `NEEDS_DECISION` if `AttentionDenseFlash` refuses an LTX shape for a reason
  §4.3 does not cover.
- `NEEDS_DECISION` if the measured speedup is far from §7's prediction.
- The A/B is reported as **pending an external resource** if `dgx:gpu0` is not
  free. It is never taken on another box, and never replaced by an estimate.

## Owed

- **True tensor cores at head_dim 128.** `vt::AttentionDenseFa2` refuses
  anything but head_dim 64 (`src/vt/cuda/cuda_flash_attn_fa2.cu:557-560`).
  Reaching the vendored FA-2 `mma.sync` path for LTX's head_dim 128 needs an
  extra `run_mha_fwd_<bfloat16_t, 128, false>` instantiation. That is
  explicitly out of scope for this row, and it is the difference between this
  fix and a materially larger one: everything below is still a scalar
  warp-per-query recurrence. Owner: this row. Issue:
  [#1551](https://github.com/mudler/vllm.cpp/issues/1551).
- **The other `vt::Attention` callers.** §3's defect shape is not LTX-specific.
  A sweep of every remaining non-decode `vt::Attention` call site belongs to its
  own row with its own issue. Owner: this row until that row exists. Issue:
  [#1552](https://github.com/mudler/vllm.cpp/issues/1552).

## Outcome

Recorded when the row reaches `DONE`.
