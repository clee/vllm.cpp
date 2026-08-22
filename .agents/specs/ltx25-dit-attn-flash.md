# LTX25-DIT-ATTN-FLASH — the LTX-2.5 DiT never opted into a fast attention op, and paid 47.84 s a forward for it

Row: `LTX25-DIT-ATTN-FLASH`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)), against the
model-matrix row `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`.
Issue: [#1549](https://github.com/mudler/vllm.cpp/issues/1549).

## Now

`ACTIVE`. The routing change landed as `90e8c3c85` (#1557). What is live now is
**§10, the pixel A/B** ([#1612](https://github.com/mudler/vllm.cpp/issues/1612)):
the swap is on `main` with no comparison of what it RENDERS, and §10 is committed
before the renders are taken so the acceptance criterion cannot be read off the
numbers it judges. The same lease settles the speed A/B that §8 still carries as
`PENDING`.

The diagnosis is confirmed against the tree and the change is scoped
to **one production call site**. It was once scoped to that call site plus a
shared-memory cap repair in `LaunchAttentionDenseFlash`; that repair is
**reverted and out of scope** — see §4.3 for why, and for who owns the bound
instead. The A/B is the confirming evidence for the whole diagnosis and is not
optional; it needs `dgx:gpu0`, which is the box the denominator was measured on,
and no other box is a valid denominator for it.

Every bare `file:line` anchor below is read at this row's **base, `6b48edb2c`**,
which is `origin/main` at the claim, and every one of them has been re-read
against that SHA rather than carried forward. Two were wrong at the base as well
as at HEAD and are corrected: the reachability entry point, and the CPU
registration block. The change itself moves some of the rest —
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
`src/vllm/multimodal/ltx2_video.cpp:4238-4244`. That tick fires immediately
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

`src/vllm/model_executor/models/ltx2_device.cpp:421` at the base, the
self-attention branch of `AttentionDev`, moves from `vt::Attention` to
`vt::AttentionDenseFlash`. After this change the branch condition is `:418` and
the swapped call `:478`, with the A/B rung's `vt::Attention` at `:476`.

`vt::AttentionDenseFlash` is registered at `src/vt/cuda/cuda_ops.cu:3839-3840`
and is head_dim-generic up to 256 (`cuda_ops.cu:3335`), so LTX's video head_dim
128 and audio head_dim 64 are both served. Its contract requires a square problem
(`key.shape[0] == query.shape[0]`, `src/vt/ops.cpp:3089-3090`), which the
self-attention branch already satisfies: that branch is entered only when
`context == nullptr`, and `AttentionDev` sets `s = tq` in exactly that case
(`ltx2_device.cpp:356`).

The dispatch **rule** is unchanged. The branch is still selected by
`context == nullptr && a.bias == nullptr`, which is upstream's own
self-attention marker, and not by what the numbers happen to be. Only the op the
branch calls changes.

### 4.2 The host call site is deliberately NOT changed

`src/vllm/model_executor/models/ltx2.cpp:959` at the base (`:966` after this
change, which adds the marker comment above it) stays on `vt::Attention`. That
arm
computes into `std::vector<float>` and hands host pointers to
`Tensor::Contiguous`, so it is CPU-only by construction. On CPU
`kAttentionDenseFlash` is registered to the **same** `AttentionKernel` as
`kAttention` (`src/vt/cpu/cpu_ops.cpp:3750-3761`), so the swap would be a
byte-identical no-op that buys nothing and moves the L2 parity reference off the
reference op. The reference arm stays on the reference kernel.

It carries a `// VT-ATTN-NAIVE:` marker recording that reason beside the call, in
the form [#1578](https://github.com/mudler/vllm.cpp/pull/1578)'s
`scripts/check-attention-rung-consistency.py` reads: a `//` comment on the call
line or within the 20 lines above it, with a substantive reason after the colon.
Same for the A/B rung in `ltx2_device.cpp` (§4.4). That checker carries `ltx2`
and `ltx2_device` on `scripts/attention-rung-allowlist.txt` because this row was
in flight when it was written; once #1578 is on `main` both stems will report
`STALE (not a failure)` — the marked-and-therefore-no-longer-excused state its
allowlist header describes — and whoever runs preflight next may delete them,
which also edits that checker's own pinned-set test.

### 4.3 The shared-memory bound: attempted here, REVERTED, and owned by #1578

**This row does not touch `LaunchAttentionDenseFlash`.** An earlier version of it
did, and the reason it no longer does is the useful part of this section.

`LaunchAttentionDenseFlash` (`src/vt/cuda/cuda_ops.cu:3330-3348`) sizes its K/V
tile at

    shmem = 2 * kFlashBc(64) * head_dim * sizeof(Tin)

and never calls `cudaFuncSetAttribute`, so it is capped at the 48 KiB of dynamic
shared memory every CUDA architecture guarantees without an opt-in. This row
originally read that as a prerequisite: raise the cap, then swap the call site.

**The swap does not need it.** LTX's DiT runs at the stream dtype, which is
**bf16 in production** (`ltx2_device.cpp:1187` after this change, `:1166` at the
base: "the device stream dtype is bf16 (production) or f32 (the L2 parity arm)"),
and the video stream's head_dim is 128:

    2 * 64 * 128 * 2 = 32,768 B

which is inside the 49,152 B no-opt-in cap with room to spare. The audio stream
at head_dim 64 asks 16,384 B. **Neither shape ever needed the opt-in**, and the
7.680 s in §7.1 was produced by a bf16 render that would have launched
identically without it.

**The cap-raise was serving a shape that does not fit at all.** What actually
exceeds 48 KiB on this path is f32 at head_dim 128 (65,536 B) and f32 at head_dim
256 (131,072 B). GB10's queried
`cudaDevAttrMaxSharedMemoryPerBlockOptin` is **101,376 B**
(`cuda_device_caps.h:46`), so f32 256 does not fit **even with the opt-in** — it
was falling back to the bit-identical `AttentionDenseFast` and reporting a launch
it never made. Raising the cap therefore bought this row one shape it does not
run (f32 128), at the price of moving a shared helper across two files and
colliding with another row on the same lines.

**So it is reverted in full**, and the four files it touched —
`src/vt/cuda/cuda_ops.cu`, `cuda_device_caps.h`, `cuda_arch_tactics.cu`,
`cuda_paged_attn.cu` — plus `tests/vt/test_ops_attention.cpp` are byte-identical
to `main`. `vt::cuda::SetDynamicSmemOptIn` stays the file-local helper of
`cuda_paged_attn.cu` that it always was.

**[#1578](https://github.com/mudler/vllm.cpp/pull/1578) owns the bound, and it
takes the opposite and correct approach.** Instead of raising the cap it makes
the ADVERTISED domain honest: `AttentionDenseFlash` declares `head_dim <= 256`
while it can only launch bf16 192 / f32 96, and #1578 narrows the declaration to
what the code can actually do and `VT_CHECK`s above it. That is the
`supports_head_size()` polarity vLLM already has
(`vllm/v1/attention/backend.py:155-163`), and it is a better answer than an
opt-in because it is a property of the code rather than of whichever device
happens to be under it. #1578 merges first; after it, bf16 head_dim 128 is inside
the declared bound and this row's swap is unaffected.

**One consequence is disclosed rather than left to be found.** With the cap-raise
gone, the f32 L2 parity arm at production geometry (head_dim 128, 65,536 B) now
reaches `AttentionDenseFlash` and cannot launch: before #1578 that is a
`cudaGetLastError` throw at `cuda_ops.cu:3352`, and after #1578 it is a
`VT_CHECK` naming the head_dim. **It fails loud in both worlds and never
silently**, which is why it is a disclosure and not a blocker. It is not reached
by anything gated: the f32 arm is a parity reference exercised at the fixture's
reduced dimensions (§5), production is bf16, and an f32 render at 2352 tokens on
the naive kernel would have taken hours anyway. It is filed under `## Owed`.

**The original diagnosis is still on the record and still true**, because it was
not a reading of the code. `src/vt/cuda/cuda_attention_cross.cu:88-96` — a kernel
written for this very model, and itself a structural port of
`AttentionDenseFlashKernel` — says:

> LTX-2.5's video stream is head_dim 128, and at f32 a fixed 64-column tile would
> ask for `2 * 64 * 128 * 4` = 64 KiB of dynamic shared memory — over the 48 KiB
> a launch gets without opting in — so the kernel would fail to launch on exactly
> the real geometry while every reduced-dimension gate (head_dim 8 and 4) passed.

Same number, same arithmetic, same reason the existing gates cannot see it. That
author halved the tile (`ChooseTileCols`). This row now does neither: it runs the
bf16 shape, which fits, and leaves the f32 shape to the row that owns the
contract.

Two things follow that are worth keeping. The first is that "every
reduced-dimension gate passed" is why this row's numeric evidence is stated as
bounding the ARITHMETIC change and nothing more (§5, §7.1, `## Owed`). The second
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
registered function pointer (`src/vt/cpu/cpu_ops.cpp:3750-3761`). Every existing
CPU golden case in `tests/vllm/models/test_ltx2_device.cpp` must stay
byte-unchanged, and if one moves, the change is wrong.

**On CUDA: NOT bit-identical, and this row does not pretend otherwise.**
`AttentionDenseFast` differs from `Attention` in how the head_dim partial sums
are grouped (`include/vt/ops.h:3304-3306`): the naive kernel reduces across a
256-thread block, the warp kernel across 32 lanes with `__shfl_xor`. The
arithmetic is the same f32 online softmax; the association order is not.
`AttentionDenseFlash` is then bit-identical to `AttentionDenseFast`
(`include/vt/ops.h:3307-3317`), so the whole difference is naive-vs-warp
regrouping.

The magnitude is therefore a floating-point reassociation of a length-`head_dim`
sum, not an algorithmic change. The binding gate is the existing CUDA
host-vs-device parity case `tests/vllm/models/test_ltx2_device.cpp:753`, which
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
`vt::GetOpProviderStats(OpId, DeviceType)` (`include/vt/op_provider.h:166-186`),
which counts dispatches through `GetOp`. This is a positive signal and it works
on the **CPU backend**, so the proof does not need a GPU.

Entry point: `vllm::Ltx2DitForwardDevice`, the production device forward called
from the denoise loop at `src/vllm/multimodal/ltx2_video.cpp:4246`. The test
drives that function, not a hand-constructed `vt::Attention` call.

**That anchor was wrong and it is the one the whole reachability argument rests
on.** This spec and the test both cited `:4055-4059`, which points at prose about
the res_2s step counter — at this row's base `6b48edb2c` and at HEAD alike, so
re-reading it at either revision would have caught it and neither did. `:4246` is
verified at both: it is the `im.on_device ? Ltx2DitForwardDevice(...)` ternary
inside the denoise loop, with the progress tick that emits §1's `last=` samples
seven lines above it at `:4238-4244`. `file::Symbol` form is what survives a
move, and `scripts/check-symbol-anchors.py` gates that form.

The assertion is **two-sided**, because a one-sided count cannot tell a routed
call from an added one:

1. `kAttentionDenseFlash` selections > 0 after one forward, and
2. `kAttention` selections == 0 after that same forward.

Half (2) is what makes it a routing proof: LTX's cross-attentions use
`kAttentionCross`, so after the swap there is no remaining `kAttention` caller
anywhere in the DiT forward. If somebody adds a second, unswapped self-attention
path later, that half goes red.

Mutation: delete the `vt::AttentionDenseFlash` call at
`ltx2_device.cpp:478` in a scratch copy, restoring `vt::Attention`, and rerun the
focused gate. Both halves must go red. The result is recorded in `## Outcome`
with the exact diff applied and the exact failure text, and the tree is restored
byte-for-byte afterwards.

The case's own process state is scope-guarded, and that is not tidiness. It
enables `vt::EnableOpProviderCallStats` and sets `VLLM_LTX2_DIT_FLASH_ATTN`, both
of which are per-process, and it contains `REQUIRE`s that unwind. Measured with a
scratch `REQUIRE(false)` after the `SetEnv` plus an observer case appended after
it: **without the guards the observer reads the knob still set and counts 8
leaked `kAttention` selections; with them it reads neither.** An unrelated case
failing because of which case failed before it is the shape this removes.

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
| `test_ltx2_device` on CUDA | **22/22 cases, 749/749 assertions, SUCCESS** |
| CUDA-vs-host f32 parity | video `8.9407e-08`, audio `4.47035e-08`, against the committed `2e-5` |
| CUDA-vs-CPU-backend bf16 | `0` on both streams |

**Which of this survives the §4.3 revert, stated rather than assumed.** The
lease also ran `test_ops_attention` at 10/10 and 88,439 assertions, and that
result is **withdrawn from this row's evidence**: it measured the cap-raise's
head_dim 64/128/256 dense-flash case, which is no longer in this branch. Two of
its arms never proved what they were read as proving in any case — f32 256 could
not fit GB10's 101,376 B ceiling and was silently answered by the bit-identical
`AttentionDenseFast`, so a numeric comparison could not tell a fallback from a
launch. The row claims nothing from that binary's `test_ops_attention` run.
[#1578](https://github.com/mudler/vllm.cpp/pull/1578) owns that file's head_dim
coverage.

**The `test_ltx2_device` rows and the render below DO survive it**, and that is
arithmetic rather than assertion. The measured binary carried the opt-in call,
but LTX renders bf16 at head_dim 128, whose tile is `2 * 64 * 128 * 2` =
**32,768 B**, and `SetDynamicSmemOptIn` returns immediately below 49,152 B
without touching the kernel. The opt-in was therefore a **no-op on every launch
these numbers came from**, so removing it cannot move them. The f32 parity case
runs the fixture's reduced dimensions and is below the cap by a wider margin
still.

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

Against the recorded **47.84 s** denominator that is 6.23x, and **6.23x is not
the number to quote.** The denominator carries two confounds this arm does not,
and both inflate the ratio. **The headline is ~6.0x:**

**Instrumentation.** The denominator ran under `job/runguard.py --stack-period 12`
(`render.log:1`), which samples `eu-stack -p` and therefore `ptrace`-stops every
thread in the process. Its own `stacks.txt` says what that cost: **523 samples,
median inter-sample delta 12.40 s against a 12.0 s period**, so ~0.40 s median
and 1.50 s maximum of stopped process per sample. At that cadence ~3.9 samples
land inside each 47.84 s forward, which is ~1.54 s, or **~3.2% of the
denominator**. The flash arm ran with no such sampler. Correcting only the
denominator gives **46.3 s / 7.680 s = 6.03x**.

**A different prompt.** The denominator's `render.log:1` carries a ~70-word
prompt; `scripts/ltx25-dit-attn-flash-ab.sh` uses one short sentence.
`ltx2_video.cpp:2253` sets `context_tokens = encoded.seq` **unpadded**, so the
DiT's four cross-attentions per block see a different number of keys in the two
arms. It is corroborated rather than inferred: `conditioning.tower` runs
**45.013 s** in the denominator against **28.426 s** in the flash arm. The sign
is the same as the sampler's — the longer prompt makes the denominator's forward
more expensive for a reason that is not the self-attention kernel — so it inflates
the ratio too, and it is not quantified here.

**So the honest statement is a range: 6.03x to 6.23x, and ~6.0x is the value to
quote**, with the sampler correction named and the prompt confound uncorrected
and pushing the same way. `6.23x` appears in this row's records only as the
uncorrected upper end of that range, never on its own. None of this is an A/B;
see below.

**The naive arm did not run, so there is NO same-binary A/B yet.** At forward 20
of the flash arm the `rc` worker was **lost** — `rc devices` then read
`dgx:gpu0 unhealthy (no contact)`, and it still did 43 minutes later.

**The cause is UNPROVEN and this row does not name one.** No memory trace was
taken, the worker's own log ends mid-forward, and the box did not come back to be
asked. Host-RAM exhaustion is the leading hypothesis only because GB10 shares
host RAM with the GPU and an unconstrained job has OOM-rebooted this box before;
that is a prior, not evidence. What IS established is that `job/ab.sh` as first
written carried **no memory guard, no sample cap and no memory trace**, unlike the
sibling campaign's `runguard.py` — so the run could neither avoid the failure nor
say afterwards what it was. That is a defect in this row's harness, and it is the
reason the next attempt can answer the question this one cannot.

Until the naive arm is taken **on the same binary in the same lease**, the
honest statement is:

- the flash arm is **measured**: 7.680 s median, n=19;
- the **~6.0x is a cross-run comparison** against a number produced by a
  different binary, in a different lease, under a stack sampler, on a different
  prompt — which is exactly the weaker form the same-binary rule exists to
  replace;
- so the A/B result is **PENDING**, not satisfied.

**The flash arm's artifacts do not say what was run, and that is a defect of this
row's harness rather than a caveat about it.** `arm-flash.log` opens at
`[render] + load` with no command line, `wd-flash/` is empty, and no
`phase-log.json` was written; the only description of the run was
`/mnt/nas_share/rc/ltx25-attnflash/job/ab.sh`, a mutable path on a share, whose
mtime is 25 minutes AFTER the run finished. The denominator has its full command
line as line 1 of its own `render.log` and this arm has nothing. So the geometry,
prompt, seed and sample cap behind 7.680 s are not verifiable from its own
evidence, which is the reason the two confounds above had to be established from
a `conditioning.tower` duration rather than read off a recipe.

Both halves are repaired. The harness is now
**`scripts/ltx25-dit-attn-flash-ab.sh`**, committed and therefore immutable per
revision, and every arm writes its own invocation — harness sha256, binary
sha256, source SHA, geometry, seed, prompt and the resolved command line — to
line 1 of its own log, plus a `harness_sha256` line into `PROVENANCE`. It also
carries a sample cap (13 per arm), a `MemAvailable` floor (12 GiB, the sibling
campaign's), a per-arm memory trace, a build cache keyed on the source SHA so a
resumed run does not re-spend 18 minutes compiling, and it runs the **naive arm
first** — the previous order took the cheap arm first and lost the box before the
expensive one, which is how a two-arm measurement became a one-arm one.

## 8. Gates

Exactly one result per gate. `PENDING` names the resource it waits on; it is
never a synonym for "probably fine".

| gate | where | result |
|---|---|---|
| CPU byte-identity | `test_ltx2_device`, `test_ltx2` | **PASS** — 22/22 and 652/652; 43/43 and 4581/4581; goldens unmoved |
| reachability, unit | `test_ltx2_device`, new case | **PASS** — flash 8 of 8, naive 0 |
| reachability mutation | in-place, restored and re-gated | **PASS** — both halves red, `CHECK( 0 == 8 )` and `CHECK( 8 == 0 )`, exit 1, compile rc 0 |
| reachability, production | the real render's own log, GB10 | **PASS** — `op=21 device=1` present, `op=18 device=1` absent (§7.1) |
| instrument scope guard, mutation | `test_ltx2_device`, in-place, restored | **PASS** — without the guards an appended observer reads the knob still set and 8 leaked `kAttention` selections; with them, neither (§6) |
| CUDA host-vs-device parity | `test_ltx2_device` on `dgx:gpu0` | **PASS** — f32 8.94e-08 / 4.47e-08 vs 2e-5; bf16 0 |
| bf16 head_dim 128 tile fits without an opt-in | arithmetic, `cuda_ops.cu:3338` | **PASS** — `2 * kFlashBc(64) * 128 * sizeof(bf16)` = 32,768 B against the 49,152 B every architecture gives without an opt-in. This is the whole of what the swapped shape needs from §4.3, and it is a property of the code, so no device is owed for it |
| the flash op's advertised head_dim bound | not this row | **NOT A GATE HERE** — §4.3's cap-raise is reverted, so `LaunchAttentionDenseFlash` and `tests/vt/test_ops_attention.cpp` are byte-identical to `main`. Owned by [#1578](https://github.com/mudler/vllm.cpp/pull/1578), which merges first |
| A/B, same binary, both arms | `dgx:gpu0` under an `rc` lease | **PENDING** — the flash arm is measured at 7.680 s median (n=19); the worker was lost before the naive arm, so no pair exists (§7.1) |
| pixel A/B at production geometry | `dgx:gpu0` under an `rc` lease, `scripts/ltx25-dit-attn-flash-pixel-ab.sh` | **PENDING** — criterion registered in §10.4; result in §10.7 |
| run-to-run control (`flash` twice) | the same lease | **PENDING** — §10.3; without it no arm-to-arm delta is attributable to the kernel |
| the comparison tool discriminates | `tests/scripts/test_ltx25_render_compare.py` | **PENDING until §10.7** — a dither passes and a one-pixel shift fails all four checks (§10.4) |
| full preflight | `scripts/agent-preflight.sh` | **PASS at HEAD** — and it was NOT before: `documentation-checkpoint` was red on two of this branch's own commits (see below) |
| `documentation-checkpoint` | CI, and locally over the branch range | **PASS at HEAD, RED before it, and the red was THIS BRANCH's** — `2aa78c69b` and `2f39a9426` each recorded a measurement in `.agents/benchmark-record.md` without writing `docs/STATUS.md` (and `docs/BENCHMARKS.md` for the second). The control on the main-only range `4c193bd55..5d548d003` is rc 0, so it was not inherited. Both commits were replaced by one that writes all three surfaces together when the branch was rebuilt, and the checker is re-run at each head rather than trusted to have stayed fixed — a job that has stopped appearing in a failing set is not the same fact as a job that passes |
| `build-newest-gcc` | CI | **PASS, and now green on `main` too** — it was red on `main` on `::getpid` in `test_qwen3_dflash2_gguf.cpp:547`, a file this change does not touch; [#1581](https://github.com/mudler/vllm.cpp/pull/1581) fixed it and this branch carries that fix through the merge. A red here after the merge is therefore this row's, not inherited |
| `build-test-cpu`, `sanitize-cpu` (both) | CI | **INHERITED** — all three fail on the same single case, `test_runner.cpp:1557`, from #1273; owned by [#1602](https://github.com/mudler/vllm.cpp/issues/1602) and [#1608](https://github.com/mudler/vllm.cpp/issues/1608). Verified against `main` with `scripts/main-baseline.py`, not by reading a push run: those are all cancelled (#274) |
| `windows-msvc-cpu` / `-vulkan` | CI | **INHERITED, baseline-less lane** — a markdown-only control PR (#1295) fails the identical step; #584/#965 own it |

**A side effect of that red is worth recording, because it was invisible.** The
`documentation-checkpoint` job runs `set -eu` and this checker is the FIRST of
three commands in the step, so `check-now-current.py` and
`check-role-discipline.py` **never ran in CI on this branch at all**. Nothing
hides behind it — both were run locally at the reviewed head and both returned
rc 0 — but a gate that stops two other gates from running is a wider failure than
its own message says.

**One gate is below the bar and it is named rather than averaged away: the A/B.**
Everything this row claims about SPEED rests on one arm, and the row does not
read as finished until the pair exists. The head_dim refusal that used to sit
beside it here is gone from the list because the change it gated is reverted, not
because it passed.

## 9. Stop conditions

- `NEEDS_DECISION` if the CUDA deviation exceeds the committed tolerances in
  `test_ltx2_device.cpp`. The tolerance is not widened to admit the change.
- `NEEDS_DECISION` if `AttentionDenseFlash` refuses the bf16 head_dim 128 or 64
  shape LTX actually runs. §4.3 shows both fit without an opt-in, so a refusal
  there falsifies this row's premise rather than needing a workaround.
- `NEEDS_DECISION` if the measured speedup is far from §7's prediction.
- The A/B is reported as **pending an external resource** if `dgx:gpu0` is not
  free. It is never taken on another box, and never replaced by an estimate.
- **The pixel thresholds in §10.4 are never widened to admit the change.** A
  failing check is a finding about a change already on `main`, filed as its own
  issue with what diverged and by how much (§10.5), and it is never repaired by
  moving a number in `scripts/ltx25-render-compare.py`.

## 10. The pixel A/B — what the model RENDERS, designed before it is read

Issue: [#1612](https://github.com/mudler/vllm.cpp/issues/1612). This section is
written and committed **before the renders are taken**, because a criterion read
off the numbers it is meant to judge is not a criterion.

### 10.1 Why this section has to exist at all

Every other model in this tree leans on a token gate. Greedy decode gives a
discrete output, and either the tokens match the oracle or they do not. **A
diffusion render has no such output.** The DiT emits latents, the VAE emits
pixels, and nothing in that chain is a symbol that can be compared for equality
against a reference by construction.

So when §5 says the swap is **not bit-identical on CUDA**, the usual net is
absent. What remains is the reduced-dimension host-vs-device parity case
(`8.94e-08` video, `4.47e-08` audio against `2e-5`), and §7.1 already states its
limit in the same breath as its result: it bounds the *arithmetic* change — a
length-`head_dim` sum reassociated — at the fixture's dimensions. Production is
bf16, head_dim 128, 2352 keys, 32 heads, 48 layers, 120 forwards. Nothing has
measured that.

### 10.2 What bf16 predicts, which is that the frames WILL differ

This is a prediction registered in advance, not a result.

`vt::Attention` and `vt::AttentionDenseFlash` run the same f32 online softmax
and differ only in association (`include/vt/ops.h:3304-3306`). Two summation
orders of an `n`-term f32 sum differ by roughly `sqrt(n) * u` in the
random-walk regime, with `u = 2^-24 = 5.96e-08` the f32 unit roundoff:

| axis reassociated | `n` | relative deviation |
|---|---|---|
| the head_dim dot product, which §5 names | 128 | `~6.7e-07` |
| the key-axis online accumulation | 2352 | `~2.9e-06` |

The DiT then **stores that result to bf16**, whose spacing is `2^-7 = 7.81e-03`
relative — four orders of magnitude coarser. A perturbation of relative size
`d` moves the rounded bf16 value only when the exact value sits within `d` of a
rounding boundary, so the per-element flip probability is `d / 7.81e-03`:
between `8.6e-05` and `3.7e-04`.

The video stream carries `2352 * 32 * 128 = 9.63e6` attention output elements
per layer and `4.62e8` over 48 layers, so **one forward injects between 4.0e4
and 1.7e5 single-ULP bf16 flips**, and a render is 120 forwards inside a
nonlinear sampler that feeds each step's output into the next.

**Three things follow, and the third is the one that shapes the gate.**

1. **Bit-identical frames are not the expected outcome.** Predicting them and
   then finding a difference would make any threshold read as a retrofit.
2. **The floor is not zero.** "Within bf16 noise" has to mean something other
   than equality, because equality is not what the arithmetic predicts.
3. **A tight a-priori pixel bound is NOT derivable.** The sampler is nonlinear
   and iterative; whether ~1e5 ULP flips per step damp or amplify over 120 steps
   is an empirical property of this model at this geometry, not something the
   error analysis above can be pushed to answer. Anyone who claims to derive one
   is deriving it from an assumption of contraction that nothing here has
   measured.

So the bound is not derived from the arithmetic. It is derived from two things
that exist independently of this experiment: a convention the video-coding field
already agreed on, and the render's own scale.

### 10.3 The design: three renders, and the third is the whole argument

One binary, built once, in one `rc` lease on `dgx:gpu0`, on one staged
checkpoint set, at `768x448/49f` (2352 tokens), seed `20260820`, with the exact
70-word prompt of the recorded 20260820 baseline.

| # | render | knob | what it is |
|---|---|---|---|
| 1 | `flash` | `VLLM_LTX2_DIT_FLASH_ATTN=1` | the arm #1549 shipped |
| 2 | `naive` | `VLLM_LTX2_DIT_FLASH_ATTN=0` | the arm it replaced |
| 3 | `flash-ctl` | `VLLM_LTX2_DIT_FLASH_ATTN=1` | **flash again**, same binary, same seed |

**(3) is not a spare. It is the control, and without it the experiment does not
answer its own question.** A difference between (1) and (2) is only attributable
to the kernel if the box produces the same render twice when nothing changes.
cuBLAS reduction splits, allocator-dependent kernel selection and any
nondeterminism anywhere in 120 forwards plus a VAE decode would otherwise sit
inside the measured delta with no way to separate them. So:

- **control == 0** (bit-identical): the noise floor is exactly zero, and every
  bit of the flash-vs-naive delta is the swapped op. The strongest attribution
  available.
- **control ~= treatment**: the swap changed nothing the machine does not change
  by itself. The strongest possible *null* result, and it can only be stated
  because the control was taken.
- **control > 0 and < treatment**: the delta is partly kernel, and the control
  is the floor the thresholds must be read against.

Every one of those is an answer. None is available from two renders, which is
what #1612 asked for and what this section deliberately exceeds.

**Order: flash, naive, flash-ctl.** The naive arm is ~6x the wall clock and it
is the one whose loss leaves no A/B at all, so it is taken while the box is
known good rather than last. The control goes last because it is the only one
recoverable cheaply: the harness caches the binary keyed on the source sha, so a
follow-up lease reaches a render in minutes rather than re-spending the build.

**Both arms are instrumented identically and neither is stack-sampled.** §7.1
established that `runguard.py --stack-period 12` cost the recorded 47.84 s
denominator ~3.2% by `ptrace`-stopping every thread. This harness runs a
`MemAvailable` watchdog and nothing else, so the speed pair it produces needs no
sampler correction and finally replaces the cross-run 6.03-6.23x range with a
same-binary ratio.

**Routing is proved per arm, two-sided, from that arm's own log.**
`VT_OP_PROVIDER_STATS=1` makes each op announce itself once when it resolves.
The flash arm must show `op=21 device=1` **and no** `op=18 device=1`; the naive
arm the reverse. A one-sided count cannot tell a routed call from an added one,
and "the knob was exported" is not evidence that the branch was taken.

### 10.4 The registered acceptance criterion

Committed as the defaults of `scripts/ltx25-render-compare.py`, so the numbers
are read by a tool that already holds the thresholds rather than compared to
them by hand afterwards.

| # | check | threshold | where it comes from |
|---|---|---|---|
| V1 | mean \|delta\|, 8-bit RGB | `<= 1.0` level | one level is the quantisation step of the artefact itself; a mean below it says the average pixel is within the PPM's own resolution |
| V2 | worst-frame PSNR | `>= 40 dB` | the video-coding "visually lossless" convention. This experiment did not choose it |
| V3 | worst-frame SSIM | `>= 0.99` | Wang et al. 2004, 11x11 Gaussian sigma=1.5 on luma. 0.98 is the usual transparency line; this is stricter, and it is the WORST frame rather than the mean |
| V4 | mean \|delta\| on luma / arm A's mean adjacent-frame MAD | `<= 0.10` | **the self-calibrating one** |
| A1 | audio PSNR vs full scale | `>= 40 dB` | same convention as V2 |
| A2 | audio Pearson r | `>= 0.999` | a waveform that has drifted in time fails this while PSNR can still look tolerable |

**V4 is the bound that is derived rather than borrowed, so it carries the
argument.** The denominator is the render's own frame-to-frame step: how much
one frame differs from the next, in the same 8-bit luma units as the numerator.
`0.10` therefore says *the two arms differ by less than a tenth of one frame of
this video's own motion*. It needs no re-argument at another geometry or another
prompt, because both terms move together — which is exactly what a constant
cannot do, and why the FA-2 arm (#1551) can take this same criterion.

**The scale is executable, not asserted.** Measured on the recorded 20260820
baseline's own frames, and pinned in `tests/scripts/test_ltx25_render_compare.py`
on synthetic fixtures so it is a gate rather than a claim:

| perturbation of a real frame | mean \|d\| | PSNR | SSIM | V4 ratio | verdict |
|---|---|---|---|---|---|
| none | 0 | inf | 1.000000 | 0.000 | bit-identical |
| +/-1 LSB on 3% of samples | 0.0199 | 65.1 dB | 0.99992 | 0.0026 | **PASS**, all four |
| one pixel of global horizontal shift | 5.183 | 28.1 dB | 0.8705 | 0.624 | **FAIL**, all four |

So the thresholds sit between a dither and a single pixel of motion, nearer the
dither: V4 at `0.10` refuses anything above a sixth of a one-pixel global shift.
A criterion that admitted the shift row would not be a criterion.

### 10.5 Reading the result, stated before there is one

- **All checks pass and the control is 0** — the swap changes the render by less
  than a tenth of its own motion step, the difference is entirely the kernel's,
  and the verdict is **within bf16 noise**. The measured values, with their
  headroom, are then what belongs in the record.
- **All checks pass and the control is comparable to the delta** — the verdict is
  **indistinguishable from run-to-run nondeterminism**, which is stronger, and
  the row additionally owes an issue for the nondeterminism itself, because a
  render that is not reproducible is its own defect.
- **Any check fails** — the verdict is **visibly different**, and that is a
  finding about a change already on `main`, not a failure of this work. It owes
  an issue naming what diverged and by how much, and it does not owe a widened
  threshold. Widening a gate to admit a change is the failure this protocol
  exists to prevent, and §9's stop conditions already say so for the numeric
  gate.

### 10.6 What this deliberately does not measure

- **PPM is 8-bit.** The comparison is on the artefact the pipeline writes, which
  is already quantised from the VAE's float output. A difference below `1/255`
  relative is invisible to it. That is the right resolution for the question
  "does it render the same video" and the wrong one for "how large is the
  latent-space deviation"; the second is the host-vs-device case's job and it is
  answered at the fixture's dimensions in §7.1.
- **One prompt, one seed, one geometry.** A sampler is chaotic and one trajectory
  is one trajectory. This bounds the swap on the trajectory production actually
  ran and recorded, and it does not claim a bound over the prompt distribution.
- **The 20260820 NAS baseline is a cross-check, never the control.** It was built
  from `a50c57d69`, an ancestor of the swap, so it is a different binary lineage:
  everything else that landed on `main` in between sits inside any delta measured
  against it. It is compared anyway, because how far two naive renders drift
  across builds bounds how much of the A/B delta could be something other than
  the kernel — but the same-binary pair is the evidence and this is context.

## Owed

- **There is NO numeric or pixel comparison at production geometry.**
  **§10 is the design that discharges this, and it is committed before the
  renders are taken.** The result lands in §10.7. Until it does, the statement
  below is still the honest one. The swap
  is not bit-identical on CUDA (§5), and the only numeric gate that exists is the
  reduced-dimension host-vs-device case — `8.94e-08` / `4.47e-08` against `2e-5`
  — which bounds the ARITHMETIC change and not the change at head_dim 128 with
  2352 keys over 48 layers. A diffusion render has no token gate to fall back on,
  and the flash arm was interrupted before it wrote any frames, so no pixel A/B
  exists either, not even against the completed 49-frame `768x448` baseline
  render already on the NAS. Owner: this row. Issue:
  [#1612](https://github.com/mudler/vllm.cpp/issues/1612).
- **The f32 L2 parity arm cannot run at production geometry any more.** With
  §4.3's cap-raise reverted, that arm reaches `AttentionDenseFlash` at head_dim
  128, whose f32 tile is 65,536 B and does not fit the 49,152 B a launch gets
  without an opt-in. It fails LOUD either way — a `cudaGetLastError` throw at
  `cuda_ops.cu:3352` today, a `VT_CHECK` naming the head_dim once
  [#1578](https://github.com/mudler/vllm.cpp/pull/1578) lands — and never
  silently, which is why it is disclosed rather than blocking. Nothing gated
  reaches it: production is bf16, and the f32 arm is a parity reference exercised
  at the fixture's reduced dimensions (§5). Owner: this row, under
  [#1612](https://github.com/mudler/vllm.cpp/issues/1612).
- **`scripts/attention-rung-allowlist.txt` will carry two STALE stems.** Once
  #1578 is on `main`, its checker sees the `// VT-ATTN-NAIVE:` markers this row
  adds and reports `ltx2` and `ltx2_device` as `STALE (not a failure)`. Deleting
  the two stems also edits that checker's pinned-set test, which is #1578's file
  and not this row's, so it is left to whoever runs preflight next — the handoff
  that allowlist's own header describes. Owner: this row until it is deleted.
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
