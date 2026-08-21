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

**The bound is a property of the DEVICE, and it is derived at runtime.** Against
GB10's queried `cudaDevAttrMaxSharedMemoryPerBlockOptin` of 101,376 B
(`cuda_device_caps.h:46`), with `kFlashBc = 64` and a tile of
`2 * 64 * head_dim * sizeof(Tin)`:

| | no opt-in (49,152 B) | with the opt-in, GB10 (101,376 B) |
|---|---|---|
| bf16 | head_dim <= 192 | head_dim <= 256 |
| f32 | head_dim <= 96 | **head_dim <= 192** |

Neither column is written into the code. Pinning 192/96 caps GB10 at two thirds
of the head_dim it can serve, and pinning 256 assumes an opt-in ceiling GB10 does
not have. `cuda_device_caps.h:46` already states the rule this row follows —
*"other architectures differ and MUST be asked, not assumed"* — so the tile is
asked about per device through the same cached seam
(`vt::cuda::GetDeviceCaps` / `vt::cuda::DynamicSmemFits`).

**And a tile that does not fit is REFUSED BY NAME.** The first version of this
fix fell back to `AttentionDenseFastKernelCuda` on the argument that the flash
kernel's own header (`cuda_ops.cu:3232-3234`) makes the two bit-identical by
construction, so the fallback is numerically free. Both halves of that are true
and it is still the wrong answer: **a silent degradation to the untiled rung is
the defect this row exists to remove.** §3 is entirely about a caller sitting on
a 500x slower kernel with nothing anywhere saying so, and a fallback with no
notice reproduces it one layer down. Worse, it is undetectable by construction:
because the two kernels agree bit-for-bit, no numeric gate anywhere can tell a
fallback from a launch, which is exactly how this row's own `test_ops_attention`
case reported 10/10 and 88,439 assertions on GB10 while flash never launched once
at f32 head_dim 256 (§7.1).

So the launcher calls `vt::cuda::SetDynamicSmemOptIn`, which **throws naming the
device and the shortfall**. That function is not a new one and not a copy: it was
`cuda_paged_attn.cu`'s file-local helper, and this row moves it to the shared seam
(declared in `cuda_device_caps.h`, defined beside `DynamicSmemFits` in
`cuda_arch_tactics.cu`) rather than writing the second copy AGENTS.md "Shared
seams" forbids. `cuda_paged_attn.cu`'s six call sites are unchanged.

Two consequences worth stating plainly:

- **f32 head_dim 128 is repaired; f32 head_dim 256 is not, and cannot be on this
  device.** 131,072 B is above GB10's 101,376 B whatever this code does. It is
  now a refusal that names both numbers instead of a bare `invalid argument`.
- **`AttentionDenseFa2` -> `AttentionDenseFlash` (`cuda_ops.cu`) is unaffected.**
  That fallback only ever catches bf16 shapes at head_dim <= 256, which is
  <= 65,536 B and inside the opt-in ceiling of every architecture that has one.

**[#1578](https://github.com/mudler/vllm.cpp/pull/1578) must reconcile onto
this.** That pull request is the opposite fix for the same defect: it narrows the
ADVERTISED bound to bf16 192 / f32 96 and hard-`VT_CHECK`s above it. Those two
numbers are the left column of the table above, so they are right for a device
with no opt-in and wrong for every device that has one — on GB10 they refuse the
f32 head_dim 128 shape LTX actually runs. The two changes touch the same lines of
`src/vt/cuda/cuda_ops.cu` and `tests/vt/test_ops_attention.cpp` and conflict.
Whichever lands second reconciles onto the runtime-queried bound; this row does
not edit that branch.

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
chooses to **opt in** instead: GB10 reports 101,376 B, so the full 64-column tile
fits at f32 head_dim 128 and the K/V reuse is kept rather than halved. Where a
device's queried ceiling cannot take the tile, this row refuses and names both
numbers; it does not narrow the tile and it does not quietly answer on another
kernel.

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

Entry point: `vllm::Ltx2DitForwardDevice`, the production device forward called
from the denoise loop at `src/vllm/multimodal/ltx2_video.cpp:4245-4249` (the call
itself is on `:4246`). The test drives that function, not a hand-constructed
`vt::Attention` call. The anchor is written at HEAD and not at this row's base:
the review found `:4055-4059`, cited here and in the test, pointing at prose
about the res_2s step counter at both revisions, and this is the citation the
whole reachability argument rests on. `file::Symbol` form is what survives a
move, and `scripts/check-symbol-anchors.py` gates it.

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
| `test_ops_attention` on CUDA | **10/10 cases, 88,439/88,439 assertions, SUCCESS** |
| `test_ltx2_device` on CUDA | **22/22 cases, 749/749 assertions, SUCCESS** |
| CUDA-vs-host f32 parity | video `8.9407e-08`, audio `4.47035e-08`, against the committed `2e-5` |
| CUDA-vs-CPU-backend bf16 | `0` on both streams |

The f32 assertion counts are the load-bearing ones: 88,439 on CUDA against 23 on
a CPU box, so the new head_dim 64/128/256 dense-flash case really ran rather than
printing its skip line. §4.3's launch failure at **head_dim 128** is therefore
repaired against a device and not against a reading of the code.

**What that run did NOT prove, and the record said it did.** At f32 head_dim 256
the tile is 131,072 B and GB10's queried ceiling is 101,376 B, so the tile never
fitted; the then-current code fell back to `AttentionDenseFast`, which is
bit-identical to the flash kernel by contract, and every one of those 20,480
assertions passed **without flash launching at 256 even once**. A numeric
comparison cannot see the difference — that is the contract — so the case
measured a fallback and reported it as a launch. The claim "GB10 reports 101,376
B, so the full tile fits" was **false at 256** and is corrected here, in
`.agents/benchmark-record.md`, and in the pull request body. It remains true at
128 (65,536 B), which is the shape LTX runs and the shape this repair is for.

The case is retitled to what it exercises and the 256 arm now asserts the
device-derived outcome: launch and match where the ceiling reaches 131,072 B,
refuse and name both numbers where it does not.

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

Against the recorded **47.84 s** denominator that is 6.23x. **The headline is
~6.0x, not 6.23x**, because the denominator carries two confounds this arm does
not, and both of them inflate the ratio:

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
and pushing the same way. None of this is an A/B; see below.

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
| f32 head_dim 128 launch | `test_ops_attention` on `dgx:gpu0` | **PASS** — 10/10, 88,439 assertions (23 on a CPU box, so the case really ran). NOT head_dim 256, which never fitted and never launched; §7.1 |
| f32 head_dim 256 refusal | `test_ops_attention` on `dgx:gpu0` | **PENDING** — the refusal replaces the silent fallback the review found, and no device has run the case since. `dgx:gpu0` is leased to another row; the operator reruns this gate |
| A/B, same binary, both arms | `dgx:gpu0` under an `rc` lease | **PENDING** — the flash arm is measured at 7.680 s median (n=19); the worker was lost before the naive arm, so no pair exists (§7.1) |
| full preflight | `scripts/agent-preflight.sh` | **PASS at HEAD** — and it was NOT before: `documentation-checkpoint` was red on two of this branch's own commits (see below) |
| `documentation-checkpoint` | CI, and locally over the range | **PASS at HEAD, RED before it** — `2aa78c69b` and `2f39a9426` each recorded a measurement in `.agents/benchmark-record.md` without writing `docs/STATUS.md` (and `docs/BENCHMARKS.md` for the second). The control on the main-only range `4c193bd55..5d548d003` is rc 0, so this was the branch's own break and not an inherited one. The two commits are replaced by one that writes all three surfaces together |
| `build-newest-gcc` | CI | **PASS for this change, red on the branch from a pre-existing break** — `main` at `e2a9e035d` fails the same job on `::getpid` in `test_qwen3_dflash2_gguf.cpp:547`, a file this change does not touch; owned by [#1565](https://github.com/mudler/vllm.cpp/issues/1565) |
| `windows-msvc-cpu` / `-vulkan` | CI | **PASS for this change, baseline-less lane** — a markdown-only control PR (#1295) fails the identical step; #503/#603 own it |

**A side effect of that red is worth recording, because it was invisible.** The
`documentation-checkpoint` job runs `set -eu` and this checker is the FIRST of
three commands in the step, so `check-now-current.py` and
`check-role-discipline.py` **never ran in CI on this branch at all**. Nothing
hides behind it — both were run locally at the reviewed head and both returned
rc 0 — but a gate that stops two other gates from running is a wider failure than
its own message says.

Two gates below the bar are named rather than averaged away: the A/B, and the
head_dim 256 refusal. Everything this row claims about SPEED rests on one arm,
and the row does not read as finished until the pair exists.

## 9. Stop conditions

- `NEEDS_DECISION` if the CUDA deviation exceeds the committed tolerances in
  `test_ltx2_device.cpp`. The tolerance is not widened to admit the change.
- `NEEDS_DECISION` if `AttentionDenseFlash` refuses an LTX shape for a reason
  §4.3 does not cover.
- `NEEDS_DECISION` if the measured speedup is far from §7's prediction.
- The A/B is reported as **pending an external resource** if `dgx:gpu0` is not
  free. It is never taken on another box, and never replaced by an estimate.

## Owed

- **There is NO numeric or pixel comparison at production geometry.** The swap
  is not bit-identical on CUDA (§5), and the only numeric gate that exists is the
  reduced-dimension host-vs-device case — `8.94e-08` / `4.47e-08` against `2e-5`
  — which bounds the ARITHMETIC change and not the change at head_dim 128 with
  2352 keys over 48 layers. A diffusion render has no token gate to fall back on,
  and the flash arm was interrupted before it wrote any frames, so no pixel A/B
  exists either, not even against the completed 49-frame `768x448` baseline
  render already on the NAS. Owner: this row. Issue:
  [#1612](https://github.com/mudler/vllm.cpp/issues/1612).
- **A bf16 head_dim-256 arm for the flash op is ungated.** §4.3's table makes
  bf16 256 the widest shape that both NEEDS the opt-in (65,536 B > 48 KiB) and
  FITS GB10, so it is the case that would exercise the advertised ceiling
  end-to-end. `test_ops_attention`'s dense-flash case is f32-only, so the ceiling
  is currently asserted at 128 and refused at 256, with the shape in between
  untested. Owner: this row, under [#1612](https://github.com/mudler/vllm.cpp/issues/1612).
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
