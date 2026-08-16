# LTX25-DECODE-THREADS — the decode runs on one core of twenty, and the seam it needs already exists

Row: `LTX25-DECODE-THREADS`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)).
Issue: [#1009](https://github.com/mudler/vllm.cpp/issues/1009).
Parent: lever 3 of the `LTX25-DECODE-SPEED` investigation
([#1006](https://github.com/mudler/vllm.cpp/issues/1006)), which filed this
issue and lists it under `## Owed`. That spec is
`.agents/specs/ltx25-decode-speed.md` on [PR
#1018](https://github.com/mudler/vllm.cpp/pull/1018) and is **not yet on
`main`**, so it is cited by pull request rather than by relative link, exactly as
the sibling dtype row ([`ltx25-decode-dtype.md`](ltx25-decode-dtype.md)) does.

Sibling, and the reason this row is riskier than it looks:
[`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) (#1008) landed at `d1b0ea3a8`
and changed the convolution's **summation order** to a blocked one. This row
adds parallelism on top of that, and parallelism is the second thing that can
change a summation order.

## Now

`ACTIVE`.

## 0. Scope

**In scope.** Route the LTX-2.5 conv video VAE's convolution loops through
`vt::cpu::ParallelForRows`, the synchronous row-chunked parallel-for that 10+
CPU kernels in this tree already use and that no line of the video VAE uses
today.

**Not in scope, and deliberately so.**

* The device arm ([#1007](https://github.com/mudler/vllm.cpp/issues/1007)).
  There is no `vt::` conv3d op on any backend; that is a much larger change and
  needs NDHWC first.
* NDHWC / memory format ([#1008](https://github.com/mudler/vllm.cpp/issues/1008)
  §5 records the verdict and the blocker, `MiniMaxH3GroupNorm3d`'s signature).
* `memory_efficient_decode.py` ([#1011](https://github.com/mudler/vllm.cpp/issues/1011)).
* SIMD. A vectorised inner tap loop is a separate change with a separate
  summation-order question, and mixing the two would make an order regression
  unattributable.
* The **audio** VAE and the video **encoder**'s non-convolution paths. The
  encoder shares `CausalConv3d`, so it inherits the change; nothing else in
  either file is touched.

**No end-to-end render number.** `dgx.casa` is unreachable and this box has no
GPU, so this row claims no render speedup and no ratio against any oracle. What
it can measure, and does, is a same-binary wall-clock A/B of the decode itself
at fixed thread counts on 20 local cores (§6).

## 1. Why there is no upstream to mirror here

Every oracle runs this decoder on an accelerator and none of them has a
host-parallel arm to port:

* Lightricks LTX-2 @ `fd4ded7f2` builds the decoder onto a device
  (`packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1139`,
  `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:267-288`);
  the whole decoder's convolution work is one `nn.Conv3d` call at
  `model/video_vae/convolution.py:312`.
* SGLang @ `f63458b5b` moves the latents to the local torch device
  (`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/decoding_av.py:71`).
* vLLM-Omni @ `a4ea67a21` states the contract outright — *"VAE(s) (always on
  GPU)"*, `vllm_omni/diffusion/models/interface.py:92`.
* `diffusers` @ `3a2f35d4e` ships no CPU decode path at all
  (`ltx2_diffusion_decoder.py:208-209`, *"No CPU path"*).

So this row is a **local seam**, not an upstream mirror, and
[`ltx25-decode-speed.md`](ltx25-decode-speed.md) §6 lever 3 records it as such.
What it does mirror is *this tree's own* CPU convolution:
`src/vt/cpu/cpu_conv2d.cpp:75-78 @ d1b0ea3a8` partitions a 2-D convolution over
`n * cout * hout` output lines through the same call, with the same comment this
row's change carries — *"independent outputs, so the partition can never change a
reduction"*.

## 2. The axis, and why it is reduction-safe

`CausalConv3d`'s output loop nest is `oc / ti / hi / wi`
(`src/vllm/model_executor/models/ltx2_video_vae.cpp:178-217 @ d1b0ea3a8`). The
`ci * kernel^3` reduction lives **entirely inside one `(oc, ti, hi, wi)` body**,
in the blocked order #1008 shipped: one `float tap` partial per input channel,
added into `float acc`.

**The parallel axis is the output line `(oc, ti, hi)`, and each unit is
`out.w` contiguous output elements.** `Volume::At(oc, ti, hi, wi)` is
`((oc*t + ti)*h + hi)*w + wi`, so output line `r` is exactly the contiguous span
`[r*out.w, (r+1)*out.w)` of `out.data`.

Three properties follow, and together they are the determinism argument:

1. **No output element is written by more than one worker.** The partition is a
   partition of `r`, and lines do not overlap.
2. **No reduction crosses a worker boundary.** Every accumulation — over `ic`,
   over `a`, `b`, `d` — is inside one `wi` iteration of one line. A worker
   executes exactly the instruction sequence the serial arm executes for that
   element, in the same order, on the same values.
3. **The result therefore does not depend on the worker count**, and it does not
   depend on which worker took which chunk either. That matters, because
   `ParallelForRows` (`src/vt/cpu/cpu_threadpool.cpp:413-458`) **steals work**
   through an atomic cursor, so the row-to-thread assignment is genuinely
   non-deterministic run to run. Bit-identity has to survive that, and it does,
   for reason 2.

This is not a new contract. `src/vt/cpu/cpu_threadpool.h:39-43` already states
it for the whole CPU backend: *"parallelism partitions OUTPUT elements only ...
No atomic accumulation into shared outputs, no reduction-order changes — results
are bit-identical to n_threads==1 by construction."* This row's job is to stay
inside that contract, not to invent one.

**What was rejected, and why.** Parallelising over the *reduction* axis `ic`
with per-thread partials and a final combine would also be a legal
convolution — and it would change the summation order as a function of the
thread count, which is exactly the defect #1008 spent its budget removing. It is
not taken, and no tolerance is widened anywhere in this row.

**`ParallelForRows` is synchronous**, so the `[&]` capture of the local `padded`,
`weight`, `out` and the loop bounds is safe: `Run` returns only after every
worker has passed the closing `Barrier()` inside `ComputeThread`
(`cpu_threadpool.cpp:234`), whose exit is a seq-cst fence (`:206-212`).

## 3. The sites

| site | what it is | parallel unit | rows |
|---|---|---|---|
| `ltx2_video_vae.cpp` `CausalConv3d`, output nest | 42 convs, ~all of the decode's FLOPs (`ltx25-decode-speed.md` §1.1) | one output line `(oc, ti, hi)` | `out_channels * out.t * out.h` |
| `ltx2_video_vae.cpp` `CausalConv3d`, pad gather | the replicate/reflect pad materialisation | one padded line `(c, ti, hi)` | `ci * pt * ph` |
| `ltx2_video_vae.cpp` `Linear3d` | the 1x1x1 conv used as `conv_shortcut` | a contiguous span of `(oc, i)` | `out_channels * in.spatial()` |

The pad gather has no reduction at all — it is a pure gather, one source element
per destination element — so it is trivially order-independent. It is included
because it is `O(ci * pt * ph * pw)` inside the same function and would otherwise
become a serial section that bounds the speedup by Amdahl's law.

Sites **not** taken, each for a stated reason:

* `PixelNorm`, `Silu`, `ApplyAdaLn`, `expand`, `drop_first_frame` — memory-bound
  elementwise passes. They are candidates, but they are not where the 7.25 TFLOP
  is, and each one added is another surface for a reviewer to check. Owed
  (§7) rather than done silently.
* `FeedSpatialNoise` — **must not** be parallelised. It consumes
  `Ltx2NoiseStream` in call order
  (`include/vllm/model_executor/models/ltx2_video_vae.h:201-210`), and that call
  order is the reproducibility contract with upstream's `torch.Generator`. The
  draw itself is already outside the loop; the loop that applies the plane could
  be partitioned, but the win is nil and the risk is a later edit moving the draw
  inside. Left alone deliberately.
* `AttnBlock3d` — the shipped decoder cannot construct an attention block at all:
  `attn_res_x` is refused by name (`ltx2_video_vae.cpp:10-14`), because upstream
  at the pinned revision cannot construct it either.

## 4. Risks

* **A partition that changes the summation order.** The one risk that can change
  the design, and the one that bound on the sibling row. Mitigation: §2's axis,
  plus the golden margins measured before and after and required to be
  **exactly equal**, not merely within tolerance. Any movement at all in a
  recorded `max|diff|` means the order moved and the design is wrong. No
  tolerance is widened; that is the stop condition (§8).
* **A result that depends on the thread count.** Mitigation: the determinism
  case in §5, which decodes the same input at five different worker counts and
  requires `memcmp == 0`.
* **A data race.** New concurrency in a file that had none. Mitigation: the
  ThreadSanitizer lane over the LTX suites (§6), because a race in a parallel
  reduction is precisely the defect this row could introduce and CI's sanitize
  lane is unreliable — it was cancelled in 4 of the last 12 `main` runs.
* **Nested dispatch.** `Threadpool::Run` throws on a dispatch from inside a
  parallel region (`cpu_threadpool.cpp:355`). The decode is called from
  `Ltx2VideoDecodeStreaming`, which is called from
  `src/vllm/multimodal/ltx2_video.cpp:3258` on the render path, and no caller in
  that chain is inside a parallel region. Checked by reading the chain, and the
  full gate would throw loudly if it were wrong.
* **A determinism test that measures nothing.** Two runs of a *serial*
  implementation are also bit-identical, so the determinism case alone is green
  before this row's change. That is why §5 ships a **second** case that observes
  the dispatch itself.

## 5. The gate

Two new cases in `tests/vllm/models/test_ltx2_vae.cpp`, both entering through
the production entry point `Ltx2VideoDecodeStreaming` — the one
`src/vllm/multimodal/ltx2_video.cpp:3258` calls on the render path, reaching
`Ltx2ConvVideoDecode` through `ltx2_video_vae_tiled.cpp:113`.

**Case A — the decode dispatches partitioned work to the CPU threadpool.** This
is the case that is RED before the change. A fresh `vt::cpu::Threadpool` is
installed with `SwapForTesting`, and its work-stealing cursor is read through
the public `ChunkAdd(0)`, which returns the current value and adds nothing. The
cursor is `0` on a fresh pool; `ParallelForRows` seeds it with `ChunkSet(nth)`
and every steal advances it (`cpu_threadpool.cpp:437-455`). So a non-zero cursor
after a decode is a direct observation that a multi-chunk partitioned dispatch
ran on that pool, and a zero cursor is the observation that none did. Before this
row the decode never touches a pool, so the case reads `0` and fails.

The same case asserts the decoded output against an analytically derived value
rather than a recorded one, so a decode that never ran cannot pass it. This is
the trap the sibling row hit and recorded: a zero-filled stub satisfies an
expectation of zero. The fixture therefore offsets `conv_out.conv.bias` off
zero, exactly as the width case does.

**Case B — the decode is bit-identical across thread counts.** The same latent
is decoded at worker counts 1, 2, 3, 5 and 8 and every result is `memcmp`-equal
to the 1-thread arm. Worker count 1 short-circuits `ParallelForRows` to
`body(0, nr)` on the caller (`cpu_threadpool.cpp:423-426`), so the 1-thread arm
*is* the pre-change code path, byte for byte. The counts are deliberately not
all powers of two: 3 and 5 do not divide the row counts, so the chunk boundaries
land in different places on every arm.

Case B also asserts that the decoded volume is not degenerate — that it holds
more than one distinct value — because an all-equal buffer, which is what a
stubbed decode returns, would satisfy a pure A-equals-B comparison.

**And the existing goldens become a threading gate for free.** The full suite
runs on the global pool, which is `hardware_concurrency` wide (20 here), so every
LTX-2.5 video golden already executes the threaded path. Their recorded
`max|diff|` values from #1008 are the before-picture, and §6 requires them to
come back **identical**.

## 6. Gates and evidence

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Reported: `CONFIGURE_EXIT`, `BUILD_EXIT`, the `: error:` count, `ctest -N`,
`CTEST_EXIT`, the full pass/fail line, `No space left` and `BFD assertion` with
positive controls, load average and free disk.

**ThreadSanitizer**, because this row adds concurrency:

```sh
cmake -S . -B build-san -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SANITIZE=thread
```

with the LTX suites run under it.

**The wall-clock A/B.** Same binary, one decode driven through
`Ltx2VideoDecodeStreaming` at a fixed synthetic decoder configuration, at
`VLLM_CPP_CPU_THREADS` 1, 2, 4, 8, 16 and 20, repeated enough times to show the
spread rather than one number, with the host load average recorded beside it.
The configuration and the harness source are recorded in `## Outcome` so the
measurement is reproducible; the shape is synthetic and is stated as such,
because the shipped checkpoint's `decoder_blocks` list comes out of a checkpoint
header this box does not have.

**What this row may not claim:** an end-to-end render speedup, any ratio against
any oracle, or a composition figure with #1008. There is no GPU here, no
large-render host, and no installed `ltx_core`.

## 7. Owed

| Item | Why it is not done here |
|---|---|
| The elementwise passes — `PixelNorm`, `Silu`, `ApplyAdaLn`, `expand`, `drop_first_frame` | Memory-bound and not where the FLOPs are. Each is order-independent and could be partitioned the same way; measuring whether it pays needs the A/B this row establishes first. |
| `AttnBlock3d` | Unreachable in the shipped decoder (`attn_res_x` is refused by name). Parallelising a path nothing can construct is dead code. |
| A SIMD inner tap loop | Separate summation-order question; see §0. |
| The composition of this row with #1008 | `ltx25-decode-speed.md` §6 warns that a threaded arm may become memory-bound where the scalar arm was ALU-bound. This row measures its own axis only. |
| An end-to-end render number | `dgx.casa` unreachable; no GPU here. |

No `.agents/issue-index.md` row is appended for #1009. That row already exists at
`.agents/issue-index.md:275` on PR
[#1018](https://github.com/mudler/vllm.cpp/pull/1018), which filed the issue and
is unmerged. `.gitattributes` sets `merge=union` on that file and
`scripts/check-agent-record.py` refuses a duplicate issue number, so appending a
second copy here would turn `main` red for every branch the moment #1018 merges —
which is exactly what a duplicate #995 row did on 2026-08-16. The sibling dtype
row made the same call for #1008 and recorded it in its pull request body.

## 8. Stop conditions

* Report `NEEDS_DECISION` rather than widening `kLtx2GoldenTol`, or any other
  tolerance, if a partition moves a golden. The answer to a moved golden is a
  partition that does not move it.
* Report `NEEDS_DECISION` rather than shipping a decode whose output depends on
  the worker count. A decode that gives different pixels at 1 thread and at 16 is
  a defect even with every golden green.
* Report the ThreadSanitizer result as it comes back. `test_ltx2_video` already
  carries a pre-existing LeakSanitizer leak under the `address,undefined` lane
  ([#1037](https://github.com/mudler/vllm.cpp/issues/1037), in the Gemma-4 rope
  cache via `DevicePool`); that one is not this row's and must not be allowed to
  mask a new report.
* Claim no number that was not measured on this box, in this session, with the
  load recorded.
