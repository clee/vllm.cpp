# LTX-2.5 — the resolution envelope, and the refusal that makes it a contract

Row: `LTX25-RESOLUTION-ENVELOPE`. Campaign: [`ltx-2-5.md`](ltx-2-5.md)
(operator-owned; **not edited by this row**). Issues:
[#919](https://github.com/mudler/vllm.cpp/issues/919) (the defect this row
fixes), [#921](https://github.com/mudler/vllm.cpp/issues/921) (the res_2s
denoising loop, listed under `## Owed` below). Sibling of
[#644](https://github.com/mudler/vllm.cpp/issues/644).

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`, `packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Verified at the local checkout `/home/mudler/_git/LTX-2`:
`git rev-parse HEAD` = `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, clean tree.
Every anchor below was read from that checkout.

---

## 0. Honesty statement — what this row does and does not claim

The dispatching brief asked for the full `TI2VidTwoStagesPipeline` and
`TI2VidTwoStagesHQPipeline` port **and** the resolution story. A ground-truth
survey against the verified pin, relayed by the coordinator mid-row, narrowed
this row to the second half. This section records the narrowing before any work
starts so that no later reader mistakes the delivered slice for the whole ask.

**This row delivers:** upstream's geometry constraint, mirrored as a refusal at
the production entry point, plus the published envelope.

**This row does not deliver, and does not claim:**

1. **The res_2s sampler.** `Ltx2Res2sStep` / `Ltx2Res2sSdeCoeff`
   (`src/vllm/model_executor/models/ltx2_pipeline.cpp:307-345`) already mirror
   `Res2sDiffusionStep` and are gated. That is one substep's SDE arithmetic, not
   the sampler. The sampler is `res2s_audio_video_denoising_loop`
   (`samplers.py:208-447`), and its exponential-integrator coefficients, its
   second transformer evaluation per step, and its bong refinement are all
   absent. §5 and #921.
2. **A raised resolution ceiling.** There is no code cap to lift. The only
   geometry guard in the LTX path today is a **lower** bound
   (`ltx2_video.cpp:1464-1471 @ 5a0ffe9e3`; `:1494-1501` after this row, which
   inserted the guard above it). The real ceiling is host memory and decode
   throughput, it is already measured, and it is already **unattributed** — §4.
   Attributing it needs the GPU, which this row must not use.

## 1. The question, and the evidence that answers it

Nothing in this tree established what resolutions LTX-2.5 supports. The only
render ever performed was 9 frames at 128x128. `ltx2-gen` exposes `--width`,
`--height` and `--frames`, and exposing a flag is not supporting a value.

### 1.1 What upstream supports, and what it enforces

**Width and height are hard-validated, and upstream raises.**
`packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py:540-551`:

```python
def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    """Assert that the resolution is divisible by the required divisor.
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    """
    divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(...)
```

It is not help text and it is not advisory. There are **ten** call sites, one at
the top of every pipeline's `__call__`, including the two this row's brief named:
`ti2vid_two_stages.py:184` (`is_two_stage=True`) and
`ti2vid_two_stages_hq.py:199` (`is_two_stage=True`), against
`ti2vid_one_stage.py:156` (`is_two_stage=False`). The CLI's `args.py` help text
carries the same rule, but the pipeline is what enforces it, and a caller who
reaches `__call__` from library code gets the `ValueError` either way.

**Where 64 and 32 come from.** The VAE spatial factor is 32
(`SpatioTemporalScaleFactors.default()` = `time=8, height=32, width=32`,
`ltx_core/types.py:31-33`). A one-stage pipeline runs at the requested size, so
its divisor is that factor. A two-stage pipeline runs stage 1 at `width // 2`,
`height // 2` (`ti2vid_two_stages.py:226-228`, `ti2vid_two_stages_hq.py:241-243`),
so the requested size must survive being halved and still divide the grid —
`32 * 2 = 64`. The divisor is the VAE factor times the worst spatial downscale
any phase applies. That is a derivation, not a coincidence, and §3 mirrors it as
one.

**Frames are NOT validated.** An explicit `num_frames` passes through
`resolve_num_frames` (`utils/blocks.py:908-928`) untouched — the function returns
it verbatim when it is not an `AutoDuration` — and lands in
`VideoLatentShape.from_pixel_shape` (`ltx_core/types.py:108-123`), which floors:

```python
frames = (shape.frames - 1) // scale_factors.time + 1
height = shape.height // scale_factors.height
width  = shape.width  // scale_factors.width
```

`snap_frames_to_grid` (`helpers.py:554-562`), which encodes
`(frames - 1) % time == 0`, is reached only from `seconds_to_clamped_num_frames`
(`:565-580`), which is the **auto-duration** path. So upstream floors an explicit
frame count exactly as we do, and validates it nowhere. §3.2 takes the
consequence.

**Upstream's own defaults**, which are the scale the envelope is measured
against (`utils/constants.py`):

| Preset | Stage 1 | Stage 2 (output) | Frames | Steps |
|---|---|---|---|---|
| `PipelineParams` / `LTX_2_PARAMS` (`:42-76`) | 512x768 | **1024x1536** | 121 | 40 |
| `LTX_2_3_PARAMS` (`:83-88`) | 512x768 | 1024x1536 | 121 | 30 |
| `LTX_2_3_HQ_PARAMS` (`:95-98`) | 544x960 | **1088x1920** | 121 | 15 |

`stage_2_height` / `stage_2_width` are `stage_1 * 2` (`:70-76`). All three
outputs are multiples of 64, as `assert_resolution` requires of a two-stage call.

### 1.2 What our code assumes, and what actually binds

Every repo-local anchor in this section is pinned `@ 5a0ffe9e3`, this row's base
SHA, because the row edits the file it cites and an unpinned anchor into a file
you are yourself moving is stale by default (#911). The post-change positions are
given where they matter.

`vllm_video_generate` resolves geometry at
`src/vllm/multimodal/ltx2_video.cpp:1401-1422 @ 5a0ffe9e3` and turns it into a
latent grid at `:1455-1463 @ 5a0ffe9e3` (`:1485-1493` after this row):

```cpp
const int64_t phase_h = height / phase.spatial_downscale;
const int64_t phase_w = width / phase.spatial_downscale;
...
vshape.frames = (frames - 1) / factors.time + 1;
vshape.height = phase_h / factors.height;
vshape.width  = phase_w / factors.width;
```

Three binding constraints, and only one of them is a check:

1. **A lower bound**, `:1464-1471 @ 5a0ffe9e3` (`:1494-1501` after this row) —
   the request must reach one latent cell. Present, correct, and gated.
2. **`frames < 1`**, `:1422 @ 5a0ffe9e3`. Present.
3. **Divisibility — absent.** Integer division is the whole of it.

**MEASURED, not reasoned.** The defect has two faces, and which one a request
gets depends on whether its floor is consistent across the phases. Both were
observed on the reduced fixture at `5a0ffe9e3`, before the guard existed:

| Request | Recipe | What happened |
|---|---|---|
| width 80 | distilled two-stage | rendered **64x64**, exit success |
| width 100 | one-stage | rendered **96x64**, exit success |
| width 96 | distilled two-stage | threw `the upsampled latent is 4x2x2x2 but phase 'refine' needs 4x2x2x3` |

The third is not a silent floor but it is not a usable error either: stage 1
floors 48 to one latent cell while stage 2 needs three, so the upsampler's shape
check fires with a true statement about latents and no mention of the width the
caller passed. One guard at the entry point closes both faces, which is why the
test case carries all three sizes.

`docs/USAGE.md:626-629 @ 5a0ffe9e3` already documents the divide-by-64 rule as
though something enforced it. Nothing does. That gap between a published promise
and the tree is the defect (#919), and it is the one thing in this story that is
a source-and-refusal question rather than a measurement one.

## 2. Scope

**In.**

- `Ltx2AssertResolution`, mirroring `helpers.py:540-551`, in the pipeline
  header/TU beside the other mirrored pipeline helpers.
- Its call at the geometry resolution in `ltx2_video.cpp`, with the divisor
  **derived** as `factors.height * recipe.max_spatial_downscale()` rather than
  hardcoded, which reproduces upstream's 64 and 32 on the two-stage and one-stage
  recipes respectively.
- A red-first test entering through `vllm_video_generate`, plus the reachability
  mutation.
- `docs/USAGE.md`: the published envelope, and the correction of the frames
  claim to what upstream and this tree both actually do.

**Out.** Named, so none is discovered later as an omission:

- The res_2s denoising loop and the HQ preset — #921, `## Owed`.
- `TI2VidTwoStagesPipeline` as a distinct recipe row. Our
  `DistilledTwoStageRecipe` (`ltx2_pipeline.cpp:1084-1131`) already carries the
  two-phase spatial-upsample shape; the non-distilled variant differs in its
  stage-1 schedule and guidance, and is a recipe row rather than a geometry
  question. Not bundled.
- Any render, any measurement, any GPU use. `dgx.casa` is running a long render
  under `flock` and the box OOM-reboots when its unified pool is exhausted.

## 3. Design

### 3.1 The refusal

One function, mirroring upstream's shape and message content:

```cpp
void Ltx2AssertResolution(int64_t height, int64_t width, int64_t divisor);
```

Called from the geometry block in `ltx2_video.cpp`, after `height` and `width`
resolve and before anything consumes them — which is where upstream calls it,
at the top of `__call__` before any work is paid for.

**The divisor is derived, not restated.** `recipe.max_spatial_downscale()`
(`ltx2_pipeline.cpp:994-1000`) already reports the worst `spatial_downscale` over
a recipe's phases: 2 for `distilled_two_stage`, 1 for `one_stage` and `dmd2`.
Multiplied by `factors.height` (32) it is 64 and 32 — upstream's two numbers,
reached by upstream's reasoning. A hardcoded pair would restate the answer and
would silently be wrong for any future recipe whose phases downscale by more.

**One divisor covers both axes**, as upstream has it (`divisor = 64 if
is_two_stage else 32`). That is the mirror and not a simplification: upstream's
VIDEO_SCALE_FACTORS is `(8, 32, 32)`, so its single spatial divisor already
covers both. The plan here first said the check would use each factor against its
own axis; that would have been a divergence dressed as future-proofing. What
landed instead asserts `factors.height == factors.width` at the call site and
fails by name if a VAE ever breaks it, so the assumption is checked rather than
carried silently — a VAE with differing axes would otherwise have its width
measured against the height factor and no test would see it.

### 3.2 Frames: the doc moves, the code does not

Upstream floors an explicit frame count and validates it nowhere (§1.1). Our code
floors identically. Mirroring means **not** adding a refusal upstream does not
have, and the honest repair is to `docs/USAGE.md`, which currently promises
enforcement of `(frames - 1) % 8 == 0`.

This is deliberately the opposite decision from width/height in the same change,
and the asymmetry is upstream's, not ours: `assert_resolution` exists and covers
two of the three axes. Recording the reason here because a later reader will
otherwise read the asymmetry as an oversight.

The rounding stays observable: `result.frame_count`, `result.width` and
`result.height` report what was rendered, so a caller who checks can still see a
floored request.

## 4. The ceiling is measured, and it is not attributed

For the envelope to be a contract it has to say what runs, not only what is
arithmetically legal. Existing evidence in this tree, none of it produced by this
row:

- **320x192 / 25f completes** on GB10 through both distilled phases
  (`docs/USAGE.md`, `docs/BENCHMARKS.md`).
- **448x256 / 25f does not.** It finishes its denoise and then loses about 59 GB
  in 24 seconds inside the decode.
- `.agents/specs/ltx25-tiled-decode.md` `## Outcome`: the decode's own heap peak
  is 361.72 MiB, which is ~170x too small to be that 60 GiB. That spec states
  plainly that the 60 GiB is **NOT attributed**.
- The same spec: `Ltx2ConvVideoDecode` at 448x256/25f took 2681 s, single-threaded,
  at 0% GPU.

`memory_efficient_decode.py` upstream is deliberately unported and is the obvious
candidate for the missing 60 GiB. Attributing it, and the reference decoder's
single-threaded throughput, are two measurement rows that need the GPU. Neither
is this row, and the envelope §6 publishes says so rather than implying the
arithmetic limit is the practical one.

Set against upstream's own defaults — 1024x1536 and 1088x1920 at 121 frames — a
320x192 practical ceiling is the story, and the envelope states both numbers next
to each other rather than only the legal one.

## 5. Tests

Red-first, entering through the production entry point per
[`.agents/reachability.md`](../reachability.md):

1. **`test_ltx2_video`**, through `vllm_video_generate` on the distilled
   two-stage recipe: a request whose width is not a multiple of 64 is refused,
   and the message names the divisor and the offending value. Red before the
   check exists, because today it renders a floored clip and returns success.
2. The one-stage divisor of 32 on the same entry point, so the derivation is
   gated on both arms and not only on the arm that ships by default.
3. A multiple-of-64 request still resolves, so the refusal is not a blanket one.
4. **Reachability mutation**: delete the `Ltx2AssertResolution` call site in a
   scratch copy and rerun the focused gate. A green gate would mean the test
   measures the function rather than the capability.

**No upstream test is ported, because there is none to port.** `Lightricks/LTX-2`
at `fd4ded7f` contains **zero** `test_*.py` files anywhere in the repository —
`find /home/mudler/_git/LTX-2 -name 'test_*.py'` returns nothing. Recorded as a
fact rather than as a silent omission of the standing "port the upstream tests in
the same change" obligation.

What replaces that obligation, at the same bar: the tests are written **against
upstream anchors** rather than ported from a suite. Only the provenance changes.
Each still fails for the intended reason before the change, still enters through
`vllm_video_generate` rather than constructing the type, and each asserted
behaviour still names the upstream `file:line` that justifies it — the divisors
and the raise from `helpers.py:540-551`, the halved stage-1 geometry from
`ti2vid_two_stages.py:226-228`, the spatial factor from `ltx_core/types.py:31-33`,
and the frames non-check from `blocks.py:908-928` with `types.py:113`.

Two of the fixture sizes are **measured rather than reasoned**: width 80 rendering
64x64 and one-stage width 100 rendering 96x64 were observed on this fixture before
the guard existed, which is why the case uses those and not the obvious 96 that
takes a different path entirely.

## 6. Records

- `docs/USAGE.md` — the envelope, and the frames correction. Keyed record;
  scoped edit reapplied by key with unrelated keys proven byte-identical.
- `docs/FEATURES.md` — **one row appended** to the LTX-2.5 gap table, and nothing
  else touched. The row was not in the original plan: this row's first pass
  declared FEATURES.md out of scope on the grounds that a refusal changes no
  feature surface, and `check-doc-checkpoint` disagreed and was right. A size that
  used to render and now refuses is exactly a change in what the project
  supports, and the file already carries the LTX-2.5 refusal table that is its
  home. Recorded rather than quietly done, because the plan said otherwise.

  The **"Temporal x2 ups gated, UNDRIVEN" cell is untouched**, byte for byte.
  This row does not drive that arm and does not change that fact. The file is a
  known lock (#595), so the edit is one appended row rather than a rewrite, to
  keep the conflict surface with the sibling rows editing it at a single line.
- `.agents/issue-index.md` — append-only, two rows appended, zero removed.
- No lifecycle change, so `docs/STATUS.md`, `docs/BENCHMARKS.md` and `## Now`
  are untouched.

## Owed

- [#921](https://github.com/mudler/vllm.cpp/issues/921) — the res_2s denoising
  loop (`samplers.py:208-447`): the `phi` / `get_res2s_coefficients` exponential
  integrator (`res2s.py:1-60`), the second transformer evaluation per step at
  `sub_sigma = sqrt(sigma * sigma_next)`, the bong anchor refinement, and the
  `Ltx2StepperKind` enumerator to select any of it. Until it lands,
  `TI2VidTwoStagesHQPipeline` cannot be served, and serving `LTX_2_3_HQ_PARAMS`
  on the Euler loop would render a plausible clip that is quietly not HQ at
  roughly half the model evaluations the preset was tuned for. No HQ recipe row
  is added by this row, so nothing can select it and nothing lands dead.
- `TI2VidTwoStagesPipeline` as a recipe row — stage 1 on the scheduler-derived
  schedule under full CFG, stage 2 on `STAGE_2_DISTILLED_SIGMAS` with guidance
  off (`ti2vid_two_stages.py:243-308`). Distinct from the distilled two-stage
  recipe that ships. Covered by #644 row scope; not separately filed, because
  #644 already owns "close every refused arm".
- Attribution of the 60 GiB decode loss and the single-threaded decode
  throughput (§4). Both need the GPU; both are measurement rows.

## Stop conditions

- Any need for the GPU. Report and stop.
- If the derived divisor disagrees with upstream's 64/32 on any shipped recipe,
  stop and report rather than special-casing a recipe.

## Outcome

**The geometry half landed; the sampler half is owed and named.**

What was measured, and what it changed. The defect was expected to be a single
silent floor and is two failure modes, split by whether a request's floor is
consistent across the phases (§1.2). Width 96 on the two-stage arm — the obvious
test value — takes the *other* path and never floors at all, so a case written
from reasoning rather than measurement would have gated the wrong thing and still
gone green. The fixture sizes are the measured ones for that reason.

What was rejected. A frames refusal, because upstream floors an explicit count
identically and validates it nowhere; adding one would have been a divergence
wearing the costume of a fix. Per-axis divisors, for the same reason — upstream
has one divisor, and the first draft's "future-proofing" was a divergence too;
what landed asserts the equality the single divisor assumes. And a `two_stage_hq`
recipe row, which would have been selectable by nothing, because the sampler that
makes it HQ is not ported (#921).

Why the divisor is a parameter rather than a constant: upstream's 64 and 32 are
derived quantities spelled as literals, and spelling them as literals here would
restate the answer instead of the reason. The one mutation that survived the
first pass (§5) was against the message's *suggested* size, not the check — worth
recording, because it is the half of a refusal that gets written once and never
tested, and a suggestion that is not itself legal is worse than none.

State: complete for its scope. No lifecycle row moves, so `docs/STATUS.md`,
`docs/BENCHMARKS.md` and `.agents/NOW.md` are untouched by design.
