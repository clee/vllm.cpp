# LTX25-DFR-PIPELINE — detail-fidelity rendering, and the upsampler it drives

Row: `LTX25-DFR-PIPELINE`. Issue:
[#986](https://github.com/mudler/vllm.cpp/issues/986). Campaign:
[#644](https://github.com/mudler/vllm.cpp/issues/644).

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`, `packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Verified at the local checkout `/home/mudler/_git/LTX-2` — `git rev-parse HEAD`
returns that SHA — **before any anchor below was read**.

---

## 0. Honesty statement — what this row claims and what it does not

Written before implementation, so neither half can be discovered later as a claim
the row did not support.

1. **The temporal upsampler becomes DRIVEN.** That is the row's headline and it
   is a claim about a code path, not about a rendered clip. `DFRPipeline`'s
   rounds loop is upstream's only consumer of a temporally-configured
   `LatentUpsampler` (`dfr_pipeline.py:235-245, 402-407`), and this row ports
   that loop and reaches it from a production entry point. Section 7 states the
   sentence the records will carry, and it is narrower than "temporal upsampling
   works".
2. **There is no real-weight temporal result and there will not be one here.**
   `/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/latent_upscale_models/`
   holds `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` and nothing
   else. Re-verified 2026-08-16 rather than inherited from
   [`ltx25-temporal-upsampler.md`](ltx25-temporal-upsampler.md) §8.5, because that
   measurement is four days old and the NAS is a shared mount. The temporal
   checkpoint named at `ltx-pipelines/docs/pipelines.md:176` is **absent**, so
   every temporal claim here is a reduced-dimension fixture claim.
3. **The GPU is not used.** A render ladder holds `dgx.casa` and the box
   OOM-reboots when the 119 GiB unified pool is exhausted. CPU gate only.
4. **Upstream ships no tests.** `find /home/mudler/_git/LTX-2 -name 'test_*.py'`
   returns **0**. `AGENTS.md` `## vLLM is the reference` requires porting
   upstream's tests in the same change; there are none to port, so the tests here
   are written against upstream ANCHORS, and §6 says what that costs and what
   guards it.

## 1. What is wrong today — measured at `0e1bee42f`

`git grep -n -i dfr -- src include tests examples docs` returns **zero product
hits**. The name appears only in records that say it is absent:
`docs/USAGE.md:791` ("`DFRPipeline`'s multi-round loop, which is not ported"),
`include/vllm/model_executor/models/ltx2_upsampler.h:30`, and
[`ltx25-temporal-upsampler.md`](ltx25-temporal-upsampler.md) §0.1, §7, §8.7. The
positive control for that grep is `git grep -c -i ltx2 -- src`, which returns 87
hits in `ltx2.cpp` alone — so the empty result is a measurement and not a
mistyped path ([#604](https://github.com/mudler/vllm.cpp/issues/604)).

`docs/FEATURES.md:166` carries `Temporal x2 ups gated, UNDRIVEN`. That cell is
**load-bearing**: `.agents/specs/ltx25-resolution-envelope.md:436` asserts it is
untouched byte for byte. This row contradicts it deliberately, by key, and §9
records which row it is contradicting.

## 2. The dependency that decides the row's shape

DFR is not an independent pipeline. Its stage 1 appends
`VideoGeneratedKeyframeSlots` (`dfr_pipeline.py:330`) and then reads
`video_state.generated_keyframes` back (`:346-348`), raising
`RuntimeError("Stage 1 did not return generated_keyframes despite requesting
slots")` if it is absent. That readback is what
[#920](https://github.com/mudler/vllm.cpp/issues/920) refused by name.

So this row cannot be done without lifting that refusal, and lifting it is not a
side effect to be performed quietly. Three things follow.

**(a) The refusal's blocker was correctly identified and this row pays it.**
[`ltx25-generated-keyframes.md`](ltx25-generated-keyframes.md) `## 4` names one
blocker — "readback with a standalone decode" — and `## Owed` names three
pieces against #920. #920 is **CLOSED**, so the debt currently has a spec bullet
and no open issue; #986 picks it up.

**(b) DFR needs TWO of those three pieces and not the third, and the difference
is a fact about DFR rather than a convenience.** The refusal says each slot frame
"must then be decoded as a STANDALONE one-frame clip — a K-frame causal decode
would blend slots that were never temporally adjacent". True, and it is a
constraint on a caller who wants the slot PIXELS. **DFR never decodes its slots.**
It hands them to the spatial latent upsampler (`dfr_pipeline.py:348`) and feeds
them straight back as `initial_keyframes` (`:364`), and in the temporal rounds it
carries them as latents through `_merge_carry_forward_keyframes` (`:527-529`).
They stay in latent space for the whole pipeline. So DFR needs the LAYOUT and the
EXTRACTION; the standalone decode belongs to a slot-output surface DFR does not
have, and it stays owed under §11 rather than being written unreached.

**(c) The #920 refusal carries a tripwire aimed at this row, and it must be
allowed to fire.** Its message declares `ABSENT HERE: GeneratedKeyframe,
generated_keyframe`, and `test_ltx2_video` parses those out of the thrown message
and re-derives them against `ltx2_conditioning.h`'s declarations with comment
lines stripped. §4a of that spec is explicit about what happens next: *"If the
readback lands, `GeneratedKeyframe` appears in the header, ABSENT goes red, and
whoever landed it is told the refusal is now false."* This row lands exactly
that. The RED is the instrument working, and the repair is to retire the refusal,
**never** to widen or delete the assertion — `AGENTS.md` `## Changing the rules or
a checker` forbids making a red gate green by deleting an assertion.

## 3. Upstream, with anchors

Every `file:line` re-derived at the pin. The needle is derived from the CLAIM
rather than read back out of the cited span, because that check is circular.

### 3.1 `dfr_layout.py` — the canvas

| Upstream | What it decides |
|---|---|
| `SEGMENT_CANDIDATES = (24, 32)` (`:12`) | the two keyframe segment lengths |
| `TILE_LEAD_SEGMENTS = 1` (`:18`) | the lead-in every non-first tile denoises through |
| `TileRange` (`:21-38`) | pixel/latent bounds, anchor and slot bags, `drop_latent_prefix` |
| `choose_segment_length` (`:40-57`) | least pad; **ties keep the LARGER** |
| `resolve_canvas` (`:60-81`) | pads `num_frames - 1` to a multiple of S; positions `[S, 2S, ..., N'-1]`, frame 0 EXCLUDED |
| `pixel_to_latent_index` (`:84-90`) | 0 is legal; anything else must sit on the x8 border |
| `_owned_segment_counts` (`:93-100`) | contiguous owned runs, **largest first** |
| `_build_tile` (`:103-134`) | the window, the two keyframe bags, the prefix drop |
| `tile_ranges` (`:137-182`) | gapless partition; `num_tiles` CLAMPED to the segment count |
| `stitch_tile_latents` (`:185-208`) | each tile contributes `latent[drop_latent_prefix:]` |
| `remap_positions_to_local` (`:211-213`) | global to tile-local |

### 3.2 `dfr_pipeline.py` — the driver

| Upstream | What it is |
|---|---|
| `_ANCHOR_KEYFRAME_STRENGTH = 0.95` (`:72`) | carried anchors pinned just short of clean |
| `_TEMPORAL_ANCESTRAL_ETA = 0.5` (`:73`) | **not** the distilled 1.0 |
| `_MAX_CONDITIONING_FPS = 60.0` (`:78`) | RoPE time base cap; playback fps is separate |
| `_keyframe_conditionings_from_latents` (`:81-98`) | anchors as `VideoConditionByKeyframeIndex` |
| `_slot_initials_from_video` (`:101-111`) | nearest latent frame per slot; `round`, not floor |
| `_merge_carry_forward_keyframes` (`:114-139`) | anchors then slots, keyed by position, slot wins |
| `_detailing_downscale_factor` (`:142-152`) | LoRA metadata, default **2** |
| stage 1 (`:317-342`) | half res, `DISTILLED_SIGMAS`, slots at `positions` |
| stage 2 (`:351-394`) | full res, `STAGE_2_DISTILLED_SIGMAS`, upsampled slot seeds, reference latent when a detailing LoRA is present |
| rounds (`:402-529`) | temporal x2, tiles, ancestral Euler, stitch, carry forward |
| the trim (`:531-540`) | `(requested - 1) * 2**rounds + 1` |
| audio (`:552-560`) | shipped from STAGE 1, cut to the video's duration |

Three of those are worth stating as behaviour rather than as a row in a table,
because each renders something plausible when it is got wrong:

- **fps doubles per round and the CONDITIONING fps is capped separately**
  (`:409`, `:414`). Upstream's reason is a statement about RoPE, not a safety
  margin, and it is quoted in `ltx2_dfr.h` beside the constant.
- **each tile gets its own ancestral noise seed**, `seed + 1000 * round + tile`
  (`:498`), with upstream's own reason: *"Tiles are positionally identical, so a
  shared ancestral seed would inject byte-identical noise into every one of
  them."*
- **image conditioning is TILE-LOCAL** (`:428-437`). `frame_idx=0` means the
  tile's first frame, so re-applying the opening image on a non-first tile would
  pin the wrong frame onto the seam. Only images that fall inside the window are
  re-attached, with their indices shifted.

### 3.3 `keyframe_slots.py` and the readback

`VideoGeneratedKeyframeSlots.__init__` (`:47-69`) and `apply_to` (`:71-150`);
`GeneratedKeyframeLayout` (`ltx_core/types.py:220-247`);
`LatentTools.clear_conditioning` (`ltx_core/tools.py:88-117`) and
`extract_generated_keyframes` (`:203-230`).

The four things `apply_to` does that a shape check cannot see:

1. `denoise_mask = 1` on the new tokens (`:118-119`), so the noiser lerps from
   the slot `latent` and **ignores** `clean_latent`. A port that seeded
   `clean_latent` instead would produce a finite, correctly shaped, unconditioned
   slot.
2. `extend_keyframes_mask(..., marked=True)` (`:121`) — upstream's ONLY marked
   call site.
3. the slot's temporal span is exactly `[t, t+1)` with `causal_fix=False`
   (`:152-174`), because the span is set explicitly. Applying the causal fix as
   well would shift every slot.
4. `GeneratedKeyframeLayout` records `first_token` (`:143-147`) so the slots are
   located exactly rather than assumed to trail — items are applied in list order
   and each appends.

## 4. Scope

**In.**

1. `ltx2_dfr.{h,cpp}` — `dfr_layout.py` in full, plus the three `dfr_pipeline.py`
   helpers that are pure index arithmetic (`_slot_initials_from_video`,
   `_merge_carry_forward_keyframes`, the target-frames trim).
2. Generated keyframe slots in `ltx2_conditioning.{h,cpp}`:
   `Ltx2GeneratedKeyframeLayout`, `Ltx2ConditionVideoByGeneratedKeyframeSlots`,
   `Ltx2ExtractGeneratedKeyframes`, and `Ltx2ClearConditioning` extracting BEFORE
   it trims.
3. A `dfr` pipeline kind in `ResolveLtx2PipelineRecipe`, and the DFR driver in
   `ltx2_video.cpp`: canvas resolution, slot conditioning per stage, the slot
   carry between stages, and the temporal rounds loop.
4. Retire the #920 refusal, which has become false, and serve the key.
5. `docs/FEATURES.md` (the UNDRIVEN cell), `docs/USAGE.md`, `.agents/issue-index.md`.

**Out, refused by name or recorded as owed rather than approximated.**

- The standalone single-frame decode of a slot — §2(b). Owed, §11.
- `_detailing_downscale_factor`'s LoRA-metadata read: #923 landed the metadata
  reader, and the arm that consumes it is the reference-latent conditioning that
  [#975](https://github.com/mudler/vllm.cpp/issues/975) still owes for two
  independent reasons. DFR's detailing IC-LoRA is therefore **refused by name**
  rather than half-built, and the refusal points at #975.
- Multi-GPU DFR (`dfr_pipeline.py` has no `_mgpu` sibling; the marker
  `kMultiGpuParallelism` already covers the family).
- Any bf16 device arm of the layout: it is integer index arithmetic and has no
  dtype.

## 5. Design

**One new translation unit for the layout.** `ltx2_dfr.cpp` holds no weights,
runs no kernel and reads no checkpoint. It is separate because every one of its
failure modes produces a correctly shaped, finite, plausible latent, so the only
instrument that can catch them is an exact gate on the indices — which is a
different kind of test from everything in `ltx2_pipeline.cpp`.

**The rounds loop reuses the phase body rather than duplicating it.** Upstream's
tile denoise is `self.stage(...)` — the same `DiffusionStage.__call__` the two
stages use, with a different stepper, a different sigma set and a tile-local
conditioning list. Mirroring that means the engine's per-phase denoise becomes a
callable the rounds loop can invoke per tile. Writing a second denoise loop by
hand is the parallel path `AGENTS.md` `## Shared seams` forbids.

**The DFR recipe is `distilled_two_stage`'s two phases with DFR's names.** Both
use `DISTILLED_SIGMAS` then `STAGE_2_DISTILLED_SIGMAS`
(`dfr_pipeline.py:281-282`), stage 1 at `width // 2` (`:319`), stage 2 with the
spatial upsample and `noise_scale = stage_2_sigmas[0]` (`:386`). That is the
shipped `DistilledTwoStageRecipe` exactly, which is a finding rather than a
shortcut: DFR differs from the distilled two-stage pipeline in its CONDITIONING
and its rounds, not in its schedule. Recorded so a reader does not go looking for
a DFR-specific sigma set that upstream does not have.

## 6. Tests and evidence

Upstream ships no tests (§0.4), so every case here is written against an anchor
and the standing trap applies: **a test asserting only upstream symbol names
cannot detect local staleness** — measured on this campaign at
[`ltx25-generated-keyframes.md`](ltx25-generated-keyframes.md) §4a, where mutation
M7 left a suite at 18/18 exit 0 while the message it checked had become false. So
the layout suite pins LOCAL facts: exact index vectors, exact `drop_latent_prefix`
values, and a stitch whose output is compared element-wise against a
hand-constructed expectation, not against a property.

Every engine-level case enters through the **production entry point** —
`vllm::multimodal::LoadVideoEngine` then `VideoEngine::Generate`, the chain
`vllm_video_generate` takes — never by constructing a type
([`reachability.md`](../reachability.md)).

**The reachability mutation is the row's headline evidence.** Delete the
production call site where the DFR driver invokes the temporal upsampler, rerun
the focused gate, and record the RED. A green gate there would mean the rounds
loop is a test-only driver and the UNDRIVEN cell should not move.

**Mutations owed**, each recorded with three facts — `git diff --stat` after
applying, whether it BUILT with the compile-error count beside it, and the exit
code. A non-building mutation establishes nothing.

| # | Mutation | Predicted signature |
|---|---|---|
| M1 | `choose_segment_length` tie takes the SMALLER candidate | wrong positions on a tied frame count |
| M2 | `_build_tile` keeps boundary 0 in `anchor_kf_global` | an anchor with no latent in the carry bag |
| M3 | `drop_latent_prefix` omits the `+1` for `own_lo > 0` | stitched T off by one per seam |
| M4 | the first tile is given a lead-in | head of the clip truncated |
| M5 | slots seed `clean_latent` instead of `latent` | slot content unconditioned, shape unchanged |
| M6 | `extend_keyframes_mask` called with `marked=false` | the trained marker misses every slot |
| M7 | the slot span uses `causal_fix=true` | every slot shifted in RoPE time |
| M8 | `extract_generated_keyframes` assumes the slots TRAIL | wrong tokens read when a keyframe also appended |
| M9 | **the temporal upsampler call site deleted** | the reachability proof |
| M10 | the per-tile ancestral seed made shared | byte-identical noise per tile |

## 7. Reachability — the sentence the records must carry

Written before implementation as a target; §10 states what was actually reached.

> The LTX-2.5 temporal x2 latent upsampler is **driven** by the DFR pipeline's
> rounds loop, reachable from `vllm_video_generate` with `pipeline_kind=dfr` and
> a positive rounds count, and from `ltx2-gen` on the same two flags. It is gated
> at REDUCED DIMENSIONS against a fixture checkpoint. The real
> `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` is not on the NAS,
> so no real-weight temporal result exists and none is implied.

Nothing in `docs/` may say more than that.

## 8. Risks

- **`docs/FEATURES.md` is a keyed record and `MAX_CELL_CHARS = 220` binds**, with
  the LTX-2.5 cells at or near it. Reapply the scoped edit BY KEY, prove unrelated
  keys byte-identical, and trim only this row's own wording — never a measurement
  or a host qualifier.
- **`ltx2_video.cpp` carries derived READER ANCHORS** gated by `test_ltx2_video`.
  A clean merge will not warn when they go stale. Re-derive at the final tree.
- **`origin/main` moves.** [#983](https://github.com/mudler/vllm.cpp/pull/983) is
  open against `ltx2_video.cpp` (the MSVC C4244 cast for #968). Merge often and
  diff the merge base, never the moving ref.
- **The #920 tripwire fires by design** (§2c). The repair is retirement, not a
  widened assertion.
- **Refusals go stale.** `ltx2_video.cpp:1379-1380` keeps a tally of refusals
  whose stated reason turned out false. Every cause this row writes or lifts is
  re-derived at the merge tree, and ruled-out causes take the
  `WHAT IS *NOT* THE REASON` shape.

## 9. The record this row contradicts

`.agents/specs/ltx25-resolution-envelope.md:436` records, as evidence for that
row, that *"The 'Temporal x2 ups gated, UNDRIVEN' cell is untouched, byte for
byte."* That was true of that row and this row makes it false, deliberately. The
row being contradicted is `LTX25-RESOLUTION-ENVELOPE`
([#919](https://github.com/mudler/vllm.cpp/issues/919), merged as `e5351776c`),
and the contradiction is in its EVIDENCE section rather than in its behaviour —
nothing that row gated changes. Named here because a later reader who greps that
sentence must find the row that moved it rather than conclude the cell drifted.

## 10. Gates

CPU-only; the GPU is not used (§0.3).

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Known-red and to be proved pre-existing rather than asserted: `windows-msvc-*`
([#584](https://github.com/mudler/vllm.cpp/issues/584)), and #968 on any branch
based on `c7cb59fbb`. A red in `agent-record` or `sanitize-cpu` is NEW
information (#873 and #904 are fixed). Load-dependent and to be re-run alone
before being charged: `test_openai_conformance`, `test_cpu_threadpool`,
`test_engine_core_proc`, `test_async_llm`
([#294](https://github.com/mudler/vllm.cpp/issues/294)),
`test_cpu_x86_llamacpp_floor` (exit 4 = `NO_QUIET_WINDOW`).

## 11. Arms

| Arm | State |
|---|---|
| bf16 / f32 reference | in scope |
| FP8 | the DFR path is resolved on the request, ahead of arm selection, so the FP8 DiT reaches it unchanged |
| NVFP4 | same |
| GGUF k-quants | **not applicable to this family.** LTX-2.5 ships no GGUF: the quantization kinds upstream defines are fp8-cast, fp8-scaled-mm, nvfp4-cast and nvfp4-prequant (`quantization_factory.py:23-26`), and no published LTX-2.5 checkpoint is a GGUF. Owed for LTX-2.5 as a whole under #644 only if such a checkpoint appears |
| detailing IC-LoRA | **refused by name**, pointing at #975 |

## Owed

- [#986](https://github.com/mudler/vllm.cpp/issues/986) — the standalone
  single-frame decode of a generated keyframe slot, and the surface that would
  return slot pixels to a caller. DFR keeps its slots in latent space (§2b), so
  nothing here reaches it and nothing here needs it.
- [#975](https://github.com/mudler/vllm.cpp/issues/975) — DFR's stage-2 x2
  spatial detailing IC-LoRA, which needs the reference-latent arm that issue
  owns.

## Stop conditions

- If the slots turn out to be reachable without a layout — if they can be assumed
  to trail — stop: §3.3's reading of `keyframe_slots.py:143-147` is wrong and the
  #920 refusal named the wrong blocker.
- If the rounds loop cannot reach the temporal upsampler from a production entry
  point, stop and report. Landing the loop unreached would repeat exactly the
  defect this row exists to retire, and `AGENTS.md` `## Nothing lands dead`
  permits it only with the naming §7 would then have to give.
- If the READER ANCHORS move, re-derive them in this row rather than editing the
  gate.

## Now

`ACTIVE`. Spec committed before implementation on `row/LTX25-DFR-PIPELINE`.
