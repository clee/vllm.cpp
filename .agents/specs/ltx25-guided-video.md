# LTX25-GUIDED-VIDEO — the guided video denoiser, as a shared seam

Row `LTX25-GUIDED-VIDEO`, under the campaign [`ltx-2-5.md`](ltx-2-5.md).
Issue [#1092](https://github.com/mudler/vllm.cpp/issues/1092).
Base: `origin/main` @ `b5756ea8c`.
Upstream: Lightricks `LTX-2` @ `fd4ded7f` (the revision every anchor below is
read at), secondary oracle `vllm-omni` (UNPINNED, #633) for the recipe tables
this tree already mirrors.

Paths are relative to `packages/ltx-pipelines/src/ltx_pipelines/` and
`packages/ltx-core/src/ltx_core/` in that checkout, as the rest of the LTX-2.5
specs use them.

## 0. Honesty statement — what this row does and does not claim

It claims: the LTX-2.5 **video** denoise path now runs upstream's guided
denoiser, that the guidance is combined in **x0** space, that one production
pipeline (`pipeline_kind = one_stage`) reaches it on its **own default
configuration**, and that the gate can see the space error that #1039 was.

It does not claim: a numerical comparison against a running oracle. There is
none — vLLM-Omni is UNPINNED (#633) and carries no LTX-2.5 recipe at all, and no
LTX-2.5 checkpoint on this host has a recorded sha256 (#1048). Every anchor
below is **source read** at `fd4ded7f`, and every number below is measured on
**this tree's own reduced fixture**. That is what an ungateable lane looks like
when it is stated instead of implied.

It does not claim to retire [#1049](https://github.com/mudler/vllm.cpp/issues/1049).
See section 6c: one of that issue's four symbols is reached by this row and three
are not, and forcing the other three would mean inventing a dispatch upstream
does not have.

## 1. Scope

**In:**

- `Ltx2GuidedDenoise` — `_guided_denoise` (`utils/denoisers.py:62-207`) in its
  own translation unit mirroring upstream's own file.
- The four passes it assembles: `cond`, `uncond`, `ptb`, `mod`.
- The `SKIP_A2V_CROSS_ATTN` / `SKIP_V2A_CROSS_ATTN` halves of
  `Ltx2DitPerturbation`, without which the `mod` pass cannot run and every video
  guider default is unreachable.
- The negative conditioning for the video path — the second half of the encode
  `GenerateAudioOnly` already performs and discards.
- The video guidance request extras that `default_1_stage_arg_parser`
  (`utils/args.py:930-1010`) exposes, gated by `allow_guidance_override`.
- The `one_stage` pipeline as the reachable consumer.

**Out, and owed rather than silently absent:**

- The other three video pipelines that need this seam
  (`a2vid_two_stage`, `ti2vid_two_stages`, `ti2vid_two_stages_hq`,
  `keyframe_interpolation`). Each is its own row; this row exists so that they
  are ordinary porting work rather than blocked.
- The **device-resident** arm of the `ptb` and `mod` passes.
  `Ltx2DitForwardDevice` (`ltx2_device.h:136`) takes no `perturbations`
  argument. Refused by name on that arm rather than run unperturbed, which would
  produce a legal-looking render whose STG term is identically zero. See §4.3.
- `BatchedPerturbationConfig`'s partial blend (`attention.py:571-572`) and
  batch > 1, which stay degenerate at the one batch size this port runs — the
  statement `ltx2.h` already carries, unchanged.

## 2. Upstream chain

The executing chain for one guided step, top to bottom:

| Step | Upstream | What it decides |
|---|---|---|
| the stage builds the model | `utils/blocks.py:480-482` — `X0Model(self._prepared_builder().build(...))` | the transformer the loop is handed is **already** an x0 model |
| the loop calls the denoiser | `utils/samplers.py:73-74` | one denoiser call per step |
| the denoiser assembles passes | `utils/denoisers.py:100-137` | `cond`, `uncond`, `ptb`, `mod`, in that order |
| the forward converts | `model/transformer/model.py:590-604` — `to_denoised(video.latent, vx, video.timesteps)` | **every** pass is x0 before any combination |
| the guider combines | `components/guiders.py:244-273`, per modality at `denoisers.py:203-204` | `cond + (cfg-1)(cond-uncond) + stg(cond-ptb) + (mod-1)(cond-modpass)`, then the rescale at `:268-271` |
| the loop post-processes | `utils/samplers.py:35` — `post_process_latent(denoised, ...)` | the conditioned tokens are pinned back **after** the guider, not per arm |
| the stepper steps | `utils/blocks.py:524-527` / `samplers.py:488-558` | Euler or ancestral, unchanged by this row |

The pass list is **shared between the two modalities and the guiders are not**.
`denoisers.py:103-137` takes the union — one `uncond` pass if *either* guider
wants one, one `ptb` pass carrying *both* modalities' perturbations, one `mod`
pass if *either* wants one — and then `:203-204` combines each modality with its
**own** guider over the same splits. A per-modality pass list would run up to six
forwards where upstream runs four, and would give the audio stream a different
video state to cross-attend to on the video-only passes. That is the single
structural fact this port has to get right, and it is why the seam takes both
guiders rather than being called twice.

The perturbation types are per direction, not per modality
(`guidance/perturbations.py:8-16`, applied at `model.py:443-458`):

| Pass | Perturbations | Reaches |
|---|---|---|
| `cond` | none | — |
| `uncond` | none | negative context on both streams |
| `ptb` | `SKIP_VIDEO_SELF_ATTN` on `video_guider.stg_blocks`, `SKIP_AUDIO_SELF_ATTN` on `audio_guider.stg_blocks` | `attention.py:557` `use_attention = not all_perturbed` |
| `mod` | `SKIP_A2V_CROSS_ATTN` and `SKIP_V2A_CROSS_ATTN`, **all blocks** | `transformer.py:335,366` `cross_attn_skip_all` |

## 3. Our baseline, derived at `b5756ea8c`

`src/vllm/multimodal/ltx2_video.cpp:3036-3045` runs **one** forward per step and
converts its velocity:

```
const Ltx2DitOutputs velocity = im.on_device ? Ltx2DitForwardDevice(...) : Ltx2DitForward(...);
const std::vector<float> v_denoised = PostProcessLatent(ToDenoised(video.latent, velocity.video, ...), video);
const std::vector<float> a_denoised = PostProcessLatent(ToDenoised(audio.latent, velocity.audio, ...), audio);
```

Everything the recipe resolved for that step is set and read by nothing:

```
$ git grep -n 'video_guidance' -- src include            # @ b5756ea8c
include/vllm/model_executor/models/ltx2_pipeline.h:526:  Ltx2MultiModalGuiderParams video_guidance;
src/vllm/model_executor/models/ltx2_pipeline.cpp:1069:  phase.video_guidance = params.video_guider;
```

The positive control for that grep is the same command for `audio_guidance`,
which returns the T2A consumer at `ltx2_video.cpp:3527`. The term and the path
set are right; the video consumer is genuinely absent. `allow_guidance_override`
(`ltx2_pipeline.h:534`) is the same shape: three recipes set it `false` and
nothing reads it.

**What is already correct and is reused unchanged:**

- `Ltx2MultiModalGuidance` (`ltx2_pipeline.cpp:479-522`) — `calculate` including
  the unbiased-`std` rescale. Reviewed under #1032/#1039.
- `Ltx2BatchedPerturbationConfig` (`ltx2_pipeline.h:380-405`) — the full
  four-type keep-mask, ported under #641 and, per #1049, constructed only by its
  own test until this row.
- `ToDenoised` (`ltx2_video.cpp:277`) and `PostProcessLatent` (`:234`).
- The T2A driver (`ltx2_t2a.cpp:322-368`), which is the **template**: it is the
  one place in this tree that already converts to x0 inside the model wrapper.

**What is missing and why nothing noticed:** a token gate cannot see it, and this
path has no token gate. An unguided render returns a finite clip of the right
size, frame count and sample rate. It is #1039's family of defect one level up:
not the wrong space, the wrong number of forwards.

## 4. Design

### 4.1 The seam — `ltx2_denoisers.{h,cpp}`

A new translation unit mirroring `ltx-pipelines/utils/denoisers.py`, rather than
another block inside `ltx2_pipeline.cpp`. Two reasons, and only the first is
about this row:

1. Upstream has that file. `AGENTS.md` §Shared seams: new capability is additive
   files mirroring the upstream structure.
2. `ltx2_pipeline.{h,cpp}` is concurrently edited by #921. A seam that four
   later rows will extend does not want to live in the file with the most
   contention.

The transformer is a **callable**, exactly as `_guided_denoise(transformer, ...)`
takes one:

```
using Ltx2X0Model = std::function<Ltx2X0Outputs(const Ltx2ModalityInput* video,
                                                const Ltx2ModalityInput* audio,
                                                const Ltx2DitPerturbation* perturbations)>;
```

That is the structural claim this row is graded on. The x0 conversion happens
**inside the caller's lambda**, which is upstream's `X0Model` wrapper
(`blocks.py:480-482` builds it; `model.py:590-604` is its forward), so the seam
combines already-denoised tensors and cannot be handed a velocity. Converting
once after the guider instead is a **different function on the default arm** —
`rescale_scale` is 0.7 on every video row — and that is #1039, on the audio arm,
in this tree, six days ago.

It also means the host forward and the device forward are the same seam with two
lambdas, and that the four later pipelines supply their own conditioning without
the seam knowing anything about keyframes, reference clips or two-stage
schedules.

### 4.2 The passes

`Ltx2GuidedDenoise` mirrors `denoisers.py:84-207` line for line:

- `v_skip`/`a_skip` from `ShouldSkipStep` (`:84-85`); both skipping returns the
  previous step's denoised pair with **no forward at all** (`:87-90`).
- `cond` always (`:100`).
- `uncond` when either guider asks or `force_uncond_pass` (`:102-109`), with the
  negative context substituted per modality and `v_neg = v_context` when a
  modality has none (`:107-108`).
- `ptb` when either guider perturbs, carrying both modalities' `stg_blocks`
  (`:111-119`).
- `mod` when either guider isolates, all blocks, both cross directions
  (`:121-137`).
- `enabled = not skip` per modality (`:151,161`), which is
  `Ltx2ModalityInput::enabled` here.
- the combination per modality with that modality's own guider (`:203-204`).

Perturbations route through `Ltx2BatchedPerturbationConfig`: one config built
over all N passes (`denoisers.py:172-176`), then `BatchSlice(i, i+1)` per pass,
then flattened into the `Ltx2DitPerturbation` the forward takes. At batch 1 the
slice is the pass's own mask, which is exactly the degeneracy `ltx2.h` already
records.

### 4.3 The cross-attention perturbation

`Ltx2DitPerturbation` grows two booleans, `video_cross_attn_skip_all` and
`audio_cross_attn_skip_all`, mirroring `TransformerArgs.cross_attn_skip_all`
(`transformer_args.py:118`). They gate the A2V and V2A branches at
`ltx2_dit.cpp`'s `if (run_a2v)` / `if (run_v2a)`, mirroring
`transformer.py:335` and `:366`. Note the polarity: `video.cross_attn_skip_all`
gates **A2V** (audio into video) and `audio.cross_attn_skip_all` gates **V2A**,
because the flag rides on the stream being *written*.

The snapshot of `vx_pre`/`ax_pre` stays outside both guards, as upstream's
`vx_pre_av = vx` at `:333` does, so a build where only one direction is skipped
still reads the pre-cross state for the other.

`ltx2.h:41-49`'s NOT-PORTED entry is corrected in the same change. Its stated
reason — "nothing upstream that this port serves constructs them" — was true for
text-to-audio, which pins `modality_scale = 1.0` (`t2a_one_stage.py:200-202`),
and is false for every video pipeline, all of which default it to 3.0.

**The device arm is refused, not degraded.** `Ltx2DitForwardDevice` has no
`perturbations` parameter, so a `ptb` or `mod` pass on that arm would have to
run unperturbed. The result is a finite clip whose STG and modality terms are
identically zero — indistinguishable from a working render. The refusal names
the missing function and the owed issue. CFG alone (a different context, no
perturbation) is served on both arms.

### 4.4 The negative conditioning

`Generate` already encodes the positive prompt into both streams
(`ltx2_video.cpp:1771-1806`). The negative half is the same chain with
`recipe.negative_prompt` (or the `negative_prompt` extra), through the same
connector, and is encoded **only when a guider asks for it**
(`do_unconditional_generation`, `guiders.py:275-277`) — at `cfg_scale = 1.0`
there is no unconditional forward and encoding it would be a wasted host-side
12B pass per request.

Two fallbacks exist for an engine with no text tower, matching the two that
already exist for the positive stream (`prompt_embeds_path`,
`audio_prompt_embeds_path`): `negative_prompt_embeds_path` and
`negative_audio_prompt_embeds_path`. This is a **local adaptation**, recorded as
one: upstream encodes `[prompt, negative_prompt]` in one `PromptEncoder` call
(`ti2vid_one_stage.py:170-178`) and has no embeds surface at all. The adaptation
is the existing one applied to the second of upstream's two encodings, not a new
concept. Without a tower and without those files, a guider that asks for the
unconditional pass is refused by name, exactly as T2A is at
`ltx2_video.cpp:3583-3593`.

### 4.5 The request extras

Mirroring `default_1_stage_arg_parser` (`utils/args.py:947-1010`), one extra per
flag, each overriding one field:

| Extra | Upstream flag | Field |
|---|---|---|
| `video_cfg_guidance_scale` | `--video-cfg-guidance-scale` | `cfg_scale` |
| `video_stg_guidance_scale` | `--video-stg-guidance-scale` | `stg_scale` |
| `video_rescale_scale` | `--video-rescale-scale` | `rescale_scale` |
| `video_stg_blocks` | `--video-stg-blocks` | `stg_blocks` |
| `a2v_guidance_scale` | `--a2v-guidance-scale` | video `modality_scale` |
| `v2a_guidance_scale` | `--v2a-guidance-scale` | audio `modality_scale` |

The audio row already exists for T2A (`ltx2_video.h:456-460`) and is reused for
the joint path. Every override is refused on a phase whose
`allow_guidance_override` is `false` — the distilled and retake recipes, whose
guidance is trained in — which is the first read that field has ever had.

An extra that is PRESENT and empty is upstream's empty list for `nargs="*"`, and
stays distinct from an ABSENT extra, which takes the params table's own value.
That distinction is already made for `audio_stg_blocks` (`ltx2_video.cpp:3540`)
and is made the same way here.

## 5. Port map

| Upstream | Here |
|---|---|
| `utils/denoisers.py:62-207` `_guided_denoise` | `Ltx2GuidedDenoise`, `src/vllm/model_executor/models/ltx2_denoisers.cpp` |
| `utils/denoisers.py:25-28` `_POSITIVE_ONLY_GUIDER` | the default-constructed `Ltx2MultiModalGuiderParams`, whose defaults are already `cfg 1.0 / stg 0.0 / modality 1.0` |
| `model/transformer/model.py:590-604` `X0Model.forward` | the caller's `Ltx2X0Model` lambda, `ltx2_video.cpp` |
| `guidance/perturbations.py:8-16` cross types | `Ltx2DitPerturbation::{video,audio}_cross_attn_skip_all` |
| `model/transformer/transformer.py:335,366` `cross_attn_skip_all` | `ltx2_dit.cpp` A2V / V2A guards |
| `components/guiders.py:244-273` | `Ltx2MultiModalGuidance` (unchanged) |
| `utils/args.py:947-1010` | the six extras in §4.5 |
| `ti2vid_one_stage.py:210-226` | the `one_stage` consumer in `ltx2_video.cpp` |

## 6. Gates

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6 && ctest --test-dir build -j4 --output-on-failure
```

Focused: `ctest --test-dir build -R 'ltx2' --output-on-failure`.

Known-red and cited by the issue that names the test, not the family:
`windows-msvc-*` (#584, no `main` baseline), `test_async_llm` (#294),
`test_engine_core_proc` (#1052), `test_serve_low_tools` (#428),
`test_cpu_x86_llamacpp_floor` exit 4 (#618).

## 6b. Reachability — the sentence the records must carry

Entry point: `vllm_video_generate` → `VideoEngine::Generate`
(`include/vllm.h`) on an engine loaded with `pipeline_kind = one_stage`, which
is a documented value of a documented load extra and needs no other flag. The
chain is `Generate` → the phase loop → `Ltx2GuidedDenoise`. No test constructs a
guider, a DiT or a modality by hand to reach it.

The reachability mutation is the deletion of that call — the
`Ltx2GuidedDenoise(...)` line in the phase loop, replaced by the single unguided
forward this row removes — with the focused gate rerun. A green gate there would
mean the suite measures the seam and not the pipeline.

## 6c. #1049 — partly retired, and the rest argued rather than deferred

| Symbol | Before | After |
|---|---|---|
| `Ltx2BatchedPerturbationConfig` | test-only | **reached**, `ltx2_denoisers.cpp` |
| `Ltx2Guidance` | test-only | still test-only |
| `Ltx2CfgDelta` | reached only via `Ltx2Guidance` | unchanged |
| `Ltx2StgDelta` | reached only via `Ltx2Guidance` | unchanged |

`Ltx2Guidance` is a **kind dispatch upstream does not have**. Every LTX-2
pipeline builds a `MultiModalGuider` and calls `calculate`; there is no object
that holds a `GuiderKind` and selects between CFG-only, STG-only and multi-modal
arms. Routing the production combination through `Ltx2Guidance(kMultiModal, ...)`
to make the symbol live would add a switch statement between the caller and the
function upstream actually calls, and would still leave `Ltx2CfgDelta` and
`Ltx2StgDelta` — the two arms nothing can select — dead. #1049 stays open, its
scope narrows to those three symbols, and the honest disposition is that they
are ported-but-unreachable arms of `guiders.py:23-27,70-74`, not a wiring gap
this row can close.

## 7. Tests to port

Upstream's own tests for this path are `pytest` over `torch` and cannot be run
here; the harness adaptation is the whole of it. What is preserved is the
**structure of what they assert** plus the four defects this tree has already
had on the sibling arm.

### 7.1 The per-arm invariant, on every arm

`cond == latent - sigma*velocity`, per pass, from the trace the render records.
Exact in x0 space; in velocity space the residual is the whole sample. The RED
prints `|x0 - velocity| = 0` **exactly**, which is unambiguous.

Non-vacuity is `REQUIRE`d, not assumed, twice: a zero latent makes the two
candidate tensors coincide, and a zero velocity on a given arm makes
`to_denoised` the identity for that arm alone.

**Every arm**, because #1039's first gate covered only the conditional pass and
three mutations survived it. This path has four arms, so it needs four rows plus
the two double-application positions.

### 7.2 The seam-level rescale control

`rescale_scale = 0.0` against `0.7` on the shipped
`Ltx2MultiModalGuidance`, measuring the disagreement between combining in x0
space and combining in velocity space. At 0.0 the linear terms are invariant and
the two are the same function; at 0.7 they are not. A gate that fires at 0.0 is
not about this defect. The existing T2A case measured 1.50e-07 against 0.352;
the video case adds the **modality** term, which the T2A control could not carry
because T2A pins `modality_scale = 1.0`.

### 7.3 The pass count

`std(cond)/std(pred)` is 1.0 to 1e-5 on this fixture in **both** spaces, so a
naive numeric assertion on the rescale difference passes whether or not the bug
exists (7.6e-07 against a span of 3.41, measured under #1039). The instrument
that works is the **count of forwards by kind**, recorded at the call and not
asserted in prose: an arm silently skipped changes a counter no output does.

### 7.4 The mutations this gate must survive

| # | Mutation | Must go RED at |
|---|---|---|
| M1 | `cond` pass left in velocity space | §7.1 cond row |
| M2 | `uncond` pass left in velocity space | §7.1 uncond row |
| M3 | `ptb` pass left in velocity space | §7.1 ptb row |
| M4 | `mod` pass left in velocity space | §7.1 mod row |
| M5 | second `ToDenoised` **below** the step-0 record | the Euler-recovery check |
| M6 | second `ToDenoised` **above** the step-0 record | the guider-replay check |
| M7 | `uncond` pass given the positive context | the replay check / a uncond≠cond check |
| M8 | `mod` pass given no cross-attn perturbation | a mod≠cond check |
| M9 | `ptb` pass given no self-attn perturbation | a ptb≠cond check |
| M10 | `PostProcessLatent` applied per arm instead of after the guider | the replay check |
| M11 | the production call site deleted (reachability) | the whole case |

Each mutation reports three facts: `git diff --stat`, whether it **BUILT** with
the compile-error count, and the exit code captured directly. A non-building
mutation reads exactly like a passing test.

## 8. Risks and decisions

**R1 — the space error, on the default arm.** `rescale_scale` is 0.7 on the
2.4/2.5 video row and 0.45 on the HQ row; both are non-zero, so a space error
hits the default. Mitigated by §7.1 and §7.2, and by the conversion living in
the caller's lambda where the seam cannot receive a velocity.

**R2 — the shared pass list.** Assembling per modality is the plausible wrong
design and it renders. Mitigated by the pass-count trace and by the seam taking
both guiders.

**R3 — cost.** Four forwards per step where there was one. That is upstream's
own cost — `denoisers.py` batches them into one call, this port runs them
serially — and it is a **correctness** row, so the throughput axis is not traded
against it. Recorded, not hidden: a `one_stage` render is now up to 4x the DiT
work per step. `distilled_two_stage`, the default recipe and the one every
benchmark on this row's campaign used, denoises with `SimpleDenoiser` upstream
(`distilled.py:266,295`) and is **unchanged** by this row.

**R4 — the device arm.** §4.3. Refused by name, owed by a new issue, rather than
run unperturbed.

**R5 — concurrent edits.** #921 touches `ltx2_pipeline.{h,cpp}` and the stepper
enum. This row's new code is in a new file; its edits to `ltx2_pipeline.h` are
additive constants only.

## 9. Stop conditions

Stop and report `NEEDS_DECISION` rather than narrowing silently if:

- the `mod` pass cannot be made to differ from the `cond` pass on the fixture,
  because then §7.4 M8 cannot go red and the isolated-modality arm is gated by
  nothing;
- the guided `one_stage` render cannot be reached without a text tower **and**
  the negative-embeds adaptation is judged out of scope, because then the
  consumer is unreachable in-tree and this becomes a seam-only row.

## Owed

- **The device-resident `ptb` and `mod` passes.** `Ltx2DitForwardDevice` takes
  no `perturbations`. Owned by this row's follow-up issue; refused by name until
  then.
- **The other four pipelines** — `a2vid_two_stage`, `ti2vid_two_stages`,
  `ti2vid_two_stages_hq`, `keyframe_interpolation`. Each needs its own row; none
  is blocked on this seam any more.
- **#1049's remaining three symbols** — see §6c.
- **An oracle-run comparison.** vLLM-Omni is UNPINNED (#633) and carries no
  LTX-2.5 recipe; no LTX-2.5 checkpoint here has a recorded sha256 (#1048). The
  guidance arithmetic is gated against upstream **source**, not against upstream
  **output**, and that is the ceiling on this row's evidence.

## Now

Spec committed. Implementation and gate follow on the same pull request.
