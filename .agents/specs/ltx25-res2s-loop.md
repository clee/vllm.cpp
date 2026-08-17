# LTX-2.5 — the `res_2s` denoising loop, and the second evaluation no shape check can see

Row: `LTX25-RES2S-LOOP`. Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned;
**not edited by this row**). Issue:
[#921](https://github.com/mudler/vllm.cpp/issues/921). Parent campaign issue:
[#644](https://github.com/mudler/vllm.cpp/issues/644). Previous owner:
[`ltx25-resolution-envelope.md`](ltx25-resolution-envelope.md) `## Owed`, which
mirrored the geometry half and explicitly did not take the sampler.

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`, `packages/ltx-pipelines`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |

Verified at the local checkout `/home/mudler/_git/LTX-2` before any anchor below
was taken: `git rev-parse HEAD` = `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
`git status --short` empty. Base: `origin/main` at `b5756ea8c`.

vLLM implements nothing in this class. `vllm-project/vllm-omni` stops at LTX-2.3
and carries no `res_2s` sampler at all, so Lightricks is the reference for this
row under AGENTS.md `## When vLLM has no implementation`, recorded in
[`.agents/oracles/`](../oracles/).

---

## 0. Honesty statement

**What lands.** The `res_2s` second-order sampler — `phi`, the RK coefficients,
the two transformer evaluations per step, the bong anchor refinement, both SDE
injections, and the terminal step — plus a `res2s_two_stage` recipe that reaches
it from `pipeline_kind`, which is a LOAD knob and therefore reaches `ltx2-gen`,
the C ABI and the server on their default configurations.

**What is measured and what is not.** Every numeric claim below is gated against
**upstream's own executing code**, not against a restatement of it: `res2s.py`,
`samplers.py`, `diffusion_steps.py` and `helpers.post_process_latent` are
imported from the checkout at the pin and run, and their outputs are the goldens.
Only three things are substituted, and each is one this port reproduces exactly:
the denoiser (a fixed quadratic), the noise **draw** (`torch.randn`, whose stream
this port does not have — see §4.3), and two media-IO modules (`av`,
`OpenImageIO`) that the import chain pulls in and nothing numeric touches.

**No render on real weights is claimed.** `dgx.casa` is contended and OOM-reboots
under a second job; a sampler is exactly the thing CPU goldens gate well. The
real-checkpoint HQ render is owed and named under `## Owed`.

**Upstream ships no tests at this pin.** `find /home/mudler/_git/LTX-2 -name
'test_*.py'` returns 0 lines. "Port the upstream tests in the same change" has
nothing to port, so §5 does the stronger thing available: run upstream's own
functions and pin their output.

---

## 1. What upstream does, with anchors on both sides

### 1.1 The pipeline, and the three things that make it HQ

`TI2VidTwoStagesHQPipeline` (`ti2vid_two_stages_hq.py:59`) differs from
`TI2VidTwoStagesPipeline` in exactly three things, and the row verified each at
the pin rather than inheriting them from #921:

1. `stepper = Res2sDiffusionStep()` (`ti2vid_two_stages_hq.py:258`), passed to
   both stages (`:285`, `:319`).
2. `loop = res2s_audio_video_denoising_loop`, passed to **both** stages
   (`:292`, `:335`).
3. `LTX_2_3_HQ_PARAMS` (`utils/constants.py:95-115`): 15 steps, stage 1 at
   `1088 // 2` x `1920 // 2`, STG off on both modalities, video rescale 0.45,
   audio rescale 1.0, cfg 3.0 / 7.0, modality 3.0.

Stage 1 runs at `width // 2, height // 2` (`:238-243`) under a `GuidedDenoiser`
with a negative context (`:271-281`); stage 2 runs at full resolution under a
`SimpleDenoiser` (`:316`) with `stage_2_sigmas` defaulting to
`STAGE_2_DISTILLED_SIGMAS` (`:193`) and re-noising to `stage_2_sigmas[0]`
(`:327`, `:332`), after a spatial upsample of the stage-1 latent (`:297`).

### 1.2 The loop runs on its own defaults, and that is a finding

`DiffusionStage.__call__` calls the loop with **six** keyword arguments and no
others — `sigmas`, `video_state`, `audio_state`, `stepper`, `transformer`,
`denoiser` (`utils/blocks.py:566-573`). Nothing in the HQ pipeline overrides any
other parameter. So every remaining knob of
`res2s_audio_video_denoising_loop` (`samplers.py:208-223`) takes its declared
default on the shipped arm:

| Parameter | Value on the HQ arm | Anchor |
|---|---|---|
| `noise_seed` | `-1` | `samplers.py:215` |
| `noise_seed_substep` | `None` -> `noise_seed + 10000` = `9999` | `samplers.py:216`, `:265-266` |
| `eta` | `0.5` (step level); substeps are **always** `0.5` | `samplers.py:217`, `:274` |
| `bongmath` | `True` | `samplers.py:218` |
| `bongmath_max_iter` | `100` | `samplers.py:219` |
| `new_noise_fn` | `_get_new_noise` (normalized), **not** `_get_plain_noise` | `samplers.py:220`, `:164-170` |
| `model_dtype` | `torch.bfloat16` | `samplers.py:221` |
| `legacy_mode` | `True` | `samplers.py:222` |

Two of these are load bearing and would be easy to get wrong by analogy with the
already-ported ancestral arm:

**The SDE noise does not depend on the request seed.** `noise_seed = -1` is a
constant, not the pipeline's `seed`. The initial latent still depends on the
seed through the noiser; the loop's own injections do not. The ancestral arm
does the opposite (`distilled.py:69-73` derives its seed from the pipeline's),
so mirroring by analogy would have been wrong. Mirrored here by giving the loop
upstream's own default parameters and having the engine call it the way
`DiffusionStage.__call__` does.

**The noise is normalized.** `euler_ancestral_denoising_loop` defaults
`new_noise_fn=_get_plain_noise` (`samplers.py:574`), a bare `torch.randn`. The
`res_2s` loop defaults to `_get_new_noise` (`samplers.py:220`), which draws in
`highest_precision_float` and then applies `(n - n.mean()) / n.std()` followed by
`_channelwise_normalize` (`samplers.py:164-170`, `:160-161`). Two loops, two noise
functions, at two adjacent lines in one file.

**`legacy_mode=True` means the timestep conversion does NOT happen.**
`_inject_sde_noise` (`samplers.py:173-205`) converts sigmas through
`timesteps_from_mask` only when `legacy_mode` is false (`:188-192`); on the HQ
arm it hands the stepper the raw schedule and applies `post_process_latent`
afterwards (`:202-203`).

### 1.3 `phi` is a cancellation cliff, not a series expansion

`phi(j, neg_h)` (`res2s.py:4-22`) returns `1 / j!` when `abs(neg_h) < 1e-10`, and
otherwise evaluates `(exp(z) - sum_{k<j} z^k / k!) / z^j` **directly**. Outside
the guard and near it, that formula cancels catastrophically, and upstream's
values are the cancelled ones. Measured by running upstream's own `phi`:

| `z` | `phi(1, z)` | `phi(2, z)` |
|---|---|---|
| `-1e-11` (guarded) | `1.0` | `0.5` |
| `-1e-10` (NOT guarded) | `1.000000082740371` | **`0.0`** |
| `-1e-9` | `0.9999999717180684` | **`0.0`** |
| `-1e-8` | `0.999999993922529` | **`1.1102230246251563`** |
| `-1e-6` | `0.9999994999843054` | `0.5000444502911705` |

This is the row's sharpest correctness trap and it points the opposite way to
intuition. A port that used a Taylor series near zero — the numerically *better*
implementation — would return `0.5` where upstream returns `0.0` and `1.11`, and
would be **wrong for this port's purpose**, which is to mirror. It also fixes the
threshold: `1e-10` is not `< 1e-10`, so `-1e-10` takes the formula branch, and
the guard's exact constant is observable. §5 pins all of it.

It reaches the coefficients directly: at `h = 1e-10`,
`get_res2s_coefficients` returns `b1 = 1.000000082740371`, `b2 = 0.0`; at
`h = 1e-8`, `b1 = -1.2204460553277836`, `b2 = 2.2204460492503126`.

### 1.4 The loop (`samplers.py:208-447`)

Per full step `i`, with `sigma = sigmas[i]`, `sigma_next = sigmas[i+1]`,
`h = hs[i]`:

1. **Terminal sigma injection, before anything else.** `n_full_steps =
   len(sigmas) - 1` is taken **first** (`:279`); then, if `sigmas[-1] == 0`, the
   schedule becomes `sigmas[:-1] ++ [0.0011, 0.0]` (`:281-282`). The schedule
   therefore grows by one and `n_full_steps` still counts the caller's steps.
2. `hs = -log(sigmas[1:] / sigmas[:-1])` on the **modified** schedule, in `hp`
   (`:284`).
3. Anchors: `x_anchor = state.latent.clone().to(hp)` (`:295-296`).
4. **Evaluation 1** at `sigmas[i]` (`:301`), then `post_process_latent` on each
   present modality (`:304-307`).
5. `a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2=0.5)` (`:312`).
6. `sub_sigma = sqrt(sigma * sigma_next)` — "a hardcode for c2 = 0.5" (`:314-315`).
7. `eps_1 = denoised_1 - x_anchor`; `x_mid = x_anchor + h * a21 * eps_1`
   (`:320-332`).
8. **Substep SDE injection** at `eta = 0.5`, with `sigmas = [sigma, sub_sigma]`
   (f64) and `step_idx = 0` (`:337-352`).
9. **Bong refinement**, when `bongmath and h < 0.5 and sigma > 0.03`
   (`:357-364`): `bongmath_max_iter` unconditional iterations of
   `x_anchor = x_mid - h * a21 * eps_1; eps_1 = denoised_1 - x_anchor`.
   There is no early exit. Both `x_anchor` and `eps_1` are carried forward.
10. **Evaluation 2** at `sigmas = [sub_sigma]`, `step_index = 0`, over the
    mid-state cast to `model_dtype` (`:369-386`), then `post_process_latent`
    (`:389-392`).
11. `eps_2 = denoised_2 - x_anchor`;
    `x_next = x_anchor + h * (b1 * eps_1 + b2 * eps_2)` (`:397-407`).
12. **Step SDE injection** at `eta`, with the loop's own **float32** schedule and
    `step_idx = i` (`:412-427`).
13. `state.latent = x_next.to(model_dtype)` (`:430-433`).

Then, when `sigmas[-1] == 0`, one final evaluation at index `n_full_steps` —
which is the injected `0.0011` — whose `post_process_latent`'d prediction becomes
the state outright (`:436-445`).

**The DiT evaluation count is therefore exactly `2 * n_full_steps + 1` when the
caller's schedule ends at 0, and `2 * n_full_steps` when it does not.** Measured
against upstream: a 5-sigma schedule ending at 0 gives 9; a 4-sigma schedule not
ending at 0 gives 6; a 2-sigma schedule ending at 0 gives 3. The already-shipped
Euler arm gives `n_full_steps` and `n_full_steps` — **half**. This count is the
discriminator this whole row rests on, because no shape check, no frame count and
no rendered clip can tell the two samplers apart.

### 1.5 The precision split is upstream's, at two levels

The loop works in `hp` = float64 on CPU (`samplers.py:262`, "float64 on CUDA/CPU
for ODE numerical stability"), and writes back to `model_dtype`. The
already-ported ancestral loop does the opposite and steps in **float32**
(`samplers.py:550-551`, `.float()` on both operands). Two loops, two precisions,
stated by upstream at both sites.

Inside the loop the SDE coefficients themselves split again, and this one is
implicit rather than stated:

* the **substep** injection is handed `torch.stack([sigma, sub_sigma])`, both
  `hp` (`samplers.py:342`), so `get_sde_coeff` runs in **float64**;
* the **step** injection is handed the loop's own `sigmas` (`samplers.py:415`,
  `:425`), which `DiffusionStage` created as **float32**
  (`ti2vid_two_stages_hq.py:268`), so `get_sde_coeff` runs in **float32**.

Mirrored rather than unified: one templated implementation, instantiated at the
two scalar types, so there is no second copy of the formula. §3.2.

---

## 2. Scope

### In

* `phi`, `get_res2s_coefficients` and the phi cache (`res2s.py:1-62`).
* `res2s_audio_video_denoising_loop` (`samplers.py:208-447`), including
  `_get_new_noise`'s normalization (`samplers.py:160-170`) and
  `_inject_sde_noise`'s legacy arm (`samplers.py:173-205`).
* `Res2sDiffusionStep.step` / `.get_sde_coeff` at **float64**, by templating the
  already-gated float32 implementation rather than copying it.
* `Ltx2StepperKind::kRes2s`.
* A `res2s_two_stage` recipe row, from `LTX_2_3_HQ_PARAMS` and
  `ti2vid_two_stages_hq.py`.
* The engine dispatch, so `pipeline_kind=res2s_two_stage` reaches the loop from
  `include/vllm.h`, `ltx2-gen` and the server.
* A `dit_evaluations` counter on `Ltx2ConditioningTrace`, because the count is
  the only observable that separates the two samplers.

### Out, and refused or recorded rather than dropped

* **The prompt enhancer, the distilled LoRA per stage, and DiffVAE.** Already out
  of scope for every LTX row here; unchanged by this one.
* **Bit-exact SDE noise against upstream.** The draw is `torch.randn` on a seeded
  `torch.Generator`; this port has `SplitMixGaussian`. Already true of the
  ancestral arm, which ships. Recorded in §4.3, not hidden.
* **`legacy_mode=False`.** Nothing upstream reaches it on this pipeline
  (`DiffusionStage` passes no `legacy_mode`), so mirroring means not building a
  selection surface for it. Recorded under `## Owed`.
* **`gradient_estimating_euler_denoising_loop`** (`samplers.py:84-152`) and
  `EulerCfgPpDiffusionStep`. Different samplers, no pipeline in scope selects
  them.

---

## 3. Design

### 3.1 A new translation unit, mirroring upstream's own file

`include/vllm/model_executor/models/ltx2_samplers.h` +
`src/vllm/model_executor/models/ltx2_samplers.cpp`, mirroring
`ltx-pipelines/utils/samplers.py` and `utils/res2s.py`. The steppers stay in
`ltx2_pipeline.{h,cpp}`, which mirrors `ltx-core/components/diffusion_steps.py`.
That is upstream's own partition: a *stepper* advances one substep, a *sampler*
decides how many substeps there are and what is evaluated between them, and they
live in different packages upstream.

### 3.2 The loop takes hooks, because upstream's takes a `denoiser`

`res2s_audio_video_denoising_loop` is a free function whose model access is a
`Denoiser` callable (`samplers.py:213-214`). Mirroring that shape is also what
makes the evaluation count gateable: a test supplies a counting denoiser and
asserts an exact number.

```
struct Ltx2Res2sHooks {
  // `denoiser(transformer, video_state, audio_state, sigmas, step_index)`
  std::function<void(const std::vector<float>&, const std::vector<float>&,
                     float sigma, std::vector<float>&, std::vector<float>&)> denoise;
  // `post_process_latent` (utils/helpers.py:461-463)
  std::function<std::vector<double>(std::vector<double>, bool is_video)> post_process;
  // `new_noise_fn` (samplers.py:220)
  std::function<std::vector<double>(int64_t count, bool is_video, bool substep)> new_noise;
};
```

`denoise` takes a **scalar** sigma rather than a schedule plus an index, because
both upstream call sites reduce to `sigmas[step_index]` at
`SimpleDenoiser.__call__` / `GuidedDenoiser.__call__` (`utils/denoisers.py:237`)
and the second call site already passes a one-element schedule with index 0
(`samplers.py:384-385`). Passing the pair would invite a caller to index it
differently from upstream.

The step arithmetic is one templated core in `ltx2_pipeline.cpp`:

```
template <typename Sigma, typename Value> std::vector<Value> Res2sStepImpl(...);
```

instantiated three times — `<float,float>` for the existing, already-gated
`Ltx2Res2sStep`; `<float,double>` for the step-level injection; `<double,double>`
for the substep injection. One formula, three dtypes, matching §1.5. The
selection is an enum named after the two upstream call sites, not a bare bool.

### 3.3 The engine hoists its per-evaluation body

`ltx2_video.cpp`'s phase loop currently builds `Ltx2ModalityInput`, runs
`Ltx2DitForward*`, and post-processes, all inline in the step loop. This row
hoists that into one `Evaluate(video_latent, audio_latent, sigma)` lambda that
**both** arms call: the Euler/ancestral loop calls it once per step, the res_2s
loop calls it through `hooks.denoise`. No second forward path is written by
hand, and the keyframe-mask guards, the frozen-sigma handling and the trace
updates are reached identically from both.

`im.trace.dit_evaluations` increments inside `Evaluate`, so it counts every arm,
across every phase.

### 3.4 The `res2s_two_stage` recipe

Two phases, from §1.1:

| | phase 0 `generate_lowres_hq` | phase 1 `refine_hq` |
|---|---|---|
| `spatial_downscale` | 2 (`:238-243`) | 1 |
| `sigmas` | empty — derived, 15 steps (`:260-267`) | `STAGE_2_DISTILLED_SIGMAS` (`:193`) |
| `noise_scale` | 1.0 | `stage_2_sigmas[0]` (`:327`) |
| `input_transform` | `kInitial` | `kSpatialUpsample` (`:297`) |
| `stepper` | `kRes2s` | `kRes2s` (`:285`, `:319`) |
| guidance | HQ params, override allowed | override not allowed (`SimpleDenoiser`, `:316`) |

Recipe level: `height`/`width` from `Ltx2Params23Hq().stage_2_*()`,
`num_inference_steps = 15`, `allow_request_sigmas = true` and
`fixed_num_inference_steps = false` (stage 1's schedule really is derived from
`num_inference_steps`), `allow_negative_prompt = true` (stage 1 builds a
`GuidedDenoiser` with a negative context, unlike the distilled arm).

`("res2s_two_stage", "2.5")` only. `LTX_2_3_HQ_PARAMS` is a plain constant that
overrides every generation-varying knob (`constants.py:91-94` says so), so there
is no `detect_params` lineage to spread it across versions, and this port's
checkpoint is 2.5.

**The name.** `res2s_two_stage` is already the string
`test_ltx2_pipeline.cpp:1186` uses as a NEGATIVE control for the refusal table.
That case must be repointed in this change, and §5 makes the repointing visible
rather than silent.

---

## 4. Risks

### 4.1 The one that renders

Serving the HQ preset's 15 steps and 0.45 rescale on the Euler loop renders a
plausible clip at **half** the model evaluations the preset was tuned for. There
is no pixel, count or shape that says so. Mitigation: `dit_evaluations`, asserted
to an exact number both at the loop and end to end through `Generate`.

### 4.2 A "better" `phi`

§1.3. Mitigation: goldens taken from upstream's own `phi` at the cliff, including
`phi2(-1e-10) == 0.0` exactly.

### 4.3 The noise stream diverges from upstream

`torch.randn` on a seeded `torch.Generator` is not reproducible here. Consequence:
`res2s` renders are not bit-comparable with upstream, exactly as the shipped
ancestral arm is not. What IS gated: the normalization `_get_new_noise` applies
after the draw, and the loop arithmetic under an injected deterministic noise
function. Stated, not hidden.

### 4.4 The bong loop is a fixed point, and 100 iterations is not arbitrary

`eps_1 <- (denoised_1 - x_mid) + h * a21 * eps_1` contracts with ratio
`h * a21`, which the `h < 0.5` guard bounds under `0.25`. It converges to machine
precision long before iteration 100, so an early exit would be numerically
invisible — which is exactly why the loop is mirrored as written and
`bongmath_max_iter` is a parameter rather than a constant folded away.

### 4.5 Stage 2 needs the spatial upsampler

`input_transform = kSpatialUpsample` reaches `Ltx2UpsampleVideoLatent`, which
refuses a spatiotemporal upsampler by name. That is the pre-existing behaviour of
`distilled_two_stage` phase 1 and is unchanged here; the HQ arm inherits both the
capability and its refusal.

---

## 5. Tests and evidence

All goldens are generated by running **upstream's own code** at the pin. The
generator is `gen_goldens.py`, recorded in this section, and the substitutions it
makes are §0's three.

### 5.1 `test_ltx2_pipeline`

1. **"ltx2 res2s phi mirrors upstream at the small-z cliff"** — `phi(1, z)` and
   `phi(2, z)` at 14 values of `z` spanning `0`, both sides of the `1e-10`
   guard, and the mid range. Asserted **exactly** (`==`) at `z = 0`,
   `z = -1e-11`, `z = -1e-10` and `z = -1e-9`, where upstream's values are
   `1.0/0.5`, `1.0/0.5`, `1.000000082740371/0.0` and `0.9999999717180684/0.0`.
   A series-expansion port fails on the third and fourth rows.
2. **"ltx2 res2s coefficients mirror upstream"** — `a21`, `b1`, `b2` at 11 values
   of `h` including `1e-12`, `1e-10` and `1e-8`, i.e. the cliff carried into the
   coefficients.
3. **"ltx2 res2s the noise normalization is applied"** — `_get_new_noise`'s two
   normalization steps against upstream's own `_channelwise_normalize` on a fixed
   input. Positive control: a zero-filled buffer must NOT reproduce the golden.
4. **"ltx2 res2s the loop evaluates TWICE per step"** — the discriminator. Four
   fixtures, a counting denoiser, and an exact expected count:

   | Fixture | `sigmas` | `n_full` | expected evaluations | forces |
   |---|---|---|---|---|
   | `BongOn` | `0.9, 0.8, 0.7, 0.62` | 3 | **6** | `h < 0.5` and `sigma > 0.03` on every step |
   | `BongOffByH` | `0.9, 0.5, 0.25, 0.12` | 3 | **6** | every `h >= 0.5`; every `sigma > 0.03` |
   | `BongOffBySigma` | `0.03, 0.028, 0.026, 0.025` | 3 | **6** | every `h < 0.5`; every `sigma <= 0.03` |
   | `TerminalZero` | `1.0, 0.75, 0.5, 0.25, 0.0` | 4 | **9** = 2*4+1 | the injected `0.0011` tail |

   The sequence of sigmas the denoiser was called at is asserted too, so a build
   that ran two evaluations at the *same* sigma fails: `BongOn` must see
   `0.9, 0.848528, 0.8, 0.748331, 0.7, 0.658787`, i.e. `sqrt(sigma*sigma_next)`
   interleaved. `TerminalZero`'s last two are `0.0165831` and `0.0011`.

   **Making the expected value impossible to hit by accident:** the counts are 6
   and 9, never 0 and never the step count, so neither a stub that evaluates
   nothing nor one that evaluates once per step can pass. `9 != 4` and `6 != 3`
   are the assertions that separate this sampler from the shipped one.
5. **"ltx2 res2s the bong refinement is reached, and only in its own branch"** —
   the same four fixtures run with `bongmath` true and false. `BongOn` and
   `TerminalZero` must **differ**; `BongOffByH` and `BongOffBySigma` must be
   **byte-identical**. Both goldens are carried, so "differ" is not asserted
   against a value this port computed. This is how each branch is forced and how
   the forcing is shown to have worked: the `h` fixture keeps every sigma above
   0.03 and the sigma fixture keeps every `h` below 0.5, so neither can be
   passing for the other's reason. `sigma > 0.03` is strict and the fixture
   starts at exactly `0.03`.
6. **"ltx2 res2s the loop reproduces upstream"** — final video and audio latents
   for all four fixtures, against upstream's own loop output. The denoise mask is
   `1,1,0,1,0,1` with a distinct clean latent, so `post_process_latent` is not the
   identity and a build that dropped the blend fails at three positions.
7. **"ltx2 the res2s_two_stage recipe is upstream's HQ preset"** — the §3.4
   table, plus that `("res2s_two_stage", "2.3")` still refuses by name.

### 5.2 `test_ltx2_video` — the production path

8. **"ltx2 video: the HQ pipeline evaluates the DiT twice per step"** — a
   `pipeline_kind=res2s_two_stage` load on the reduced-dimension fixture,
   `engine->Generate(...)`, and `trace.dit_evaluations` asserted against the
   number the two phases' schedules imply. Compared **against the same render on
   `distilled_two_stage`**, which must report strictly fewer, so the assertion
   cannot pass by both arms being the same.

### 5.3 Reachability

The production entry point is `vllm_video_generate` -> `VideoEngine::Generate` ->
`Ltx2VideoEngine::Generate` -> the phase loop's `kRes2s` dispatch. The mutation
is §8: delete the dispatch, rerun, show RED.

---

## 6. Gates

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Reported with `CONFIGURE_EXIT`, `BUILD_EXIT`, `: error:` count, `ctest -N`,
`CTEST_EXIT`, the pass/fail line, load and free disk, and positive controls for
`No space left` and `BFD assertion`.

Known-red and not this row's: `windows-msvc-*` (#584). Load-dependent:
`test_async_llm` (#294), `test_engine_core_proc` (#1052), `test_serve_low_tools`
(#428), `test_cpu_x86_llamacpp_floor` exiting 4 as `NO_QUIET_WINDOW` (#618).

---

## 7. Stop conditions

* Return `NEEDS_DECISION` rather than narrowing if the `res2s_two_stage` recipe
  cannot be made reachable from `pipeline_kind` without changing the C ABI.
* Do not take the GPU. A sampler is gated by CPU goldens; `dgx.casa` is contended
  and OOM-reboots under a second job.
* Do not "fix" `phi`. §1.3.

---

## 8. What the mutation pass found

Filled in by the implementation. Each row records `git diff --stat`, whether the
mutation **BUILT** with its compile-error count, and the exit code captured
directly — because a mutation that fails to build and one that never applied both
read exactly like a passing test.

---

## Owed

* [#921](https://github.com/mudler/vllm.cpp/issues/921) is closed by this row.
* A real-checkpoint HQ render on `dgx.casa`, and a rendered-clip comparison
  against the Euler arm at the same preset. Not attempted here (§0).
* `legacy_mode=False` (`samplers.py:188-192`) — the `timesteps_from_mask`
  conversion inside `_inject_sde_noise`. Unreachable upstream from any pipeline in
  scope; no selection surface built.
* Bit-exact SDE noise against upstream's `torch.randn` stream (§4.3), which the
  already-shipped ancestral arm owes on the same grounds.

## Now

`ACTIVE` — implementation in flight on `row/LTX25-RES2S-LOOP`.
