# LTX-2.5 — the prompt-side AdaLN path (`use_prompt_adaln_single`)

Row: `LTX25-PROMPT-ADALN`. Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned; not
edited by this row). Issue:
[#644](https://github.com/mudler/vllm.cpp/issues/644), row 0.

Upstream pins:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`) | `fd4ded7f` |
| huggingface/diffusers | `3a2f35d4` |

Both are read from local checkouts at those revisions, and the golden generator
IMPORTS and EXECUTES the LTX-2 modules rather than restating them.

## 0. What is wrong today

`src/vllm/model_executor/models/ltx2_loader.cpp:988` sets, unconditionally:

```cpp
declared.use_prompt_adaln_single = false;
```

and `:573` / `:626` do the same on the two manifest paths. The flag defaults
**TRUE** in both references:

- diffusers `src/diffusers/models/transformers/transformer_ltx2.py:1185` —
  `use_prompt_adaln_single: bool = True`
- LTX-2 `packages/ltx-core/src/ltx_core/model/transformer/model.py:77` — same
  default, and `model_configurator.py:76` / `:138` read it as
  `config.get("use_prompt_adaln_single", True)`

The shipped FP8 DiT carries the 18 tensors the flag builds (12
`prompt_adaln_single.*`, 6 `audio_prompt_adaln_single.*`; see
`tests/vllm/models/ltx2_fp8_dit_manifest.inc:232-240,286-294`), so the flag is
TRUE for the checkpoint this campaign renders. `ltx2.cpp:274-276` refuses those
tensors by name, so a real render needs `allow_unported_modules=1`
(`src/vllm/multimodal/ltx2_video.cpp:570`) — which reaches the loader lines above
and **silently clears the flag**.

Net effect: every render drops the timestep-conditioned half of the prompt K/V
modulation, keeping only the static `prompt_scale_shift_table`. Nothing observes
it: shapes are unchanged, values stay finite, and the goldens were generated with
`use_prompt_adaln_single=False` (`scripts/gen-ltx2-goldens.py:149`), so the gate
agrees with the defect.

Campaign history this row does **not** re-derive: `ltx-2-5.md` §1.2 already
RETRACTED the "prompt K/V carry no timestep term" claim and recorded that the
shipped checkpoint carries a `[4096, 256]` prompt timestep embedder. What was
never closed is *using* the tensors.

## 1. What upstream does, with anchors

### 1.1 The module

`model.py:222-227` (video) and `:252-257` (audio):

```python
self.prompt_adaln_single = (
    AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2)
    if self.cross_attention_adaln and self.use_prompt_adaln_single
    else None
)
```

`AdaLayerNormSingle` (`adaln.py:19-45`) is the same brick the port already has
(`Ltx2AdaLayerNormSingle`): `emb.timestep_embedder.linear_1 [dim, 256]`,
`linear_2 [dim, dim]`, `linear [coefficient * dim, dim]`. Coefficient **2** here,
not `adaln_embedding_coefficient()` — shift and scale for the K/V only.

diffusers twin: `transformer_ltx2.py:1255-1259`, `num_mod_params=2`.

Registration order inside `_init_video` puts it between `adaln_single` and
`proj_out`, which is where `EnumerateLtx2DitTensors` already reserves its slot
(the `VT_CHECK` at `ltx2.cpp:274-276`).

### 1.2 The producer

`transformer_args.py:274-277`, inside `TransformerArgsPreprocessor.prepare`:

```python
prompt_timestep = None
if self.prompt_adaln is not None:
    prompt_timestep, _ = self._prepare_timestep(
        modality.sigma, self.prompt_adaln, batch_size, modality.latent.dtype
    )
```

Three things this fixes in one line, each of which a shape check cannot see:

1. The input is **`modality.sigma`**, `(B,)` (`modality.py:54`) — the per-sample
   scalar noise level — **not** `modality.timesteps`, which is per-token `(B, T)`.
2. `_prepare_timestep` (`transformer_args.py:173-186`) multiplies by
   `timestep_scale_multiplier` before the embedder, exactly as the port's
   `PrepareTimestep` already does for the main AdaLN.
3. The result is viewed to `(B, -1, 2 * dim)`, i.e. `(B, 1, 2 * dim)` — one row
   broadcast over the prompt tokens.

Wired into both preprocessor kinds at `model.py:313`, `:333` (multimodal) and
`:348`, `:364` (single-modality).

diffusers twin: `transformer_ltx2.py:1536-1547`. diffusers passes `sigma` already
scaled from the pipeline (`pipeline_ltx2_image2video.py:1481` — `sigma=timestep`,
and `timestep` is the scheduler's 0..1000 value), so the two references agree on
the value reaching the embedder; only the place the x1000 happens differs. This
port mirrors LTX-2, so the multiply happens here.

### 1.3 The consumer

`transformer.py:427-447` (`apply_cross_attention_adaln`):

```python
kv_modulation = prompt_scale_shift_table[None, None].to(...)              # :441
if prompt_timestep is not None:                                           # :442
    kv_modulation = kv_modulation + prompt_timestep.reshape(
        batch_size, prompt_timestep.shape[1], 2, -1)                      # :443
shift_kv, scale_kv = kv_modulation.unbind(dim=2)                          # :444
...
encoder_hidden_states = context * (1 + scale_kv) + shift_kv               # :446
```

Reached from `_apply_text_cross_attention` (`:223-251`), which is called for the
video stream at `:288-296` and the audio stream at `:317-325`, passing
`video.prompt_timestep` / `audio.prompt_timestep`. Every block, both streams.

diffusers twin: `transformer_ltx2.py:677-693` (`get_mod_params` over
`prompt_scale_shift_table`), threaded at `:1648-1649`.

Layout consequence: the flat `[B, 1, 2 * dim]` row is read as `[2, dim]` with
**shift first, scale second** — the same order the static table already uses in
`ModulateContext` (`ltx2_dit.cpp:118-129`).

## 2. Scope

**In.**

1. `Ltx2DitParams::use_prompt_adaln_single` is honoured end to end: contract,
   binding, host forward, device forward.
2. The 18 tensors enter `EnumerateLtx2DitTensors` / `BindLtx2DitWeights` when
   `cross_attention_adaln && use_prompt_adaln_single`.
3. `temb_prompt` / `temb_prompt_audio` computed from each stream's own `sigma`
   and threaded into every block's text cross-attention on both the host
   (`ltx2_dit.cpp`) and device (`ltx2_device.cpp`) paths.
4. The three loader `= false` assignments are deleted, and replaced by a guard
   (§3.2) that makes a future silent clear impossible.
5. The `ltx2.cpp:274-276` refusal is deleted for these two families.
6. Goldens executed from upstream at reduced dims, with a mutation proving the
   new term is load-bearing, and a measured magnitude.

**Out.** Anything the campaign already records as owed: keyframe absolute
position embedding (still genuinely unported, still what
`allow_unported_modules` is for), the caption projections, guidance
perturbations, and the bf16/FP8/NVFP4 stream dtypes on the host forward.

## 3. Design

### 3.1 The seam

`Ltx2DitWeights` gains two `Ltx2AdaLayerNormSingleWeights` members;
`Ltx2BlockArgs` / `BlockArgsDev` gain a `[batch, 1, 2 * width]` prompt-modulation
pointer per stream, `nullptr` when the flag is off. `ModulateContext` and its
device twin take that pointer and add `prompt_mod[b, {0,1} * width + c]` to the
table row before applying `context * (1 + scale) + shift`. `nullptr` gives the
existing static-only behaviour byte-for-byte, which is what keeps every current
golden valid.

The **order of the two additions** is upstream's: the table and the timestep row
are summed FIRST (`:443`), and only then does `(1 + scale)` apply. Folding it the
other way would round differently — the same trap `ProcessOutput`
(`ltx2_dit.cpp:530-540`) already documents.

### 3.2 The prompt-K/V cache, and what replaces the cleared flag

`Ltx2DitForward` already refuses a cache when the flag is on
(`ltx2_dit.cpp:672-676`); with the flag no longer cleared, that refusal becomes
*reachable* rather than dead, and it is correct: the K/V now carry a timestep
term. Nothing in the shipped pipeline passes a cache (`grep` over
`ltx2_pipeline.cpp` finds none), so no caller regresses.

**What replaced the `= false`.** The loader clearing existed so
`EnumerateLtx2DitTensors` would not throw. With the tensors ported the contract
simply includes them, so the assignment has no job left. In its place the loader
asserts the invariant the clearing used to violate:

> the resolved `use_prompt_adaln_single` must equal whether the FILE carries
> `prompt_adaln_single.linear.weight`

A future edit that re-clears the flag then hits a named refusal instead of
quietly dropping 18 tensors. This is deliberately an *equality*, not a one-sided
check: clearing the flag with the tensors present is the defect this row fixes,
and setting it with the tensors absent would bind missing weights.

### 3.3 `allow_unported_modules`

After this row it is scoped to genuinely-unported modules only: the sole flag it
still clears in a config copy is `use_keyframes_abs_pos_embedding`
(`ltx2_loader.cpp:979-984`), whose module really is unported. The guard in §3.2
is what makes that scoping structural rather than a comment — the extra `= false`
cannot come back without going red.

`Ltx2AdoptDeclaredDitParams`'s contract-equality check becomes load-bearing in a
second way: a checkpoint whose config declares `use_prompt_adaln_single=false`
while its shapes carry the tensors now produces two DIFFERENT contracts and is
refused, instead of both sides being forced to the same cleared value.

## 4. Memory format

Mirrors the existing L2 parity forward exactly: f32 host, f32/bf16 device stream
with the `prompt_scale_shift_table` read at F32 (`ltx2_device.cpp:508-511`). The
prompt modulation is one `[batch, 1, 2 * width]` buffer per stream per forward —
`2 * 4096 * batch` floats for the video stream at full size, computed once
outside the block loop, not per block. No new per-token buffer, so no per-token
byte cost.

## 5. Tests

1. `EnumerateLtx2DitTensors` with the flag ON reproduces upstream
   `named_parameters()` verbatim — names, order, ranks, dims — for a model built
   with `use_prompt_adaln_single=True`. This is the 18-tensor contract.
2. Full dual-stream DiT forward, flag ON, against upstream's executed output.
3. **The mutation**: the same forward with the prompt-AdaLN contribution zeroed
   must go RED against that golden. A term that is present but inert is not a
   port.
4. The flag-OFF goldens stay byte-identical, proving the new path is off when
   upstream's is off.
5. The prompt-K/V cache stays refused with the flag on (existing case).

## 6. Measured magnitude

Recorded in §Outcome, on two fixtures that answer two different questions.

The reduced-dimension generator gives the relative change in the modulated prompt
context and in the DiT's outputs, flag ON vs OFF. That is a GATE FLOOR: a number
below round-off would mean the term is inert and the mutation in §5.3 could not
bite. It is **not** the answer to "does this matter", because both the static
table and the prompt-AdaLN MLP are drawn from the same synthetic init scale, so
every ratio it produces is a property of the fixture.

"Does this matter" is answered on the SHIPPED checkpoint's own weights, run
through upstream's `AdaLayerNormSingle`. That measurement is required before the
row's Outcome may state a magnitude.

## 7. Risks

- **The goldens agree with the defect.** Every existing LTX golden was generated
  with the flag off, so no existing case can fail whatever this row does. The new
  flag-ON case is the only instrument, which is why §5.3 mutates it rather than
  asserting it.
- **`sigma` vs `timesteps`.** Using the per-token `timesteps` instead of the
  per-sample `sigma` produces a same-shaped, finite, wrong result at batch 1 with
  uniform timesteps. The golden runs a batch of 2 with per-token timesteps that
  differ from sigma, so the two are distinguishable.
- **Order within the `[2, dim]` row.** Swapping shift and scale is finite and
  same-shaped. The golden's random weights make it observable.

## 8. Stop conditions

- The flag-ON forward cannot be made to match upstream to the existing
  `kRoundOff` bound: stop and report rather than widening the bound.
- The mutation in §5.3 stays green: the path is not reached, and the row is not
  done. Escalate the mutation's magnitude before concluding anything about
  reachability (issue #604).

## Outcome

### What was measured — (a) the gate floor, on SYNTHETIC weights

The generator emits these into `tests/vllm/models/ltx2_goldens.inc` and prints
them on stderr, from the SAME shared weight stream on both arms (keyed by
parameter name, so every common weight is bit-identical and the difference is the
term and nothing else):

| Quantity | Flag ON vs OFF |
|---|---|
| timestep term vs the static table it is added to | `max\|term\|` 0.0252 vs `max\|table\|` 0.0487 — 51.7% |
| block-0 modulated prompt K/V | `max\|on-off\|` 0.0310 — 5.82% of `max\|off\|` |
| DiT video output (2 blocks) | 1.4567e-4 — 0.04% of `max\|off\|`, **73x** the gate's 2e-6 floor |
| DiT audio output (2 blocks) | 7.367e-5 — 0.03%, **37x** the floor |

**ALL FOUR ROWS ARE GATE-FLOOR NUMBERS, AND NONE OF THEM ANSWERS "DOES THIS
MATTER".** Corrected 2026-08-13 (issue #644) — this section originally billed the
first two as the answer and disclaimed only the last two as synthetic-bounded.
They have identical provenance: `prompt_scale_shift_table` and every
prompt-AdaLN MLP parameter are drawn from the same `param_spec` rule at
`scale=0.05` (`scripts/gen-ltx2-goldens.py:100-106`), so the ratio between them
is a property of the FIXTURE, not of the conditioning. Vary only the MLP init and
it moves with it: 0.005 → 4.1% / 0.49%, 0.05 (committed) → 51.7% / 5.82%,
0.2 → 1450% / 142%. What these rows are FOR is the mutation below: 73x and 37x
above round-off is what makes a zeroed term detectable.

### What was measured — (b) the SHIPPED checkpoint, which is the answer

Measured 2026-08-13 by loading the real tensors into upstream's own
`AdaLayerNormSingle(inner_dim, embedding_coefficient=2)` (`adaln.py:19-45`, built
by `model.py:223-227` / `:253-257`) and evaluating it on `sigma *
timestep_scale_multiplier` — the file's own config gives 1000 — exactly as
`transformer_args.py:274-278` and `:177` do, then comparing against all 48
`prompt_scale_shift_table` / `audio_prompt_scale_shift_table` tensors it is
summed with at `transformer.py:441-443`.

- File: `/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`
  (7876 tensors, 1,179,408-byte header; the prompt-AdaLN tensors are BF16 there,
  so they are read directly with no dequantisation step of ours in the path).
- Upstream: `ltx_core` at `fd4ded7f`, imported BY PATH from
  `/home/mudler/_git/LTX-2` with `ltx_core.__file__` asserted under that checkout.
- Sigmas: a uniform grid over the whole range, `linspace(0, 1, 101)`, and
  separately the sampler the file's own scheduler config names —
  `LinearQuadraticScheduler().execute(8)` (`components/schedulers.py:60-88`).

| | video (dim 4096) | audio (dim 2048) |
|---|---|---|
| `rms\|table\|`, 48 blocks | 0.017553 | 0.021925 |
| `rms\|term\|`, uniform σ | 0.236446 | 0.347171 |
| **term/table, RMS** | **1347%** | **1583%** |
| `max\|term\|` / `max\|table\|` | **7119%** | **2817%** |
| term/table RMS, LinearQuadratic 8-step | 1275% | 1492% |

Split by row, on the uniform grid: shift 605% (video) / 461% (audio), scale 1732%
/ 2044%. The scale row is where it concentrates, which is the row that multiplies.

**On the shipped model the timestep term DOMINATES the static table; the table is
the perturbation.** What the pre-row renders applied was
`context * (1 + ~0.018 rms) + ~0.018` where upstream applies
`context * (1 + ~0.32 rms) + ~0.10`. Stated as the quantity actually consumed at
`transformer.py:446`, for a context of unit rms: the modulated context has rms
1.0035 static-only against 1.0915 upstream (video, **+8.8%**) and 1.0033 against
1.3170 (audio, **+31.3%**).

**The synthetic fixture UNDERSTATES the real defect by two orders of magnitude.**
Comparing like with like — the fixture's 51.7% is a `max\|term\|` / `max\|table\|`
ratio, and that same ratio on the shipped weights is 7119% (video) and 2817%
(audio): **138x and 54x** larger. Recorded because the original Outcome quoted
"roughly half the magnitude" from the fixture as if it described the checkpoint.

Reproduce with `scripts/measure-ltx2-prompt-adaln.py --ltx2 <LTX-2 checkout>
--checkpoint <the file above>`, committed by this repair so the number is
re-runnable rather than transcribed. It asserts `ltx_core.__file__` under the
named checkout before it reads anything, and nothing of ours is in its numeric
path — the only vllm.cpp input is which tensors to read.

### The mutations

All five run on the committed head, restored byte-for-byte afterwards (source
md5s re-checked). Exit status is the authority; assertion COUNTS are recorded
because doctest's summary and the exit code disagree in both directions.

| # | Mutation | Result |
|---|---|---|
| M1 | host `ModulateContext` ignores `prompt_mod` | RED — 3/35 cases, 6/2435 assertions, exit 1 |
| M2 | device `TextCrossAttentionDev` takes the static-only branch always | RED — 1/15 cases, 6/523 assertions, exit 1 |
| M3 | re-add `use_prompt_adaln_single = false` before the loader guard | RED — the guard throws by name; assertion count DROPS 4826 → 4815, exit 1 |
| M4 | prompt AdaLN driven by `m.timesteps` instead of `m.sigma` | RED — 2/35 cases, 4/2435 assertions, exit 1 |
| M5 | shift and scale rows swapped within the `[2, width]` row | RED — 2/35 cases, 4/2435 assertions, exit 1 |

M3 is the one that cannot be reached by any INPUT, and that is stated rather than
papered over: `ParseLtx2DitParamsFromManifest` derives the flag from the same
manifest the guard reads, so they agree by construction unless an assignment
intervenes — which is exactly the edit the guard exists to catch. The
input-driven half of the same rule lives in `Ltx2AdoptDeclaredDitParams`, where a
config that disagrees with the shapes now produces two different contracts and is
refused; that one is gated by a test with real inputs in both directions.

### The gate

`BUILD_EXIT=0` on every build; build logs grepped for `No space left|BFD assertion`
(0 hits) and `df -h /` logged (88% used, 52G free at the end). Case AND assertion
counts against the `cefacd2d0` baseline, measured by reverting the working tree to
HEAD, rebuilding the four targets and running them, then re-applying the diff and
re-checking its md5 (`03324d42…`, identical before and after):

| Suite | HEAD `cefacd2d0` | this row | delta |
|---|---|---|---|
| `test_ltx2` | 30 cases / 1627 assertions | 35 / 2435 | +5 cases, +808 assertions |
| `test_ltx2_loader` | 24 / 4817 | 26 / 4826 | +2 cases, +9 (new cases minus the assertions the retired unported-family claims took with them) |
| `test_ltx2_device` | 13 / 498 | 15 / 523 | +2 cases, +25 assertions |
| `test_ltx2_video` | 30 / 502 | 30 / 502 | unchanged — see below; 502 is the SKIPPED default |

**What `30 / 502` does and does not say (corrected 2026-08-13).** The
shipped-checkpoint case is env-gated: with `LTX2_CHECKPOINT_ROOT` unset it prints
`SKIPPED` and returns at `test_ltx2_video.cpp:917-921`, so `30 / 502` means the
whole real-header case DID NOT RUN — not "it ran and no assertion counted the
module". With the variable pointing at the Lightricks tree the same binary
measures **30 cases / 8734 assertions**, both before and after this repair
(re-measured on this branch, exit 0 in both configurations). Any future quote of
this suite's count owes the configuration alongside it.

The `test_ltx2_video` fixture had to move: it declared a config that omits
`use_prompt_adaln_single` (mirroring the shipped NVFP4 DiT) while its SHAPES said
false, so the config/shape equality check refused it — correctly. It now carries
the module, which is the shipped shape and puts the whole video engine on the new
path.

### What was rejected

- **Widening `modulate`'s kernel contract** with a `rows_per_src_row` divisor, to
  express "one row per batch element broadcast over that element's tokens" in one
  launch. Rejected: it changes a kernel's semantics for a dimension that is 1 or 2
  in every shipped call, and would owe its own red-before evidence. The device
  path loops over batch and offsets the pointers instead.
- **Narrowing the static prompt table to the stream dtype** on the flag-ON path,
  which is literally what upstream's `.to(dtype=x_normed.dtype)` does. Rejected as
  out of scope: the existing static-only path deliberately keeps the table at F32
  (`ltx2_device.cpp`, "a narrowed table would be the dtype rule applied
  backwards"), and changing that polarity is a separate decision. The flag-ON path
  routes the sum through `ada_value`, which stores at the stream dtype — the same
  rounding every other table+modulation sum in that file already has.

### Why the defaults are what they are

`Ltx2DitParams::use_prompt_adaln_single` keeps its `true` default, which is now
honoured rather than overwritten. It matches `model.py:77`,
`model_configurator.py:76`/`:138` and diffusers `transformer_ltx2.py:1185`, and
it matches both shipped DiTs: the FP8 file carries no config at all (so the
default decides), and the NVFP4 file's config OMITS the key — verified by reading
both headers off the NAS. So neither shipped checkpoint is refused by the new
config/shape equality check.

`allow_unported_modules` keeps existing, because
`keyframes_abs_pos_embedding` is genuinely unported and a real render still needs
the opt-in for it. What changed is that it can no longer switch a ported feature
off: the loader asserts the flag against the file instead of clearing it, and
`Ltx2AdoptDeclaredDitParams` clears exactly one flag, for a module nothing
applies.

### The keyframes claim next door, corrected 2026-08-13

`ltx2.h` carried, in the same paragraph this row rewrote, *"LTX-2.5's checkpoint
does not carry the parameter"* about `keyframes_abs_pos_embedding`. It is FALSE —
the same class of claim as the `use_prompt_adaln_single=false` assertion this row
exists to remove — and the tree already contradicted it twice
(`.agents/model-matrix.md`, `tests/vllm/multimodal/test_ltx2_video.cpp:913-914`).
Read straight off both files' headers, and run through upstream's own loader and
configurator:

| | FP8 (`vonkaiser`) | NVFP4 (first-party) |
|---|---|---|
| carries `keyframes_abs_pos_embedding` | YES — `F8_E4M3 [1, 4096]` + F32 scale | NO |
| declares the flag in `__metadata__` | **no `__metadata__` AT ALL** | `true` |
| `LTXModelConfigurator.from_metadata` | **RAISES** `KeyError: 'caption_channels'` | builds it, `[1, 4096]` |

So the two files each contradict one half of the retired claim, and neither
supports it. Two corrections to the reasoning that came with the finding, both
measured rather than read:

- The FP8 file does not "resolve the flag `False` at `model_configurator.py:82`".
  Upstream never reaches line 82 on it: `_build_caption_projections` indexes
  `caption_channels` on the empty config first and raises. That file ships no
  config, so what its flag resolves to is decided entirely out of band — and the
  tensor it carries is trained (`.agents/specs/ltx-2-5.md` §3.1 reads its bytes).
- On the NVFP4 file the flag IS on and the module IS built, but the tensor is
  absent, so it keeps `torch.zeros(1, inner_dim)` (`model.py:217-219`) through
  `load_state_dict(..., strict=False)`
  (`loader/single_gpu_model_builder.py:98`) — a genuine no-op there.

It is also not a keyframe-only feature: `transformer_args.py:269` applies it on
every `prepare` whose `keyframes_mask` is set, and `tools.py:186-196` sets that
mask unconditionally on the target's first latent frame. (Diffusers' own pipeline
does not consume it — `.agents/specs/ltx-2-5.md` §3.1 records that — but `ltx_core`
is what this campaign ports, and `ltx_core` does.)

**The refusal keying does NOT change, and that is the decision, not an omission.**
`ltx2_loader.cpp`'s `RefuseUnported` fires on the TENSORS the file carries. Keying
it on the resolved flag instead would, on the FP8 DiT, read a DEFAULT rather than
the file — because that file declares nothing — and would therefore load it
silently while discarding a trained `[1, 4096]` parameter. Tensor presence is the
only signal that file actually carries, and refusing loudly with an opt-in is
strictly safer than resolving quietly. No behaviour changed, so no new gate is
owed; the refusal MESSAGE changed, because it asserted the implication that is
false in both directions.

### The claims repair's own gate (2026-08-13)

Nothing executable changed except three refusal MESSAGES, so the numbers are
expected to be identical and the point is that they are:

- `BUILD_EXIT=0`; build logs grepped for `No space left|BFD assertion` — 0 hits;
  `df -h /` 92% used, 37G free at the end.
- `ctest -N` = **423**. Full `ctest -j8` = 422/423 with `test_serve_low_tools`
  starved under `-j` (a known parallel flake); serially **1/1 PASS, exit 0**.
  2 skipped (`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`) as
  on the baseline.
- Suite counts, unchanged from the row above: `test_ltx2` 35/2435,
  `test_ltx2_loader` 26/4826, `test_ltx2_device` 15/523, `test_ltx2_video`
  30/502 skipped-default and **30/8734** with `LTX2_CHECKPOINT_ROOT` set. Exit 0
  on all five runs.
- `tests/vllm/models/ltx2_goldens.inc` REGENERATED from
  `scripts/gen-ltx2-goldens.py` against `ltx_core` `fd4ded7f`: every golden VALUE
  byte-identical, the diff is the comment block alone. That re-proves provenance
  as well as the wording.

An earlier full run was voided rather than reported: another session's
disk-pressure cleanup deleted `build/` while ctest was at 421/423, and the last
two tests recorded `Not Run — Failed to change working directory`. A run whose
tree vanished under it is not a result; it was rebuilt and re-run from scratch.

### Two divergences from upstream, recorded rather than fixed

- **We are stricter than upstream about a config that disagrees with its file.**
  Upstream loads with `load_state_dict(..., strict=False)`
  (`loader/single_gpu_model_builder.py:98`), so a config declaring
  `use_prompt_adaln_single=false` over a file that carries the module would build
  no module, drop 18 tensors on the floor and run flag-OFF without a word. §3.2's
  equality check refuses that. Ours is better; it is still a DIVERGENCE, not a
  mirror, and it is named here so it is not later mistaken for ported behaviour.
- **We refuse `keyframes_abs_pos_embedding` by tensor presence** where upstream
  would take an out-of-band config's word for it (above). Same shape of
  divergence, same reason it stands.

## Now

`DONE` — landed on `row/LTX25-PROMPT-ADALN`.
