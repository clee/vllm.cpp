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

Recorded in §Outcome: the relative change in the modulated prompt context and in
the DiT's outputs, flag ON vs OFF, at the reduced dimensions. This is the answer
to "does this matter"; a number below round-off would mean the term is inert and
the whole row is decoration.

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

## Now

`ACTIVE` — implementation on `row/LTX25-PROMPT-ADALN`.
