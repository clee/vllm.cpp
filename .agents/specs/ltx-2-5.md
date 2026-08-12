# LTX-2.5 — 21B joint video+audio flow-matching DiT, and the generalized video seam

**Rows:** `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model` (model-matrix),
`ROAD-V1-LTX25` (roadmap portfolio).
**Issue:** [#435](https://github.com/mudler/vllm.cpp/issues/435).
**Branch:** `row/MODEL-DIFFUSION-LTX25` (ONE PR for the whole campaign — developer-directed
2026-08-11; AGENTS.md retired per-class line budgets, so size is a review judgement).
**Upstream (architecture):** Lightricks `LTX-2` — `packages/ltx-core/src/ltx_core/`.
**Upstream (serving oracle):** vLLM-Omni `vllm_omni/diffusion/` — see §3, it does NOT yet
carry 2.5.
**Checkpoints:** `Lightricks/LTX-2.5` (gated: auto), `vonkaiser/LTX-2.5-FP8-NVFP4` (ungated).
**Status:** L0 — spec committed. Implementation phases L1–L7 below.

---

## 0. Honesty statement — what is and is not claimed

LTX-2.5 is **not an autoregressive LLM**. It is a joint video+audio **diffusion
transformer**: one request runs a flow-matching denoise loop in which a 21.00B DiT is
forwarded once per step over two coupled modality streams, and the resulting latents are
decoded to frames plus a waveform by two VAEs. There is no KV cache in the LLM sense, no
sampler, no logits, and **no token-exact gate** — the SACRED near-tie methodology this
project uses for decoders does not apply, exactly as recorded for MiniMax-H3
([minimax-h3](minimax-h3.md) §0).

Three things are recorded as **owed** here, before any work starts, so they cannot be
discovered later:

1. **The speed gate lands `PENDING`.** vLLM-Omni's only route to 2.5 is its
   `DiffusersAdapterPipeline`, which is a black box — `supports_step_execution = False`,
   `supports_request_batch = False`
   (`vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py:68-69`).
   A throughput number taken through it is **not vLLM's production configuration**, which
   AGENTS.md §Gates requires as the denominator. Correctness is gateable through it;
   throughput is not. The axis stays open with a named next step (§9).
2. **DiffVAE is refused, never silently downgraded.** The higher-quality video decoder is
   `NADiffusionDecoder`, built on neighborhood attention. Until its row lands, asking for it
   fails with a message naming the missing piece; it does not quietly fall back to the Conv
   VAE and return a worse render as if it were the requested one.
3. **No render-quality claim.** Structural e2e (correct shapes, finite values, valid MP4)
   is not a quality result. H3 taught this directly: its fp4-resident e2e *ran* and produced
   a valid mp4 while the frames were a non-scene patch grid.

## 1. Architecture — measured, not inferred

Read by HTTP range request from the ungated
`vonkaiser/LTX-2.5-FP8-NVFP4` → `transformer/ltx-2.5-22b-distilled-fp8.safetensors`
(6124 tensors, 881,048-byte header; no payload downloaded — the same technique used for
H3's manifests).

| Field | Value |
|---|---|
| Parameters | **21.00B** — blocks 18.560B + audio connector 2.016B + global 0.427B |
| Blocks | **48**, 386.7M each |
| Video stream | hidden **4096**, 32 heads x 128 |
| Audio stream | hidden **2048**, 32 heads x 64 |
| `in_channels` / `out_channels` | 128 video (`patchify_proj` [4096,128], `proj_out` [128,4096]); 128 audio |
| Video FFN | `ff.net.0.proj` [16384, 4096] → `ff.net.2` [4096, 16384], **NO bias** |
| Audio FFN | `audio_ff.net.0.proj` [8192, 2048] → [2048, 8192], **WITH bias** |
| Activation | `gelu-approximate` (`model_configurator.py:31`) |
| Norms | `standardization_norm=rms_norm`, `qk_norm=rms_norm`, `norm_elementwise_affine=False` |
| Audio connector | 8 x 1-D transformer blocks + `learnable_registers` [128, 2048] |
| Quant (FP8 arm) | F8_E4M3 + **per-tensor F32 `weight_scale`**; biases/norms BF16; 1775 FP8 tensors |

The filename says `22b`; the measured count is **21.00B** and the Diffusers card says ~19B.
The measured number is the one this spec uses.

**The `ff_bias` cross-check.** `ff` carries no bias while `audio_ff` does. That is exactly
`ff_bias=false` / `audio_ff_bias=true`, whose defaults
(`model_configurator.py:78-80`) are documented as *"Default True keeps backwards
compatibility: pre-2.5 checkpoints lack these keys and retain FFN biases. LTX 2.5 (gemma4)
sets ff_bias=false."* Checkpoint and source agree — that agreement, not either alone, is
what AGENTS.md §"Verify against both the running oracle and its source" asks for.

### 1.1 The block — `BasicAVTransformerBlock` (`transformer.py:87`)

Per block, two coupled streams:

| Tensor | Shape | Role |
|---|---|---|
| `scale_shift_table` | [9, 4096] | 3 groups of 3: `slice(0,3)` self-attn pre-mod, `slice(3,6)` FFN pre-mod, `slice(6,9)` cross-attn `shift_q, scale_q, gate` (`transformer.py:240,273,401`) |
| `audio_scale_shift_table` | [9, 2048] | same, audio (`:302,320,410`) |
| `prompt_scale_shift_table` | [2, 4096] | **prompt K/V modulation — see §1.2** |
| `audio_prompt_scale_shift_table` | [2, 2048] | same, audio |
| `scale_shift_table_a2v_ca_video` | [5, 4096] | audio↔video cross-attn modulation (`:337,378`) |
| `scale_shift_table_a2v_ca_audio` | [5, 2048] | same (`:347,369`) |

Attentions per block: `attn1` (video self), `attn2` (video↔text, cross_dim 4096),
`audio_attn1`, `audio_attn2` (cross_dim 2048), plus the two cross-modal
`audio_to_video_attn` (`transformer.py:154`) and `video_to_audio_attn`
(`transformer.py:166`).

**Per-head gated attention** is on for every one of them: `to_gate_logits` is
`torch.nn.Linear(query_dim, heads, bias=True)` (`attention.py:513-514`) — hence the
`[32, dim]` weights in the checkpoint, one logit per head — applied *after* the attention
output as `out = self.gated_attention_function(x, out, self)` (`attention.py:577`). H3 has
no analogue; getting this wrong yields a plausible-but-wrong render rather than an error.

### 1.2 The free win — prompt K/V is timestep-independent

`get_ada_values` modulates from `scale_shift_table` **using `timestep`**
(`transformer.py:192-197`). The prompt path does not: at `transformer.py:441`,

```python
kv_modulation = prompt_scale_shift_table[None, None].to(device=..., dtype=...)
```

— no timestep term at all. `model_configurator.py:74-76` states the consequence directly:
*"KV-cacheable checkpoints set `use_prompt_adaln_single=false`, dropping the
timestep-dependence of the cross-attention K/V so they can be computed once per prompt and
reused across steps."* The 2.5 checkpoint carries only the static
`prompt_scale_shift_table [2, dim]` and no prompt-side timestep MLP, so **the cross-attention
K/V for all 48 blocks is computed once per request and reused for every denoise step.**

This is a property of the checkpoint, not an optimization we invent, so it ships as the
default path (AGENTS.md: parity enablers ship as defaults). L2 gates it by asserting the
cached and recomputed paths are **bit-identical**.

**Correction, 2026-08-12 — the earlier claim here was too strong.** This section previously
said the bit-identity gate meant the cache "cannot silently diverge". L2's fresh review
disproved that, and the distinction is the whole point of the feature:

- Against a changed **timestep**, the cache cannot diverge. That is the property the
  checkpoint gives us, and it is real.
- Against a changed **prompt**, it silently could. The gate ran the forward twice with
  IDENTICAL inputs, so it only ever proved "same in, same out"; the cache carried no prompt
  identity and its only validity check was on SIZE. A probe that swapped in a different
  prompt of equal token count found the cache did not notice.

The failure that implies is not academic: a pipeline or server reusing one cache across two
requests whose prompts differ but tokenize to the same length renders the **second request
with the first request's prompt**, with no error, no shape mismatch and no finiteness
failure. The repair carries a content fingerprint on the cache and refuses by name on
mismatch. Recorded rather than quietly amended, because "we gate that" was written here
before it was true.

### 1.3 How this differs from MiniMax-H3

| | MiniMax-H3 (ported) | LTX-2.5 |
|---|---|---|
| Modalities | ONE packed sequence, per-row token tags | **two streams + explicit audio↔video cross-attn** |
| FFN | SwiGLU 14336 | gelu-approximate 16384, no bias |
| Gated attention | none | **per-head, every attention** |
| Text encoder | Qwen3-VL-32B-derived | **Gemma-4 12B + projections** |
| Video decode | ViT3D | Conv VAE **or** DiffVAE (neighborhood attn) |
| Extras | — | latent spatial/temporal x2 upsamplers, duration head |

### 1.4 Text conditioning is a MULTI-LAYER aggregate, not the last hidden state

Recorded 2026-08-11 while briefing L3, from the real TE checkpoint
(`vonkaiser` `gemma4-12b-with-proj-nvfp4-torchao.safetensors`, 1688 tensors) read against
`text_encoders/gemma/feature_extractor.py`.

`feature_extractor.py` takes hidden states shaped `[batch, seq_len, hidden_dim, num_layers]`,
normalizes them, and concatenates **across the LAYER dimension** to
`[batch, seq_len, hidden_dim * num_layers]`. The checkpoint confirms it: the two projections
take **94080** input features, and 94080 = 1920 x 49 — the model's 1920-wide per-layer state
across its 48 layers plus one.

| Tensor | Shape |
|---|---|
| `text_embedding_projection.video_aggregate_embed.weight` / `.bias` | [4096, 94080] / [4096] |
| `text_embedding_projection.audio_aggregate_embed.weight` / `.bias` | [2048, 94080] / [2048] |

There are at least TWO normalization variants and the right one is selected from config, never
guessed: `_norm_and_concat_padded_batch` (per-batch, per-layer masked mean and range, an `8 *`
scale, `eps = 1e-6`) and `norm_and_concat_per_token_rms` (per-token RMS, "for V2 models"). Both
are padding-side agnostic and ZERO padded positions.

**Why this is a trap and not a detail:** getting the variant, the mask handling, the reduction
axes or the layer order wrong yields conditioning that is finite, correctly shaped and WRONG.
It renders a plausible video for the wrong prompt, which no shape or finiteness check catches.
L3 gates the variant selection explicitly.

Two further facts the loader must respect, both measured:

- **The tokenizer is embedded AS A TENSOR** — `tokenizer_json` U8 [32,169,626] (~32 MB), plus
  `hf_asset__{chat_template,generation_config,processor_config,tokenizer_config}`. A loader that
  assumes a sibling `tokenizer.json` file fails on this checkpoint.
- **The TE quantization is torchao NVFP4, NOT compressed-tensors.** `weight` U8 packed,
  `weight_scale` F8_E4M3 grouped, `weight_scale_2` F32 scalar, plus a `torchao_nvfp4` U8 [240]
  marker per quantized module. H3's NVFP4 arm is compressed-tensors, so the layouts must be
  verified before any reuse rather than assumed equal (L6).

The checkpoint also carries the FULL multimodal Gemma-4 (`vision_model.*`,
`multi_modal_projector`, `audio_projector`); text-only conditioning is the scope, but the loader
must not choke on their presence.

### 1.5 The audio VAE is NOT end-to-end causal

Recorded 2026-08-12 from L4, which measured it rather than assuming it: its first causality
probes FAILED, and upstream agreed with the failure.

`causality_axis` governs the audio decoder's **convolutions**, but its `AttnBlock`s attend over
the whole (time, mel) map, so a last-frame perturbation reaches every output frame. The Conv
video decoder is not end-to-end causal either, for a different reason: `res_x_y`'s shortcut norm
is a one-group GroupNorm over (C,T,H,W) whose statistics span time.

This matters because "causal" is exactly the kind of property a port assumes and never checks.
Both are now gated in two parts: the shipped config asserts the GLOBAL reach, and a stripped
config isolates the convolution-only reach, with upstream itself supplying the expected windows
([5,8] audio, [3,4] video).

## 2. Scope

**In:** the DiT forward (both streams, gated attention, AV cross-attention, split and
interleaved RoPE); the Gemma-4 12B text encoder with its two caption projections; the Conv
video VAE, the audio VAE and its vocoder; the flow-matching pipeline including the distilled
two-stage recipe and the latent spatial x2 upsampler; the duration head; the FP8 and NVFP4
arms; the generalized `VideoEngine` seam with H3 moved behind it; `/v1/videos`; e2e on
dgx.casa.

**Out (recorded as owed, not silently dropped):** DiffVAE / `NADiffusionDecoder` (own row —
new neighborhood-attention kernel); the temporal x2 upsampler; LoRA fusion; multishot;
`int8-convrot` (ComfyUI-only quantization); multi-GPU / CFG parallelism.

## 3. The oracle problem, and its resolution

vLLM-Omni does **not** support LTX-2.5. Its recipe table keys on
`("one_stage","2")`, `("one_stage","2.3")`, `("distilled_two_stage","2")`, `("dmd2","2")`,
`("dmd2","2.3")` (`vllm_omni/diffusion/models/ltx2/ltx2_recipes.py:162-166`), and
`resolve_ltx_pipeline_recipe` raises `ValueError` on anything else. Upstream issues
[vllm-omni#6066 "[New Model]: LTX-2.5"](https://github.com/vllm-project/vllm-omni/issues/6066)
(filed 2026-08-11) and [#4985 "Align and expand LTX support"](https://github.com/vllm-project/vllm-omni/issues/4985)
are open.

But its `DiffusersAdapterPipeline` is fully generic — it calls
`DiffusionPipeline.from_pretrained(model_id, **load_kwargs)`
(`pipeline_diffusers_adapter.py:116`) — so vLLM-Omni **can** execute 2.5 through
`--load-format diffusers` against `Lightricks/LTX-2.5-Diffusers` with diffusers installed
from main.

**Resolution (developer-directed 2026-08-11):**

- **Binding correctness oracle:** vLLM-Omni + diffusers adapter. Keeps AGENTS.md
  §"vLLM is the reference" intact.
- **Immediate cross-check:** Lightricks `ltx-pipelines`, the model author's own runtime.
  Available as soon as the `Lightricks/LTX-2.5` auto-gate is accepted, and it keeps every
  phase unblocked while `-Diffusers` access is pending.

Both are recorded per brick. Where they disagree, the disagreement is the finding.

**BINDING-ORACLE PARITY IS PENDING FOR EVERY BRICK LANDED SO FAR** (recorded 2026-08-12, from
L2's review). L1–L5 gate against the CROSS-CHECK (`ltx_core` executed at reduced dimensions),
not against the binding oracle, because `Lightricks/LTX-2.5-Diffusers` access is still
awaiting manual approval. That is legitimate under §3 and §6 and it is what "immediate
cross-check" is for — but it must be stated, not left implicit. Concretely:

| Axis | State |
|---|---|
| DiT / VAE / text-encoder parity vs `ltx_core` (cross-check) | gated, per-brick max abs diff recorded |
| DiT / VAE / text-encoder parity vs vLLM-Omni (BINDING oracle) | **PENDING** on `-Diffusers` access |
| Throughput vs vLLM's production configuration | **PENDING**, structurally, per §0 |

`docs/BENCHMARKS.md` records only the SPEED axis as pending, which understates it; the
correctness axis against the binding oracle is pending too. Neither is a failure, and neither
is a pass. §3's instruction to "record the vllm-omni SHA inline with every golden" is
therefore N/A so far rather than satisfied, and saying so is the point.

**There is still no vllm-omni parity PIN** — `.agents/upstream-sync.md` covers the vLLM repo
only. This spec inherits H3's open gap (model-matrix, H3 row: *"OPEN: there is no vllm-omni
parity PIN"*) and records the vllm-omni SHA used for every golden inline with that golden.

## 4. Checkpoint access and placement

Verified against the HF API on 2026-08-11 with the session token:

| Repo | `gated` | Consequence |
|---|---|---|
| `Lightricks/LTX-2.5` | `auto` | Accepting the license opens it. Holds the **first-party NVFP4 DiT**, 18.72 GB |
| `Lightricks/LTX-2.5-Diffusers` | restricted (manual) | Needed for the **binding oracle**; request submitted |
| `vonkaiser/LTX-2.5-FP8-NVFP4` | none | **Unblocks L1–L2 today**: FP8 DiT + NVFP4 Gemma-4 TE |

All artifacts land under `$CHECKPOINT_ROOT` = `/mnt/nas_share/checkpoints` (per `.env`), so
dgx.casa and the cluster nodes mount one copy rather than each pulling 30 GB.

**Best GB10 arm** (119 GiB unified):

| Component | Source | Size |
|---|---|---|
| DiT, NVFP4 | `Lightricks/LTX-2.5` `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18.72 GB |
| Gemma-4 12B TE, NVFP4 | `vonkaiser` `gemma4-12b-with-proj-nvfp4-torchao.safetensors` | 7.40 GB |
| Video VAE + audio VAE, bf16 | `Lightricks/LTX-2.5` | 1.83 GB |
| Latent spatial upsampler x2, bf16 | `Lightricks/LTX-2.5` | 1.00 GB |
| | | **~29 GB** |

Comfortably inside the pool, and materially smaller than H3's ~41 GB GGUF arm.

## 5. Design — the generalized seam

`include/vllm.h` already models this shape: `vllm_video_model_params` carries `dit_path`,
`encoder_path`, `video_vae_path`, `audio_vae_path` as separate artifacts (ABI v12, ROW 2).
It is only the *internals* that are H3-typed — `vllm::multimodal::MiniMaxH3VideoEngine` —
plus two H3-specific fields (`partition` = fl2va/ref2va, and H3's 50-step default).

**L1 introduces `vllm::multimodal::VideoEngine`**, an abstract seam with a
checkpoint-detected registry, and moves H3 behind it **unchanged**. Per AGENTS.md
§"Shared seams", a capability not reachable through the shared surface is not done, and new
models are additive files. ABI goes to **v18 by ADDING fields only** — v12 video callers keep
working byte-identically, which the existing `test_capi` v12 section already guards.

**Correction, 2026-08-11.** Earlier revisions of this section said "v13". That was wrong: it
read the VIDEO SLICE's own v12 label as if it were the ABI counter, when `VLLM_ABI_VERSION`
was already **17** and v13 shipped long ago as `vllm_complete_tokens`. The additive
requirement was always the real one and is unchanged; only the number moves, 17 -> 18. L1
found this while implementing, which is the delegation loop working as intended.

Reuse is the point. Already ours and shared, not re-implemented: the flow-matching denoise
loop, AdaLN block plumbing, 3D RoPE construction, VAE CNN infrastructure
(`minimax_h3_vae_cnn.cpp`), WAV writing, PPM frame serialization, the ffmpeg mux argv
composer, the NVFP4 resident-weight path and Marlin W4A16 dispatch, and — for the text
tower — `gemma4.cpp` / `gemma4_weights.cpp`.

Genuinely new: dual-stream AV cross-attention, per-head gated attention, gelu-approximate
FFN, the audio embeddings connector, the two-stage distilled recipe with its latent
upsampler, and the duration head.

## 6. Phases

All on `row/MODEL-DIFFUSION-LTX25`, one PR.

| Phase | Scope | Gate |
|---|---|---|
| **L0** | This spec; issue #435; checkpoint inventory; oracle stand-up | spec committed |
| **L1** | `VideoEngine` interface + registry; H3 behind it unchanged; ABI v18 additive | H3 frames+WAV **byte-identical** to pre-refactor on the committed fold fixture; v12 `test_capi` green |
| **L2** | DiT layout + forward: dual stream, gated attn, AV cross-attn, split/interleaved RoPE, prompt-KV cache | reduced-dim CPU parity vs upstream modules; cached vs recomputed prompt K/V **bit-identical** |
| **L3** | Gemma-4 12B TE + caption projections (4096 video / 2048 audio) | parity vs upstream TE; reuses `gemma4.cpp` |
| **L4** | Conv video VAE + audio VAE + vocoder | per-brick parity vs upstream decoders |
| **L5** | Pipeline: sigma schedule, distilled two-stage, latent spatial x2 upsampler, duration head | recipe values EXACT vs upstream |
| **L6** | NVFP4 DiT + NVFP4 TE arms; GB10 load-time residency | quantized vs bf16 wiring gate; residency per the ATS finding |
| **L7** | e2e on dgx.casa under `flock`; `/v1/videos` route | valid MP4+WAV; speed axis recorded `PENDING` per §0 |

## 7. Tests

Mirroring H3's method, which is what made its bricks trustworthy: upstream's modules are
pure Python, so they are **executed at reduced dimensions on CPU** as the oracle, with both
sides rebuilding weights and inputs from an identical deterministic stream so **no weight
byte is checked in**. A generator script freezes upstream outputs; the C++ test asserts
against them.

Specific traps this port must gate, each of which produces a *plausible but wrong* result
rather than an error:

- **Per-head gate application order** — gating before vs after `to_out` differ silently.
- **`ff` bias presence** — reading a bias that is not there, or skipping one that is.
- **Cross-modal projection asymmetry** — `audio_to_video_attn.to_q` is `[2048, 4096]` while
  `to_k`/`to_v` are `[2048, 2048]` and `to_out` is `[4096, 2048]`. Transposing any of these
  still type-checks against a square assumption.
- **AdaLN slice mapping** — `slice(0,3)` / `slice(3,6)` / `slice(6,9)` are not interchangeable.
- **Prompt-KV caching** — asserted bit-identical against recomputation.
- **RoPE `split` vs `interleaved`** (`LTXRopeType`) and the optional float64 frequency path.
- **F32 `scale`-guard on tolerances** — per this project's doctest finding, `Approx` needs
  `.scale(0.0)` or a 1.19e-5 absolute floor silently accepts anything.

## 8. Risks

| Risk | Mitigation |
|---|---|
| `-Diffusers` manual access never granted | `ltx-pipelines` cross-check keeps every phase unblocked; binding oracle recorded as pending, not faked |
| Upstream lands 2.5 in vllm-omni mid-campaign | Good outcome — re-anchor goldens to the native path and record the SHA; the diffusers-adapter goldens stay as the earlier evidence |
| Gemma-4 TE differs from our ported Gemma-4 | L3 gates the TE against upstream independently before wiring; a delta is a finding, not an adaptation |
| GB10 unified-memory OOM reboots the box | Never run a large oracle alongside ctest; `flock $HOME/gpu.lock`; park `local-ai-worker` |
| Contention with the 3 other coordinators | `flock` on every GPU-executing step; named tmux; single-load steady state |

## 9. Stop conditions

- A brick whose upstream reference cannot be executed → record `NEEDS_CONTEXT`, do not guess
  the semantics from shapes.
- A parity delta that is not round-off → stop and report; never widen a tolerance to pass.
- Any temptation to declare a performance ceiling → forbidden by AGENTS.md; keep the gap open
  and name the next traceable hypothesis.
- Speed axis: stays open until vllm-omni carries native 2.5 (tracked upstream at #6066) or
  another production-configuration denominator is ratified.

## Now

L0 committed. Next: L1 — introduce `vllm::multimodal::VideoEngine` and move MiniMax-H3
behind it unchanged, gated on frames+WAV being byte-identical to the pre-refactor fold
fixture.
