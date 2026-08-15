# LTX-2.5 IC-LoRA — the adapter this port could not read, and the second blocker the refusal did not name

**Row:** `LTX25-IC-LORA` (a row of the `#644` full-port campaign).
**Issue:** [#923](https://github.com/mudler/vllm.cpp/issues/923).
**Branch:** `row/LTX25-IC-LORA`.
**Parent spec:** [`ltx-2-5.md`](ltx-2-5.md) — operator-owned, NOT edited by this row.
**Upstream root (primary):** Lightricks/LTX-2 @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
verified at the working checkout `/home/mudler/_git/LTX-2` with a clean tree.
**Entry point ported from:** `packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py`.

vLLM is not the reference here and cannot be: vLLM-Omni carries no LTX-2.5, and
`ICLoraPipeline` has no vLLM analogue at all. The oracle is the `ltx-2` pin
recorded in [`../oracles/`](../oracles/), used as the secondary oracle the rule
allows where vLLM implements nothing.

---

## 0. What is claimed, and what is not

This row builds **the LoRA adapter path**: reading an IC-LoRA safetensors file,
reading its metadata, and fusing its delta into the DiT at load, on every dtype
arm the loader materializes. That is the thing the reference refusal names as
absent, and it is the shared prerequisite for `ic_lora.py`, `hdr_ic_lora.py` and
`dubit.py` alike.

It does **not** ship video-to-video. Three statements up front so nothing is
discovered later.

1. **The reference-video arm has a SECOND blocker, and the brief for this row did
   not account for it.** The refusal at `ltx2_video.cpp:1336-1346` names one
   cause — the LoRA metadata is unread — and that cause is true and is closed
   here. But `ltx2_video.cpp:1305-1335`, thirty lines earlier, documents the
   other one: every *appending* conditioning item is blocked on token-append
   machinery the engine does not have. `VideoConditionByReferenceLatent` appends
   (`reference_video_cond.py:97-100`, ported at `ltx2_conditioning.cpp:265` via
   `AppendTokens`), so the reference arm needs it too. §6 sizes it and §7 records
   it as owed. Closing only the metadata half and lifting the refusal would ship
   a wrong render.
2. **No render-quality claim, and no real-weights claim.** The evidence here is
   numeric parity against upstream's fusion arithmetic plus mutation gates on
   synthetic fixtures. No IC-LoRA checkpoint was fused on real weights, because
   this row ran without GPU authority (`dgx.casa` was under a long render).
3. **No speed claim.** The parent spec's `PENDING` speed axis is untouched.

## 1. The gap, re-verified against the current tree

Re-derived at `95b7366`, not read out of the record.

| claim | check | result |
|---|---|---|
| the reference conditioning MATH landed | `Ltx2ConditionVideoByReference`, `src/vllm/model_executor/models/ltx2_conditioning.cpp:221` | present, gated by `test_ltx2_vae` |
| the video VAE ENCODER landed and is reachable | `Ltx2ConvVideoEncode`, called at `src/vllm/multimodal/ltx2_video.cpp:1645` | present, reached |
| `ref_video_dir` reaches the engine | `src/capi/vllm_c.cpp:1642`, `include/vllm/multimodal/video_engine.h:96` | present; a DIR of `frame_%06d.ppm`, so no codec is involved |
| arbitrary safetensors `__metadata__` is readable | `SafetensorsFile::Metadata()`, `include/vllm/model_executor/model_loader/safetensors_reader.h:55-57` | present, returns the whole map |
| anything reads a LoRA file | `grep -rn "lora_A\|lora_B" src include` | **only `include/vllm/lora/` — see below** |
| anything fuses a LoRA into LTX weights | `grep -rn kLoraFusion src include` | **only the refusal, `ltx2_pipeline.cpp:1178`** |

**The tree's existing `include/vllm/lora/` is not this mechanism and this row does
not route through it.** It is vLLM's runtime *punica* subsystem — f32,
slot-indexed, per-token, hanging off `LinearMethodBase` for text decode
(`punica.h:15-24`, `layers.h:26-28`). LTX does not apply an adapter at runtime;
it **fuses at load** (`loader/fuse_loras.py:119-150`). Mirroring vLLM's LoRA here
would mirror the wrong upstream. Recorded as a deliberate divergence from the
shared-seam preference, with the reason: the seam cannot express the behaviour,
which is exactly the case AGENTS.md allows extending or bypassing for.

## 2. Upstream anchors

Every anchor below was re-derived at the final tree and asserted unique; §5.3
records the method and the result.

### 2.1 The pipeline

| what | anchor |
|---|---|
| `ICLoraPipeline.__init__` | `ic_lora.py:71-173` |
| stage 1 takes the LoRAs, **stage 2 takes none** | `ic_lora.py:104-125` (`loras=tuple(loras)` at `:108` against `loras=()` at `:119`) |
| reference scale factors read from LoRA metadata | `ic_lora.py:150-173` |
| conflicting factors across LoRAs raise | `ic_lora.py:158-163` and `:167-172` |
| `__call__` | `ic_lora.py:175-349` |
| `conditioning_attention_strength` range check | `ic_lora.py:230-233` |
| reference appended LAST, after the image conditionings | `ic_lora.py:377-402` |
| the CLI's `--lora` | `utils/args.py:600-611` |

### 2.2 The adapter format and its fusion

| what | anchor |
|---|---|
| `LoraPathStrengthAndSDOps(path, strength, sd_ops)` | `loader/primitives.py:160-167` |
| key shape `<prefix>.lora_A.weight` / `.lora_B.weight` → `<prefix>.weight` | `loader/fuse_loras.py:183-186` and `:196-198` |
| the delta: `sum((B * strength) @ A)` | `loader/fuse_loras.py:99-116` |
| **aggregation dtype is bfloat16** | `loader/fuse_loras.py:71` (`bf16_fuse_rule`) |
| the bf16 fuse: `deltas.add_(weight)` then cast to the weight dtype | `loader/fuse_loras.py:61-68` |
| a LoRA naming a key the model lacks is SKIPPED, not an error | `loader/fuse_loras.py:135-137` |
| metadata `reference_downscale_factor`, default 1 | `iclora_utils.py:30-38` |
| metadata `reference_temporal_scale_factor`, default 1 | `iclora_utils.py:41-49` |

### 2.3 The quantized arms

Upstream carries a fuse rule per policy, and **all four aggregate in bfloat16**:

| arm | rule | anchor |
|---|---|---|
| bf16 | add, cast back | `loader/fuse_loras.py:61-71` |
| fp8 scaled-mm | dequant by `weight_scale`, add in f32, **re-quantize** with a fresh scale | `quantization/fp8_scaled_mm.py:167-189` |
| fp8 cast | fused add-round (stochastic on CUDA+Triton) | `quantization/fp8_cast.py:204-239` |
| NVFP4 | dequant to bf16, add, **re-quantize**, emit weight + both scales | `quantization/nvfp4/fuse.py:13-50` |

## 3. Design

### 3.1 The insertion point, and why every arm is served by one of them

Both quantized arms in this tree **dequantize to bf16 inside one function** before
anything else sees a byte. `MaterializeDitTensor`
(`src/vllm/model_executor/models/ltx2_loader.cpp:424-499`) has four branches —
F32 memcpy, BF16 memcpy, `DequantFp8ToBf16` at `:461-469`, and
`Ltx2DequantNvfp4ToBf16` at `:470-498` — and the last two both `return
vt::DType::kBF16`. The header states the policy outright: "The default
materialization is **bf16**, which is the checkpoint's own model dtype"
(`ltx2_loader.h:76-81`).

So the fusion hook goes **immediately after `MaterializeDitTensor` returns**, and
one hook covers FP8, NVFP4, BF16 and F32. It is reached from both callers:
`Ltx2LoadDitFromSafetensors` (`ltx2_loader.cpp:614-621`) and
`Ltx2StreamDitToDevice` (`:667-684`). On the device arm it runs *before*
`backend.Copy`, which preserves that arm's stated invariant that one host buffer
is live at a time (`:668-670`).

**This is a deliberate divergence from upstream's per-arm rules, and it is forced
by an existing design rather than chosen here.** Upstream re-quantizes because it
keeps FP8/NVFP4 weights resident for its quantized kernels; this tree does not
keep them, so there is nothing to re-quantize into. The consequence is recorded
rather than hidden: our fused weight **skips upstream's lossy quantize round
trip** and is therefore slightly more precise than upstream's on the FP8 and
NVFP4 arms. It costs no extra bytes — the weight was already bf16 — so this is
not the too-wide-dtype failure that rule guards against. The tree has no FP8 or
NVFP4 *quantizer* at all (`grep` for one returns nothing; only `DequantFp8ToBf16`
and `Ltx2DequantNvfp4ToBf16` exist), so mirroring the round trip is not available
to be chosen.

### 3.2 The memory format

Mirrored deliberately, per `.agents/porting.md` §"Mirror the memory format".

| ask | upstream answer | here |
|---|---|---|
| what dtype does the delta accumulate in? | **bfloat16** — every one of the four `FuseRule`s sets `aggregation_dtype=torch.bfloat16` | bf16 |
| is the first product rounded differently from the rest? | yes: `matmul(B * strength, A).to(dtype)` first, `addmm_(B, A, alpha=strength)` after (`fuse_loras.py:110-116`) | mirrored |
| what dtype is the fused weight stored as? | the weight's own dtype | the materialized dtype, unchanged |

An f32 accumulator would be the exact defect that rule exists for: the tokens
would still match and every golden would still pass. It is bf16 here on purpose,
and §5.2 mutates that choice to prove the gate can see it.

### 3.3 Surface

- Load extras `lora_path` and `lora_strength` on the `ltx-2.5` family, mirroring
  upstream's `(path, strength)` pair. LoRAs are a **constructor** argument
  upstream (`ic_lora.py:104-114`), not a `__call__` argument, so a load extra is
  the faithful shape and a per-generation field would not be.
- `--lora PATH [STRENGTH]` on `ltx2-gen`, mirroring `utils/args.py:600-611` and
  its `DEFAULT_LORA_STRENGTH`.
- No ABI change. Both extras ride the existing `extra_keys`/`extra_values`
  parallel arrays, which `include/vllm.h:926-927` records as existing for exactly
  this.
- **Exactly one LoRA**, with a second refused by name. Upstream's own `dubit.py`
  enforces the same (`dubit.py:364-365`) and `hdr_ic_lora.py` takes exactly one
  (`hdr_ic_lora.py:271-272`). N-LoRA fusion is recorded as owed in §7 rather than
  half-built; the conflict-detection loop this row ports (`ic_lora.py:155-173`)
  is what N-LoRA needs and is written to take a list already.

### 3.4 `kLoraFusion` stops being a marker

`Ltx2UnportedPipelineFeature::kLoraFusion` is classified as a
declared-out-of-scope marker and its message asserts `DECLARED, NOT REQUESTABLE`
(`ltx2_pipeline.cpp:1178-1181`). Once `lora_path` exists that sentence is false.
Issue [#691](https://github.com/mudler/vllm.cpp/issues/691) records that the
ledger test gates the message **text** and not the property, and predicts this
exact divergence in its own words. So the enumerator is **removed**, not
reclassified: there is no longer an unported LoRA-fusion feature to name, and a
refusal for a served capability is worse than no refusal.

## 4. Risks

| risk | mitigation |
|---|---|
| the fused delta lands on the wrong tensor because LoRA keys carry a ComfyUI prefix the contract strips | normalize onto `Ltx2TensorSpec::name` explicitly and **refuse by name** when a LoRA names a module the contract lacks after normalization, rather than silently skipping — see §4.1 for where this deliberately departs from upstream |
| a LoRA that touches nothing at all fuses green and renders identically | the adapter load refuses when **zero** contract tensors matched; a no-op LoRA is a user error, not a successful load |
| editing `ltx2_video.cpp` above line 1100 shifts the gated READER ANCHORS | the new readers go below the existing ones where possible; the anchors are re-derived at the final tree and the gate is run |
| `docs/FEATURES.md` is a keyed record other agents are editing concurrently | reapply by key and diff unrelated keys byte-for-byte before commit |

### 4.1 One deliberate departure

Upstream **skips** a LoRA key whose target weight is absent
(`fuse_loras.py:135-137`, `if original_weight is None: continue`). It can afford
that: its state dict is the whole model. Here the contract is a fixed enumerated
set with unported modules already stripped, so a skip would silently absorb both
a genuinely-inapplicable key *and* a misnamed one, and the second is the failure
this project keeps paying for. The load therefore refuses by name on an unmatched
key. This is a divergence, it is argued here and in the commit, and it is the
narrower behaviour.

## 5. Tests and gates

### 5.1 Ported from upstream — there is nothing to port, and that is measured

The rule is to port the upstream tests in the same change. **The pinned upstream
ships none, anywhere in the repository**, so the obligation is discharged as
not-applicable rather than skipped. Measured at
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, with a positive control so a null
result cannot be a wrong search term:

| probe | result |
|---|---|
| `find . -name 'test_*.py' -o -name '*_test.py'` | **0** |
| `find . -type d -name 'test*'` | **0** |
| `find . -name conftest.py` | **0** |
| `grep -rl "import pytest\|import unittest" --include='*.py' .` | **0** |
| *control:* `find . -name '*.py'` | **280** |

The control is what makes the zeros admissible: the same traversal that found no
tests found 280 Python files. So the fusion arithmetic is gated against
upstream's expression read from source (`fuse_loras.py:99-116`) and executed by
hand, which is stated as what it is — a source-derived value gate, not a ported
test.

### 5.2 Written here, each RED first

| case | proves |
|---|---|
| the fused weight equals `W + (B*strength) @ A` | the arithmetic, against values computed from the upstream expression |
| the accumulator is bf16, not f32 | a rank-2 adapter whose f32 and bf16 accumulations differ in the stored bf16 result; **this is the memory-format gate** |
| `strength` scales the delta linearly | strength is read, not ignored |
| a LoRA naming an absent module refuses by name | §4.1 |
| a LoRA matching zero tensors refuses by name | §4 |
| the metadata factors are read, absent ⇒ 1 | `iclora_utils.py:30-49` |
| a second `lora_path` refuses by name | §3.3 |
| **the FP8 arm fuses** — an FP8 fixture plus a LoRA changes the materialized weight | that the hook is after dequant, on the arm most users run |
| **the NVFP4 arm fuses** | the same for NVFP4 |

### 5.3 Reachability

The production entry point is `vllm_video_engine_load` → `LoadVideoEngine` →
`Ltx2VideoEngine::Load` → `Ltx2LoadDitFromSafetensors`. The smallest failing test
enters **there**, through `LoadVideoEngine` with a `lora_path` load extra on the
engine fixture, and asserts the rendered conditioning digest moves. The
reachability mutation deletes the fusion call site in `ltx2_loader.cpp` and
re-runs that case; a green gate would be the finding.

### 5.4 Gate

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6
ctest --test-dir build -j4 --output-on-failure
```

Known-red on `main` at the base SHA `95b7366`, proven pre-existing by running
`scripts/agent-preflight.sh` on a byte-for-byte unmodified tree
(`git status --porcelain` empty) before any edit: the seven-checker `#873` family
— `check-release-binary-contract`, `check-release-workflow`,
`check-test-registration`, `test_check_release_binary_contract`,
`test_release_manifest`, `test_release_pipeline`, `test_check_test_registration`.

## 6. The second blocker, sized

Recorded here so the next agent does not re-derive it.

Serving a reference video needs the engine's phase loop to carry a **grown** token
sequence and trim it back. What is and is not in the way:

- **The DiT is not in the way.** `Ltx2ModalityInput::tokens` is a per-call field,
  and the DiT already accepts a self-attention strength mask
  (`ltx2.h:458-462`, implemented `ltx2_dit.cpp:588-593`, mirroring
  `_prepare_self_attention_mask`, `transformer_args.py:208-237`).
- **The conditioning item is not in the way.** It is ported and gated.
- **`Ltx2LatentState` having no attention-mask field is not in the way for the
  DEFAULT arm.** At `conditioning_attention_strength == 1.0` with no mask,
  upstream computes `attn_mask = None` (`iclora_utils.py:157-160`) and
  `update_attention_mask` returns `None` (`mask_utils.py:141-143`), so the
  default arm needs no mask at all.
- **The phase loop is in the way.** One `Ltx2VideoTokenCount(vshape, 1)` feeds
  the sigma schedule, the `Ltx2ModalityInput` and `Ltx2VideoUnpatchify`
  (`ltx2_video.cpp:1305-1322` documents this for the keyframe arm, and it is the
  same obstruction). It must instead carry `state.tokens` through denoise and
  trim to the target count before unpatchify, mirroring `clear_conditioning`
  (`ltx_core/tools.py:88-105`).

That work is **shared with the last-frame keyframe arm**, which is blocked on the
identical machinery. It is therefore its own row rather than a tail of this one.

## 7. Owed

Each is owed by this row and named in the commit and pull request bodies.

| owed | issue |
|---|---|
| token-append grow-and-trim in the phase loop; until it lands, the reference-video and reference-image arms stay refused, with the refusal rewritten to name this cause instead of the metadata one this row closed | filed as part of this row's delivery, see the pull request body |
| the `conditioning_attention_mask` / `conditioning_attention_strength < 1.0` arm, which needs `Ltx2LatentState` to carry a mask and `build_attention_mask`'s block structure (`mask_utils.py:170-243`) | same |
| N-LoRA fusion (more than one adapter) | same |
| GGUF k-quant LoRA fusion — not applicable rather than owed: the LTX-2.5 DiT ships FP8 and NVFP4, and no GGUF LTX DiT exists to fuse into | n/a |
| a real-weights IC-LoRA fusion measurement | blocked on GPU authority; this row had none |

## 8. Stop conditions

- Stop and report `NEEDS_DECISION` if closing the metadata half would require
  lifting the reference refusal before the token-append machinery exists. **This
  fired**; §0.1 and §6 are the result, and the refusal is rewritten rather than
  lifted.
- Stop if `docs/FEATURES.md` cannot be reapplied by key with unrelated keys
  byte-identical.
- Do not use the GPU. `dgx.casa` was under a long render for this row's duration.

## Now

`ACTIVE` — the adapter path is implemented and gated; the reference arm stays
refused on the cause named in §6.
