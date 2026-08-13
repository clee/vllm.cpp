# LTX-2.5 — retire the arms that do not exist, and refuse the extra that is ignored

Row: `LTX25-RETIRE-DEAD-ARMS`. Campaign: [`ltx-2-5.md`](ltx-2-5.md) (operator-owned;
not edited by this row). Issues:
[#644](https://github.com/mudler/vllm.cpp/issues/644) items D–I and N,
[#611](https://github.com/mudler/vllm.cpp/issues/611).
Pattern this row is an instance of: [#604](https://github.com/mudler/vllm.cpp/issues/604).

Upstream pins:

| Reference | Revision | Local checkout verified at |
|---|---|---|
| Lightricks/LTX-2 | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` | `git rev-parse HEAD`, 2026-08-13 |
| huggingface/diffusers | `3a2f35d4efa4c059c8bfb3bc0d6c906264895c81` | `git rev-parse HEAD`, 2026-08-13 |

Every anchor below was re-derived from those two checkouts by this row. A prior
grounding pass reached the same conclusions; it is an input, not a result.

## 0. What is wrong today

`Ltx2UnportedPipelineFeature` (`include/vllm/model_executor/models/ltx2_pipeline.h:580-587`)
is a ledger of seven arms this port refuses by name. Read as a whole it claims
more than is true, in four separate ways:

1. **One arm is fabricated.** `kMultishot` refuses "multishot generation" and its
   enum comment cites *"ltx-pipelines multishot entry points"*. No such entry
   point exists. Neither does the symbol, the string, or the concept.
2. **One arm's anchor names something upstream does not do.** `kCfgParallelism`
   is anchored to `ltx-pipelines/multigpu`, which contains no CFG batching at
   all — the three real parallelisms are orthogonal to guidance, and the recipe
   this port runs uses no guidance in the first place.
3. **One arm is stale.** `kVideoEngineWiring` says the end-to-end wiring "is
   phase L7, not L5". L7 landed in `cefacd2d0`.
4. **Five of the seven have no product call site.** Recording them as "refused by
   name" overstates what exists: nothing a caller can send reaches them. They are
   declared-out-of-scope markers, and the code did not say so.

Separately, and user-visible: `duration_head_path` is accepted by
`kKnownLoadExtras` (`src/vllm/multimodal/ltx2_video.cpp:257`) and **read by
nothing**. A caller who points at a duration head gets the recipe default and is
told nothing.

## 1. What the references actually contain

### 1.1 `multishot` — FABRICATED

Searched as a **subject**, not as our own phrasing (the self-confirming-grep trap,
#604):

| Query | LTX-2 `fd4ded7f` | diffusers `3a2f35d4` |
|---|---|---|
| `multishot` / `multi_shot` / `multi-shot` (case-insensitive) | 0 hits | 0 hits |
| `\bshots?\b` in `*.py` `*.md` `*.json` `*.yaml` | 5 files, none a generation mode | — |
| `multi[_ -]?(gpu\|scale\|stage\|shot\|clip\|prompt\|segment)` | `multi-GPU` / `multi_gpu` only | — |
| `scene` / `storyboard` | `ltx-trainer/scripts/split_scenes.py` (PySceneDetect, a **training-data** preprocessor) | — |

The five `shot` files, with what the word means in each:

- `ltx-core/duration_head/duration_head.py:1,5` — "predicts **shot** duration",
  i.e. the natural length of ONE camera take.
- `ltx-core/duration_head/__init__.py:1` — same.
- `ltx-pipelines/utils/blocks.py:804` — `DurationPredictor`, "Predicts **shot**
  duration (in frames)".
- `ltx-trainer/src/ltx_trainer/captioning.py` — captioning prose.
- `LTX-2/README.md:59,136` — an example prompt ("a medium close-up **shot**") and
  the prompting guide ("think like a cinematographer describing a **shot** list").

Upstream's "shot" is a single continuous take. There is no multi-shot generation
mode, and nothing that composes several takes into one output.

diffusers says the same thing in its own vocabulary, which is the stronger check
because the word is far more common there: every `shot` in
`src/diffusers/pipelines/ltx2/` is a **camera shot type** in the prompt-enhancement
guidance — `utils.py:217`, "Shot type (exactly one: extreme wide shot / wide shot
/ medium shot / medium close-up / close-up / extreme close-up)" — or the duration
head's own docstring, `duration_head.py:83`, "Predicts the natural duration of the
**shot** implied by a caption". Its sixteen `ltx2/` modules are
`pipeline_ltx2`, `_condition`, `_diffusion_decode`, `_hdr_lora`, `_ic_lora`,
`_image2video`, `_latent_upsample`, plus the components. None is multi-shot.

`ltx-pipelines`' actual entry points, from `docs/pipelines.md` ("Full reference
for all 11 pipelines") and the module list:
`ti2vid_one_stage`, `ti2vid_two_stages`, `ti2vid_two_stages_hq`,
`ti2vid_two_stages_mgpu`, `ti2vid_two_stages_hq_mgpu`, `distilled`,
`distilled_mgpu`, `ic_lora`, `hdr_ic_lora`, `a2vid_two_stage`, `t2a_one_stage`,
`dubit`, `retake`, `dfr_pipeline`, `keyframe_interpolation`. None is multi-shot.

**Disposition: RETIRE.** This is a defect in our record, not a gap in our port.
There is nothing to owe, so recording it as owed is the error. The enumerator is
removed and the retirement recorded in the header, in this spec, and in the
commit message — which is where an exception's reason lives (AGENTS.md, "there is
no waiver registry").

### 1.2 `int8-convrot` — real absence, deliberately out of scope

The inference quantization kinds LTX-2 defines, exhaustively
(`ltx-pipelines/utils/quantization_factory.py:23-27`, a `str`-valued enum with an
`assert_never` on the match):

```python
class QuantizationKind(str, Enum):
    FP8_CAST = "fp8-cast"
    FP8_SCALED_MM = "fp8-scaled-mm"
    NVFP4_CAST = "nvfp4-cast"
    NVFP4_PREQUANT = "nvfp4-prequant"
```

No int8 arm, and no rotation of any kind: `convrot` / `conv_rot` / `hadamard` /
`quarot` / `spinquant` are 0 hits across the repository. `rotation` hits only EXIF
image orientation (`media_io/decode.py:32`) and the VAE's per-slab spatial
rotation (`video_vae/transformer/det_attn_rope.py:61`) — neither a quantization
transform.

`int8` upstream is **training-only**, in two places, and neither is an inference
weight format:

- `ltx-trainer/src/ltx_trainer/gemma_8bit.py:33-36` — bitsandbytes `LLM.int8()`
  for the Gemma backbone during LoRA training.
- `ltx-trainer/src/ltx_trainer/quantization.py:11-15` — optimum-quanto precisions
  (`int8-quanto`, `int4-quanto`, …) for the trainer.

Every other `int8` match in the repository is `uint8`: pixel buffers, packed NVFP4
nibbles, block-streaming staging.

**Disposition: KEEP, re-anchored.** The refusal message is honest — it is a
ComfyUI-ecosystem quantization, not an LTX-2 arm — but the enum comment did not
say it had been checked. It now records the absence at these pins so nobody
re-audits it. Its *kind* changes: it is a declared-out-of-scope marker, not a
reachable refusal.

### 1.3 CFG parallelism — the name describes something upstream does not do

`cfg` (case-insensitive, word) is **0 hits** in both multi-GPU trees:
`ltx-pipelines/src/ltx_pipelines/multigpu/` and
`ltx-core/src/ltx_core/multigpu/`.

The three parallelisms upstream actually implements:

| Form | Anchor | What it splits |
|---|---|---|
| Sequence parallel | `multigpu/sp_builder.py:25` (`SequenceParallelBuilder`), `ltx-core multigpu/transformer/sequence_parallel.py`, all-to-all attention | the token axis of one denoise step |
| Tiled data parallel | `multigpu/tdp_builder.py:25` (`TiledDataParallelBuilder`) | spatial tiles, **upscale stage only** |
| Distributed VAE decode | `ltx-core multigpu/vae/distributed_decoder.py:204-256` (`DistributedVideoDecoder.decode_video`) | latent tiles across ranks, driver blends |

Upstream states the purpose in its own words
(`ltx-pipelines/docs/multigpu/README.md:7`, inside the ⚠️ block at `:5-16`):

> **Multi-GPU (MGPU) is a latency tool, not a memory tool.**

and adds that the mutable transformer is a **full replica on every GPU**, so it
cannot make a checkpoint fit.

And CFG is not in the path this port runs at all. The distilled recipe denoises
with `SimpleDenoiser` at both stages (`ltx-pipelines/distilled.py:266`, `:295`),
documented as "**single transformer call, no guidance**"
(`utils/denoisers.py:3`); the guider it degenerates to is
`MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0)`
(`utils/denoisers.py:25-26`), "only runs the conditioned pass and returns cond
unchanged". A `cfg_scale` of 1.0 is one pass, so there is no second pass to place
on a second GPU.

**Disposition: RENAME + re-anchor.** `kCfgParallelism` → `kMultiGpuParallelism`,
anchored to the three real forms, with the reason it is out of scope stated as
what it is: a single-node multi-GPU **latency** feature, on a port whose target is
one GB10.

### 1.4 `kLoraFusion` — real upstream, correctly refused

Verified present: `ltx-core/loader/primitives.py:160`
(`class LoraPathStrengthAndSDOps(NamedTuple)`), exported at
`loader/__init__.py:14,46`, consumed by `loader/single_gpu_model_builder.py:21`
and `block_streaming/builder.py:34,90`, fused by `loader/fuse_loras.py`. The
anchor stands; only its *kind* is corrected (marker, not reachable refusal).

### 1.5 `kVideoEngineWiring` — LANDED

`cefacd2d0` ("feat(ltx-2.5): LTX-2.5 joint video+audio DiT, and a video seam that
is no longer MiniMax-only (#435) (#641)", 2026-08-13) shipped exactly this: the
composition through `vllm::multimodal::VideoEngine`, reachable through the C ABI
as video family `ltx-2.5`. `include/vllm/multimodal/ltx2_video.h:12` already
speaks of the refusal in the past tense.

**Disposition: RETIRE.** A refusal whose subject shipped is a false statement, not
a record of debt.

### 1.6 The five with no product call site

`grep` over `src/`, `include/`, `examples/`, `tests/` for every enumerator:

| Enumerator | Product call site |
|---|---|
| `kTemporalUpsampler` | `src/vllm/model_executor/models/ltx2_upsampler.cpp:395` |
| `kBetaScheduler` | `src/vllm/model_executor/models/ltx2_pipeline.cpp:199` |
| `kLoraFusion` | **none** |
| `kMultishot` | **none** |
| `kInt8ConvRot` | **none** |
| `kCfgParallelism` | **none** |
| `kVideoEngineWiring` | **none** |

The five are enumerated only by `tests/vllm/models/test_ltx2_pipeline.cpp:1237-1244`.
There is no request field, load extra, or CLI flag that asks for a LoRA, an
int8-convrot checkpoint, or a second GPU, so no caller can trip them.

That is not a defect on its own — a declared boundary is worth having. The defect
is calling it a refusal. The header and the messages now distinguish:

- **reachable refusal** — a product path constructs the condition and throws;
- **declared-out-of-scope marker** — a record of what upstream has and this port
  does not, reached only by the ledger test.

## 2. `duration_head_path` (#611)

### 2.1 The full `kKnownLoadExtras` audit

Every key the family accepts, and whether any code reads it. Reader anchors are in
`src/vllm/multimodal/ltx2_video.cpp` unless noted.

| Key | Constant | Reader | Status |
|---|---|---|---|
| `audio_prompt_embeds_path` | `kLtx2AudioPromptEmbedsExtra` | `:901`, `:914` | READ |
| `pipeline_kind` | `kLtx2PipelineKindExtra` | `:737` | READ |
| `model_version` | `kLtx2ModelVersionExtra` | `:721` | READ |
| `allow_unported_modules` | `kLtx2AllowUnportedExtra` | `:570` | READ |
| `max_phase` | `kLtx2MaxPhaseExtra` | `:739` | READ |
| `dit_config_path` | `kLtx2DitConfigPathExtra` | `:625` | READ |
| `prompt_embeds_valid_rows` | `kLtx2PromptValidRowsExtra` | `:942` | READ |
| `encoder_config_path` | `kLtx2EncoderConfigPathExtra` | `:796` | READ |
| `upsampler_path` | (literal) | `:771` | READ |
| `duration_head_path` | (literal) | **none** | **ACCEPTED AND IGNORED** |

Nine of ten are wired. `duration_head_path` is the only defect, so the sweep this
row owes is complete and closes the "the sweep that found this one did not cover
them all" clause of #611.

One documentation gap found by the same sweep and fixed here:
`docs/USAGE.md:1650-1654` lists the LTX-2.5 extras and omits `encoder_config_path`
entirely, though it is defined and read.

### 2.2 Why it is inert

`ltx2_duration_head.h` / `ltx2_duration_head.cpp` port `DurationHead` and
`AttentionPooler` and gate them as bricks. Nothing in `ltx2_video.cpp` constructs
one. The AUTO-duration path (`resolve_num_frames`, `ltx-pipelines/utils/blocks.py`)
therefore cannot run, and `Generate` computes `frames` from
`duration_seconds * fps` directly (`ltx2_video.cpp:1152`).

The stated reason for this moved twice already (#604's pattern, and the finding
that produced #611). The reason recorded here is the current one: **no head is
constructed**, so a supplied path is a file the engine never opens.

### 2.3 Fix

Refuse `duration_head_path` **by name** when it is supplied and non-empty, with a
message that says the head is unported, that the recipe default would otherwise be
substituted silently, and what to use instead (`num_frames`, or `duration`, which
is exact arithmetic against the recipe frame rate).

Rejected alternative: dropping the key from `kKnownLoadExtras`. That produces
"unknown load extra", which is *wrong* — the key is defined by this family and its
meaning is understood; what is missing is the implementation. AGENTS.md requires
"a message naming the missing piece", and "unknown key" does not name it.

Rejected alternative: reading it and constructing the head. That is the real fix
and it stays owed — it needs the connector-output plumbing the head consumes
(`duration_head.py:89-118` takes audio and/or video connector token states), which
is a different row. Refusing is the cheap correct answer until then.

## 3. Scope

**In.**

1. Retire `kMultishot` and `kVideoEngineWiring`; record both retirements in the
   header, this spec, and the commit message.
2. Rename `kCfgParallelism` → `kMultiGpuParallelism` and re-anchor its message to
   the three real parallelisms plus the reason CFG is not in our path.
3. Re-anchor `kInt8ConvRot`'s comment and message to record the verified absence
   at these pins.
4. Split the ledger into reachable refusals and declared-out-of-scope markers, in
   the header comment, in the messages, and in the test.
5. Refuse `duration_head_path` by name (#611).
6. `docs/USAGE.md`: the extras paragraph, corrected on both counts.

**Out.**

- `.agents/specs/ltx-2-5.md` — operator-owned. This row does not edit it. Its §2
  "Out" list still names `multishot`; correcting that is the operator's edit, and
  this spec is the record it would cite.
- Constructing a duration head, and the AUTO-duration path. Stays owed (#611
  remains open after this row, retitled by the fix rather than closed by it —
  see §7).
- `kTemporalUpsampler` (row `LTX25-TEMPORAL-UPSAMPLER`), image conditioning (row
  `LTX25-IMAGE-COND-FIX`), tiled decode (row `LTX25-TILED-DECODE`), AdaLN claims
  (row `LTX25-ADALN-CLAIMS`). No file in this diff is theirs.

## 4. Tests

RED first for the behavioural change; the records changes are gated by the ledger
test's own assertions.

1. **`ltx2 duration_head_path is REFUSED by name` (new,
   `tests/vllm/multimodal/test_ltx2_video.cpp`).** Builds a valid load, adds
   `extras["duration_head_path"]`, and requires a throw whose message names the
   key, names the duration head as the missing piece, and names the alternative.
   **RED before the fix**: the load succeeds, because nothing reads the key. That
   is the defect stated as a test.
2. **`ltx2 an accepted load extra is READ by something` (new).** Asserts the
   inventory of §2.1 does not silently grow: every key in `kKnownLoadExtras`
   either round-trips through a reader or is refused by name. Implemented as the
   two known-inert keys being refused and the rest being accepted, so adding a
   tenth decorative key fails.
3. **`ltx2 every L5 out-of-scope feature is refused BY NAME`** (existing, updated).
   The list drops from 7 to 5 entries — a CHANGED COUNT, reported as such — and
   splits into `reachable` and `markers`, with the marker messages required to say
   they are not requestable. Adds a guard that no refusal message mentions
   `multishot` again.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Removing an enumerator breaks an out-of-tree caller | `Ltx2UnportedPipelineFeature` is internal (`include/vllm/model_executor/`), not part of `include/vllm.h`. Grep shows 3 call sites, all in-tree. |
| Refusing `duration_head_path` breaks a working caller | It cannot: no code reads it, so no caller was getting anything from it. A caller who passes it today is being silently ignored, which is the bug. |
| The ledger test's case count changes and reads as a regression | Stated up front in the gate report: 7 → 5 entries in one case, plus 2 new cases. The exit code is the authority. |
| A concurrent LTX row edits the same header | Four live rows named; none owns `ltx2_pipeline.h:575-590`, `ltx2_video.cpp:250-300`, or the ledger test case. `LTX25-TEMPORAL-UPSAMPLER` owns `kTemporalUpsampler`, which this row does not touch. |

## 6. Stop conditions

- Return `NEEDS_DECISION` if `multishot` turns out to exist under vocabulary not
  searched here — the retirement is then wrong and the arm must be re-anchored
  instead.
- Return `NEEDS_DECISION` on a collision with a live LTX row.
- Do not close #611 if the refusal cannot be made to fail RED first.

## 7. Outcome

**Every claim in §1 was re-derived by this row against the two pinned checkouts;
none was taken from the prior grounding pass.** Results:

- `multishot`: **fabricated, confirmed.** 0 hits for the term in either reference;
  the only upstream sense of "shot" is one camera take (§1.1). Retired.
- `int8-convrot`: **absent upstream, confirmed.** Four inference quantization kinds
  exist and none is int8; int8 is training-only (§1.2). Kept, re-anchored, so the
  absence does not have to be re-audited.
- CFG parallelism: **the name was wrong, the exclusion is right.** 0 `cfg` hits in
  either multi-GPU tree; upstream's own README calls MGPU a latency tool; the
  distilled recipe runs `SimpleDenoiser` with no guidance at all, so there is no
  CFG pass to parallelize (§1.3). Renamed and re-anchored.
- `kVideoEngineWiring`: **stale, confirmed.** L7 landed in `cefacd2d0`. Retired.
- Five enumerators had no product call site (§1.6). The ledger now says which
  kind each is, because "refused by name" overstated a marker.
- `duration_head_path` was the **only** unread key of ten (§2.1). Now refused by
  name. The full inventory is the durable half of this row: it means the next
  person asking "which extras are decorative?" reads a table instead of grepping.

What this row deliberately did **not** do: construct a duration head. #611 stays
open for that, with its user-visible half — silent substitution of the recipe
default — closed.

## Now

Row `LTX25-RETIRE-DEAD-ARMS` is `DONE`. The ledger carries five entries, split by
kind; `duration_head_path` is refused by name with a RED-first test; the
`kKnownLoadExtras` inventory is recorded in §2.1. `.agents/specs/ltx-2-5.md` §2
"Out" still lists `multishot` and is the operator's to correct, citing §1.1 here.
