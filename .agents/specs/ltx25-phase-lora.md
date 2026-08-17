# LTX25-PHASE-LORA — the adapter set belongs to a PHASE, not to the load

Row `LTX25-PHASE-LORA`. Issue
[#1118](https://github.com/mudler/vllm.cpp/issues/1118). Campaign
[`ltx-2-5.md`](ltx-2-5.md), under roadmap row `ROAD-V1-LTX25`.

Upstream pin: Lightricks/LTX-2 `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
verified with `git rev-parse HEAD` in `/home/mudler/_git/LTX-2` on 2026-08-17.

## Now

`ACTIVE` -> `DONE` with this change. The row exists because #1118 blocks #1093
and #1096 and bounds #921; it is the seam those three need, and it closes the
divergence #1117 shipped and filed in the same breath.

## Scope

**In.** One seam: a phase declares which of the load's adapters it runs, and the
engine brings the resident DiT to that state at the phase boundary. One
consumer: `a2vid_two_stage`, whose stage 1 upstream runs WITHOUT the distilled
adapter. One record repair: the reference-conditioning refusal in
`ltx2_video.cpp` gives two reasons and the second becomes false here.

**Out.** `ti2vid_two_stages` (#1093) and `keyframe_interpolation` (#1096) — both
additionally need checkpoints that are not on the NAS. The HQ arm's per-phase
STRENGTH (#921). N-adapter subsets. The reference clip's pixel path, which is
the reference refusal's FIRST reason and stays refused.

## Upstream chain

Four pipelines build two `DiffusionStage`s from ONE checkpoint and give them
DIFFERENT adapter sets. Each line below was read at the pin.

| Pipeline | stage 1 | stage 2 |
|---|---|---|
| `a2vid_two_stage.py` | `loras=tuple(loras)` (`:107`) | `(*loras, *distilled_lora)` (`:114`) |
| `ti2vid_two_stages.py` | `loras=tuple(loras)` (`:140`) | `(*tuple(loras), *distilled_lora)` (`:151`) |
| `ti2vid_two_stages_hq.py` | `(*loras, distilled_lora_stage_1)` (`:154`) | `(*loras, distilled_lora_stage_2)` (`:165`) |
| `ic_lora.py` | `loras=tuple(loras)` (`:108`) | `loras=()` (`:119`) |

The two `DiffusionStage.from_checkpoint` calls are `ti2vid_two_stages.py:136`
and `:147`, and `ic_lora.py:104` and `:115`. `ic_lora.py` is the mirror image of
the other three: the adapter rides stage 1 and stage 2 runs bare.

`distilled.py:131` builds ONE stage set, which is why `distilled_two_stage`,
`dfr` and `retake` have never needed this seam and why no gate here has ever
asked the question. **Those recipes must not change**, and the default value of
the new field is chosen so that they cannot.

The fusion arithmetic itself is unchanged and already ported:
`loader/fuse_loras.py:99-116` and `:119-150`, anchored in
[`ltx2_lora.h`](../../include/vllm/model_executor/models/ltx2_lora.h).

## Our baseline

Measured on `d1e5e9bc0`, the base this row branches from.

- `git grep -n 'dit_options.loras' -- src` returns exactly ONE line,
  `src/vllm/multimodal/ltx2_video.cpp:831`. Positive control:
  `git grep -c loras -- src` returns five FILES, seven hits of them in
  `ltx2_video.cpp`. (The dispatch that opened this row said "5 in that same
  file"; the five is a file count.)
- That one site runs inside `Ltx2VideoEngine::Load`, under
  `if (!lora_path.empty())`, and feeds
  `Ltx2LoadDitFromSafetensors` / `Ltx2StreamDitToDevice` at `:833-834`.
- `Ltx2DitCheckpoint im.dit` (`ltx2_video.cpp:593`) is built once and outlives
  every generation. The phase loop (`:2775`) and the DiT forward inside it
  (`:3599-3602`) read `im.dit.weights` with no notion of a phase.
- `Ltx2PhaseRecipe` (`ltx2_pipeline.h:606-629`) carries thirteen per-phase
  fields and none of them names an adapter.
- `ltx2_pipeline.h:686-691` already states the gap in the tree's own words, and
  names #1118: "What this flag CANNOT express is upstream's placement".

So the defect is not that the adapter is unreachable. It is that the adapter has
exactly one placement, and upstream has four.

## Port map

### The field

`Ltx2PhaseRecipe` gains `Ltx2PhaseLoraScope loras = kAllAdapters`.

The enum has two enumerators, and two is the whole space this engine can select
because `Ltx2ResolveLoraReferenceFactors` refuses more than one adapter by name
(`ltx2_lora.h:167-172`, mirroring `dubit.py:364-365` and
`hdr_ic_lora.py:271-272`):

- `kAllAdapters` — the phase runs every adapter the load supplied. This is the
  DEFAULT, so every recipe that exists today keeps the behaviour it was gated
  with, and `distilled.py:131`'s single stage set stays single.
- `kNoAdapters` — the phase runs the base weights.

**A per-phase STRENGTH is deliberately NOT added.** `ti2vid_two_stages_hq.py`
needs one (0.25 at `:92-96`, 0.5 at `:97-101`), and no recipe this row ships
would set it, so adding the field now lands a branch nothing can select — the
argument `ltx2_lora.h:41-44` already makes for the second product form. Owed
below, against #921.

### The mechanism

`Ltx2RebindDitLoras`, new in `ltx2_loader.h` / `.cpp`, brings an
ALREADY-LOADED checkpoint to the adapter state a given phase wants:

```cpp
void Ltx2RebindDitLoras(vt::Queue* queue, const SafetensorsFile& file,
                        const Ltx2DitLoadOptions& options, bool fuse,
                        Ltx2DitCheckpoint& checkpoint);
```

For every contract tensor the adapters target it re-materializes the tensor from
`file` through the same `MaterializeDitTensor` the load uses, fuses the adapters
into it when `fuse`, and writes the result back **into the buffer the view
already points at**. The view pointer never moves, so `checkpoint.weights` — a
pure view struct (`ltx2.h:273-295`) — stays valid and is not re-bound. `queue`
is non-null exactly when the checkpoint was staged to a device, in which case the
write back is a `Copy` into the same device allocation.

Three properties make this the shape this row chose:

1. **It is EXACT.** Each phase gets `round_bf16(W + delta)` computed from the
   pristine base, through the same code path the load uses. Not
   `Wx + s*B(Ax)`.
2. **It costs no extra RESIDENT memory.** One tensor of scratch, plus the
   adapter's own A/B factors for the duration of the rebind. No second weight
   set ever exists.
3. **It is one call site.** `FuseLorasInto` (`ltx2_loader.cpp:536-547`) already
   collapses both load arms onto one `Ltx2FuseLoraIntoTensor` call; the rebind
   reuses it rather than writing a parallel fuse.

### The consumer

`A2VidTwoStageRecipe` (`ltx2_pipeline.cpp:1510`) sets
`stage1.loras = kNoAdapters` (`a2vid_two_stage.py:107`) and leaves stage 2 at
the default (`:114`).

Reachable from a production entry point on its default configuration:
`include/vllm.h` -> `LoadVideoEngine` with the documented `pipeline_kind` and
`lora_path` load extras -> `Ltx2VideoEngine::Generate`. `ltx2-gen
--pipeline-kind a2vid_two_stage --lora-path ... --audio-path ...` is the same
two calls through the ABI. The `/v1/videos` route cannot drive it, because
`VideoGenParamsFromRequest` never writes `gen.extras` (#928) — stated so the
reach claim excludes it rather than overstating it.

**Which upstream list our one slot IS.** For this recipe the supplied adapter is
upstream's `distilled_lora`, which is what `requires_distilled_lora`
(`ltx2_pipeline.h:692`) already declares by mirroring `--distilled-lora
required=True` (`utils/args.py:1140-1153`). Upstream's separate user `loras`
list, which rides BOTH stages, has no spelling here and cannot until the
adapter arity refusal lifts. Owed below.

### The record repair

`ltx2_video.cpp`'s reference-conditioning refusal gives TWO reasons. Reason 2 —
"THE REFERENCE ITEM BELONGS TO STAGE 1, AND STAGE 2 MUST RUN UNFUSED ... this
engine holds ONE `Ltx2Dit`, fused at load, that every phase of the recipe runs"
— becomes FALSE with this change and is rewritten to say what closed it, in the
form that block already uses for its three earlier ruled-out reasons.

**The refusal is NOT retired.** Reason 1, the reference clip's pixel path
(`iclora_utils.py:112-117`, `:87-89`, `:144-148`), is untouched by this row and
still holds. Retiring the whole refusal here would ship an arm whose geometry
nothing supplies. The dispatch that opened this row proposed retiring it as the
proving consumer; that is rejected on the evidence above and `a2vid_two_stage` is
the consumer instead.

## Tests to port

Upstream has no test for adapter placement — the placement IS the pipeline
constructor, and `ltx-pipelines` ships no unit test over it. So these are ported
in the sense that every assertion cites the upstream line it mirrors, and the
harness is ours.

### The distinguishing gate

`tests/vllm/multimodal/test_ltx2_video.cpp`, in the a2vid section, driven
entirely through `LoadVideoEngine` and `Generate`:

| Render | `lora_strength` | `max_phase` | Assertion |
|---|---|---|---|
| A | 1.0 | 0 | `A == B` |
| B | 0.0 | 0 | — |
| C | 1.0 | (none) | `C != D` |
| D | 0.0 | (none) | — |

`max_phase` is a documented LOAD extra (`kLtx2MaxPhaseExtra`,
`ltx2_video.h:202`, read at `ltx2_video.cpp:1029`), so `max_phase = 0` renders
stage 1 ALONE through the production path. Strength `0.0` is the control rather
than "no adapter", because `requires_distilled_lora` refuses an a2vid load with
no `lora_path` at all; strength 0 fuses a zero delta and is already gated as
equal to the base model ("the IC-LoRA strength reaches the PIXELS, and 0 is a
no-op").

**Both rows are load-bearing, and either one alone is passable by a defect.**
`A == B` says the adapter is NOT on stage 1 — it REDs under today's load-time
fusion. `C != D` says the adapter IS on stage 2 — it REDs under an
implementation that simply stopped fusing. A gate asserting only "a LoRA was
applied" passes on the current defect, which is why neither row stands alone.

### The exactness gate

`tests/vllm/models/test_ltx2_loader.cpp`: load a checkpoint with the adapter and
without it, keep both byte images, then on a third checkpoint rebind off and
compare byte-for-byte against the unfused image, and rebind on and compare
byte-for-byte against the fused image. This is what makes the "exact, not
`Wx + s*B(Ax)`" claim executable rather than asserted, and it fails on any
rebind that reconstructs the base by SUBTRACTING the delta instead of
re-materializing it.

### Mutations required to pass

1. `stage1.loras = kNoAdapters` deleted from `A2VidTwoStageRecipe` — reverts to
   single fusion. The distinguishing gate must go RED.
2. The rebind call deleted from the phase loop. Must go RED.
3. `dit_options.loras.push_back` deleted (the standing reachability mutation,
   `ltx25-ic-lora.md` section 5.3). Must go RED.

Every mutation prints four facts: `git diff --stat`, whether it BUILT, the
compile-error count, and the exit code captured directly. A mutation that fails
to build, or that never applied, reads as a passing test.

## Gates

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build -j6 && ctest --test-dir build -j4 --output-on-failure
```

Whole binaries, never a `--test-case` filter: a filter matching zero cases
prints `SUCCESS!` at exit 0, and at least one LTX case name contains a comma,
which doctest `-tc` splits on. Report `CONFIGURE_EXIT`, `BUILD_EXIT`, the
`: error:` count, `ctest -N`, `CTEST_EXIT`, the pass/fail line, and
`No space left` / `BFD` greps with positive controls, plus load and free disk.

`READER ANCHORS` (`ltx2_video.cpp:386-387`) is gated by `test_ltx2_video` and
shifts whenever the readers above line 1335 move. `Load` is edited by this row,
so the list WILL move. Re-derive with the test's own walk and paste the list it
prints; arm the instrument first by inserting a line above the anchors and
confirming MISMATCH.

**No GPU.** This fleet is leased with `rc`, and a weight-lifetime seam is
correctly gated by the CPU goldens. No real-weights result is claimed by this
row.

## Dependencies

- #1117 (`LTX25-A2VID-RECIPE`), landed at `d1e5e9bc0`. This row's consumer is
  its recipe, and its `requires_distilled_lora` flag is what makes the supplied
  adapter identifiable as the distilled one.
- #923 (`LTX25-IC-LORA`), landed. Supplies `ltx2_lora.h` and the one fuse site.

Blocks #1093, #1096. Bounds #921.

## Work breakdown

1. Spec, committed before any product code.
2. The enum and the field, defaulted so no existing recipe moves.
3. `Ltx2RebindDitLoras` and its exactness test.
4. The a2vid consumer and the distinguishing gate.
5. The reference-refusal reason-2 rewrite.
6. Anchors re-derived, mutations run, full gate.

## Risks and decisions

**Three shapes were costed. The third was chosen.**

| Shape | Resident memory | Exactness | Verdict |
|---|---|---|---|
| A second resident `Ltx2DitWeights` | DOUBLES the DiT: 18.7 GB nvfp4, 21.0 GB fp8, ~39 GB bf16 | exact | REJECTED |
| Unfused runtime LoRA, phase-selectable | + the adapter | `Wx + s*B(Ax)` against upstream's `round_bf16(W + s*BA)x` | REJECTED |
| Re-materialize the affected tensors at the phase boundary | none | exact | CHOSEN |

**The premise that shape A mirrors upstream is FALSE, and correcting it removes
the only argument that made shape A look principled.** The dispatch that opened
this row, and #1118 itself, both describe upstream as paying "two
`DiffusionStage.from_checkpoint` calls" as though it held two models. It does
not. Both calls name the SAME file:

```text
a2vid_two_stage.py:103   self.stage_1 = DiffusionStage.from_checkpoint(
a2vid_two_stage.py:104       model_paths.transformer(),
a2vid_two_stage.py:107       loras=tuple(loras),
a2vid_two_stage.py:114   stage_2_loras = (*tuple(loras), *tuple(distilled_lora))
a2vid_two_stage.py:115   self.stage_2 = DiffusionStage.from_checkpoint(
a2vid_two_stage.py:116       model_paths.transformer(),
a2vid_two_stage.py:119       loras=stage_2_loras,
```

`ti2vid_two_stages.py:136`/`:147` has the identical shape. So upstream pays two
MATERIALIZATIONS of one checkpoint with different adapter tuples, which is what
shape C is. Shape C is therefore the faithful mirror rather than the cheap
substitute, and shape A would invent a heavier architecture than the reference.

Shape A is also not viable on one GB10 regardless: 119 GB unified,
`vm.overcommit_memory=1`, zero swap, and no OOM line — the box goes down when
memory is oversubscribed, twice in the week this row was written. Shape B is a
rounding divergence AND a different GEMM path, so it would change every arm's
numerics to serve one recipe. Shape C spends wall-clock at a boundary that
happens once or twice per render, and on this fleet wall-clock is not the
constraint that reboots the box.

**The field is a per-phase SET, not a per-phase boolean, and the difference is
load-bearing.** Upstream needs two placements, not one:

| Placement | Pipelines | `ltx-pipelines/CLAUDE.md` |
|---|---|---|
| adapter on stage 2 ONLY | TI2Vid, A2Vid, Keyframe | `:48` |
| adapter on BOTH stages | HQ, DFR | `:49`, `:50-51` |

`Ltx2PhaseLoraScope` expresses both: stage 1 `kNoAdapters` with stage 2
defaulted gives the first, and both phases defaulted gives the second. Two
enumerators are the COMPLETE space here rather than a boolean standing in for a
set, because the adapter arity is capped at one by a gated refusal
(`ltx2_lora.h:167-172`, case "ltx2 lora: more than one adapter refuses BY
NAME"), so the powerset of the load's adapters has exactly two members. The day
that cap lifts, the third value goes here.

**What shape C actually costs, stated rather than waved at.** The rebind
re-reads the adapter file and re-materializes only the tensors that adapter
targets, so the transient peak is the resident DiT plus the adapter's A/B
factors plus one tensor. It is not free, and it is not a full reload: every
tensor no adapter names is untouched, because for those the fused and unfused
images are equal by construction.

**Why re-materialize rather than subtract the delta.** Subtracting would be one
pass over the target tensors with no file read, and it would be WRONG:
`round_bf16(round_bf16(W + d) - d)` is not `W`. The exactness gate is written to
fail that implementation specifically.

**Why the default is `kAllAdapters`.** Any other default silently changes
`distilled_two_stage`, `dfr`, `retake`, `one_stage` and `res2s`, all of which are
gated, and upstream gives all of them one stage set (`distilled.py:131`). A
default that moves a landed arm would be the same class of defect this row
exists to fix.

## Owed

- **Per-phase adapter STRENGTH**, which `ti2vid_two_stages_hq.py:92-101` needs
  and this field cannot express. Owned by #921.
- **N-adapter per-phase SUBSETS**, upstream's `(*loras, *distilled_lora)` where
  `loras` is the user's own list and rides both stages. Blocked on the adapter
  arity refusal (`ltx2_lora.h:167-172`), which is upstream-faithful for the
  pipelines that take exactly one and is not lifted here.
- **A real-weights comparison against upstream's own render** on the same
  checkpoint, take and seed — upstream's stage 1 on base weights against ours.
  This is the instrument `ltx25-a2vid-recipe.md` section 4.4 named, and it needs
  the distilled adapter checkpoint, which `find /mnt/nas_share/checkpoints
  -iname '*lora*'` returns nothing for. Owed against #1093's checkpoint
  dependency.
- **The reference clip's PIXEL path**, reason 1 of the refusal this row
  narrows. Still owed by #975.
