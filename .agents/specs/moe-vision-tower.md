# MoE vision tower (M2/M3): image and video on the GDN-hybrid MoE backbone

**Rows:** `MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation`
**Issue:** [#891](https://github.com/mudler/vllm.cpp/issues/891)
**Lifecycle:** `READY`
**Owner:** unassigned

## Scope

Run the Qwen3.5 vision tower on the **MoE** backbone, and gate image and video
token-exact — the arm the dense row already has and this one does not.

In scope:

- stop dropping the checkpoint's `model.visual.*` tensors in the MoE loader;
- the tower forward on the MoE backbone, gated on mm input so the text path is
  byte-identical;
- image→text and video→text token-exact gates on `Qwen/Qwen3.6-35B-A3B`.

Out of scope: audio (this family ships none — no `audio_config`, no
`audio_token_id`), the deepstack path (`deepstack_visual_indexes: []`, compiled
out upstream at `qwen3_vl.py:1709-1716`), any speed claim, and any claim about
executing `Qwen/Qwen3.8-2.4T-A95B`.

## What is missing, precisely

**The tower forward, not the plumbing.** M0/M1 landed the mm input pipeline and
the processor-parity gate passes, so images and video already reach the model
correctly. They have nowhere to go on this backbone.

The dense arm is the template and the proof it is tractable: a forked GDN-hybrid
VL forward, gated on mm input, achieving **image 32/32** and **video 32/32**
token-exact while leaving the text path byte-identical.

## Upstream chain

| Anchor | Contract to mirror |
|---|---|
| pinned vLLM `qwen3_5.py` `Qwen3_5MoeForConditionalGeneration` | The MoE conditional-generation class composes the same vision tower as the dense one over a different text backbone. The tower is not MoE-specific. |
| pinned vLLM `qwen3_vl.py:1709-1716` | `deepstack_visual_indexes: []` compiles the deepstack path out for this family. |
| local: the dense arm's VL forward (`MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`, M3-b/M3-d) | Fork gated on mm input so text stays byte-identical; reuse the windowed tower and video MRoPE rather than writing a second one. |
| local `.agents/specs/mm-tools-scoping-2026-07-10.md` | Records that the 35B's `model.visual.*` tensors are present and **silently dropped** by the loader today. |

Read the dense arm's forward before writing anything. If the two towers diverge
in anything but which backbone consumes their output, that divergence is itself
a finding.

## Design

1. **Load the tower.** The 35B bf16 checkpoint carries 333 `model.visual.*`
   tensors. They are dropped today. A silent drop is the defect class this
   project keeps rediscovering — make their absence a refusal, not a shrug.
2. **Fork the forward, gated on mm input.** Exactly as the dense arm does. When
   no mm input is present the text path must execute the identical instruction
   sequence it does today.
3. **Reuse, do not reimplement.** The windowed tower, the video MRoPE and the
   processor are shared with the dense arm. A second implementation would drift.

## Risks

- **Text regression on a 315/315 gate.** The MoE text row is gated and heavily
  used. Mitigated by byte-identical goldens plus a re-run text gate — not by a
  green suite.
- **Silent modality drop.** If the tower loads but is never invoked, image input
  degenerates to text-only and still produces plausible tokens. The gate must
  compare against the oracle's mm output, not merely check it did not crash.
- **Divergence from the dense tower.** Two implementations of one tower will
  drift; the dense one is already gated at 32/32 both modalities.
- **Memory.** The 35B bf16 is 71.9 GB plus a tower on a 119 GiB box, and a
  global OOM there has already killed unrelated processes.

## Tests

1. Image→text on `Qwen/Qwen3.6-35B-A3B`, token-exact vs the pinned oracle, using
   the **same harness** as the dense arm's 32/32 run.
2. Video→text likewise.
3. Text inertness: the MoE text gate stays 315/315, goldens byte-identical.
4. Loader: a checkpoint whose `visual.*` tensors are absent is refused naming
   them, rather than silently loading a text-only model.

## Gates

- Focused suites plus the full serial gate.
- **Binding: image and video token-exact vs the pinned oracle at 35B.**
- Text inertness proven by re-run, not argued.
- The CPU suite is necessary and **not sufficient**: a tower that loads and is
  never invoked passes every offline test and still answers image prompts from
  text alone.

## Evidence required

- RED capture before the tower runs.
- Real counts per modality, or an explicit statement that one did not run.
- Golden md5 before/after and the text gate's count from an actual run.

## Stop conditions

- If the dense arm's tower cannot be reused, stop and return `NEEDS_DECISION`
  rather than writing a second one.
- If text inertness cannot be held byte-identical, stop — that gate outranks
  this feature.
- Do not claim the 2.4T runs. This row proves the MoE vision path at 35B.

## Now

Row is `READY`. Spec committed; implementation not started.
