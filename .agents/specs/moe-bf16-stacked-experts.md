# MoE bf16 / 3-D-stacked routed experts: the arm Qwen3.8 needs

**Rows:** `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm`,
`MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation`
**Issue:** [#740](https://github.com/mudler/vllm.cpp/issues/740)
**Lifecycle:** `READY`
**Owner:** unassigned

## Scope

Teach `LoadQwen3_5Moe` to read **3-D stacked, unquantized (bf16)** routed experts,
selected by tensor presence, and gate it token-exact on a real published Qwen
bf16 MoE checkpoint.

In scope:

- a stacked bf16 reader for the routed experts, dispatched by what the shard
  index actually contains;
- narrowing `CheckMoeExpertLayoutSupported` so it stops refusing what now loads
  and keeps refusing what remains unsupported;
- a token-exact greedy gate on `Qwen/Qwen3.6-35B-A3B` (bf16) against the pinned
  oracle;
- byte-identical inertness for the NVFP4 path serving the gated 27B / 35B /
  Coder rows.

Out of scope: the shared expert, the attention tower, GDN, the compressed-tensors
`weight_packed` spelling, GGUF, MTP for 3.8, any speed claim, and any claim about
executing `Qwen/Qwen3.8-2.4T-A95B` itself.

## Why this is gateable, and why that was previously misjudged

[#490](https://github.com/mudler/vllm.cpp/issues/490) recorded this arm as owed
but framed it as hardware-blocked, because Qwen3.8 is ~4.8 TB bf16 against
GB10's 128 GB. That framing was too narrow: **what needs proving is the layout,
not the scale.**

`Qwen/Qwen3.6-35B-A3B` (bf16) carries the identical shape at a size that fits:

| | Qwen3.8-2.4T-A95B | Qwen3.6-35B-A3B bf16 |
|---|---|---|
| routed experts | 3-D stacked | 3-D stacked |
| scale tensors | none | none |
| backbone prefix | `model.` | `model.language_model.` |
| size | ~4.8 TB | **71.9 GB, 26 shards** |

71.9 GB fits GB10's unified memory and vLLM runs it, so this arm gets a real
token-exact gate. The differing prefix is a bonus rather than a complication: the
same run exercises `ResolveQwen3_5BackbonePrefix` (landed in #490) on the
namespace it was written for.

**This will be the first time the MoE loader reads a published Qwen bf16 repo.**
Every existing gate reads the requantized `nvidia/Qwen3.6-35B-A3B-NVFP4`, whose
experts are per-expert and quantized — which is precisely why the gap survived
undetected until Qwen3.8 exposed it.

## Upstream chain

| Upstream anchor | Contract to mirror |
|---|---|
| pinned vLLM `vllm/model_executor/models/qwen3_5_moe.py` expert loading | Stacked `experts.gate_up_proj` / `experts.down_proj` are the native published layout; the loader slices per expert rather than requiring per-expert tensors. |
| pinned vLLM `vllm/model_executor/layers/fused_moe/layer.py` weight loader | Expert-parallel slicing of a stacked tensor, and the gate/up split within `gate_up_proj`. |
| local `src/vllm/model_executor/models/gemma4_weights.cpp:326` | In-tree precedent for dispatching between stacked and per-expert layouts by tensor presence. |
| local `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp` | The dense arm already routes BF16 / FP8 / NVFP4 by presence; the MoE arm should mirror that shape, not invent a new one. |

Read the pinned oracle's own slicing order before writing any of this. The gate
is token-exact, so a transposed or mis-split `gate_up_proj` will show up as
divergence, not as a load error.

## Design

1. **Dispatch by presence, once.** Probe the shard index for the stacked spelling
   versus the per-expert one, decide a single time, and thread the decision —
   mirroring `ResolveQwen3_5BackbonePrefix`, which resolves the namespace once
   rather than per lookup. A per-lookup fallback would let a checkpoint load half
   from each layout and still appear to succeed.
2. **The stacked reader slices, it does not materialize.** `gate_up_proj` is
   `[num_experts, 2 * moe_intermediate, hidden]`; `down_proj` is
   `[num_experts, hidden, moe_intermediate]`. Confirm both against the real index
   rather than trusting this spec. The gate/up split within the stacked tensor
   must match upstream's ordering exactly.
3. **Alignment.** Stacked slices are offsets into an mmap'd safetensors payload
   and carry no alignment guarantee. Route every typed read through
   `vt::LoadUnaligned` (`include/vt/unaligned.h`), which is the seam
   [#627](https://github.com/mudler/vllm.cpp/issues/627) exists for and which has
   already been bypassed twice by new loaders.
4. **Narrow the refusal, do not delete it.** `CheckMoeExpertLayoutSupported`
   currently refuses three shapes. The stacked-expert branch becomes supported;
   the unquantized-`lm_head` and per-expert-but-unquantized branches must be
   re-examined on their own merits rather than removed as collateral. Every
   refusal test that changes meaning gets updated deliberately, with the reason
   recorded.

## Risks

- **Regression on three gated rows.** This is the shared loader behind 27B / 35B
  / Coder. Mitigated by byte-identical goldens plus re-run SACRED gates — not by
  a green suite.
- **Silent mis-slicing.** A wrong gate/up split or expert stride loads cleanly and
  produces wrong logits. Only the token-exact gate catches it, which is why the
  synthetic unit test is necessary but explicitly not sufficient.
- **Refusal over-narrowing.** Making the refusal permissive enough to accept
  stacked bf16 could also accept a genuinely unsupported shape. Each remaining
  refusal branch needs its own surviving test.
- **Checkpoint availability.** 71.9 GB over 26 shards must be staged before the
  gate can run; that is a prerequisite, not a finding.

## Tests

1. Synthetic stacked-layout load through the production `LoadQwen3_5Moe`, both
   residency paths, RED-first.
2. Byte-equality: the same logical weights expressed stacked and per-expert must
   load to identical bytes.
3. Refusal tests: each shape that remains unsupported still refuses, naming the
   missing piece; the stacked shape no longer does.
4. Inertness: 27B / 35B / Coder suites unchanged, golden md5 unchanged.

## Gates

- Focused suites plus the full serial gate.
- **Binding: token-exact greedy on `Qwen/Qwen3.6-35B-A3B` bf16 vs the pinned
  oracle.** Both arms identical prompts, token counts, sampling and batching.
- SACRED 27B / 35B / Coder re-run on the GPU box; goldens byte-identical.
- The synthetic tests alone do not close this row. A stacked reader that loads
  without error and produces wrong logits passes every CPU test in this list.

## Evidence required

- RED capture for the stacked-layout test before the reader exists.
- The token-exact gate's real counts, or an explicit statement that it did not
  run and why.
- Golden md5 before/after, and SACRED counts from an actual run.

## Stop conditions

- If the stacked slicing order cannot be established from the pinned oracle's
  source, stop and return `NEEDS_DECISION` rather than inferring it from shapes —
  a plausible-looking guess produces wrong logits, not an error.
- If narrowing the refusal would require deleting a test rather than updating it,
  stop and say so.
- Do not claim Qwen3.8 runs. This row makes the architecture loadable and proves
  it at 35B; the 2.4T checkpoint remains unrunnable on size alone.

## Now

Row is `READY`. Spec committed; implementation not started.
