# GDN `output_gate_type`: resolve the gate activation from config, not from a default

**Rows:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`,
`MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation`
**Issue:** [#489](https://github.com/mudler/vllm.cpp/issues/489)
**Lifecycle:** `READY`
**Owner:** unassigned

## Scope

Parse the GDN output-gate activation from the model config and thread it to the
gated RMSNorm, replacing the unconditional silu we compute today.

In scope:

- add `output_gate_type` to `HfConfig`, parsed from the resolved text config;
- mirror upstream normalization exactly: absent -> `"silu"`; `"swish"` -> `"silu"`;
  accept `{"silu", "swish", "sigmoid"}`; **reject** any other value at config load
  rather than silently defaulting;
- thread the resolved value to the GDN `vt::RmsNormGatedArgs::sigmoid_gate` call
  sites in the Qwen3.5 family (dense + MoE + MTP), which already carry the
  plumbing;
- prove the existing 27B / 35B / Coder paths stay **byte-identical**.

Out of scope: the attention output gate (`attn_output_gate`, a different
mechanism), the `RMSNormGated` `group_size` / `norm_before_gate` knobs (upstream
bakes both for Qwen GDN), any new kernel, any performance claim, and any change
to models outside the GDN family.

## Upstream chain

| Upstream anchor | Contract to mirror |
|---|---|
| pinned vLLM `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:452-456` | `getattr(config, "output_gate_type", "silu")`; `"swish"` collapses to `"silu"`; the set `{silu, swish, sigmoid}` is asserted, anything else is an error. |
| pinned vLLM `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:464` | The resolved string is the `activation=` of `RMSNormGated`; nothing else consumes it. |
| local `include/vt/ops.h:466-472` | `RmsNormGatedArgs::sigmoid_gate` already selects sigmoid vs silu; `norm_before_gate=True`, `group_size=None` are baked in, matching upstream. |
| local `.agents/specs/gdn-semantics.md:32-36` | Records that both current gate models resolve to silu and that the sigmoid golden is a spare — this row converts that documented assumption into enforced behavior. |

This is an **at-pin** port. `git diff 555967922 origin/main --
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` touches only ROCm
kernel availability and a spec-decode `a`/`b` `index_select` fix; lines 452-464
are identical at the pin and at upstream main, so no pin movement is implied.

## Design

`output_gate_type` is a **config-resolution** concern, so normalization and
validation live in `hf_config.cpp` next to the other Qwen3.5-family defaults,
not at the call site. The model layer consumes a resolved boolean.

1. `HfConfig` gains `std::string output_gate_type` (canonicalized, so it only
   ever holds `"silu"` or `"sigmoid"` after load — `"swish"` is collapsed at
   parse time exactly as upstream does before the assert).
2. Parsing reads the key from the **resolved text config**, so a nested VL
   wrapper and a flat text-only config behave alike.
3. An unrecognized value throws at config load with a message naming the key and
   the accepted set. Upstream asserts; we surface the same failure as a refusal
   rather than a silent fallback, per the standing rule that an unimplemented or
   unrecognized arm is refused with a message naming the missing piece.
4. The Qwen3.5 GDN sites pass `sigmoid_gate = (output_gate_type == "sigmoid")`
   into `RmsNormGatedArgs`. Because every current gate checkpoint resolves to
   silu, that expression is `false` on every gated path and the emitted work is
   unchanged.

The canonicalize-at-parse choice matters: it means exactly one place can ever
decide what the gate is, so a future call site cannot reintroduce the default by
forgetting to normalize.

## Risks

- **Silent inertness.** The change could be wired but never reach the kernel, and
  every existing gate would still pass because they are all silu. Mitigated by a
  RED-first test that drives the `sigmoid` arm through the model path and proves
  the output differs from the silu arm — not by asserting a config field.
- **Regression on gated rows.** The GDN call sites are on the 27B/35B hot path.
  Mitigated by byte-identical golden comparison, not by "tests pass".
- **Wrong resolution surface.** Reading the key from the top-level doc instead of
  the resolved text config would work for flat configs and silently miss nested
  VL wrappers. Covered by a nested-config test.

## Tests

Ported/authored in the same change:

1. `test_hf_config.cpp` — absent key resolves to `silu`; `"swish"` resolves to
   `silu`; `"sigmoid"` resolves to `sigmoid`; an unrecognized value throws; the
   key is picked up from a **nested** `text_config` as well as a flat config.
2. A GDN-path numerics test that runs the same input through the silu and sigmoid
   arms and asserts the outputs **differ** — the mutation-proof that the config
   actually reaches the kernel. Must be RED before the wiring lands.
3. Inertness: existing Qwen3.5 dense/MoE/MTP tests unchanged and green.

Per the standing trap, assert on the doctest `Status` line and the assertion
count, not on `assertions:` alone; and use `.scale(0.0)` for any small-magnitude
comparison.

## Gates

- Focused: the two test targets above, plus the Qwen3.5 dense/MoE/MTP suites.
- Full gate on the row before push.
- **Byte-identical evidence:** the 27B/35B goldens' md5 must be unchanged. A
  passing token gate is necessary but not sufficient here — the whole defect
  class is a numerics change that a silu-only corpus cannot see.

## Evidence required

- RED capture of the sigmoid-arm numerics test before the wiring.
- Green focused + full gate after.
- Golden md5 before/after showing no drift on the gated rows.

## Stop conditions

- If upstream's `RMSNormGated` `activation` turns out to select anything beyond
  the silu/sigmoid split that `vt::RmsNormGatedArgs` models, stop and return
  `NEEDS_DECISION` rather than widening the vt surface.
- If the sigmoid arm cannot be driven end-to-end through a model path, stop and
  report rather than settling for a config-field assertion, which would prove
  nothing about the kernel.

## Now

Row is `READY`. Spec committed; implementation not started.
