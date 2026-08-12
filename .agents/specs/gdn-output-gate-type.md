# GDN `output_gate_type`: resolve the gate activation from config, not from a default

**Rows:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`,
`MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation`
**Issue:** [#489](https://github.com/mudler/vllm.cpp/issues/489)
**Lifecycle:** `ACTIVE`
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

Row is `ACTIVE` on `row/MODEL-GDN-OUTPUT-GATE-TYPE`. Implemented at `5da2e364`,
reviewed independently (design confirmed correct and complete: all 13 non-test
gated-RMSNorm constructions enumerated, no tail unwired, the fp8 `FusedRecipe`
value-copy proven unable to alias the constexpr original), then repaired for the
three findings that review returned. Owed before `DONE`: the operator's own
rerun of the row gate, the GPU arms this host cannot execute, and an `## Outcome`
section.

### What landed

`LoadHfConfig` resolves `output_gate_type` from the **resolved text config** and
canonicalizes it there — absent -> `silu`, `"swish"` -> `silu`, `"sigmoid"`
preserved, anything else refused naming the key and the accepted set. The three
Qwen3.5 GDN tails (`GdnBlock`, `GdnBlockPaged`, `GdnBlockPagedMixedSpec`)
consume the resolved boolean through `vt::RmsNormGatedArgs::sigmoid_gate`; the
fp8 glue-fused arm binds a value copy of `kRmsNormGatedQuantFp8` because
`sigmoid_gate` is a structural recipe flag.

Absent and present-but-unusable are **different states**. `getattr` substitutes
its default only for a missing attribute, so `"output_gate_type": null`, `""`,
and a non-string value reach upstream's assert and error; the loader probes for
the key rather than routing through `GetString`, which flattens absence and null
to the same empty string. The first implementation collapsed both into `silu`,
contradicting `docs/USAGE.md`, which already shipped the refusal contract.

The refusal is unconditional rather than gated on a GDN architecture, where
upstream's assert lives. No known checkpoint carries the key outside the GDN
family; if one appears, widening is a scoped follow-up with its own test.

### Evidence

**RED first, F1** (`test_hf_config`, before the probe replaced `GetString`) —
`1 failed | 19 skipped`, `assertions: 16 | 10 passed | 6 failed`,
`Status: FAILURE!`; every failure of the form
`CHECK_THROWS_WITH_AS( vllm::LoadHfConfig(f.path()), "output_gate_type",
std::runtime_error ) did NOT throw at all!` in the two new subcases (present
`null`, present `""`, flat and nested). Green after: `210 | 210 passed`.

**Polarity, the F2 repair.** The original numerics tests asserted only that the
two arms DIFFER. Review inverted the resolution
(`return cfg.output_gate_type != "sigmoid";` — a silu checkpoint driving the
sigmoid kernel, this row's own bug class reversed) and both focused suites
stayed GREEN. The repair adds a reference that never consults our gate:
silu(0) = 0·sigmoid(0) = 0 exactly while sigmoid(0) = 0.5, and the gate input is
`z = h @ in_proj_z` with no bias, so zeroing that projection makes a silu tail
annihilate the whole GDN block output while a sigmoid tail does not. The paged
cases assert the silu arm is exactly zero and the sigmoid arm is not; the dense
case asserts the silu arm is BIT-identical to an independently constructed model
whose GDN `out_proj` is zeroed, and that the sigmoid arm is not.

Re-running review's inversion now goes RED in both focused suites, both halves
flipping together:

- `test_qwen27_dense_forward` — `9 | 8 passed | 1 failed`,
  `assertions: 583 | 581 passed | 2 failed`, `Status: FAILURE!`;
  `CHECK( silu_differs == 0 )` reads `240 == 0`, `CHECK( sigmoid_gap > 0.0 )`
  reads `0 > 0`. The pre-existing "arms differ" case still passes with
  `max|diff| = 0.12374` — which is exactly why it could not see this.
- `test_qwen3_5_gdn_spec_routing` — `6 | 4 passed | 2 failed`,
  `assertions: 52 | 44 passed | 8 failed`, `Status: FAILURE!`;
  `CHECK( silu_nonzero == 0 )` reads `384 == 0` (`mixed=false`) and `640 == 0`
  (`mixed=true`) at both 27B and 35B GDN dims, with `max_sigmoid == 0`.

The mutated file was restored byte-for-byte (`md5 ee95ae2743...`, empty
`git diff`) and both suites returned to `583 | 583 passed` and `52 | 52 passed`.

**Inertness.** `tests/parity/goldens` is untouched — `git diff` against the
spec commit `3b99c1db` over that tree is empty. Rollup md5 over the qwen3*/
qwen36*/gdn* goldens: `886f4202f9e4fea2af611f1642f84a08`; individually
`qwen36_logits_27b f6b07d2df97f0ea6938202414e00a011`,
`qwen36_logits_35b 3a4d27ce010310c5cdb3435f59aebcad`,
`qwen36_gdn_layer_27b a86b3dbd8086f3684bb9bc04d51cdd32`,
`qwen36_gdn_layer_35b 1320c83388c6220426a660858d90322c`,
`qwen3coder_greedy 444895f5dc423427510251b1dcdad13e`.

**Not run here.** This host has no GPU: the CUDA/fp8 arms of all three tails,
the `FusedChain` fp8 recipe copy on device, and every dgx gate are UNRUN and
owed at the operator's rerun.
