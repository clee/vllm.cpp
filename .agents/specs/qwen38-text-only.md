# Qwen3.5/3.8 text-only arms: `Qwen3_5MoeForCausalLM`, `Qwen3_5ForCausalLM`

**Rows:** `MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm`,
`MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` (both NEW beyond-pin rows; this row
adds them to [`model-matrix.md`](../model-matrix.md) with the inventory counts
bumped, mirroring the `MuseGlimmer`/`KimiK3` beyond-pin precedent)
**Issue:** [#490](https://github.com/mudler/vllm.cpp/issues/490)
**Lifecycle:** `READY`
**Owner:** unassigned

## Scope

Load text-only checkpoints of the Qwen3.5-family GDN-hybrid backbone we already
run — the arms upstream calls `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`.
`Qwen/Qwen3.8-2.4T-A95B` is the motivating checkpoint; it is the same
architecture at larger scale, not a new one.

In scope:

- register the two text-only architecture strings against the existing dense and
  MoE factories;
- accept both the VL-prefixed (`model.language_model.`) and clean (`model.`)
  weight namespaces in the Qwen3.5 dense and MoE loaders;
- resolve a **flat** (non-nested, no `vision_config`) text config through the
  existing path;
- prove the 27B / 35B / Coder gates stay byte-identical.

Out of scope: any speed claim, any GGUF/quantized arm for 3.8, the vision tower
(a text-only checkpoint has none), MTP weights for 3.8, advancing the parity pin,
and **any support claim for the 2.4T checkpoint itself**, which this hardware
cannot execute (see Gates).

## Why this is not a new port

`config.json` for `Qwen/Qwen3.8-2.4T-A95B` declares `Qwen3_5MoeForCausalLM` /
`model_type: qwen3_5_moe_text`. Against Qwen3.6-35B-A3B, which we run token-exact
315/315, every structural knob is identical — `head_dim` 256,
`linear_key/value_head_dim` 128, `linear_num_key_heads` 16,
`full_attention_interval` 4, `attn_output_gate` true, `partial_rotary_factor`
0.25, `rope_theta` 1e7, `mtp_num_hidden_layers` 1, `linear_conv_kernel_dim` 4,
and `vocab_size` 248320 (the same tokenizer). The differences are scale only:
hidden 2048->8192, layers 40->92, attention heads 16->64, KV heads 2->4, linear
V-heads 32->128, experts 256->512, top-k 8->10, moe/shared intermediate
512->2048. All of these are read from config, not hardcoded
(`qwen3_5_common.cpp:40-47`; the only expert constraint is `num_experts > 0` at
`qwen3_5_weights.cpp:618`).

The 3.8 config also carries `output_gate_type: "swish"`, which normalizes to
silu. That key is handled by its own row (issue #489) and is not re-litigated
here.

## Upstream chain

| Upstream anchor | Contract to mirror |
|---|---|
| upstream `vllm/model_executor/models/registry.py:202-203` @ `ad5d29db7` | `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM` are registered text-only arms of the same `qwen3_5` module. |
| upstream `vllm/model_executor/models/qwen3_5.py:439-449` @ `ad5d29db7` | `Qwen3_5ForCausalLM` is the shared base unchanged; `Qwen3_5MoeForCausalLM` is that base plus the MoE hyperparameters — no separate backbone. |
| upstream `vllm/model_executor/models/qwen3_5.py:296-300` @ `ad5d29db7` | `WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})` — the canonical namespace is `model.`, and the VL-prefixed form is accepted and rewritten. |

**Ahead-of-pin, stated as such.** Our parity pin is `555967922`, whose registry
carries only the `ForConditionalGeneration` entries. The text-only arms arrived
upstream in **PR #50210 / `ad5d29db7`**, which is post-pin. This row is a
deliberate forward port of one upstream PR, not a mirror of the pin, and it does
not advance the pin or reconcile anything else in that range. That is visible
debt argued here and in the commit, not a silent divergence.

## Design

Weight-name evidence, read from the published indices of both checkpoints:

| | Qwen3.6-35B-A3B | Qwen3.8-2.4T-A95B |
|---|---|---|
| embed | `model.language_model.embed_tokens.weight` | `model.embed_tokens.weight` |
| layer | `model.language_model.layers.0.linear_attn.*` | `model.layers.0.linear_attn.*` |
| experts | `...mlp.experts.gate_up_proj` (3D stacked) | `...mlp.experts.gate_up_proj` (3D stacked) |
| shared | `...mlp.shared_expert_gate.weight` | `...mlp.shared_expert_gate.weight` |
| head | `lm_head.weight` | `lm_head.weight` |

The names are identical modulo the prefix — same 3D-stacked experts, same shared
expert gate, same top-level `lm_head`. So the loader body is already correct and
the only structural change is where it looks.

1. **One prefix decision, resolved once.** The Qwen3.5 loaders currently
   concatenate the literal `model.language_model.` in 4 places
   (`qwen3_5_weights.cpp:560,632,633,659`) and 3 more in
   `qwen3_5_dense_weights.cpp`. Replace the literal with a single resolved
   backbone prefix, chosen once per checkpoint by probing which namespace the
   shard index actually contains, then used everywhere. Mirrors upstream's single
   `WeightsMapper` rather than scattering a fallback into each lookup — a
   per-lookup fallback would let a checkpoint load half from one namespace and
   half from the other and still appear to succeed.
2. **Registration is additive.** Two `REGISTER_VLLM_MODEL` entries pointing at
   the existing dense and MoE factories. No factory, forward, or KV-cache change:
   `ModelRegistry::Resolve` is exact-match with no aliasing
   (`model_registry.cpp:217-231`), so the strings must be present literally.
3. **Config resolution already works.** `ResolveTextConfig` falls through to the
   top-level document when there is no `text_config`
   (`hf_config.cpp:113-122`), and `qwen3_5_moe_text` is already in
   `IsQwen35Family` (`:128-132`), so the `partial_rotary_factor` 0.25 default
   applies to a flat config. MRoPE is mm-path-only and every text caller passes
   `nullptr` (`qwen3_5.cpp:7540`), so a config without `mrope_section` is
   unaffected. Both facts get a test rather than an assumption.

## Risks

- **Regression on gated rows.** These loaders serve 27B/35B/Coder. A prefix bug
  breaks checkpoints we currently gate. Mitigated by byte-identical golden md5,
  not by a green suite.
- **Half-resolved namespace.** Probing per lookup instead of once could load a
  mixture. Mitigated by design point 1 and a test with a deliberately mixed
  index, which must be refused.
- **Untestable scale.** 92 layers / 512 experts is far past anything we can
  instantiate. Mitigated by testing config resolution and name mapping directly,
  and by *not* claiming the checkpoint runs.
- **Ahead-of-pin drift.** The forward-ported arm could diverge if upstream
  changes it before our next sync. Recorded in the porting inventory as
  ahead-of-pin so the next sync cycle reconciles it deliberately.

## Tests

1. Architecture dispatch: a flat Qwen3.8-shaped config resolves
   `Qwen3_5MoeForCausalLM` to the MoE registration, and `Qwen3_5ForCausalLM` to
   the dense one. RED first — today both raise unsupported.
2. Config resolution on the real 3.8 shape: flat doc, no `vision_config`, no
   `mrope_section`; assert the scale fields and the 0.25 rotary default.
3. Weight-name mapping: a clean (`model.`) index and a VL-prefixed
   (`model.language_model.`) index both resolve every expected backbone tensor
   name; a mixed index is refused.
4. Inertness: 27B/35B/Coder suites unchanged, golden md5 unchanged.

## Gates

- Focused: the targets above plus the Qwen3.5 dense/MoE suites.
- Full gate on the row before push.
- **The run gate is OWED and must be recorded as owed.** 2.4T bf16 is ~4.8 TB;
  the only other released variant is `Qwen/Qwen3.8-2.4T-A95B-FP8` at ~2.4 TB;
  GB10 has 128 GB unified and no smaller Qwen3.8 sibling exists. There is
  therefore **no token-exact oracle run for this checkpoint**, and the row may
  not reach `DONE` on the strength of dispatch and mapping tests. What this row
  can honestly claim is that the architecture is registered and the weight
  namespace resolves — nothing about generated tokens.

If a text-only checkpoint small enough to execute appears (any
`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` that fits GB10), that becomes the
run gate and closes this axis.

## Evidence required

- RED capture of the dispatch test before registration.
- Green focused + full gate after.
- Golden md5 before/after for 27B/35B/Coder showing no drift.
- The owed run gate recorded explicitly in the row and in `docs/STATUS.md`.

## Stop conditions

- If the prefix cannot be resolved once per checkpoint without touching the
  per-tensor lookup contract, stop and return `NEEDS_DECISION` rather than
  scattering fallbacks through the loader.
- If any 27B/35B/Coder golden md5 moves, stop — that is a regression on a gated
  row, and this row carries no evidence that could justify it.
- Do not implement an MTP arm, a quantized arm, or a GGUF arm for 3.8 on
  speculation; refuse them with a message naming the missing piece and record
  them as owed.

## Now

Both rows are `PARTIAL` (2026-08-12). Registration, the once-per-checkpoint
backbone-namespace resolution and the tests above are landed on
`row/MODEL-QWEN38-TEXT-ONLY`; full CPU gate green (396/396, 1 skipped:
`test_voxtral_e2e`, fixture absent) and `tests/parity/goldens` md5-unchanged.

**Next step is the OWED run gate, and nothing else advances these rows.** It
needs a text-only `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` checkpoint that
fits GB10; none exists today. Until one does, the honest claim stays "the
architecture is registered and the weight namespace resolves". Also owed, and
deliberately NOT implemented on speculation: the MTP, quantized and GGUF arms for
3.8.

## Outcome

**Measured.** Architecture dispatch for both strings, config resolution on the
flat 3.8 shape (scale fields plus the family's 0.25 partial-rotary default with
no `text_config`, `vision_config` or `mrope_section`), and weight-namespace
resolution on a clean index, a VL-prefixed index, a vision-inclusive VL index, an
index carrying `mtp.*`, a mixed index and an empty one. The strongest of these is
not a name-mapping assertion: two synthetic one-layer checkpoints with
byte-identical payloads and only the namespace differing load to byte-identical
weights through the production `LoadQwen3_5Dense`.

**Rejected.** A per-lookup namespace fallback — it would let a checkpoint bind
half its tensors from each namespace and still appear to load, which is exactly
the failure a name-mapping test cannot see. Also rejected: a blanket
"starts with `model.`" probe, because `model.visual.*` on a vision-inclusive 27B
checkpoint would have made it look like a flat text checkpoint and turned a
checkpoint we gate today into a refusal. Only the three structural backbone
spellings vote.

**Why the defaults are what they are.** The per-layer public seams
(`LoadQwen3_5MoeLayer`, `LoadQwen3_5DenseLayer`) default `backbone_prefix` to the
VL spelling, so every 27B/35B/Coder caller is byte-identical by construction
rather than by re-measurement. The text-only arms register with
`kQwen3_5TextInfo` (hybrid YES, multimodal NO) because upstream's
`Qwen3_5ForCausalLMBase` inherits `IsHybrid` but not `SupportsMultiModal`; the
`ForConditionalGeneration` wrappers remain the multimodal registrations.

**What was NOT established.** Any claim about generated tokens, memory or speed
for `Qwen/Qwen3.8-2.4T-A95B`. That checkpoint cannot be executed on this
hardware and was never run.
