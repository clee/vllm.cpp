# LTX-2.5 — `keyframes_abs_pos_embedding`, the module that refuses both shipped checkpoints

Row: `LTX25-KEYFRAMES-ABS-POS`
Issue: [#658](https://github.com/mudler/vllm.cpp/issues/658)
Campaign: [#644](https://github.com/mudler/vllm.cpp/issues/644)
Pin: `Lightricks/LTX-2 @ fd4ded7f`

## 0. What is wrong today — measured, not inferred

**Neither shipped LTX-2.5 DiT can be loaded inside the contract.** Both are
refused, from opposite directions, and both refusals are correct given that the
module is unported. Two renders on `dgx.casa` established this, and the
checkpoint headers were read directly to confirm why:

| checkpoint | header facts | refusal |
|---|---|---|
| `vonkaiser-fp8-nvfp4/.../ltx-2.5-22b-distilled-fp8.safetensors` | 6124 tensors, **no `__metadata__` at all**, carries `model.diffusion_model.keyframes_abs_pos_embedding` `F8_E4M3 [1,4096]` + `F32` scale | "carries modules this port does NOT carry: `keyframes_abs_pos_embedding`" |
| `lightricks-ltx-2.5/.../ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 7876 tensors, `__metadata__` present, **declares `use_keyframes_abs_pos_embedding=true`**, carries **no** keyframe tensor | `ltx2.cpp:192` — "`use_keyframes_abs_pos_embedding` is not ported … the checkpoint does not carry `keyframes_abs_pos_embedding`" |

So the only way to render from a real checkpoint today is
`allow_unported_modules`, which loads the ported subset. That is honest about
what it drops, but it means **every render we can currently produce is missing a
trained term** — see §1. "It renders with the escape hatch" is not the port
being complete.

**This is a mirror, not a product decision.** The reference defines the
behaviour exactly, so it gets ported; nothing here is escalated.

## 1. What upstream does, with anchors

**The parameter** — `model/transformer/model.py:215-219`:

```python
# Marks tokens whose latent encodes a single standalone pixel frame. Zero-initialized, so a
# checkpoint that predates it behaves identically until the parameter is trained.
self.keyframes_abs_pos_embedding = (
    torch.nn.Parameter(torch.zeros(1, self.inner_dim)) if self.use_keyframes_abs_pos_embedding else None
)
```

**The consumer** — `model/transformer/transformer_args.py:38-43`, called once at
`:269`:

```python
embedding = embedding_provider()
if embedding is None:
    return hidden_states
mask = (keyframes_mask > 0).to(dtype=hidden_states.dtype)
return hidden_states + mask * embedding.to(dtype=hidden_states.dtype)
```

**The mask** — `tools.py:184` plus `_first_frame_keyframes_mask` (`:186-195`)
marks the target's **first latent frame unconditionally**. Upstream's own comment
says so in terms: *"the reference implementation marks it unconditionally --
independently of whether any keyframe slots exist."*

**Therefore, on a trained checkpoint, upstream adds a learned `[1, inner_dim]`
per-token bias to every token of the first latent frame, on every forward,
whether or not a keyframe was supplied.** The module is ~three lines of maths; it
is small and it is unconditional, which is exactly why its absence is easy to
miss and expensive to leave.

**Do not port `enable_keyframes_abs_pos_embedding` / `supports_…`
(`model.py:166-173` and `:175-200`).** Both are **defined and never called**
anywhere in the checkout — one grep hit each, the definition, re-confirmed with a
positive control in the same command. Re-verify before relying on it; do not take
it from this spec. But **read `supports_…`**: §2 shows it returns `False` on the
NVFP4 arm both before and after the load, which is what settles that arm's
behaviour even though nothing calls it.

## 2. Why this is not "just a zero"

The zero-init comment above is what made this look inert, and it is the trap.
Two independent reasons it is not:

1. **On the FP8 DiT the tensor is trained and present.** `F8_E4M3 [1,4096]` with
   an `F32` scale is not `torch.zeros`.
2. **On the NVFP4 DiT the parameter is never materialised at all.** Upstream
   builds on the **meta device** (`loader/helpers.py:90-91`, `create_meta_model`)
   and loads with `assign=True` (`loader/single_gpu_model_builder.py:98`), so an
   absent key stays on `meta` — reading it raises. A reviewer executed this. It
   is not a zero that is harmlessly added; it is a parameter that does not exist.

The polarity does mean a *genuine* zero would be inert, since the term is
**added**, not multiplied. That is why the FP8 case is the one that changes
output.

**SETTLED BY EXECUTION on 2026-08-13, and the answer decides the NVFP4 arm.**
Two reviewers had disagreed — one read the parameter as *zero-initialized*
(`model.py:200`) and therefore "an exact no-op there"; the other **ran** it. Both
readings are consistent with the source; only one is consistent with what runs.
An implementer then executed upstream's own `create_meta_model`
(`loader/helpers.py:84-95`) against the real NVFP4 `__metadata__`, read through
upstream's own `read_model_metadata` / `SafetensorsModelStateDictLoader`
(`sft_loader.py:58-74`):

```
keys matching 'keyframes_abs_pos' in the file : []   (0 of 7876 entries)
config.transformer['use_keyframes_abs_pos_embedding'] = True
keyframes_abs_pos_embedding: shape=(1, 4096) f32 device=meta is_meta=True
supports_keyframes_abs_pos_embedding (BEFORE load) : False
after load_state_dict(sd, strict=False, assign=True):
  neighbour patchify_proj.weight : device=cpu is_meta=False   <- materialised
  keyframes_abs_pos_embedding    : device=meta is_meta=True
  in missing_keys                : True
  reading the value RAISES : RuntimeError: Tensor.item() cannot be called on meta tensors
supports_keyframes_abs_pos_embedding (AFTER load)  : False
```

The materialised **neighbour** is what makes this mean something: the loader did
run and did populate the model; only the absent key stayed on `meta`.

**Therefore, on the first-party NVFP4 DiT, upstream never reaches the add at
all** — `supports_keyframes_abs_pos_embedding` is False before *and* after the
load. So the correct mirror on that arm is **to load and apply nothing**, which
is neither a refusal nor an invented zero. Refusing it, as `ltx2.cpp:192` does
today, is stricter than upstream; synthesising a zero and adding it would be
inventing behaviour that upstream's own guard exists to prevent.

Two consequences for this row. **The NVFP4 arm's refusal is retired outright**,
not replaced by a no-op path with a fabricated tensor. And a render taken today
on the NVFP4 DiT with `allow_unported_modules` is, **for this module only**,
upstream-equivalent — the flag is declared, the tensor is absent, and upstream
would apply nothing either. That is a narrow claim about one module and must not
be restated as "the render is upstream-equivalent".

**The flag is not where you would look for it.** A raw read of `__metadata__`
returns `None`; it lives at `config.transformer`, which is why upstream's
JSON-decoding reader is needed to see it at all.

**Re-derive these anchors at HEAD before relying on them.** The guards are
`model.py:166-173` and `:175-200`, with the quoted phrases at `:170` and `:182`
— corrected from an earlier brief that said `167-182` / `175-193`.

**What is not in dispute, because it was measured on the bytes:** the vonkaiser
FP8 copy is **trained** — `F8_E4M3 [1,4096]`, **4096 of 4096 bytes non-zero**,
plus its `F32` scale. Whatever the NVFP4 arm turns out to do, that arm changes
output.

## 3. Scope

**In.**

1. Port the parameter and its application: load `keyframes_abs_pos_embedding`
   (and its scale on the quantized arm), and add `mask * embedding` at the same
   point upstream does.
2. Port the mask rule: the target's **first latent frame** is marked
   unconditionally, mirroring `_first_frame_keyframes_mask`.
3. Retire the two refusals — `ltx2.cpp:191-196` (flag declared, tensor absent)
   and the loader's "carries modules this port does NOT carry" entry — and
   remove the force-clear of the flag under `allow_unported_modules`
   (`ltx2_loader.cpp:974-981` / `:1019-1022`; re-derive these anchors at HEAD,
   they have moved twice this campaign).
4. Correct `include/vllm/model_executor/models/ltx2.h:47-49`, which says *"LTX-2.5's
   checkpoint does not carry the parameter"* — false for the FP8 DiT.

**Out.**

- Keyframe *conditioning* as a user-facing feature (supplying keyframe slots).
  This row is the unconditional first-frame bias only, which is what both
  checkpoints need in order to load at all.
- Any change to the image-conditioning arm's own behaviour beyond gaining the
  bias it should always have had. That arm is [#657](https://github.com/mudler/vllm.cpp/pull/657)/[#666](https://github.com/mudler/vllm.cpp/pull/666).

## 4. Memory format — check this explicitly

Upstream casts **both** operands to `hidden_states.dtype`
(`transformer_args.py:42-43`). There is no `f32` escape on this path. Mirror that
polarity: the bias is applied in the model dtype. **A token gate cannot catch a
dtype that is too wide** — it stays numerically correct while moving twice the
bytes — so check the memory format against the reference explicitly rather than
inferring it from a passing golden.

On the FP8 arm the tensor is `F8_E4M3` with an `F32` scale: dequantize per the
existing tower convention, and state which one you used.

## 5. Tests

RED-first, and the RED must be *the intended failure*, not a load refusal.

1. **A golden from upstream**, generated by executing `ltx_core` at the pin on a
   fixture with a known `keyframes_mask`, asserting the first latent frame's
   tokens differ from the rest by exactly the bias and that every other token is
   untouched.
2. **The unconditional rule**: a generation with **no keyframe supplied** must
   still mark the first latent frame. This is the half most likely to be ported
   as "only when keyframes exist", which would be silently wrong on every render.
3. **Both real checkpoints load without `allow_unported_modules`.** Env-gated on
   `LTX2_CHECKPOINT_ROOT`, and note in the report that CI does not set it — CI
   runs ~5.7% of `test_ltx2_video`'s assertions, so this test is host-local
   evidence, not a gate.
4. The two retired refusals must no longer fire, and the tests that asserted them
   must be **replaced by tests of the new behaviour**, never merely deleted.
   Deleting an assertion to turn a gate green is forbidden.

**Mutations that must be run and recorded:** zero the bias (golden must go RED);
apply the mask to the wrong frame (RED); make the mask conditional on a supplied
keyframe (RED — this is test 2's whole point); apply in `f32` and store back wide
(must be caught by the memory-format check, and if it is not, say so — that is a
finding about the gate, not a pass).

## 6. Risks

- **Every existing LTX-2.5 golden changes**, because the bias is unconditional
  and non-zero on the FP8 arm. That is the correct outcome, but it means goldens
  must be **regenerated from the generator against the pin**, with the
  regeneration itself proved reproducible, and the diff explained per value
  rather than accepted wholesale.
- **The FP8 arm is the only one whose output changes.** The NVFP4 arm is settled
  in §2 by execution: upstream never reaches the add there, so this row **retires
  that refusal and applies nothing**. The risk to guard is the tempting
  middle option — allocating a zero tensor because the code path wants one, and
  adding it. That is not a mirror; upstream's own `supports_…` guard exists
  precisely to stop a model reaching the add without a trained parameter.

## 7. Stop conditions

- If porting the mask changes token counts or phase geometry, stop: that is a
  different row.
- If the FP8 dequant convention for a `[1,4096]` + scalar-scale tensor does not
  already exist, stop and report rather than inventing a second convention.
- `dgx.casa` is required only for the two real-checkpoint loads, and it rebooted
  three times on 2026-08-13; treat GPU evidence as best-effort and gate the row
  on the fixture goldens.

## Now

`READY`. Spec committed ahead of implementation; a fresh implementer works from
this file, and a fresh reviewer — not the implementer — reviews the head.
