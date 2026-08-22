# LTX25-DIT-ATTN-FA2-HD128 — the tensor-core rung LTX-2.5 could not reach, because one template was never instantiated

Row: `LTX25-DIT-ATTN-FA2-HD128`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)), against the
model-matrix row `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`.
Issue: [#1551](https://github.com/mudler/vllm.cpp/issues/1551), filed by
[#1549](https://github.com/mudler/vllm.cpp/issues/1549) and carried under `##
Owed` in [`ltx25-dit-attn-flash.md`](ltx25-dit-attn-flash.md) until this row
existed.

## Now

`ACTIVE`. The gap is confirmed against the tree and the change is three product
edits plus one vendored translation unit. The A/B needs `dgx:gpu0` — the box
both prior LTX numbers were measured on — and no other box is a valid
denominator for it.

Every bare `file:line` anchor below is read at this row's base,
**`73ada0df8de72853a20f9f0e5b00e33b1ab02a6c`**, which is `origin/main` at the
claim. Anchors written `file::Symbol` survive the moves this change makes and
are what `scripts/check-symbol-anchors.py` gates.

## 0. The one-sentence finding

`vt::AttentionDenseFa2` is the only op in this tree that reaches a tensor core
for dense non-causal attention, and it refused every head dim but 64
(`src/vt/cuda/cuda_flash_attn_fa2.cu` `LaunchDenseFA2Bf16`, and the dispatch
test in `cuda_ops.cu::AttentionDenseFa2KernelCuda`), so LTX-2.5's head_dim-128
DiT could not enter it — not because the kernel is missing, but because one
explicit template instantiation was never compiled.

## 1. Where this row starts

[#1549](https://github.com/mudler/vllm.cpp/issues/1549) moved the LTX-2.5 DiT
self-attention off `vt::Attention` and onto `vt::AttentionDenseFlash`, and one
DiT forward at `768x448/49f` went from **47.84 s** (n=119, median 47.91 s) to
**7.680 s** (n=19, median). That row's own `## Owed` names what it left:

> **True tensor cores at head_dim 128.** `vt::AttentionDenseFa2` refuses
> anything but head_dim 64. Reaching the vendored FA-2 `mma.sync` path for LTX's
> head_dim 128 needs an extra `run_mha_fwd_<bfloat16_t, 128, false>`
> instantiation. That is explicitly out of scope for this row, and it is the
> difference between this fix and a materially larger one: everything below is
> still a scalar warp-per-query recurrence.

That is this row. `AttentionDenseFlash` removed the redundant global K/V traffic
— the naive kernel re-read K and V once per (query, head) — but it did not
change the arithmetic unit. It is still one warp per query walking the keys
through a dependent online-softmax chain, in scalar FMAs. No `mma.sync`
anywhere.

## 2. Why the gap was a build-list entry and not a kernel

This is the part worth writing down, because it makes the size of the change
much smaller than the issue's framing suggests and the reader should be able to
check that claim rather than take it.

The vendored FlashAttention-2 tree at `src/vt/cuda/flash_attn/` already carries
the complete head_dim-128 machinery:

- `src/vt/cuda/flash_attn/src/flash_fwd_launch_template.h::run_mha_fwd_hdim128`
  is upstream's own generic launcher, unmodified, and it has been in this tree
  since the vendoring.
- `flash_fwd_split_hdim128_bf16_sm80.cu` and its causal sibling already
  instantiate `run_mha_fwd_splitkv_dispatch<bfloat16_t, 128, {true,false}>` and
  ship on every build. That is the kernel Qwen3-dense decode runs. The kernel
  traits, the `mma.sync` bodies, the softmax and the epilogue at head_dim 128
  are therefore already compiled, already exercised, and already gated.

What was missing is the **non-split entry point** at that head dim:
`run_mha_fwd_<bfloat16_t, 128, false>`, upstream's plain batch `mha_fwd`. The
split-KV dispatch is what a PAGED caller needs; a dense, non-paged,
single-request attention wants the batch forward, which is also what vLLM
dispatches for this shape (`vllm/model_executor/models/whisper.py`
`WhisperEncoderAttention:255` -> `forward:298-317` ->
`flash_attn_varlen_func`, `causal=False`).

`flash_fwd_hdim64_bf16_sm80.cu` — added for the Whisper encoder — is the
precedent and says so in its own header: it is a three-line file asking the
existing generic launcher for one more `(Headdim, entry-point)` pair. This row
adds the same file at 128.

**One thing genuinely differs at 128, and it is the reason the head dim is worth
stating rather than assuming.** On a non-`sm8x` target `run_mha_fwd_hdim128`
selects `Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, T>`, whose
`kSmemSize` is over CUDA's default 48 KiB launch cap. That is **not** the #1544
problem. `run_flash_fwd` raises the cap itself with
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
immediately before the launch. The vendored path carries its own opt-in; the
scalar `LaunchAttentionDenseFlash` is the one that does not, which is exactly
what [#1578](https://github.com/mudler/vllm.cpp/pull/1578) narrowed its
advertised bound to. **The two head_dim domains are therefore not nested**, and
§6 records the one consequence.

## 3. Scope

**In scope.**

1. `src/vt/cuda/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu` — the
   `run_mha_fwd_<bfloat16_t, 128, false>` instantiation, and its entry in
   `_FA2_KERNEL_SRCS` in `CMakeLists.txt`.
2. `cuda_flash_attn_fa2.cu::LaunchDenseFA2Bf16` — admit head_dim 128, dispatch
   to the matching instantiation, and keep refusing everything else **by name**.
3. `cuda_ops.cu::AttentionDenseFa2KernelCuda` — widen the shape gate to
   `{64, 128}`; every other term (bf16, non-causal, MHA) is unchanged.
4. `ltx2_device.cpp` — the DiT self-attention calls `vt::AttentionDenseFa2`, and
   the A/B knob becomes three-way so each rung is separately runnable from one
   binary.
5. `include/vt/ops.h` — the op's advertised head_dim domain.
6. Tests: head_dim-128 parity, key-range, causal refusal, fall-through and A/B
   knob cases in `tests/vt/test_ops_attention_dense_fa2.cpp`; the ported
   upstream tolerance rule (§5); the reachability case in
   `tests/vllm/models/test_ltx2_device.cpp` retargeted and widened.

**Out of scope, and named rather than silently absent.**

- **The causal hd-128 non-split instantiation.** LTX's DiT self-attention is
  bidirectional and no causal non-split hd-128 caller exists. Adding it is one
  line; adding it now would be compiling a kernel nothing reaches, which is the
  failure `AGENTS.md` §"Nothing lands dead" names. `LaunchDenseFA2Bf16` refuses
  causal by name until one appears.
- **f32.** FA-2 has no f32 arm at all. f32 callers fall through, as they always
  have.
- **The other head dims.** 96, 160, 192 and 256 have upstream launchers and no
  caller here. Same reason.
- **[#1552](https://github.com/mudler/vllm.cpp/issues/1552)**, the sweep of the
  remaining `vt::Attention` call sites. Different defect shape, its own row.

## 4. The dispatch, and why the model asks for the op unconditionally

`ltx2_device.cpp` calls `vt::AttentionDenseFa2` with **no shape test of its
own**. That is deliberate and is the op's own contract: it is TOTAL. Its fast
path is bf16 / head_dim in {64,128} / non-causal / MHA with the vendored kernels
compiled, and every other shape falls through to `AttentionDenseFlash`
bit-exactly. Four fall-through cases in
`tests/vt/test_ops_attention_dense_fa2.cpp` hold that, and this row adds two
more at head_dim 128.

Writing a shape test at the call site would be a **second copy of the op's
domain**, and the copy is what goes stale when the instantiation set moves. The
model states what it wants — the fastest correct dense non-causal attention —
and the op resolves it.

Both LTX streams are inside the fast path at the production dtype: video is 32
heads x 128, audio is 32 heads x 64, both bf16, both non-causal, both `h_k ==
h`. The f32 parity arm and the whole CPU backend take the fall-through and are
unchanged.

**The A/B knob is three-way**, because a two-arm knob cannot measure a
three-rung ladder, and because each arm must be able to state which rung it ran
from its own log rather than from its timing — the timing being the quantity
under measurement:

| `VLLM_LTX2_DIT_FLASH_ATTN` | op called | `VT_OP_PROVIDER_STATS` id |
|---|---|---|
| unset (default) | `vt::AttentionDenseFa2` | `kAttentionDenseFa2` |
| `flash` | `vt::AttentionDenseFlash` | `kAttentionDenseFlash` |
| `0` | `vt::Attention` | `kAttention` |

`=0` keeps exactly the meaning #1549 gave it and `docs/ENVIRONMENT.md` records,
so the 47.84 s denominator stays reachable from this binary. `flash` is the arm
this row's ratio is taken against.

The alternative — reusing the global `VT_FA2_DENSE=0` for the flash arm — was
rejected. It works, but both arms then resolve `kAttentionDenseFa2` and the
op-provider log cannot tell them apart, so the only evidence of which kernel ran
would be the wall clock. Three ops give three ids.

## 5. Numerics: the gate is upstream's rule, not a bound fitted to the result

FA-2 is **not** bit-identical to `AttentionDenseFlash`, and this row does not
pretend otherwise. `mma.sync` reassociates both the QK^T and the P.V reductions,
and the vendored kernel exponentiates with `exp2f` on a log2-scaled score where
the scalar kernels use `expf` on a linearly-scaled one. A diffusion model has no
token gate to fall back on, so the deviation has to be bounded directly.

**Two comparisons, and only one of them is a gate.**

**(a) The arm-to-arm rel-L2, against the ALREADY-COMMITTED bound.**
`kRelL2Bound = 1.0e-2` and `kMaxAbsVsRmsBound = 0.15` are in
`tests/vt/test_ops_attention_dense_fa2.cpp` at this row's base and were derived
for the hd-64 case from the **output dtype**: bf16's relative resolution is
`2^-8 = 3.9e-3`, so 1e-2 is about 2.5 bf16 ulps. That is a property of the store
width, not of a head dim, a sequence length or a measurement. This row reuses
those constants unchanged at head_dim 128, at LTX's own geometry (T=2352, H=32,
D=128) and at two ragged shapes. **Widening either constant for this head dim
would be fitting the bound to the result and is a stop condition, not a repair.**

**(b) The PORTED upstream tolerance rule, which is the actual gate.** (a) tells
you how far the two arms diverge and nothing about which is right; both could be
wrong together. Upstream FA-2 answers that question with a rule stated against a
more accurate reference — `tests/test_flash_attn.py::test_flash_attn_output`
(`vllm-project/flash-attention @ 2c839c33`):

```python
assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
```

`out_ref` is the higher-precision reference, `out_pt` the reference kernel at
the tested dtype, `out` the FA-2 result. **Only the harness is adapted**:
`out_ref` is a `double` host reference computed in the test file, because
porting torch's fp32 SDPA is not possible here, and `out_pt` is our own
`AttentionDenseFlash` rather than torch's bf16 SDPA. **The factor of 2 and the
max-abs statistic are upstream's and are not re-derived.**

The rule is run at LTX's production **sequence length** (T=2352) with a reduced
head count (H=2), and that reduction is stated rather than hidden: the reference
is `O(T^2 * D)` in scalar double per head, heads are independent in this op, and
the sequence length is the axis the reduction order actually accumulates over.
Reducing T instead would reduce the thing being measured.

The case also `REQUIRE`s that the incumbent's own error against the reference is
non-zero, because otherwise the rule reads `x <= 0` and is a mute switch wearing
a strict gate.

**What neither comparison bounds, and this row inherits the gap rather than
closing it by assertion.** Both are per-op comparisons at one attention. Neither
bounds the accumulated deviation over 48 blocks x 120 forwards of a denoise
trajectory, and no pixel comparison of an LTX render against an alternative
attention rung exists anywhere in this tree —
[#1612](https://github.com/mudler/vllm.cpp/issues/1612) already owes that for
#1549's swap and this row does not discharge it. §8 records what was attempted.

## 6. Risks

- **The advertised-domain asymmetry.** The FA-2 arm has no 48 KiB bound (the
  vendored launcher opts in for itself); the fall-through arm does. bf16
  head_dim 128 is inside both. **f32 head_dim 128 is inside neither** — f32 is
  not an FA-2 shape and its `AttentionDenseFlash` tile is 65,536 B — so it
  reaches that op's named refusal. That is the LTX f32 parity arm at production
  geometry, which #1549 already disclosed and #1612 already owns. This row does
  not change it and does not silently paper over it.
- **Compile time and binary size.** One more `run_flash_fwd` instantiation,
  with the same `BOOL_SWITCH` fan-out every other vendored TU already pays.
  Measured in §8.
- **A partial revert.** One stream moved back to flash while the other stays on
  FA-2 would leave the video stream — the whole point of this row — without
  tensor cores. The reachability case's negative half now names
  `kAttentionDenseFlash` as well as `kAttention` for exactly this.

## 7. Tests and gates

| Gate | What it holds |
|---|---|
| `test_ops_attention_dense_fa2` hd-128 parity | FA-2 vs flash at T=2352/H=32/D=128 and two ragged shapes, against the base's own committed bounds |
| `test_ops_attention_dense_fa2` upstream rule | §5(b): `max\|fa2-ref\| <= 2 * max\|flash-ref\|` against a `double` host reference |
| `test_ops_attention_dense_fa2` hd-128 key range | perturbing V's tail must move the output; a clamped `seqlen_k` or a mis-stride goes red |
| `test_ops_attention_dense_fa2` hd-128 causal | a causal request must LEAVE the op; kills a gate written `(d==64 && !causal) \|\| d==128` |
| `test_ops_attention_dense_fa2` fall-through | head_dim 80 AND 192 must fall through bit-exactly; {64,128} is a set, not an interval |
| `test_ops_attention_dense_fa2` `VT_FA2_DENSE=0` at hd-128 | the same-binary rollback really routes back, and the ON arm is NOT that same answer |
| `test_ltx2_device` reachability | through `Ltx2DitForwardDevice`: `kAttentionDenseFa2` selected exactly `2 * layers * batch`, `kAttentionDenseFlash` 0, `kAttention` 0 |
| A/B on `dgx:gpu0` | per-forward median at `768x448/49f`, fa2 arm against flash arm, one binary, from the engine's own `last=` lines |

**Reachability mutation.** Replace `vt::AttentionDenseFa2` with
`vt::AttentionDenseFlash` at `ltx2_device.cpp`'s self-attention branch in a
scratch copy and rerun `test_ltx2_device`. The case must go red on the positive
half. A gate that stays green without the call site measures a class, not a
capability.

## 8. Evidence

Recorded when measured. Nothing is written here that was not run.

## 9. Stop conditions

- **The numeric gate cannot be met at the committed bounds.** Report the
  measured deviation and stop. Do not widen `kRelL2Bound`, do not widen
  upstream's factor of 2, and do not reduce the geometry until it passes.
- **The A/B cannot run on `dgx:gpu0`.** A different box is not a valid
  denominator for the 7.680 s flash number. Record the speed axis `PENDING` with
  the reason and land the correctness half, or hold the row.
- **The instantiation does not build for `sm_121a`.** Report the compiler
  output. Do not disable a guard to get past it.

## Owed

- **The pixel comparison of a full render across attention rungs.** Inherited
  from #1549 and owned by
  [#1612](https://github.com/mudler/vllm.cpp/issues/1612), not by this row. This
  row's numeric evidence is per-op and does not bound a 120-forward denoise
  trajectory.

- **The distilled NVFP4 DiT's recorded revision disagrees with its own download
  sidecar.** `docs/USAGE.md` pins
  `Lightricks/LTX-2.5 @ 6c7e5e573ac1667efc83407806fe9b0b93730e60` for
  `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`, while
  that file's `huggingface_hub` `.metadata` sidecar on the shared checkout
  records `8a4ff96f581e72bedc1b44367581c49d544a05f1`. Both can be true — a later
  re-download explains it, and the two bf16 DiT rows have no sidecar at all, so
  nothing local contradicts them. Found while adding the three missing LTX-2.5
  rows and deliberately NOT folded in: choosing a revision without knowing which
  bytes that row's author measured replaces a possibly-stale pin with a
  definitely-unverified one, which is worse. This row does not run that model
  arm and has no way to re-derive it. Owner: this row.
  Issue: [#1702](https://github.com/mudler/vllm.cpp/issues/1702).

## Outcome

Recorded when the row reaches `DONE`.
