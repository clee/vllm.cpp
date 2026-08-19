# Per-op bf16 rounding polarity in the `vt` elementwise seam

Issues: [#1322](https://github.com/mudler/vllm.cpp/issues/1322) (the filed gap),
[#1342](https://github.com/mudler/vllm.cpp/issues/1342) (Vulkan/Metal silu
spelling), [#1343](https://github.com/mudler/vllm.cpp/issues/1343)
(`RmsNormPlusAdd` arm asymmetry).
Row: `VT-ACT-ROUND-POLARITY`.

## Now

`SPIKE`. No product code changed. [#1322](https://github.com/mudler/vllm.cpp/issues/1322)
names two ops as one defect; the pinned primary oracle splits them. One is a
real divergence that upstream pins **bit-exactly**. The other is not a
divergence at all — upstream made exactly the proposed edit, called it a bugfix,
and **reverted it three weeks later**, and the reverted-to form is what the pin
carries and what `vt` already computes.

This spec records the ground truth, scopes the half that is real, and stops.
The implementing wave is a separate row because the change spans six providers
and every bf16 gated multilayer perceptron in the tree, and neither of those can
be gated from the shared checkout.

## Gap verification

Checked before writing anything, because several changes this session were found
already landed.

| surface | result |
|---|---|
| `git log --grep 1322` | only `94e740514`, a Music3 spec commit that *cites* the issue; no implementation |
| open pull requests | none touches `vt::RmsNorm`, `vt::SiluAndMul` or their providers |
| `origin/fix/fused-chain-interp-rmsnorm` | ancestor of `origin/main`; its RmsNorm content is a `-ffp-contract=off` build pin, not a polarity change |
| `origin/perf/rmsnorm-fp4` | ancestor of `origin/main` |
| `origin/row/BACKEND-VULKAN-RMSNORM` | PR [#185](https://github.com/mudler/vllm.cpp/pull/185), **MERGED** — a decode workgroup-shape speedup (7.85 → 1.59 ms/token), not a polarity change. Not an ancestor by sha because it was squash-merged, so the pull request state is the authority, not `--is-ancestor` |

Nothing to reconcile. The gap is real and unowned.

## Scope

**In scope for the implementing row.** The gated activation family —
`vt::SiluAndMul`, `vt::GeluAndMul`, `vt::MoeSiluMul`, and the SwiGLU epilogues
of `kMoeGateUpSwiGLUGrouped` and `kMoeGroupedGemmBf16GateUpSilu` — rounds
nothing between the activation and the multiply. Upstream rounds `act(gate)` to
the input dtype first, in both of the implementations it ships for accelerators,
and asserts the two agree bit-exactly.

**Out of scope, with evidence.** `vt::RmsNorm`, `vt::RmsNormGated` and the
quant-fused siblings. See `## The refuted half`.

**Not touched.** `.agents/specs/minimax-music3.md`, and the files of the two
Music3 pull requests in repair
([#1330](https://github.com/mudler/vllm.cpp/pull/1330),
[#1328](https://github.com/mudler/vllm.cpp/pull/1328)). #1330's tolerance rests
on this divergence existing; the activation change would move part of it, which
is a fact for that row's owner and not an edit this row may make.

## The refuted half — `vt::RmsNorm` is already faithful

The dispatch anchors the obligation on a reference that rounds `x*inv` to the
weight dtype before the affine multiply. That reference exists. It is not the
one that runs.

At the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, upstream carries
two RMSNorm implementations with **different** polarity:

| upstream implementation | final expression | polarity |
|---|---|---|
| `vllm/ir/ops/layernorm.py::rms_norm` (native) | `x = x.to(weight.dtype) * weight` | rounds first |
| `vllm/ir/ops/layernorm.py::fused_add_rms_norm` (native) | `x = x.to(weight.dtype) * weight` | rounds first |
| `csrc/libtorch_stable/layernorm_kernels.cu::rms_norm_kernel` | `dst.val[j] = static_cast<scalar_t>(x * s_variance * w)` | **f32, one rounding** |
| `csrc/libtorch_stable/layernorm_kernels.cu::fused_add_rms_norm_kernel` (both specializations) | `Converter::convert(x * s_variance * wf)`, `(scalar_t)(x * s_variance * w)` | **f32, one rounding** |
| `csrc/cpu/layernorm.cpp::rms_norm_impl` | `fp32_out = fp32_x * fp32_s_variance * fp32_w;` | **f32, one rounding** |

`vllm/kernels/vllm_c.py::rms_no_var_size` selects the C kernel whenever
`variance_size` is unset **and `weight.dtype == x.dtype`**. A bf16 model on a
CUDA-alike device therefore always executes the f32 form, and so does the CPU
backend. The native form is reached only when the dtypes differ — which is
`GemmaRMSNorm`, whose weight is `self.weight.float() + 1.0`, so there the
cast-back is a cast to f32 and rounds nothing at all.
`layernorm.py::GemmaRMSNorm`'s own docstring names this as difference 2 from the
standard class.

**Upstream made the proposed change and reverted it.**

| sha | date | subject | direction |
|---|---|---|---|
| `4d51588e2` | 2026-04-26 | `[Feat] DeepSeek V4 Rebased (#40860)` | round-first → f32-across-weight, unremarked inside a large feature commit |
| `124fac10c` | 2026-05-29 | `[Bugfix] Fix RMSNorm kernels to multiply in weight's native dtype (#42379)` | f32-across-weight → round-first, in `layernorm_kernels.cu` **and** `layernorm_quant_kernels.cu` — precisely the edit this row was dispatched to make |
| `225936a1d` | 2026-06-18 | `[CI Bug] Revert #42379 to fix CI \`Multi-Modal Models (Extended Generation 1)\` (#46070)` | back to f32-across-weight |

All three are ancestors of the pin, and the reverted-to state is what the pin
carries. Three later commits touch the file and none alters the store
expression.

Upstream's own kernel test `tests/kernels/core/test_layernorm.py::test_rms_norm`
compares `forward_native` against the kernel at `atol=1e-2, rtol=1e-2` and never
asserts they agree exactly, so upstream does not treat the two polarities as one
behaviour.

`vt::RmsNorm` computes `v * inv * wj` in f32 and stores once, on every provider.
That is the executed upstream kernel. Making it round first would move us
**away** from the pinned oracle onto a form upstream removed from its own tree:
no golden captured from a running vLLM would improve, and the bf16 ones would
get worse.

**The honest reading of the history.** The f32 form arrived as a side effect,
was labelled a bug, and survives only because reverting unbroke a CI job. It is
current and deliberate-by-revert, and it disagrees with upstream's own Python
reference, which was never changed to match. That makes it a candidate to
re-examine at the next pin advance — it does not make it something to diverge
from now. The rule is mirror the pin, and `vt` mirrors it.

## The measured half — the gated activations round `act(gate)`

Upstream has one polarity here on every accelerator path, and pins it.

| upstream implementation | final expression | rounds `act(gate)`? |
|---|---|---|
| `activation.py::SiluAndMul.forward_native` | `F.silu(x[..., :d]) * x[..., d:]` | yes — `F.silu` on a bf16 tensor yields bf16 |
| `activation_kernels.cu::silu_kernel` | `return (T)(((float)x) / (1.0f + expf((float)-x * alpha)))` | yes — explicit narrowing to `T` |
| `activation_kernels.cu::compute` | `return (scalar_t)(ACT_FN(gate, alpha) * ((float)up + beta))` | yes — `ACT_FN` already returned `scalar_t` |
| `activation_kernels.cu::packed_silu_kernel` | `return cast_to_packed<packed_t>(fval)` | yes — the vectorized path narrows identically |
| `activation.py::GeluAndMul.forward_native` | `F.gelu(x[..., :d], approximate=…) * x[..., d:]` | yes |
| `activation_kernels.cu::gelu_kernel` / `gelu_tanh_kernel` | `return (T)(…)` | yes |

The fused mixture-of-experts path is the same kernel, not a second one:
`fused_moe.py::fused_experts_impl` calls `activation.py::apply_moe_activation`,
which calls `torch.ops._C.silu_and_mul`.

**Upstream asserts this bit-exactly.**
`tests/kernels/core/test_activation.py::test_act_and_mul` runs the CUDA kernel
and `forward_native` over the same input and, for `silu_and_mul`,
`mul_and_silu`, `gelu`, `gelu_tanh` and `fatrelu`, asserts

```python
torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)
```

with the comment that these implementations "are equivalent to the native
PyTorch implementations, so we can do exact comparison". Only
`swigluoai_and_mul` and `swiglustep_and_mul` get a tolerance. So the rounding of
`act(gate)` is not an artefact of one kernel — it is a pinned contract between
two implementations, and it is the one upstream test in this area a port can
reproduce exactly. That test is the smallest failing test the implementing row
should port.

`src/vt/cuda/cuda_ops.cu` already carries the standing note `// Upstream csrc
counterpart: csrc/activation_kernels.cu (act_and_mul_kernel<silu>) — align
post-MVP`, so the divergence was known when the kernel landed and deferred, not
decided.

**One upstream implementation disagrees, and it is nearly dead.**
`csrc/cpu/activation.cpp::activation_kernel` and
`csrc/cpu/cpu_fused_moe.cpp::silu_and_mul` keep silu in f32 and round once, as
`vt` does. But `SiluAndMul.forward_cpu` returns `forward_native` on every CPU
architecture except POWERPC, so that kernel is reached only there. It is not the
mirror source.

**The rounding target is the INPUT dtype, not the output dtype.** Upstream's
`scalar_t` is the dtype of `x`, and `SiluAndMul.forward_cuda` allocates `out`
with `dtype=x.dtype`, so upstream never has the two differ. Our seam permits an
f32 input with a bf16 output. Mirroring upstream means rounding through the
**input** dtype, which leaves every f32-input path bit-identical to today and
confines the change to bf16-in paths. That is a property to assert directly, not
to infer from values: it is what keeps the existing f32 goldens in
`tests/vt/test_ops_activation.cpp` unmoved, and a token gate cannot see it.

**Size of the effect, as an analysis figure and not a gate.** Over 2^20 pairs of
independently bf16-rounded standard-normal `(gate, up)`, replicating the tree's
round-to-nearest-even bf16 codec: **27.47% of outputs differ, every one of them
by exactly 1 bf16 unit in the last place**. So the change is pervasive and
uniformly small — which is the profile that a token gate is least able to see
and that accumulates across layers.

## The `vt` inventory

Every provider of the RmsNorm family keeps f32 across the weight multiply, so
the family is internally consistent and consistent with the pin:

| op | providers | polarity | verdict |
|---|---|---|---|
| `kRmsNorm` | CPU, CUDA (3 kernels), ROCm, Vulkan, Metal, Tenstorrent host | f32, one rounding | matches the pin — no change |
| `kRmsNormGated` | CPU, CUDA (2), ROCm, Vulkan | f32, one rounding | matches `RMSNormGated.forward_static` and its triton kernel, both all-f32 — no change |
| `kRmsNormQuantFp8`, `kRmsNormGatedQuantFp8` | CPU, CUDA | bf16 hop before the fp8 quant | the stated contract (`ops.h`), and it matches `layernorm_quant_kernels.cu` at the pin — no change |
| `kRmsNormGatedGroup` | CPU, CUDA | **rounds `x*inv` to `input_dtype` before `*weight`** | already the round-first polarity, deliberately mirroring `mamba_mixer2.py::Mixer2RMSNormGated` — no change |

`kRmsNormGatedGroup` is the important one for the framing of #1322. The issue
says the seam "has no way to express the other behaviour". It does: `cpu_ops.cpp`
already has `RoundThrough(DType, float)` and `cuda_mamba2_ssd.cuh` already has
`M2RoundThrough`, and one op already uses them for exactly this purpose. What is
missing is not an expressive mechanism but a general elementwise
tensor-times-tensor multiply, and the activation fix does not need one — it needs
one `RoundThrough` call inside each existing kernel.

The gated activations, which do need to change:

| op | providers | polarity today |
|---|---|---|
| `kSiluAndMul` | CPU, CUDA, ROCm, Vulkan, Metal | f32, one rounding — **diverges** |
| `kSiluAndMul` | Tenstorrent | `ttnn::silu` materializes a bf16 tile before `ttnn::multiply` — **already rounds**, so the providers are *already inconsistent with each other* |
| `kGeluAndMul` | CPU, CUDA, ROCm (2) | f32, one rounding — **diverges** |
| `kMoeSiluMul` | CPU, CUDA, ROCm | f32, one rounding — **diverges** |
| `kMoeGateUpSwiGLUGrouped` | CPU, CUDA | f32 SwiGLU epilogue into an f32 output — diverges only where the output is narrow |
| `kMoeGroupedGemmBf16GateUpSilu` | CUDA | f32 SwiGLU epilogue — **diverges** |
| `kSiluMulFp4Quant`, `kSiluAndMulFp4Quant` | CPU, CUDA | bf16 hop on the **product**, by contract; the `silu` itself is unrounded — **diverges in the same way**, and the contract comment needs re-reading against upstream when it is fixed |

## Reachability

`vt::SiluAndMul` is reached from a production entry point on the default
configuration through the shared multilayer-perceptron seam:
`include/vllm/model_executor/layers/linear.h::UnquantizedMlpGateUpMethod`
matmuls into a bf16 `[M,2I]` buffer and calls `vt::SiluAndMul(d.q, act.t(),
gate_up.t())` with both tensors bf16. Sixteen further production call sites exist
across `qwen3_5.cpp`, `qwen3_dflash.cpp`, `laguna.cpp`, `dense_nvfp4_gemm.h`,
`minimax_h3_device.cpp`, `minimax_h3_encoder_device.cpp`,
`minimax_h3_video_vae_device.cpp` and `minimax_music3_device.cpp`. There is no
unreached-slice question here. The question is the opposite one, and it is why
this is not a small change.

## Blast radius

Every bf16 SwiGLU or GeGLU multilayer perceptron in the tree changes value: the
dense path of every registered text model, the shared expert of every
mixture-of-experts model, and the H3 and Music3 device arms. Goldens captured
from a running vLLM should improve; goldens captured from our own output shift
and each shift has to be explained rather than absorbed.

Six providers move together or not at all, because `AGENTS.md` forbids a
hand-written parallel path and holds the providers to being consistent with each
other. Four of them — CUDA, ROCm, Metal, Tenstorrent — cannot be executed from
the shared checkout at all, and the CPU arm needs an idle box.

That is why this spec stops here.

## Owed

- The implementing row: the six-provider edit, the ported form of upstream's
  `test_act_and_mul` exactness assertion as the red-first test, and the full
  gate. Owns [#1322](https://github.com/mudler/vllm.cpp/issues/1322).
- [#1342](https://github.com/mudler/vllm.cpp/issues/1342): Vulkan
  `vt_silu_and_mul.comp` and Metal `metal_msl.h` both document a 1:1 port of the
  CPU kernel's `gate / (1 + exp(-gate))` spelling and both emit
  `gate * vt_sigmoid(gate) * up` instead. Found here, not fixed here.
- [#1343](https://github.com/mudler/vllm.cpp/issues/1343):
  `src/vt/fused_ops.cpp::RmsNormPlusAdd` rounds `out` between the norm and the
  add on its composed arm and does not on its ROCm arm, so one API has two
  numeric behaviours. Found here, not fixed here.
- The `include/vt/ops.h` doc comment on `vt::RmsNorm` says "unlike upstream
  `forward_native`", which reads as a divergence and is the reason this gap was
  filed against that op. It should say that the kernel which actually runs
  agrees with us. Deferred with the implementing row rather than taken here,
  because editing that header rebuilds the tree and this row lands no code.

## Stop conditions

- A golden that gets **worse** after the activation change stops the row and is
  reported, not absorbed. It means either the change is wrong or that golden was
  captured from our own output.
- No tolerance is widened to make a golden pass. A golden that must be re-taken
  is re-taken from the pinned oracle, with its provenance recorded.
- `vt::RmsNorm` is not touched without a new pin and a fresh reading of
  `layernorm_kernels.cu`. If a later pin restores vllm#42379, that is a pin
  advance with its own reconciliation, not a licence to pre-empt it.

## Outcome

Deferred to the implementing row. What this row establishes:

- The RmsNorm half of #1322 is **refuted** against the primary oracle at the
  pin, with the upstream revert as the decisive evidence. Nobody needs to derive
  it again.
- The activation half is **confirmed**, with an upstream bit-exact test to port
  and a measured 27.47%-of-elements, 1-ULP effect.
- The seam's expressive gap named in #1322 is narrower than filed: the rounding
  helper already exists on two providers and one op already uses it. A general
  elementwise multiply is not required for this fix.
