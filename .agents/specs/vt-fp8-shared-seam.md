# VT-FP8-SHARED-SEAM — the FP8 W8A8 linear path becomes a shared seam

Issue: [#940](https://github.com/mudler/vllm.cpp/issues/940).
Owning row: `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` (the row that forces it; [#517](https://github.com/mudler/vllm.cpp/issues/517),
spec [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md)).
Related: [`vt-fp8-w8a8-cpu-arm.md`](vt-fp8-w8a8-cpu-arm.md) — #468/#842's op-tier
registrations, whose §Residual gap names exactly the model-tier half this row
closes; [`perf-fp8-alpha-fold.md`](perf-fp8-alpha-fold.md) — the folded-alpha
lever whose arithmetic this seam carries unchanged.

AGENTS.md §"Shared seams": *"If a shared seam cannot represent the upstream
behavior, extend it or record one exact tracked exception. Never hand-roll a
parallel path."*

NVFP4 already honours that:
`include/vllm/model_executor/models/dense_nvfp4_gemm.h` is a real seam with a
policy layer at `compressed_tensors/schemes/nvfp4.h`. FP8 W8A8 did not. The
weight struct `Fp8Weight` was in a header
(`qwen3_5_weights.h:273 @ c7cb59fbb`), but everything that made it *usable* sat
inside one translation unit:

| Entry point | Was (`qwen3_5.cpp @ c7cb59fbb`) |
|---|---|
| `ResidentFp8` | `:1458` |
| `DenseCublasLtFp8Enabled` | `:1478` |
| `MatmulFp8CutlassD` | `:1495` (CUDA guard `:1497-1498`) |
| `MatmulFp8CutlassPreQuantD` | `:1517` (CUDA guard `:1519-1520`) |

A second model therefore had three options and the policy forbids two: include a
`.cpp`, copy the entry points, or extend the seam. This row extends it.

## Scope

**IN.**

1. `include/vllm/model_executor/models/dense_fp8_gemm.h` — the four definitions
   above, moved VERBATIM, mirroring `dense_nvfp4_gemm.h`'s shape (preamble with
   the upstream chain, `namespace vllm::dense_fp8`, header-only, the CUDA guard
   travelling with the code it guards).
2. `include/vllm/model_executor/layers/quantization/fp8.h` — `Fp8W8A8LinearMethod`
   + `MakeLinearMethod(const OwnedTensor&, const Fp8Weight&)`, mirroring
   `compressed_tensors/schemes/nvfp4.h`'s `Nvfp4W4A16LinearMethod` +
   `MakeLinearMethod`.
3. `src/vllm/model_executor/models/qwen3_5.cpp` — calls the extracted seam. The
   ~14 call sites are UNCHANGED text; what changes is that the three names they
   call are now two one-line type adapters plus a `using`.
4. `tests/vllm/model_executor/layers/test_linear_method.cpp` — the two CPU cases
   that make the policy layer's selection and its reach into the seam
   falsifiable.

**OUT.**

- **Wiring NemotronH to this seam.** That is A2-Q under #517, and it is what the
  seam exists for, but a model port inside an extraction would make the
  byte-identity gate below unreadable.
- **Any numerics, tolerance, guard or dispatch condition.** Specifically the
  CUDA-only refusal keyed on `kMatmulFp8CublasLt` is carried across unchanged —
  see §Residual gap.
- `ResidentFp8Qkv` / `ResidentFp8Qkvz` and the merged-QKV(z) fp8 path. They are
  35B-specific merged-operand builders, not the general linear seam, and #940
  does not name them.
- The missing device-upload accounting recorded under §Found, not fixed.

### Why the seam is templated on `Dev`/`DBuf`

`qwen3_5.cpp` carries its own anonymous-namespace `Dev`/`DBuf` — the KNOWN
DUPLICATION `dense_nvfp4_gemm.h:45-50` records, deliberately, because unifying
the device-glue families is a separate gate-model-touching refactor. Those types
are layout-identical to `dense_attn::Dev`/`DBuf` but are *distinct types*, so a
non-template header could only have been COPIED into qwen3_5.cpp, not called by
it.

That copy is the failure mode #940 exists to prevent: a seam sitting dead beside
the production path, where "byte-identical" is vacuously true because nothing
routed through it. Templating on the two glue types instead gives ONE definition
with two instantiations — qwen3_5.cpp instantiates it with its own types
(generating the code it had), the `vllm::layers` policy layer instantiates it
with the shared ones. The mutation table below is what turns that from a claim
into a measurement.

## Upstream chain

Our loader (`LoadFp8Raw`, `qwen3_5_weights.cpp:423`) accepts both the
compressed-tensors and the ModelOpt spelling of the same per-tensor FP8 W8A8
checkpoint and reduces them to one `Fp8Weight`. Upstream, all three entry points
delegate to the same shared `fp8_linear.apply_weights`, which is why one method
here covers every spelling:

| Concern | Upstream (pin `555967922`) |
|---|---|
| compressed-tensors scheme | `compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:60,201-207` |
| ModelOpt scheme | `modelopt.py:444,531-537` |
| generic fp8 linear | `fp8.py:267,446` |
| static per-tensor act quant | `utils/quant_utils.py:124` `kFp8StaticTensorSym`, handed to `init_fp8_linear_kernel` at `modelopt.py:511-512` → our `vt::QuantFp8Static` |
| per-tensor scaled epilogue | the folded `alpha = input_scale * weight_scale` |
| scheme selection | `base_config.py:180` `get_quant_method` → `MakeLinearMethod` |

**Header placement.** `quantization/fp8.h`, not `quantization/schemes/fp8.h`.
#940's text names `schemes/nvfp4.h` for a file that actually lives at
`compressed_tensors/schemes/nvfp4.h`, so it is using a shorthand, and inventing a
`quantization/schemes/` directory would mirror nothing upstream. `fp8.py` sits
directly under `quantization/` in vLLM, and `layers/quantization/
modelopt_mixed_precision.h` is the local precedent for a header-only policy
header in that same place.

## Design

`dense_fp8_gemm.h` is header-only and carries, in order:

- `DenseCublasLtFp8Enabled()` — `VT_DENSE_CUBLASLT_FP8`, default ON. Unchanged
  lever, unchanged spelling, unchanged default.
- `ResidentFp8(DevT, const Fp8Weight&)` — the lazy one-shot upload of the raw
  fp8 `[N,K]` bytes, owned by the (const) weight's `mutable shared_ptr`.
- `MatmulFp8CutlassD<DBufT>(DevT, x, w, out_dtype)` — `QuantFp8Static` then the
  folded-alpha fp8 GEMM (cuBLASLt by default, cutlass under the lever).
- `MatmulFp8CutlassPreQuantD<DBufT>(DevT, a_fp8, w, out_dtype)` — the same GEMM
  over an activation a preceding fused epilogue already quantized. Upstream this
  is the `x: torch.Tensor | QuantizedActivation` overload of the same apply.

`quantization/fp8.h` adds nothing computational: `Apply` is
`MatmulFp8CutlassD<DBuf>`, `ApplyPreQuantized` is
`MatmulFp8CutlassPreQuantD<DBuf>`, and `MakeLinearMethod` selects on
`Fp8Weight::Empty()` exactly as the NVFP4 factory selects on
`Nvfp4Weight::Empty()`.

### Residual gap — carried, not closed

`MatmulFp8CutlassD` refuses on a host queue because its guard asks for
`kMatmulFp8CublasLt`, registered for kCUDA only, while the ops it would actually
run (`kQuantFp8Static`, `kMatmulFp8Cutlass`) DO have CPU reference arms since
#468/#842. That mismatch is recorded in
[`vt-fp8-w8a8-cpu-arm.md`](vt-fp8-w8a8-cpu-arm.md) §Residual gap and pinned at
`tests/vt/test_ops_fp8_cpu.cpp:445-453`. Widening it is a dispatch change; an
extraction that quietly widened it would be invisible to a byte-identity gate,
which is the whole reason it is out of scope here. The new CPU case re-pins it
at the model tier so it stays visible rather than assumed closed.

## Risks

- **The seam lands dead beside the production path.** Mitigated by construction
  (one definition, instantiated) and MEASURED by mutations M3–M4 below, which
  perturb the header and require a Qwen3.5 CUDA arm to go red.
- **A behaviour change hides inside a move.** Mitigated by moving the bodies
  verbatim and by the assertion-count identity below; the only textual change is
  `MakeTensor` → `dense_attn::MakeTensor` (identical body,
  `dense_device_glue.h:47` vs `qwen3_5.cpp:583`).
- **Template instantiation differs from the inline code.** Both instantiations
  are `inline` in the same TU as before; the CPU full gate and the CUDA arm are
  what test this rather than the argument.

## Tests and gates

Qwen3.5 byte-identity is the gate. Assertion counts before and after must be
identical for every pre-existing suite; the two NEW cases are additive and
declared as such.

| Suite | Before (`c7cb59fbb`) | After |
|---|---|---|
| `test_qwen3_5_gdn_spec_routing` | 6 cases / 52 assertions | 6 / 52 |
| `test_ops_fp8_cpu` | 4 / 56 | 4 / 56 |
| `test_qwen27_paged_forward` | 29 / 765 | 29 / 765 |
| `test_qwen27_dense_forward` | 9 / 583 | 9 / 583 |
| `test_linear_method` | 6 / 76 | 8 / 88 (+2 new cases) |

Full CPU gate: clean configure + `cmake --build build -j 12` (0 warnings under
`-Werror`) + `ctest -j 4`.

Mutation table (the "the seam is LIVE" proof) is recorded in §Evidence.

## Found, not fixed

`ResidentFp8` — and its merged siblings `ResidentFp8Qkv` / `ResidentFp8Qkvz` —
`Alloc` + `Copy` the fp8 weight bytes to the device without calling
`vllm::load_stats::AddDeviceUpload` and without the post-upload
`AdoptDeviceBytesAsHost` step. Both are performed by every other resident-weight
helper in the same file: `ResidentWeight` (`qwen3_5.cpp:1009,1016 @ c7cb59fbb`)
and `ResidentNvfp4` (`:1106,1111,1116,1121`), and `dense_nvfp4_gemm.h:294-328`
carries the comment explaining why the pair is mandatory ("this is the one
host->device move of those bytes and it must be accounted and followed by the
same post-upload residency step every other qualifying weight gets", ENG-LOAD
-DIRECT-UPLOAD / #150).

Consequences, both plausible and neither measured here: the 35B fp8 tower's
upload is missing from load accounting, and its device pages are never re-tagged,
which is the shape of the GB10 weight-residency ATS penalty. Carried across
UNCHANGED — repairing it inside an extraction would be exactly the behaviour
change hidden in a move that this row's gate cannot see. Filed separately.

## Stop conditions

- The extraction cannot be made behaviour-preserving — return `NEEDS_DECISION`
  with the demonstration rather than adapting the numerics.
- A mutation of the extracted code leaves every Qwen3.5 arm green: the seam is
  not on the production path and the byte-identity claim is vacuous. Do not
  land; re-wire until a mutation bites.

## Outcome

Landed as a pure extraction. See §Evidence in the PR for the before/after
assertion counts, the four-row mutation table, and the full-gate result.

## Now

`DONE` — the seam exists and Qwen3.5 routes through it. Wiring
`MODEL-NEMOTRON-H`'s 46 FP8 W8A8 projections to it is A2-Q under #517 and is
NOT part of this row.
