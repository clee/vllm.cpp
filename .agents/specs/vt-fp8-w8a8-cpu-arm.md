# VT-FP8-W8A8-CPU-ARM — register the static FP8 W8A8 path on the CPU backend

Issue: [#468](https://github.com/mudler/vllm.cpp/issues/468).
Owning row: `PERF-27B-LMHEAD-FP4` (the row #468 is filed against in the
[roadmap issue table](../roadmap_v1.md)).
Related: [`perf-fp8-alpha-fold.md`](perf-fp8-alpha-fold.md) (the lever whose
model-layer wiring #468 found untestable),
[`portable-fusion-framework.md`](portable-fusion-framework.md) §3b/§6 (the
"backend-negotiated fp8 tail" this row retires on CPU).

#468 lists two ways to close its coverage debt. This row is **option 1**: *"a CPU
registration for the fp8 matmul sufficient to exercise the wiring (which has
value well beyond this lever — it would make the whole fp8 model path
CPU-testable)"*. It does not do option 2, and it does not touch the parked
`VT_GDN_FP8_ALPHA_IN_CONV` lever.

## Scope

**IN — three registrations' worth of surface, in two files.**

1. `src/vt/cpu/cpu_ops.cpp`: register `OpId::kQuantFp8Static` on
   `DeviceType::kCPU`. The fp8-e4m3fn codec this needs already lives in that file
   (`F32ToFp8`, saturating RNE, whose comment already claims it bit-matches
   `vllm::F32ToF8E4M3`; `kFp8Max = 448.0F`), so the kernel is the scale
   application and nothing else.
2. `src/vt/cpu/cpu_ops.cpp`: register `OpId::kMatmulFp8Cutlass` on
   `DeviceType::kCPU` as a **correctness reference** — f32 accumulate, one folded
   `alpha`. It makes no performance claim and is not a lever.
3. `include/vt/ops.h` + `src/vt/cuda/cuda_matmul_fp8_cutlass.cu` +
   `include/vt/fused_recipe.h`: comment repairs the first two items create the
   need for (below).

**IN — the consequential test changes.** `tests/vt/test_fused_chain_additivity.cpp`
asserts, for all three fp8-terminal recipes, that the FULL Tier-0 composite
**THROWS** on CPU because `vt::QuantFp8Static` is unregistered there. Registering
it makes those three `CHECK_THROWS` false. They are not deleted: each is
**replaced by the strictly stronger byte-exact assertion** the registration now
makes possible — full composite == standalone-op-sequence golden, fp8 output
included — and the catalog's `cpu_full` flag flips to `true` for those three
rows. That is a change of a checker's claim, which is why this spec exists and
why the RED-before/GREEN-after evidence below is mandatory.

**OUT.**

- `OpId::kMatmulFp8CublasLt` and `OpId::kMatmulFp8CublasLtAlphaVec` stay
  CUDA-only. A "cuBLASLt" kernel on the CPU backend would be a lie in the name,
  and the alpha-vec op's whole claim is about a cuBLASLt epilogue pointer mode.
  **This has a consequence that must not be glossed** — see *Residual gap*.
- Any CUDA behavior. Not one line of the CUDA fp8 kernels changes; only a
  comment that describes them wrongly.
- `src/vllm/model_executor/models/nemotron_h*.{h,cpp}` and
  `tests/vllm/models/test_nemotron_h_*` — owned elsewhere, tracked by #517.
- The parked `VT_GDN_FP8_ALPHA_IN_CONV` lever on
  `row/PERF-ALPHA-IN-CONV-PARKED`, and the 8 `mixed_scale` call sites. #468's
  option 2 (an argument-forwarding test) is not attempted here.

### Residual gap — stated up front, not discovered later

The model-layer entry points `MatmulFp8CutlassD` and
`MatmulFp8CutlassPreQuantD` (`src/vllm/model_executor/models/qwen3_5.cpp`) both
gate on `vt::OpRegistered(vt::OpId::kMatmulFp8CublasLt, d.q.device.type)` and
refuse with *"the fp8 W8A8 path is CUDA-only"*. Registering `kMatmulFp8Cutlass`
and `kQuantFp8Static` on CPU therefore makes the **op seam** CPU-reachable and
the **fusion catalog's fp8 terminal** CPU-reachable; it does **not** on its own
make `MatmulFp8CutlassD` execute on a CPU queue. Closing that last step is a
separate scoped change (either widening the predicate to the op actually
selected, or registering a CPU arm for the cuBLASLt op id) and stays open under
#468. This row does not claim it.

## Upstream chain

Pinned oracle `/home/mudler/_git/vllm` @ `5559679229bc961848b121ccdeaa8fa5d79bec98`
(0.26.0.dev0), the pin recorded in [`upstream-sync.md`](../upstream-sync.md).
Verified present at that SHA on 2026-08-14; the line numbers below were re-derived
at HEAD rather than copied from an earlier record.

| What | Upstream anchor |
|---|---|
| Dispatch to the static fp8 linear method is unconditional | `vllm/model_executor/layers/quantization/modelopt.py:2527-2528` — `if quant_algo == "FP8": return ModelOptFp8LinearMethod(self.fp8_config)`; exclusions are checked first at `:2519-2523` |
| The method is **static, hard-coded** — no dynamic fallback inside it | `modelopt.py:510-513` — `init_fp8_linear_kernel(activation_quant_key=kFp8StaticTensorSym, weight_quant_key=kFp8StaticTensorSym, …)` |
| `input_scale` is created per merged shard and collapsed to a **scalar** | `modelopt.py:502-508` (`PerTensorScaleParameter`, one entry per output partition) then `:528` — `layer.input_scale = Parameter(layer.input_scale.max(), …)` |
| Per-token dynamic is a **different class**, reached by a different `quant_algo` | `ModelOptFp8PcPtLinearMethod`, `modelopt.py:540`, selected at `:393-395` |
| The quant math | `csrc/quantization/w8a8/fp8/common.cuh:58-77` — `x = val * scale` under `is_scale_inverted=true` (`:62`), else `x = val / scale` (`:64`); then `fmaxf(-448, fminf(x, 448))` (`:68`); then the hardware RNE convert (`:71-76`) |
| The reciprocal is formed ONCE, outside the elementwise math | `csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:31` and `:38` — `1.0f / scale[…]` |
| Per-tensor selection = one group over the whole tensor | `common.cu:204-210` — `scale.numel() == 1` ⇒ `group_m = num_tokens`, `group_n = hidden_size` |

So upstream's shipped path is **reciprocal-multiply**, not divide: the `val / scale`
arm at `common.cuh:64` is the `is_scale_inverted=false` branch, and the caller that
feeds the static per-tensor path hands it an already-inverted scale
(`common.cu:31`). Our CUDA kernel mirrors that
(`cuda_matmul_fp8_cutlass.cu:331-333`: `const float inv = 1.0f / input_scale;`
then `LoadIn(x, i) * inv`). The CPU kernel added here does the same. The two are
**not** interchangeable in f32: `x / s` and `x * (1/s)` differ by up to one ulp
before the fp8 round, and near an e4m3 tie that ulp changes the emitted byte.

**Our one documented deviation is PRESERVED, not revisited.** Upstream applies
`scale_a` and `scale_b` as two epilogue scalars
(`vllm/model_executor/layers/quantization/utils/scaled_mm/cutlass.py` `ScaledEpilogue`);
we fold `alpha = input_scale * weight_scale` into one f32
(`include/vt/ops.h`, the `MatmulFp8Cutlass` contract). The CPU reference folds it
the same way, so it is a reference for the op we actually ship.

## Design

### `QuantFp8StaticKernel` (CPU)

```
inv = 1.0f / input_scale            // formed ONCE, mirroring common.cu:31
out[i] = F32ToFp8(LoadF32(x, i) * inv)
```

`LoadF32` handles the f32 and bf16 `x` dtypes the op contract admits, matching the
CUDA kernel's `LoadIn` overloads (bf16 is widened to f32 *before* the multiply, so
the two backends round at the same point). `F32ToFp8` supplies the saturation
(`a >= 448 ⇒ 0x7E`, which is the encoding of 448 — clamp-then-convert and
saturating-convert coincide because 448 IS the largest finite e4m3fn value) and
the round-to-nearest-even. Row-chunked through the existing `ForRows` so it
inherits the file's thread pool; the op is elementwise, so chunking cannot move a
value.

### `MatmulFp8CutlassKernel` (CPU)

```
acc  = Σ_k Fp8ToF32(a[m,k]) · Fp8ToF32(b[n,k])     // f32 accumulate
out[m,n] = alpha · acc                              // ONE folded scalar
```

Shaped exactly like the neighbouring `MatmulNvfp4Fp4Kernel`: the A row is decoded
once per M and reused across N. **This is a correctness reference, not a
performance path** — it is a naive triple loop, it makes no speed claim, and
nothing routes a production model through it. It is stated as such in the code.

It is *not* a bit-mirror of the CUDA GEMM and does not claim to be: the CUDA arm
reduces K in tensor-core order and rounds its epilogue through bf16, so the two
agree to fp8/bf16 tolerance, not to the byte. Only the QUANT half is a bit-mirror
claim, and that is what G2 tests.

### Comment repairs

Three comments become false the moment these registrations land, and one is
already false today:

- `include/vt/ops.h` (`QuantFp8Static` contract): *"CUDA only (the 35B W8A8 path
  is CUDA-resident)"* → CUDA + CPU.
- `include/vt/ops.h` (`MatmulFp8Cutlass` contract): *"CUDA-only (sm120a)"* →
  CUDA (sm120a) + a CPU reference.
- `include/vt/fused_recipe.h` (`kQuantFp8` opcode): *"CUDA-only"*.
- **Already false, and the reason this one matters most:** `include/vt/ops.h`
  states the contract as `clamp(x[i] / input_scale, …)` and
  `src/vt/cuda/cuda_matmul_fp8_cutlass.cu` repeats it as `fp8_e4m3(clamp(x[i]/input_scale, …))`,
  while the code three lines below multiplies by the reciprocal. **The code is
  right and the comment is wrong.** Left alone, it invites someone to "correct"
  a default-ON 35B path into a divide, which is a token-visible change nothing
  currently forbids. The comment is repaired to the reciprocal form with the
  upstream anchor beside it.

## Risks

| Risk | Mitigation |
|---|---|
| A CPU registration silently becomes a reference-tier fallback on unified-memory accelerators (`MaybeInstallReferenceTier`, `src/vt/op_provider.cpp:204-225`), turning a hard refusal into a slow silent success on e.g. Metal/Vulkan | Not silent by construction: the tier is announced once per (op, device) on stderr as `[vt reference-tier] … running the PORTABLE CPU fallback (correct but slow)` and counted (`op_provider.cpp:516-528`). Recorded here as an intended, visible consequence rather than left to be discovered |
| The three `CHECK_THROWS` in `test_fused_chain_additivity.cpp` are "fixed" by deletion, weakening the additivity proof | They are REPLACED by full-composite byte-exact checks and the `cpu_full` flags flip to `true`; the case's `== 9` catalog count guard is untouched. Assertion count goes UP, not down, and that is recorded in the evidence |
| `F32ToFp8`'s comment claims it bit-matches `vllm::F32ToF8E4M3`, and the claim was never tested against an INDEPENDENT reference | G1 is exactly that test, and its reference is derived from the format definition + upstream's clamp, not from either of our two codecs |
| A CPU-vs-CUDA byte divergence exists and nobody sees it, because this box has no GPU | G2 is declared, and reported PENDING with the reason rather than skipped. It is not counted as satisfied |

## Tests and gates

New file `tests/vt/test_ops_fp8_cpu.cpp`, wired into `tests/CMakeLists.txt`.

**G1 — bitwise, zero tolerance.** `vt::QuantFp8Static` on a CPU queue must
produce bytes identical to an **independently written** reference:
`clamp(x · (1/input_scale), ±448)` then e4m3fn round-to-nearest-even. The
reference is built from the *format*, not from our code: it enumerates all 256
e4m3fn encodings, decodes each to an exact `double` from the field layout, and
picks the nearest with an even-significand tie-break by scanning. That is a
different algorithm from both `F32ToFp8` (frexp + `nearbyint`) and
`vllm::F32ToF8E4M3`, so agreement is evidence rather than tautology. Not
expressible as `doctest::Approx` — `Approx`'s `scale` term defaults to 1.0 and
gives it a ~1.19e-5 absolute floor, which is meaningless for a byte compare.
Coverage includes both `x` dtypes, values that overflow ±448 in both signs,
subnormal-range values, exact ties, and zeros of both signs.

**G2 — CPU vs CUDA, bitwise.** The CPU registration must agree byte-for-byte
with the CUDA kernel on the same input. **PENDING on this host: it has no GPU
(`nvidia-smi` absent, no CUDA device).** The case is committed and CUDA-gated so
it binds on the next GPU holder; it is not claimed as satisfied here. Gate hosts
are `dgx.casa` (GB10/sm_121) and `192.168.68.23` (Thor/sm_110).

**G3 — the GEMM.** `vt::MatmulFp8Cutlass` on CPU against a `double` reference
that reproduces upstream's LOSSY pipeline — clamp, e4m3 RNE, dequant, then exact
accumulation. A reference doing exact arithmetic on the pre-quant floats would
make a WRONG implementation look better than upstream and pass; the tolerance
then bounds only the K-reduction order, which is the only thing that legitimately
differs.

**Mutations that must prove the gate is armed.** Each applied ALONE to a restored
tree, rebuilt, run, restored; the compiler exit status is recorded beside every
result, because a mutation that fails to BUILD reads as a passing test.

| # | Mutation | Must |
|---|---|---|
| M1 | `input_scale` ignored (activation left unscaled) | G1 RED |
| M2 | divide instead of reciprocal-multiply | G1 RED — and if it does NOT, that is a reportable finding: the gate would be blind to the exact defect the stale comment invites |
| M3 | saturation removed (no clamp to ±448) | G1 RED on an overflowing input |
| M4 | round-to-nearest-even replaced by truncation | G1 RED |
| M5 | `alpha` reduced to `weight_scale` only | G3 RED |

`Status:` and the case count are read on every mutation run, not just
`assertions:` — a failing doctest binary can print `0 failed` when a case throws.

**Full gate:** `scripts/agent-preflight.sh` and the CPU ctest suite, with `df -h`
before and after. `test_op_parity` is RED on `main` at the base SHA
(base-inherited, #755/#672) and is subtracted as a known baseline red.

## Evidence

Host: `mudler-ubuntu-box`, x86_64, GCC 13.3.0, **no CUDA device** (`nvidia-smi`
absent). Build: `cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF`, no `CMAKE_BUILD_TYPE`
(so no `NDEBUG`, asserts live), `-Wall -Wextra -Werror`. Date 2026-08-14.
Base SHA `b1cd4d8f6bb7ec5f0bd923a75dcc140becc7fdd8`.

| Gate | Result |
|---|---|
| G1 bitwise quant | **PASS** |
| G2 CPU vs CUDA bitwise | **PENDING — no GPU on this host.** Committed and CUDA-gated; the case prints the reason and still asserts the CPU registration so it can never be vacuous. Owed on `dgx.casa` (GB10/sm_121) or `192.168.68.23` (Thor/sm_110) |
| G3 fp8 GEMM vs lossy `double` reference | **PASS** |

`test_ops_fp8_cpu`: 4 cases / 56 assertions / `Status: SUCCESS!`.
`test_fused_chain_additivity`: 1 case / **25** assertions (was **21**), `SUCCESS!`.

**The three replaced assertions are a FORCED consequence, and it is measured, not
argued.** Checking out the BASE version of that test (the one carrying the three
`CHECK_THROWS`) and running it against the new registrations gives
`test cases: 1 | 0 passed | 1 failed`, `assertions: 21 | 18 passed | 3 failed`,
`Status: FAILURE!` — exactly three failures, which are exactly the three
`CHECK_THROWS`, with the other 18 assertions untouched. The registration makes
those three claims false; it does not make them optional. They were replaced by
the stronger byte-exact form, not deleted.

(The implementation commit message says "19 -> 25". That arithmetic was wrong;
the measured base count is 21. Corrected here rather than by rewriting pushed
history.)

### Mutations — all applied ALONE to a restored tree, compiler exit recorded

| # | Mutation | compile | `test cases` | `assertions` | `Status` |
|---|---|---|---|---|---|
| M0 | both registrations removed (the RED-BEFORE) | 0 | 4 \| 0 passed \| **4 failed** | 6 \| 1 \| **5 failed** | **FAILURE!** |
| M1 | `input_scale` ignored | 0 | 4 \| 3 \| **1 failed** | 56 \| 32 \| **24 failed** | **FAILURE!** |
| M2 | divide instead of reciprocal-multiply | 0 | 4 \| 3 \| **1 failed** | 56 \| 50 \| **6 failed** | **FAILURE!** |
| M3 | saturation removed | 0 | 4 \| 3 \| **1 failed** | 56 \| 28 \| **28 failed** | **FAILURE!** |
| M4 | RNE replaced by truncation | 0 | 4 \| 3 \| **1 failed** | 56 \| 28 \| **28 failed** | **FAILURE!** |
| M5a | kernel ignores `alpha` | 0 | 4 \| 3 \| **1 failed** | 56 \| 53 \| **3 failed** | **FAILURE!** |
| M5b | caller folds `weight_scale` only | 0 | 4 \| 3 \| **1 failed** | 56 \| 53 \| **3 failed** | **FAILURE!** |
| — | restored (green-after) | 0 | 4 \| **4 passed** \| 0 | 56 \| **56** \| 0 | **SUCCESS!** |

Re-run in full on the post-merge tree (merge of `origin/main` @ `5da1d7f2f`):
identical results, every row, and the restored tree green at 56/56.

M0's assertion count DROPS to 6 rather than staying at 56 — a changed case count
is signal, and it is why `Status:` is read alongside `assertions:`.

**M1, M2 and M5a first failed to BUILD, not to assert** (`-Werror=unused-parameter`
/ `-Wunused-variable` on the now-dead `input_scale` / `inv_scale` / `alpha`).
A mutation that fails to build reads as a passing test, so each was re-expressed
with an explicit `(void)` and re-run. Only the `compile_exit=0` rows above are
verdicts.

### What the mutation series measured, beyond pass/fail

**M1 (`input_scale` ignored) changes ~99.7% of output bytes** at every scale
except 1.0, where ignoring the scale is correctly a no-op: 4308/4319, 4311/4319,
4311/4319 and 4299/4319 words at scales 0.5, 0.035, 0.0092 and 7.25, and 0/4319
at 1.0. Overall 17229/21595 = 79.8%.

**M2 (divide vs reciprocal-multiply) is nearly invisible, and that is the
finding.** Over 20000 random values in [-2, 2] the two forms NEVER disagree, at
any of 14 scales tried. The difference only appears where an input lands on an
e4m3 tie after scaling — which is precisely what G1's constructed tie population
is for — and even there it is scale-dependent: over that population **10 of 18
candidate scales expose it at all**. Of the five scales G1 originally shipped
with, **only 0.0092 did, at 24 of 209 words**, so the mutant died by 2
assertions and the assertion protecting the reciprocal form was one scale-list
edit away from being silently disarmed. 0.13 (78/209) and 0.77 (82/209) were
measured as the strongest detectors and added, with the numbers recorded beside
them in the test. M2 now dies by 6 assertions across 3 scales.

This is the concrete answer to the question the spec's comment-repair section
raised: the gate CAN see the exact defect the stale comment invites, but only
because the input population contains constructed exact ties and the scale list
contains a detector. Neither is decoration.

**M3 exposed defense-in-depth in the pre-existing codec.** `F32ToFp8` saturates
in TWO places — the early `a >= kFp8Max` return and a late `exp_field > 15`
overflow guard — so removing either alone leaves saturation intact and the
mutant survives. M3 removes both, which is what "saturation removed" has to mean
for this codec, and then it dies by 28 assertions. Recorded because a reviewer
mutating only the obvious guard would wrongly conclude the gate is blind.

## Stop conditions

- G1 RED against the independent reference on any covered input ⇒ stop and
  report. It would mean either the new kernel or the pre-existing `F32ToFp8`
  disagrees with upstream's format, and the second is a bug in shipped code that
  gets its own issue and fix in-flow.
- A mutation that does not RED ⇒ report the survivor rather than strengthening
  the mutation until it dies. A surviving mutant is a finding about the gate.
- Any need to touch `kMatmulFp8CublasLt`, the model layer, or CUDA behavior ⇒
  `NEEDS_DECISION`, not a silent scope widening.
- Disk below ~10 GiB free ⇒ stop and report; an ENOSPC build leaves the PREVIOUS
  binary in place and makes checkers emit false policy refusals.
