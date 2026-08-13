# NVFP4 nibble order — two producers, two conventions, one shared dequant

**Row:** `MODEL-DIFFUSION-LTX25` phase L9a (`.agents/specs/ltx-2-5.md` §4.1, §4.2).
**Issue:** [#435](https://github.com/mudler/vllm.cpp/issues/435).
**Branch:** `row/LTX25-L9A-NVFP4-LINEAR`.
**Authority:** operator, 2026-08-13, after L9a returned `NEEDS_DECISION` rather than
building the arm it was briefed to build.
**Status:** spec committed BEFORE implementation, per AGENTS.md §"Spec before code".

---

## 0. Why this needs its own spec

`DequantNvfp4ToBf16` (`nvfp4_dequant.h:59`) is shared by the MiniMax-H3, Laguna
(Qwen3.5), DeepSeek-V4 and Qwen3-32B NVFP4 arms and by the `vt` fp4 GEMM reference.
Phase L9a needs it to read a checkpoint packed the other way round. A change there is
not LTX-2.5 loader work, so it is not made inside the LTX row — it is specified,
gated, and reviewed on its own terms.

## 1. The fact, measured and sourced

E2M1 packs two 4-bit values per byte. **Which logical element lands in which nibble is
a producer convention, and the two producers we consume disagree.**

| Producer | Element `2j` goes in | Source |
|---|---|---|
| **torchao** (`NVFP4Tensor`) | the **LOW** nibble | `torchao/prototype/mx_formats/kernels.py:160` — `(uint8_data[::2] \| uint8_data[1::2] << 4)`; its own inverse at `:137-139` stacks `first_elements = data & 0b1111` first |
| **vLLM** (reads torchao/ModelOpt) | the **LOW** nibble | `nvfp4_emulation_utils.py:321-324` — `low = a_flat & 0x0F` then `torch.stack((low, high))`; the Triton path agrees at `:101-108` (`tl.interleave(low_result, high_result)`) |
| **Lightricks `nvfp4-prequant`** | the **HIGH** nibble | `ltx-kernels/docs/NVFP4.md:27-29` — "`hi_first=True` (default) puts element `2j` in the **high** nibble … Pre-quantized checkpoints used with `nvfp4-prequant` are expected in the default order"; `ltx-core/quantization/nvfp4/linear.py:6` |
| **"Star Ultimate Model Converter Pro"** (`lilcheaty/MiniMax-H3-NVFP4`) | the **HIGH** nibble | already recorded at `minimax_h3.h:1500-1515` |

Our `DequantNvfp4ToBf16` is low-first (`nvfp4_dequant.cpp:74-75`), which is correct for
torchao and ModelOpt and wrong for the other two.

**Read low-first, a high-first file has every adjacent pair transposed.** It is finite,
correctly shaped, and wrong — the exact failure class this project keeps recording. L9a
measured it on the first-party LTX-2.5 NVFP4 DiT against the independent `vonkaiser` FP8
oracle: correct reading corr **0.9956** / 9.46% rel rms; wrong nibble order corr **0.032**.
See `.agents/specs/ltx-2-5.md` §4.1 for the full table and the control.

### 1.1 §4.2 is ANSWERED, and the answer is "no shipped defect"

`ltx-2-5.md` §4.2 recorded an open question: our torchao text-encoder arm reads low-first,
but its gate compared against goldens made by our own low-first helper — consistency, not
correctness. **torchao's source settles it: low-first.** Three mutually consistent and
independent witnesses (torchao's packer, torchao's unpacker, vLLM's two readers). The
shipped TE arm is correct. This spec's test work converts that self-referential gate into
a source-anchored one so it cannot regress into a question again.

## 2. Scope

**In.** One parameter on `DequantNvfp4ToBf16` selecting nibble order, **defaulting to
today's low-first behaviour**; the LTX-2.5 loader's producer discrimination and its use of
that parameter; the shape assertion admitting the cuBLAS-padded framing of the swizzled
scale; a correlation gate that proves the DiT dequantizes to the right numbers.

**Out.** H3's `MiniMaxH3Nvfp4SwapNibbles` (§3.1 says why it stays); the fp4-resident
device GEMM path (`cuda_matmul_nvfp4.cu`), which stays low-first-only; MXFP4 and GGUF
NVFP4, which have their own packing and are untouched; the TE's tower wiring, owned by
`row/LTX25-L10-TEXT-TOWER`.

## 3. Design

```cpp
enum class Nvfp4NibbleOrder {
  kLowFirst,   // element 2j in the LOW nibble — torchao, ModelOpt, vLLM. DEFAULT.
  kHighFirst,  // element 2j in the HIGH nibble — Lightricks nvfp4-prequant.
};

void DequantNvfp4ToBf16(const uint8_t* packed, const uint8_t* weight_scale_fp8,
                        float weight_scale_2, int64_t out_dim, int64_t in_dim,
                        uint16_t* out_bf16,
                        Nvfp4NibbleOrder order = Nvfp4NibbleOrder::kLowFirst);
```

A defaulted trailing parameter makes "every existing caller is untouched" true **by
construction** rather than by inspection. The proof obligation in §5 is that the gates
agree.

### 3.1 Why NOT H3's nibble-swap-at-load, which is prior art

`minimax_h3.h:1500-1517` already solved this once, by swapping the two nibbles of every
byte at load so the file becomes standard low-first. That is a genuinely different design
and it was right **for H3**, whose bytes also feed a Marlin fp4-**resident** path: one
transform fixes the bf16 dequant and the device GEMM together.

LTX-2.5 has no fp4-resident arm — `Ltx2StreamDitToDevice` dequantizes to bf16 and uploads
— so a swap would allocate and rewrite a second copy of 9.4 GB of packed weights to feed a
consumer that could just as well have read them in place.

Two mechanisms for one concept is a real smell and it is recorded rather than hidden:
**if LTX-2.5 ever grows an fp4-resident NVFP4 device path, this decision must be revisited
and the load-time normalization adopted**, because a host-side dequant parameter cannot fix
a device GEMM that reads the packed bytes itself. That is the tracked condition, written
where it will be found.

### 3.2 The producer discriminator (operator-RATIFIED, 2026-08-13)

The scale layout and the nibble order are decided together, from the **`torchao_nvfp4`
marker's presence or absence**. This is in-file evidence, not provenance guessing: torchao
always writes that marker, so its absence excludes torchao.

| Marker | Required `weight_scale` shape | Resolves to |
|---|---|---|
| present, `is_swizzled_scales=true` | `[32*ceil(N/128), 16*ceil(G/4)]` (`to_blocked` framing) | torchao: unswizzle, **low**-first |
| absent | `[round_up(N,128), round_up(G,4)]` (cuBLAS-padded framing) | `nvfp4-prequant`: unswizzle, **high**-first |
| anything else | — | **REFUSE BY NAME** |

`G = in_features / 16`. No "probably" branch: a marker with the wrong shape, an absent
marker with the wrong shape, or a marker declaring an unimplemented combination all refuse.

**Both framings describe the SAME bytes.** `[32*ceil(N/128), 16*ceil(G/4)]` and
`[round_up(N,128), round_up(G,4)]` have equal element counts and the identical layout;
they are two 2-D dresses on one flat buffer. `Ltx2UnswizzleNvfp4BlockScale` is already
framing-agnostic — it keys off logical `rows`/`cols` and a byte count, and its index
formula matches `ltx-kernels/csrc/nvfp4/quantize.cu:26-31` term for term — so **only the
shape assertion changes**, not the permutation.

**The assumption must be legible at the code site.** The marker-absent arm is an inference
from a producer's signature, not a fact read off the file, and the comment there says so
and cites the evidence: the 0.9956 correlation and `ltx-kernels/docs/NVFP4.md:27-29`.

### 3.3 Why the shape cannot discriminate alone

For every quantized layer in the first-party DiT, `N % 128 == 0` and `G % 4 == 0`, so the
cuBLAS-padded shape **numerically equals the linear `[N, G]` shape**. A shape test can
therefore never separate "swizzled, padded framing" from "linear". That is why the marker
is primary and the shape only corroborates, and it is why the original linear-arm brief was
unimplementable as written.

## 4. Risks

| Risk | Mitigation |
|---|---|
| A defaulted parameter silently changes an existing caller | Every NVFP4 gate re-run with **assertion counts compared**, not just Status (§5) |
| The high-first arm is wrong and output is merely finite | The correlation gate (§5.2) requires the 0.9956 / 9.46% signature; finiteness is not accepted as evidence |
| Marker-absent inference generalizes to a file it should not | Refuse by name on any other marker/shape combination; no fallback branch |
| Two nibble mechanisms drift apart | §3.1 records the condition under which H3's is adopted here |
| The correlation gate needs a 21 GB and an 18.7 GB file | It reads byte RANGES from the two headers, materializes ~128 rows, and is opt-in behind an env var like the other shipped-checkpoint cases |

## 5. Tests and gates

### 5.1 RED first
A case that dequantizes the real first-party DiT bytes and asserts the FP8-correlation
signature must FAIL before the change, for the right reason (the loader refuses the shape),
and pass after.

### 5.2 The correlation gate — the one nobody had
The `vonkaiser` FP8 DiT and the first-party NVFP4 DiT quantize the same base weights, so
the FP8 file is an **independent oracle** for the NVFP4 read. The gate:

- dequantizes the same module from both files (rows 0..127),
- asserts Pearson correlation **>= 0.99** and relative rms error **<= 0.15**,
- asserts a **control**: the same NVFP4 read against a DIFFERENT module's FP8 weights
  correlates **< 0.2**, so the gate proves it can tell right from wrong rather than
  passing on any two finite arrays,
- and emits the VALUES, never a boolean.

Without the control the gate is `7.0(c)` again — a fixture that cannot separate a correct
implementation from a plausible wrong one.

### 5.3 Nibble order, source-anchored
A unit case pinning both orders against hand-computed bytes, plus a generator-side anchor
on torchao's `pack_uint4` line and `ltx-kernels`' `hi_first` documentation, so a convention
change upstream is reported as a source change (§1.1's repair).

### 5.4 The discriminator
Every row of §3.2's table, including each refusal, asserted by name.

### 5.5 Unchanged-behaviour proof (operator's non-negotiable)
These must be green with **identical case and assertion counts**, reported. Baselines
captured on this branch at `f400413e`, Release, CUDA=OFF:

| Gate | Baseline |
|---|---|
| `test_nvfp4_dequant` | 4 / 47 |
| `test_gguf_nvfp4` | 14 / 2352 |
| `test_qwen36_weights` | 7 / 45 |
| `test_qwen3_forward` | 7 / 1557 |
| `test_minimax_h3` | 79 / 57395 |
| `test_ops_nvfp4_matmul` | 4 / 1 |
| `test_ops_moe_grouped` | 6 / 3 |
| `test_ops_nvfp4_fp4` | 22 / 919 |
| `test_ltx2_loader` | 20 / 2363 |
| `test_ltx2_device` | 13 / 498 |

Counts only RISE, and only in the two suites this change adds cases to. **If any other
gate moves, STOP.** `Status:` is grepped every run: a thrown case DROPS the assertion
count while still printing "passed".

### 5.6 The real artifact
The first-party NVFP4 DiT stages onto the GPU and forwards, on dgx.casa under
`flock -w 2700 $HOME/gpu.lock`, naming the file, its tensor count and its derived geometry,
with finite non-degenerate output.

## 6. Stop conditions

- The high-first read does not reproduce the correlation signature → STOP; the encoding is
  not merely a nibble order and the premise is wrong again.
- Any existing NVFP4 gate changes case or assertion count → STOP.
- A checkpoint appears that needs a third convention, or a marker/shape combination outside
  §3.2 → refuse by name and report; do not add a branch.
- Any temptation to accept "the forward ran and the output was finite" as the correctness
  result → forbidden; §5.2 is the result.

## 7. Outcome

**What was measured.** The first-party NVFP4 DiT is SWIZZLED and HIGH-nibble-first, and both
halves were established against an oracle that is not ours — the `vonkaiser` FP8 DiT of the
same base weights. Re-derived after a full-disk incident on the build host and identical to
every digit:

| reading | rms | corr vs FP8 | rel rms |
|---|---|---|---|
| LINEAR / low-first (what the original brief asked for) | 0.013564 | 0.000414 | 1.786 |
| LINEAR / high-first | 0.013564 | 0.257746 | 1.558 |
| SWIZZLED / low-first (the shipped dequant) | 0.009208 | 0.032296 | 1.394 |
| **SWIZZLED / high-first** | **0.009208** | **0.995560** | **0.0946** |

The committed gate re-measures this from real bytes at 0.994968 corr / 0.100672 rel rms, with
the control at 0.00362 and the deliberately-wrong nibble order at 0.00514.

**What was rejected.**

1. *A linear-layout arm.* The premise was wrong; building it would have made the file load and
   render silently wrong. L9a stopped and returned `NEEDS_DECISION` instead.
2. *Discriminating by SHAPE.* Gated as impossible for this artifact: all 1176 quantized modules
   have a cuBLAS-padded scale shape numerically identical to the linear one
   (`ambiguous_with_linear == 1176`). The marker leads; the shape only corroborates.
3. *H3's nibble-swap-at-load.* Correct for H3, which also feeds a Marlin fp4-resident path, and
   wasteful here: LTX-2.5 dequantizes to bf16, so a swap would rewrite 9.4 GB of packed weights
   to feed a consumer that could read them in place. §3.1 records the condition that reverses
   this decision.
4. *"The forward ran and the output was finite."* The gate asserts absmax is IDENTICAL under
   both nibble orders, so every magnitude summary is blind to the defect by construction. Only
   the correlation separates them.

**Why the default is low-first.** It is what torchao's `pack_uint4` writes
(`kernels.py:160`), what torchao's own unpacker reads (`:137-139`), and what vLLM's two
independent readers assume (`nvfp4_emulation_utils.py:321-324` and `:101-108`). Every caller
predating this change consumes a ModelOpt or torchao checkpoint, so a defaulted parameter makes
"nothing else moves" true by construction — and the ten-gate sweep confirmed it, eight of them
byte-identical.

**§4.2 closed, no defect.** torchao is low-first on three independent witnesses, so the shipped
text-encoder arm was already correct. Its gate is now anchored to those sources rather than
comparing our helper to itself.

**Gates.** Ten NVFP4 gates plus nine LTX/video gates green; the only counts that moved are the
two suites this change adds cases to (`test_nvfp4_dequant` 4/47 -> 5/69, `test_ltx2_loader`
20/2363 -> 24/4793). Three mutations were run and all three turned the suite red: forcing
low-first (3 cases), resolving a marker-less padded file as torchao (3 cases), and reinstating
an apostrophe in a refusal (1 case). Tree restored byte-for-byte after each.

**On the GPU (dgx.casa, GB10, one `flock -w 2700 $HOME/gpu.lock` hold 02:31:03 to 02:37:15).**
Both arms stage device-resident and forward, `Status: SUCCESS!`:

| arm | file | assertions | output absmax |
|---|---|---|---|
| NVFP4 | `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` (18,721,432,024 B, 7876 tensors) | 5626 | video 0.275391 / audio 1.02344 |
| FP8 | `ltx-2.5-22b-distilled-fp8.safetensors` (6124 tensors) | 5624 | video 0.300781 / audio 2.14062 |

The FP8 figures reproduce the previously recorded GB10 run to the digit, which is what makes
the NVFP4 figures next to them trustworthy rather than merely new. The two arms differ because
they are different quantizations AND different configurations: the NVFP4 file declares a config
(`double_precision_rope`, `av_ca_timestep_scale_multiplier = 1000`, both asserted after
adoption), while the FP8 file declares none and runs under manifest defaults.

**Not obtained.** The device case's MESSAGE text (printed tensor count, staging seconds, config
source) was cut by a `tail` in the harness on the first run, and a second run to capture it
timed out on the GPU lock after 45 minutes (`FLOCK_EXIT=1`) against a saturated box. The
underlying facts are asserted rather than printed — the geometry CHECKs are among the 5626 that
passed — and 7876 tensors / 48 blocks / video 4096 / audio 2048 / 21.004B parameters are gated
from the committed manifest. Cosmetic, and not worth further contended GPU time while another
row is rendering.
