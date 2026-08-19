# A2-Q2b — the REAL-CHECKPOINT per-block gate, and `lm_head` on NVFP4

**Issue:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Sibling / predecessor:** [`nemotron-h-a2q2-nvfp4-moe-lmhead.md`](nemotron-h-a2q2-nvfp4-moe-lmhead.md)
(A2-Q2a — the MoE arm, landed on the synthetic gate).
**State:** IMPLEMENTED on `row/A2-Q2b-lmhead-nvfp4`; the GPU legs of §3 are pending a
`dgx:gpu0` window. See `## Now`.

---

## 0. Why this exists as its own unit

A2-Q2 was split twice, and this file owns the second split.

The first split (A2-Q2a / A2-Q2b) was taken because keeping `lm_head` on the
host **preserves A2-R's attributability**: both arms end in the identical host
projection, so a token difference stays attributable to the MoE arm.

The second split — deferring the **real-checkpoint per-block numeric gate** out
of A2-Q2a — was a scheduling decision, and it is recorded here rather than left
implicit. A2-Q2a's device MoE arm is **unreached** (G-SAFE refuses before it,
see that spec's `## Owed`), and end-to-end NemotronH is blocked on **A2-P**, not
on the quantized arms. Holding A2-Q2a's branch for a 21 GB checkpoint load on a
contended, reboot-prone box would have held the critical path behind a gate that
unlocks nothing. So A2-Q2a landed on the synthetic NVFP4 fixture, and the
expensive gate came here.

**This is a deferral, not a cancellation.** The synthetic result is explicitly
bounded (A2-Q2a §13.6.1) and cannot stand in for what this unit measures.

---

## 1. Scope

| In A2-Q2b | Out |
|---|---|
| the per-block numeric gate on the REAL checkpoint: every one of the 23 MoE layers against `trace.mixer[l]` | anything A2-P owns (paging, carried state, batching, the G-SAFE narrowing) |
| `lm_head` through the NVFP4 dense route, with the `## 5. Owed` residency decision (#984) applied as A2-Q2a applied it | the FP8 mamba arm — A2-Q1 |
| hybrid-vs-host token identity, and the disclosure that A2-R's attributability property ENDS when `lm_head` moves | fixing [#984](https://github.com/mudler/vllm.cpp/issues/984) or [#962](https://github.com/mudler/vllm.cpp/issues/962) |
| the §5.3 mutations A2-Q2a left owed (Q2-M3 … Q2-M7) | any throughput number, on any axis |

---

## 2. What it must measure, and why the synthetic arm does not

A2-Q2a's fixture reported **bit-exact** device-vs-host agreement (`0` over 512
elements, against a routed-scale separation of `0.6`). That is a real result and
a NARROW one. Its output is bf16, its contraction is K=128, its E2M1 codes are
exactly representable and its group scales are powers of two — precisely the
conditions under which a bf16 store absorbs genuine reduction-order differences.

So this unit must measure what the synthetic arm structurally cannot:

- **the real geometry** — H=2688, I=1856, E=128, top_k=6. The contraction is 21x
  longer, so the reduction order genuinely differs.
- **the real scale VALUES** — `weight_scale_2` and the fp8 group scales the
  checkpoint ships are neither powers of two nor uniform, unlike the fixture's.
- **the other Marlin thread configs.** The fixture resolves on `{128,64,128}`
  (up) and `{64,128,128}` (down) only; the real shapes may select others.

**A red here after a bit-exact synthetic is an EXPECTED outcome, not a
contradiction.** It is what A2-Q2a §13.6.1 predicts. Report it as a result; never
widen a band to absorb it.

---

## 3. The gate

Per-block numeric equivalence against the host reference via `NemotronHTrace`,
on the real checkpoint, at **every** one of the 23 MoE layers, plus `lm_head`
against the host projection on the same gathered rows, plus hybrid-vs-host token
identity.

**Bands are MEASURED, and the guard is a PROPERTY** — the shape A2-Q2a arrived
at after its first attempt failed on its own instrument:

- Every comparison reports **how many elements it examined**, and the caller
  asserts that count **against the geometry**, never against the buffer's own
  size (which would agree with itself if the buffer were short). A maximum over
  zero elements is `0.0`, and so is a bit-exact comparison; without the count
  the two are indistinguishable.
- Agreement, separation and the property guard must report the **same** count,
  or the band between them is fiction.
- The band must **admit exact agreement**. A2-Q2a's first band was
  `sqrt(agreed * separation)`, which collapses to 0 when the arms agree exactly
  and failed on the best possible outcome. `separation / 2` with an explicit
  `REQUIRE(separation > 0)` has the intended property and keeps `<` strict.
  Never relax `<` to `<=`: that accepts a band of 0.

---

## 4. ★ Operational — the host working set is what takes the box down

`gpu_memory_utilization` does **not** bound host RAM on GB10, and `nvidia-smi`
attributes only the device side. **Both instruments are blind to the transient
host working set**, which is what actually reboots the machine
(`vm.overcommit_memory=1`, zero swap: the kernel grants memory it cannot back and
the box reboots rather than OOM-killing a process).

So:

1. Take `$GPU_LOCK` with a **blocking** `flock` and wait. Never race.
2. **Check `free -g` headroom INSIDE the locked region**, not before acquiring
   it. A blocking flock says the previous holder released; it never says the box
   recovered. Abort loudly below a stated floor (A2-Q2a used 60 GB).
3. **Sample `free -g` on a loop for the whole load and record the PEAK**, not the
   final value. The figure that matters is transient.
4. Build `-j 4`. One log per run. Never hang holding the lock.
5. Verify the configure log reads `ENABLED for [121a]` — a `DISABLED` line or a
   `[121]` **voids** the result rather than failing it.

Budget to size against: ~17.6 GiB host weights + 16.5 GB device arena + page
cache for a 21 GB checkpoint. Checkpoint at
`${CHECKPOINT_ROOT}/nemotron-3.5-lightning-30b-nvfp4`, with `CHECKPOINT_ROOT` =
`/usr/local/nas_share/checkpoints` — `/usr/local` is COS_PERSISTENT and survives
a reboot; `/mnt` is the ephemeral root overlay of the immutable OS and does not.

**This row has lost four GB10 windows to environment rather than to code** —
`121` instead of `121a`, an unconstrained build, a CUTLASS fetch with no egress,
and an 8h19m outage ended by a human power cycle. Size the plan for that.

---

## 5. Owed

- [#962](https://github.com/mudler/vllm.cpp/issues/962) — NVFP4 Marlin disagrees
  with itself on sm_110 (`bitdiff=15/32768`). The **Thor leg stays PENDING** on
  it; do not quote a number from a kernel that contradicts itself.
- [#984](https://github.com/mudler/vllm.cpp/issues/984) — the address-keyed
  Marlin repack cache. A2-Q2a routed around it by never calling either
  `MarlinDenseResidentFor`; `lm_head` must do the same or say why not.
- The §5.3 mutations A2-Q2a left owed: **Q2-M3** (expert stride off by one),
  **Q2-M4** (`routed_scaling_factor` folded into the logits), **Q2-M5** (shared
  expert added before the routed scale), **Q2-M6** (`MoeRelu2` → plain relu),
  **Q2-M7** (device call site deleted). Q2-M1 and Q2-M2 were run in A2-Q2a.
  Q2-M3 matters most: `src/vt/ops.cpp:874-895` validates **no extent of
  `b_q_weight` and nothing at all about `b_scales`**, so a stride defect is
  silent at the op boundary and only the numeric gate can see it.


- [#1410](https://github.com/mudler/vllm.cpp/issues/1410) — `check-runner-routing-consistency.py`
  cannot resolve a cross-TU free-function device forward, so NemotronH still
  classifies HOST although `NemotronHPagedForward` now assigns both
  `fl.device_tensor` and `fl.device_storage`. The allowlist entry is NARROWED
  to that instrument limit rather than removed. Not fixed in this flow because
  it changes checker semantics, which `AGENTS.md` routes to its own row.
- **The 23-layer MoE per-block sweep of §3 and the `lm_head` real-checkpoint
  numeric leg** run only on `dgx:gpu0` and are PENDING a window, not waived.
  What this change lands is the implementation, the CPU-buildable arms, and the
  synthetic device gate that A2-Q2a's preamble argues for. `## Now` records the
  exact state of each leg, so a `PENDING` here is a scheduled measurement with
  a named blocker rather than a gate nobody ran.
- **Q2-M3 … Q2-M7** (A2-Q2a's owed mutations) are unchanged by this row and
  stay owed. They gate the MoE arm, not `lm_head`.

---

## 6. Now

**Implemented.** The device `lm_head` arm lands on `row/A2-Q2b-lmhead-nvfp4`.

### What was measured BEFORE anything was built

The row's premise — that host re-expansion dominates NemotronH decode and that
`lm_head` is a large share of it — was ARITHMETIC when this row was dispatched.
It is now a measurement, taken at the single dequant seam
(`NemotronHOwned::DenseBf16`) on the REAL 21 GB checkpoint through the
production ABI driver (`examples/nemotron_h_gen`), one decode step, T=1,
top_k=6, 23 MoE layers:

| group | shape | calls | elements | per call | % |
|---|---|---|---|---|---|
| routed expert `up_proj` | `[1856, 2688]` | 138 | 688 472 064 | 4 988 928 | 22.36% |
| routed expert `down_proj` | `[2688, 1856]` | 138 | 688 472 064 | 4 988 928 | 22.36% |
| shared expert `down_proj` | `[2688, 3712]` | 23 | 229 490 688 | 9 977 856 | 7.45% |
| shared expert `up_proj` | `[3712, 2688]` | 23 | 229 490 688 | 9 977 856 | 7.45% |
| **`lm_head`** | **`[131072, 2688]`** | **1** | **352 321 536** | **352 321 536** | **11.44%** |
| mamba `out_proj` (FP8) | `[2688, 4096]` | 23 | 253 231 104 | 11 010 048 | 8.23% |
| mamba `in_proj` (FP8) | `[10304, 2688]` | 23 | 637 034 496 | 27 697 152 | 20.69% |
| **TOTAL** | | **369** | **3 078 512 640** | | **100%** |

`138 == 6 * 23` exactly, which is what confirms this is the decode shape and not
a prefill aggregate.

Three things follow, and the third is why the row proceeded:

1. **The dispatching estimate was wrong in both of its numbers, in the same
   direction.** It put `lm_head` at 131072 x 4096 = 537e6 elements and at ~43%
   of the population. `hidden_size` is 2688, not 4096: the true count is
   352 321 536, and the true share of that population is 28.35%.
2. **The "~1.24e9 elements / ~2.49 GB per token" figure is real and now has a
   name.** It is not the host arm's total (3.079e9 / 6.157 GB). It is exactly
   `mamba + lm_head` = 1 242 587 136 elements = 2.485 GB — the residue AFTER
   A2-Q2a moved the MoE arm to the device. It matches to four significant
   figures, which is what identifies which regime the number describes.
3. **`lm_head` is the LAST one.** Against the three-leg discriminator on
   `dgx:gpu0` (`/workspace/a2d1-discriminate/20260819T200231Z`: device mamba ON
   1.554 s/token and 108.2x vs vLLM, OFF 10.319 s/token and 718.1x), the mamba
   arm is worth 6.64x and is in flight on A2-D1. `lm_head` is on the HOST in
   every one of those three legs. Once the mamba arm lands, `lm_head` is
   352 321 536 of 352 321 536 — 100% of the host re-expansion left in a decode
   step. It is also the largest SINGLE re-expansion in the model by 12.7x
   (352.3e6 in one call against 27.7e6 for mamba `in_proj`), so its 704.6 MB
   transient bf16 buffer is the allocation that matters most on a
   unified-memory box that reboots rather than OOM-kills.

The refutation is therefore narrow and the conclusion survives: the estimate's
share was wrong, the direction was right, and the case is STRONGER after the
discriminator than before it.

### Leg status

| Leg | State |
|---|---|
| seam extension (caller-owned `MarlinDenseResident`) | DONE, built on BOTH arms (see below) |
| device `lm_head` arm + `DeviceLmHeadEligible` | DONE, built on BOTH arms |
| production wiring in `NemotronHPagedForward` -> device `ForwardLogits` | DONE |
| host arm retained as the gate's operand + the non-NVFP4 fallback | DONE |
| routing-allowlist entry narrowed, [#1410](https://github.com/mudler/vllm.cpp/issues/1410) filed | DONE |
| the CPU-reachable half (`n_out`, the shared `final_normed` download) gated through `GPUModelRunner` | DONE — `test_nemotron_h_paged_forward.cpp` §12 |
| synthetic device `lm_head` numeric gate (measured band, asserted counts) | WRITTEN; runs on CUDA only. **NEVER RUN** |
| `nvcc` build of the Marlin KERNEL + any execution of the device arm | PENDING a `dgx:gpu0` window |
| real-checkpoint `lm_head` numeric leg + token identity | PENDING a `dgx:gpu0` window |
| reachability deletion mutation | PENDING the same window |
| 23-layer MoE per-block sweep (§3) | PENDING; owed above |

### What "built" means here, corrected

The first submission of this row said "the CPU build compiles the `#else` arms
only, so it does not compile the Marlin path at all", and recorded the CUDA
build of the Marlin arm as blocked on a GPU window. **That was wrong, and it
overstated the blocker.** `include/vt/cuda/marlin_repack.h` includes only
`<cstdint>`, `<cstddef>` and `<vector>`, so the HOST side of the Marlin arm
needs no CUDA toolkit at all. Measured on this box, which has no `nvcc`:

```
c++ -std=c++20 -I include -I src -isystem third_party -DVT_MARLIN_NVFP4=1     -Wall -Wextra -Werror -c -o nhd_marlin.o     src/vllm/model_executor/models/nemotron_h_device.cpp
-> rc 0, 0 errors, 0 warnings, a 1 269 696-byte object
```

Both arms are therefore compiled, and both are compiled `-Werror`. What
genuinely needs `nvcc` is the Marlin **kernel** and every **execution** of the
device path. Landing without those is acceptable and is what the PENDING rows
above record; claiming the host arm could not be compiled was not.

### The fresh review, and what it found

A fresh reviewer returned FINDINGS on the first submission. The two blocking
ones are recorded here because both are about EVIDENCE, and evidence is what a
spec is for.

1. **`tests/vllm/models/test_nemotron_h_moe_device.cpp` had never compiled.**
   `NemotronHHostWeights` was used unqualified and was missing from the file's
   using-block, so the file failed `-Wall -Wextra -Werror` with 9 errors on a
   plain CPU build — the PR head at `rc 1`, the merge-base version of the same
   file at `rc 0` on the same command. **The red-first result claimed for the
   synthetic numeric gate therefore did not exist and could not have existed.**
   The declaration is repaired and the file now compiles at `rc 0` on both
   arms, but the gate itself is still CUDA-only and still has never executed;
   the table above says `NEVER RUN` rather than restating a red nobody saw.
   `vllm_cpp_add_test(test_nemotron_h_moe_device)` is deliberately registered
   with NO CUDA guard, and that is what surfaced this: a case that skips at run
   time still has to parse and type-check on every CPU build.
2. **The production source asserted a protection that does not exist.** A
   comment at the device branch claimed the allowlist entry was removed and
   that "the routing checker, not a comment, is what now holds this branch in
   place". All three parts were false: the entry is narrowed and still present,
   and deleting the entire `if (DeviceLmHeadEligible(...)) { ... }` block leaves
   `scripts/check-runner-routing-consistency.py` at `rc 0` with byte-identical
   output ("3 host-logits off-framework (3 allowlisted)"), reproduced on the
   repaired tree with the file restored byte-for-byte afterwards (identical
   sha256). **Nothing automated holds that branch.** The checker is not widened
   to make it — that changes checker semantics, and [#1410](https://github.com/mudler/vllm.cpp/issues/1410)
   owns it with its own red-before. The comment now says so, and the
   reachability deletion mutation stays PENDING rather than claimed.

The reviewer also found a real defect that a token gate structurally cannot
see. `DeviceLmHeadEligible` restated the shared dispatcher's selection clauses
and dropped `MarlinW4A16Enabled()`, so under an explicit `VT_NVFP4_MARLIN=0`
the predicate said eligible while `MatmulNvfp4W4A16D` took its naive
redundant-dequant arm — computing the SAME logits while re-uploading the whole
`[131072, 2688]` operand on every decode step, because `LmHeadNvfp4View` hands
out a stack temporary that `ResidentNvfp4`'s weight-keyed cache can never hit.
The repair is structural rather than a patched clause: the three clauses now
live once, in `dense_nvfp4::MarlinW4A16Selects`, and both the dispatcher and
the model call it, so they cannot drift. `DeviceLmHeadD` additionally refuses
BY NAME if the seam's `fallback_gemms` counter moves across its own call, and
the synthetic gate asserts the same counter — the counter is demonstrably armed
rather than assumed, because `tests/vllm/models/test_qwen3_forward.cpp:497`
already asserts on CPU that it reaches exactly `5 * num_hidden_layers` when the
dispatcher does fall back. The behavioural red for this class needs CUDA and is
PENDING with the rest.
