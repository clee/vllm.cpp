# A2-D1 — NemotronH decodes on the single-step recurrent kernels

**Issue:** [#1311](https://github.com/mudler/vllm.cpp/issues/1311).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Governing spec:** [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md); this unit
is residual work on the arm [`nemotron-h-a2q1-fp8-mamba.md`](nemotron-h-a2q1-fp8-mamba.md)
built.
**Base:** `row/A2-Q1-fp8-mamba` @ `56e2dc36182fa3c5aa4adaa4abe45dc453335e54`
(PR [#1289](https://github.com/mudler/vllm.cpp/pull/1289)), merged forward onto
`origin/main` `edbc47ce0`. `NemotronHMamba2MixerDevice`, the function this unit
changes, does not exist on `main`.
**Pinned oracle:** `${VLLM_SOURCE}` @ `5559679229bc961848b121ccdeaa8fa5d79bec98`
(vLLM 0.26.0.dev0), per [`upstream-sync.md`](../upstream-sync.md).
**Lifecycle at this commit:** unchanged. The row stays `INVENTORIED`.

---

## 0. The divergence

vLLM's Mamba2 mixer branches on `has_decode` (`mamba_mixer2.py:981`) and runs a
different pair of kernels on the decode rows:

| | vLLM decode | vLLM prefill |
|---|---|---|
| conv | `causal_conv1d_update` `:1012` | `causal_conv1d_fn` `:869` |
| SSD | `selective_state_update` `:1087` | `mamba_chunk_scan_combined` `:890` |
| state I/O | `conv_state_indices` / `state_batch_indices`, IN PLACE at the slot | gather, scan, scatter |

A2-Q1 ran the **prefill pair over both halves**: `vt::CausalConv1dFwd` +
`vt::Mamba2ChunkScan` wrapped in `GatherNemotronHState` /
`ScatterNemotronHState`, for every row of every step.

`vt::Mamba2StateUpdate` — the port of `selective_state_update` — has been
registered on CUDA (`cuda_gdn.cu:6672`) and CPU (`cpu_ops.cpp:3408`) and gated
since [`mamba2-ssd.md`](mamba2-ssd.md) §W2 landed on 2026-08-13, with **zero
callers under `src/vllm/`**. That is a live `AGENTS.md` §"Nothing lands dead"
case, and closing it is half of why this unit exists.

The tree already carried the warning. `qwen3_5.cpp:4730-4746`, on the GDN decode
path: *"Passing the state indices to the op eliminates the per-request
gather+scatter — the two host<->device copies per sequence per layer that
dominate the decode memcpy tax."* NemotronH did the thing that comment warns
against.

### 0.1 What it costs, as arithmetic

Counted from `cuda_mamba2_ssd.cuh:577-645` at `T=1`, `chunk_size=128`,
`nchunks=1`. **Not profiled** — #1311 says so and this spec repeats it:

| per mamba layer per token | chunk scan | state update |
|---|---|---|
| kernel launches | 5 | 1 |
| `cudaMallocAsync`/`FreeAsync` | 5 + 5 | 0 |
| `cudaMemsetAsync` | 2, of 2.50 MiB | 0 |
| scratch | 4.56 MiB | 0 |
| `M2ChunkScanKernel` grid | 524,288 elements to compute 4,096 | 4,096 |

x23 mamba layers = ~105 MiB scratch, 230 driver alloc/frees and 115 launches per
token in the SSD alone, plus the gather/scatter state churn upstream's decode
half never performs.

### 0.2 The same arithmetic, re-derived here, and where it disagrees with #1311

Re-counted from `cuda_mamba2_ssd.cuh:596-641` at the DRIVER geometry
(`H=64 P=64 N=128 G=8 cs=128 S=1 nchunks=1 T=1`), because a number that is
quoted often enough starts being treated as measured
([[a-number-quoted-often-becomes-treated-as-measured]]). **This is arithmetic,
not a profile.**

| scratch term | elements | bytes |
|---|---|---|
| `dtv` = `H*nchunks*cs` | 8,192 f32 | 32 KiB |
| `dac` = same | 8,192 f32 | 32 KiB |
| `states` = `nchunks*H*P*N` | 524,288 f32 | 2.00 MiB |
| `cb` = `nchunks*G*cs*cs` | 131,072 f32 | 512 KiB |
| `passed` = `n_states * state_elem` | 524,288 f32 | 2.00 MiB |
| **total per call** | | **4.5625 MiB** |

Zeroed per call: `cb` + `passed` = 2.50 MiB. `M2ChunkScanKernel`'s grid is
`nchunks*H*cs*P` = 524,288 elements to produce `T*H*P` = 4,096 — a factor of
`cs` = 128.

Per TOKEN over the 23 mamba layers, decode:

| | chunk scan (before) | state update (after) |
|---|---|---|
| SSD kernel launches | 115 | 23 |
| SSD `Alloc`/`Free` | 230 | 0 |
| SSD `cudaMemsetAsync` | 46, zeroing 57.5 MiB | 0 |
| SSD scratch | 104.9 MiB | 0 |
| gather/scatter launches | 92 | 0 |
| small metadata H2D | 138 | 0 |

**Where this disagrees with #1311.** The issue estimated "roughly +414 MiB/token"
of gather/scatter state churn. Counting the three SSM movements — gather
(read page + write working), scatter (read working + write page) and the
`final_states` copy-back — at 4.00 MiB each gives 12.0 MiB per layer per token,
so **~276 MiB/token for the SSM plus ~7 MiB for the conv, about 283 MiB/token**,
not 414. The disagreement is recorded rather than reconciled to the larger
figure: neither number is measured, and the direction of the change does not
depend on which is right.

---

## 1. Scope

| In A2-D1 | Out of A2-D1 |
|---|---|
| the decode/prefill split of the device mamba arm in `NemotronHPagedForward` | the MoE block (`NemotronHMoeBlockDevice`) — [#1312](https://github.com/mudler/vllm.cpp/issues/1312) |
| routing decode rows to `vt::CausalConv1dUpdate` + `vt::Mamba2StateUpdate` at their cache slots | `runner.cpp` / `sampler.cpp` — [#1313](https://github.com/mudler/vllm.cpp/issues/1313) |
| narrowing the gather/scatter to the prefill rows | the HOST mamba fallback branch, which stays exactly as it was |
| gating the chunk-scan/state-update equivalence at `n_groups=8` | lifting G-SAFE's `num_reqs <= 1` — A2-B |
| a recording seam for which arm ran | speculative decode rows — #810 W5 |

**Prefill keeps the chunk scan.** The selection is the metadata's own
`num_decodes` / `num_prefills` split, never the token count: a one-token PREFILL
of a fresh request is also `T == 1`, carries no state in, and can have
`prefill_has_initial_state == 0`, which neither single-step kernel can express.

---

## 2. Design

`NemotronHMamba2MixerDevice` gains one optional parameter, a
`NemotronHMambaDecodeSlots` descriptor carrying the two FULL cache pages and the
per-row slot indices. Non-null replaces steps 3 and 5 of the block with the
single-step kernels and forbids the gathered carry; null leaves every existing
caller byte-identical.

The paged forward's device-mamba branch becomes:

1. `nd = gdn_meta.num_decodes`, `np = gdn_meta.num_prefills` (already computed
   by A2-P, mirroring `mamba_attn.py:523-532`).
2. if `nd > 0`: ONE batched mixer call over the leading `nd` tokens with the
   decode slots. No gather. No scatter. No `final_states` copy-back.
3. if any rows remain: gather **those rows only**, loop as A2-Q1 did, scatter
   **those rows only**.

`MetaSubView` narrows the `[R]` index vectors to a contiguous sub-range, the
same shape `SubView` already has in `qwen3_5.cpp` for the GDN decode arm.

### 2.1 Why the decode arm needs no `has_initial` mask

A decode continues a sequence by definition, which is why upstream leaves
`has_initial_state` `None` on a decode-only step (`gdn_attn.py:405`) and why
A2-P already sets the mask to 1 for every row below `num_decodes`. The gather's
zeroing obligation (A2-P's `NemotronHRecurrentIo` note) applies to rows whose
mask is 0, and a decode row never has one.

### 2.2 The recording seam

The two arms compute the same recurrence, so they produce the same tokens and a
token gate cannot tell them apart. `NemotronHMambaArmCounts` counts launches at
the `vt::` call sites — not at the branch condition — so a gate entering through
`ModelRegistry::Forward` observes what the step RAN. It mirrors
`RecordGdnOutActivationDTypes` (`qwen3_5.cpp`).

---

## 3. Risks

**R1 — a wrong carry is silently wrong tokens, not a crash.** The recurrent
state is exactly what a mis-slotted or mis-grouped update corrupts, and the
result is finite, plausible and wrong. Mitigated by the A3 token gate on the
real checkpoint, run on BOTH arms of the same binary.

**R2 — the equivalence at `n_groups=8` was ASSUMED.** `mamba2-ssd.md` §8.3
declares an equivalence contract, but §8.3 is device-vs-host; the only
decode-vs-prefill case ran `H=4 G=2`, i.e. `heads_per_group = 2`. NemotronH runs
`heads_per_group = 8`. A head-to-group map correct at 2 and wrong at 8 passed
every existing case. **Gated by this unit before the swap is relied on** — see
§4.

**R3 — the instrument could report zero for the wrong reason.** A counter that
is never incremented reads exactly like a kernel that is never launched. Gated
on CPU (§4) so a broken recorder reds in seconds rather than on a GPU window.

**R4 — the decode rows must lead the batch and be one token each.** Asserted at
the call site against `gdn_meta`, refusing rather than mis-slicing.

---

## 4. Tests

| # | Case | Where | Runs on |
|---|---|---|---|
| T1 | decode == chunked prefill at the driver group count, two shapes incl. `T=1 chunk=128` | `tests/vt/test_ops_mamba2_state_update.cpp` | CPU |
| T2 | the same on device, both arms CUDA, native provider asserted | same file | CUDA |
| T3 | the arm recorder counts what the step launched, through `ModelRegistry::Forward` | `tests/vllm/models/test_nemotron_h_paged_forward.cpp` | CPU |
| T4 | A3 e2e token gate + same-binary `VT_NEMOTRON_H_MAMBA_DECODE_STEP` A/B | `scripts/nemotron-h-a2q1-dgx-gate.sh` | GPU |

T1/T2 report the comparison's SCALE and worst absolute difference
unconditionally and assert the scale is O(1): the inherited 5e-3 atol would
accept everything if the tensors compared were ~1e-7
([[count-based-tolerances-bound-nothing]]).

### 4.1 Mutations

| id | mutation | must be caught by |
|---|---|---|
| A2D1-M1 | clamp the state-update group index to `min(h/hpg, 1)` | T1 — and NOT by the pre-existing `H=4 G=2` case, which is the point |
| A2D1-M2 | delete the decode branch of the paged forward | T4's counter readout |
| A2D1-M3 | select the decode arm on `T == 1` instead of on `num_decodes` | a one-token prefill of a fresh request |
| A2D1-M4 | stop incrementing the counters | T3 |

---

## 5. Gates

**The A3 e2e gate must read `96/96 mode=decode STRICT PASS` on the default
(state-update) arm.** That is the acceptance condition; no throughput number is
accepted before it.

The A/B is same-binary: `VT_NEMOTRON_H_MAMBA_DECODE_STEP=1` (default) against
`=0`, on one build, on an idle leased box, sampling the DECODE WINDOW ONLY. A
sampler started with the process swallows a multi-minute engine load and has
already produced one void number on this row.

---

## 6. Stop conditions

- The A3 gate does not read `96/96 STRICT PASS` on the default arm → the swap is
  wrong; do not tune, do not widen, revert to the chunk scan and reopen #1311
  with the divergence.
- T1 or T2 fails → the equivalence contract does not hold at `n_groups=8` and
  the premise of #1311 is refuted. `NEEDS_DECISION`, not a wider tolerance.
- The counters read zero on both arms → the instrument is broken; T3 is the
  triage, not the GPU log.

---

## Now

T1, T2 and T3 committed and green. **T4 MET on `thor:gpu0` (sm_110): the A3 gate
reads `96/96 mode=decode STRICT PASS` on the single-step arm**, with every
decode step launching 23 state-update rows and 0 chunk scans, 0 gathers and 0
scatters. The `dgx:gpu0` sm_121a leg and the same-binary A/B delta are still
owed.

## Owed

- The `dgx:gpu0` sm_121a leg of T4, and the ON/OFF A/B delta on one binary. The
  sm_110 leg is MET (see `## Now`); dgx was queued behind a two-hour job.
  Owned by this row, tracked by
  [#1311](https://github.com/mudler/vllm.cpp/issues/1311). NO vLLM ratio may be
  quoted for arch 110: the 0.014369 s reference is GB10's.
- The batched decode arm (`nd > 1`) is written and unreachable while G-SAFE pins
  `num_reqs <= 1`; A2-B owns lifting that, and the code is indexed rather than
  hardcoded so the lift is a count change.

## Outcome

Pending; this section is authored when the row reaches `DONE`.
