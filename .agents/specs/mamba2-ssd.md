# Mamba2 SSD — the generic selective-scan core `KERNEL-SSM-MAMBA` never got

**Claim:** `CLAIM-KERNEL-SSM-MAMBA-SSD`. **Kernel row:** `KERNEL-SSM-MAMBA`
(existing, stays `INVENTORIED` at this spec commit — see §8).
**Issue:** [#496](https://github.com/mudler/vllm.cpp/issues/496).

**Model rows it unblocks (all `INVENTORIED`, all waiting on this one kernel):**
`MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` (model-matrix.md:248, the
immediate driver), `MODEL-TEXT-mamba2-mamba2-for-causal-lm` (:234),
`MODEL-TEXT-granitemoehybrid-granite-moe-hybrid-for-causal-lm` (:214),
`MODEL-TEXT-falcon-h1-falcon-h1-for-causal-lm` (:194),
`MODEL-TEXT-zamba2-zamba2-for-causal-lm` (:280),
`MODEL-TEXT-jamba-jamba-for-causal-lm` (:227),
`MODEL-TEXT-lfm2-lfm2-for-causal-lm` (:229),
`MODEL-TEXT-lfm2-moe-lfm2-moe-for-causal-lm` (:230),
`MODEL-TEXT-plamo2-plamo2-for-causal-lm` (:264),
`MODEL-TEXT-olmo-hybrid-olmo-hybrid-for-causal-lm` (:251).

**Base:** `main` HEAD `e1087a8812c9b7d96fca5a813981f378fcace638`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).

**Signal (honest, up front):** this is a **host-reference-first kernel brick**,
mirroring the KDA lane ([kda-kernel-delta.md](kda-kernel-delta.md)) and the
DeepSeek-V4 DSA lane. W1 lands the SSD numerics as portable CPU references
gated against a from-first-principles double-precision reference and against
the sequential recurrence they are supposed to be algebraically equal to. The
W1 gate is **host-reference + structural review, NOT a dumped-oracle rel-L2**.
The device (CUDA) arm is W2, and the real end-to-end token gate is a NAMED
residual owned by the model row, not by this one.

---

## 0. Scope (headline verdict)

`KERNEL-SSM-MAMBA` is named "General Mamba selective scan, causal convolution,
SSD, and linear attention kernels" (kernel-matrix.md:157). Only the second and
fourth clauses were ever built. Everything landed under the row is the **GDN**
arm: the gated-delta-rule recurrence, its causal conv, its state gather/scatter
and the sm_120 tiling campaign. The row's own matrix text concedes it —
*"these GDN tests do not prove generic Mamba support"* — and repeats
"generic Mamba lifecycle stays `INVENTORIED`" at four checkpoints
(kernel-matrix.md:167-209).

**Gated delta rule is not Mamba2.** GDN carries a delta-removal term
`(I − β kkᵀ)` and a per-head scalar decay; Mamba2's SSD is a diagonally-decayed
gated linear recurrence with **no** removal term, driven by `A_log`, a
per-token `dt` and a skip `D`, with `B`/`C` shared across `n_groups` head
groups. `kGdnPrefill` cannot be reshaped into it. This row owns exactly two new
numerical objects:

1. **the chunked SSD scan** (`mamba_chunk_scan_combined_varlen`) — the varlen
   prefill path: chunk cumulative decay, chunk-local state, inter-chunk state
   passing, and the chunk-scan output combine;
2. **the single-token selective state update** (`selective_state_update`) — the
   decode path, including the scattered-cache-slot indexing our GDN decode
   already models.

Plus one small third thing that is genuinely different from what we have:

3. **the silu-gated GROUP RMS norm** (`Mixer2RMSNormGated.forward_native`,
   `mamba_mixer2.py:100-149`) — `x · silu(gate)` followed by an RMS norm over
   `group_size = intermediate_size / n_groups` slices. Our `kRmsNormGated` is
   the GDN/KDA **sigmoid** gate over the whole row; the activation and the
   reduction extent both differ, so it is a sibling op, not a parameter.

**Out of scope, explicitly.** The Nemotron-3.5-Lightning model port — non-gated
`relu²` MoE grouped GEMM, ModelOpt `MIXED_PRECISION` per-module loading, the
MTP head, `layers_block_type` layer dispatch — is a separate row on
`MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`. So is TP sharding of
`n_groups` (`mamba_v2_sharded_weight_loader`, `mamba_mixer2.py:174-236`);
this row lands `tp_world_size == 1` and states the shard seam it leaves open.
So is ReplaySSM (`use_replayssm`, `selective_state_update_replayssm_output_only`)
and Mamba **v1** (`mamba_mixer.py`), neither of which any scoped model needs.

## 1. Upstream chain (`file:line` @ `555967922`)

### 1.1 The layer — `vllm/model_executor/layers/mamba/mamba_mixer2.py`

| What | Anchor |
|---|---|
| `Mixer2RMSNormGated` silu-gated group RMS norm | `:69-172` (native `:100-149`) |
| `MambaMixer2.__init__` (proj split, `A_log`/`D`/`dt_bias`) | `:250-547` |
| `conv_ssm_forward` — the prefill/decode split | `:687-1104` |
| decode/prefill token split (`num_decodes`, `num_prefills`) | `:738-790` |
| varlen prefill call site | `:870` (and warmup `:654`) |
| decode call site | `:1087` |
| `get_state_dtype` / `get_state_shape` | `:1105-1141` |
| `mamba_type` → `MambaAttentionBackendEnum.MAMBA2` | `:1142-1144` |

### 1.2 The SSD ops — `vllm/model_executor/layers/mamba/ops/`

| Kernel | File | Role |
|---|---|---|
| `mamba_chunk_scan_combined_varlen` | `ssd_combined.py:157-235` | varlen entry; `_mamba_chunk_scan_combined_fwd` at `:27-156` is the 5-stage pipeline |
| `_chunk_cumsum_fwd` + `_chunk_state_fwd` | `ssd_chunk_state.py` (407 L) | per-chunk `dA_cumsum` (with `dt_softplus`, `dt_limit`) and chunk-local state |
| `_state_passing_fwd` | `ssd_state_passing.py` (146 L) | inter-chunk recurrence, consumes `initial_states`, honours `seq_idx` boundaries |
| `_bmm_chunk_fwd` | `ssd_bmm.py` (209 L) | `CB = C·Bᵀ` per chunk, f32 accumulate (`ssd_combined.py:124`) |
| `_chunk_scan_fwd` | `ssd_chunk_scan.py` (525 L) | combines intra-chunk attention-like term + inter-chunk state term + `D` skip |
| `selective_state_update` | `mamba_ssm.py:497+` | decode; `state_batch_indices`/`null_block_id` scattered slots |
| `_layer_norm_fwd` (gated) | `layernorm_gated.py` (172 L) | the fused form of §0.3 |

Contract points that must be mirrored, not re-derived:
`chunk_size` must be a power of 2 (`ssd_combined.py:48`); `seq_idx` is
per-chunk, `seq_idx.shape == (nchunks,)` (`:60-61`, `:189`); `initial_states`
is `(batch, nheads, headdim, dstate)` (`:79`, `:194`); `CB` accumulates in
**f32** regardless of activation dtype (`:124`); the SSM state dtype is a
separate knob from the activation dtype (`state_dtype`, `:46,119,176`).

### 1.3 State layout — `vllm/model_executor/layers/mamba/mamba_utils.py`

`mamba2_state_shape` (`:174-199`) returns exactly two shapes:
`conv_state = (conv_dim/tp, conv_kernel − 1 + num_spec)` where
`conv_dim = intermediate_size + 2·n_groups·state_size`, and
`temporal_state = (num_heads/tp, head_dim, state_size)`.
`mamba2_state_dtype` (`:73-81`) gives `(conv_dtype, ssm_dtype)` independently,
which is the same conv-then-temporal ordered `MambaSpec` pair our runner already
allocates (porting-inventory.md:109).

### 1.4 The config that drives it

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (`model_type: nemotron_h`,
52 layers = 23 mamba / 6 attention / 23 moe): `mamba_num_heads=64`,
`mamba_head_dim=64`, `n_groups=8`, `ssm_state_size=128`, `conv_kernel=4`,
`chunk_size=128`, `mamba_hidden_act=silu`, `use_conv_bias=true`,
`use_bias=false`, `mamba_ssm_cache_dtype=float32`. Wired at
`nemotron_h.py:373-389`.

## 2. Our baseline — reuse vs new (our `file:line`)

### REUSE (landed; the SSD recurrence rides on these unchanged)

- `MambaSpec`, the ordered conv-then-temporal state pair, and its exact
  `page_size_bytes` — `include/vllm/v1/kv_cache_interface.h`,
  `src/vllm/v1/kv_cache_interface.cpp:102-127`.
- Hybrid KV coordinator and the per-group managers (full-attn left→right +
  Mamba right→left single recurrent state) —
  `src/vllm/v1/core/kv_cache_coordinator.cpp`,
  `single_type_kv_cache_manager.cpp:652`.
- All three causal-conv arms: `vt::CausalConv1dFwd`, `CausalConv1dUpdate`,
  `CausalConv1dSpecUpdate` (`include/vt/ops.h:95-97`) — Mamba2's conv is the
  same op with the same persistent `conv_state`; only the channel split around
  it differs.
- Scattered state slots: `kGdnStateGather` / `kGdnStateScatter`
  (`include/vt/ops.h:186-187`) — the `state_batch_indices` / `null_block_id`
  semantics of `selective_state_update` are the ones GDN decode already models
  (`ops.h:1856`).
- `kQkvSplit`-style projection splitting, `kMatmulBT`, `kSiluAndMul`.

### NEW (this brick)

- `vt::Mamba2ChunkScan` — varlen chunked SSD prefill.
- `vt::Mamba2StateUpdate` — single-token decode selective update.
- `vt::RmsNormGatedGroup` — silu-gated, `n_groups`-wise RMS norm (§0.3).

### NEW, but NOT this brick (named residuals)

- The CUDA arm of all three (W2) — W1 is CPU host references only.
- `n_groups` TP sharding + `extra_groups_for_head_shards`
  (`mamba_utils.py:187`).
- Spec-decode `num_spec > 0` state rows; the conv side already supports them
  (`CausalConv1dSpecUpdate`), the temporal side does not.
- ReplaySSM, Mamba v1, `mamba_cache_mode=align` prefix retention
  (already an open T1 item, porting-inventory.md:80).

## 3. Port map (upstream → local)

| Upstream | Local (new) | Note |
|---|---|---|
| `ssd_chunk_state.py::_chunk_cumsum_fwd` | `src/vt/cpu/cpu_mamba2_ssd.cpp` | `dt_softplus` + `dt_limit` clamp, then `dA_cumsum` per chunk |
| `ssd_chunk_state.py::_chunk_state_fwd` | same TU | chunk-local `(nheads, headdim, dstate)` |
| `ssd_state_passing.py::_state_passing_fwd` | same TU | consumes `initial_states`, resets on `seq_idx` change |
| `ssd_bmm.py::_bmm_chunk_fwd` | same TU | **f32 accumulate**, non-negotiable |
| `ssd_chunk_scan.py::_chunk_scan_fwd` | same TU | intra + inter + `D` skip |
| `ssd_combined.py::mamba_chunk_scan_combined_varlen` | `vt::Mamba2ChunkScan` in `src/vt/ops.cpp` + `include/vt/ops.h` | the 5 stages above, in upstream order |
| `mamba_ssm.py::selective_state_update` | `vt::Mamba2StateUpdate` | scattered slots + NULL row, mirroring `GdnDecode` |
| `mamba_mixer2.py:100-149` | `vt::RmsNormGatedGroup` | silu gate + group RMS |
| `mamba_utils.py::mamba2_state_shape` | `src/vllm/v1/kv_cache_spec_registry.cpp` | a second `MambaSpec` producer; no new spec type |

Every new symbol carries the `file:line` it was ported from, per
[porting.md](../porting.md). Nothing here is written from scratch, so nothing
is owed to porting-inventory §9.

## 4. Tests to port

From `tests/kernels/mamba/` @ `555967922`, parameters, dtypes and tolerances
preserved, harness adaptation documented where unavoidable:

| Upstream | Local | Covers |
|---|---|---|
| `test_mamba_ssm_ssd.py` | `tests/vt/test_ops_mamba2_ssd.cpp` | the chunked scan across `chunk_size`, `n_groups`, `seq_idx` boundaries, `initial_states` |
| `test_mamba_ssm.py` | `tests/vt/test_ops_mamba2_state_update.cpp` | `selective_state_update` incl. scattered/NULL slots |
| `test_mamba_mixer2.py` | `tests/vt/test_ops_mamba2_gated_norm.cpp` | the gated group RMS norm |

Plus two tests upstream does not have, because our gate is not a dumped oracle:

- **Chunked == sequential.** The SSD chunked scan must equal a naive
  per-token recurrence written independently in double precision. This is the
  test that actually catches a wrong `dA_cumsum` or a dropped inter-chunk term,
  and it is the reason this row can gate without the GPU.
- **Chunk-boundary invariance.** The same sequence scanned at
  `chunk_size ∈ {8, 16, 32, 64, 128}` must agree to the f32 tolerance. A
  state-passing defect is invisible at one chunk size and loud across five —
  the same failure shape as [[h3-video-decode-temporal-and-tiling-compose]],
  where the gates ran below one chunk and saw nothing.

**RED first.** Each test is committed failing for the intended reason with the
red output captured, before the implementation. The reviewer mutates the
claimed guarantee in a scratch copy — drop the `initial_states` term, widen
`dt_limit`, swap the `CB` accumulator to bf16 — and proves each mutation is
caught. `Approx` comparisons use `.scale(0.0)`
([[doctest-approx-scale-term-floor]]); pass/fail is read from the `Status`
line, not the `assertions:` count ([[doctest-assertions-line-hides-thrown-cases]]).

## 5. Gates

**W1 (host reference).** Focused: the three new test binaries green, with the
chunked==sequential and chunk-boundary-invariance arms passing at the stated f32
tolerance. Full: `ctest` clean, re-run **serially** for the known parallel-flaky
set ([[flaky-under-parallel-ctest]]), clean `-Werror` on a **clean rebuild**
([[incremental-build-masks-werror]]), and a Debug arm so asserts are live
([[release-gate-masks-asserts]]).

**W2 (device).** CUDA arm byte-compared against the W1 host reference on the
Nemotron-3.5 shapes (`nheads=64, headdim=64, dstate=128, ngroups=8, chunk=128`),
`compute-sanitizer` clean, on dgx under `flock $HOME/gpu.lock` with
`local-ai-worker` parked ([[localai-worker-down-on-dgx]]).

**Not this row's gate.** A token-exact end-to-end comparison against the pinned
vLLM needs the model port, so it is stated here as a residual and owned by
`MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`. When it runs, the oracle
identity is asserted before any number is believed — `vllm.__version__` +
flashinfer, abort on mismatch ([[oracle-identity-must-be-asserted]]) — and the
fixture is checked to be the checkpoint the changed path actually loads
([[sacred-27b-gate-loads-wrong-checkpoint]]).

**No performance claim is made in this row.** The SSD kernel's throughput
against vLLM's Triton pipeline is a separate measurement with its own
both-arms `nsys` requirement ([[profile-both-arms-before-choosing-a-lever]]);
a host reference is not a speed result and will not be reported as one.

## 6. W-breakdown

| W | Content | Exit |
|---|---|---|
| W0 | This spec. Upstream chain read end to end, contract points extracted, reuse-vs-new settled | spec committed, issue linked in three places |
| W1 | RED tests → `vt::Mamba2ChunkScan`, `vt::Mamba2StateUpdate`, `vt::RmsNormGatedGroup` CPU references → focused green → full gate → fresh scoped review | host references gated, reviewed by a fresh agent |
| W2 | CUDA arm for all three, byte-compared to W1 | device green + `compute-sanitizer` clean on dgx |
| W3 | `MambaSpec` producer for Mamba2 shapes; het-KV group construction proven on the Nemotron-H layer pattern | spec/shape tests green; hands off to the model row |

W1 is the load-bearing one. W0 and W3 are small; W2 is bounded by the shapes W1
already gates.

## 7. Risks / decisions

- **The chunked scan is where correctness hides.** Chunk cumsum, state passing
  and the scan combine are individually plausible and jointly wrong in ways a
  single-chunk test cannot see. Mitigated by the two extra tests in §4, which
  are the point of the design, not extras.
- **f32 discipline.** `CB` accumulates in f32 upstream (`ssd_combined.py:124`)
  and the SSM state has its own dtype knob. Going *wider* than the oracle is
  numerically correct and therefore invisible to a token gate while doubling
  traffic ([[token-gates-cannot-see-dequant-fallbacks]]), so the memory format
  is checked against the oracle explicitly, per [porting.md](../porting.md),
  and every f32 buffer on this path carries a one-line reason.
- **A bf16 output arm can absorb a real reduction-order defect**
  ([[bf16-store-absorbs-reduction-order-defects]]). Every new test sweeps an
  **f32 output arm** as well.
- **`n_groups` sharding is deferred, not forgotten.** W1 asserts
  `tp_world_size == 1` and refuses otherwise with a message naming
  `extra_groups_for_head_shards`, rather than silently computing a wrong split.
  An arm that is not implemented is refused, never discovered later.
- **Do not "reuse" GDN by parameterising it.** The two recurrences differ in
  structure, not in constants. A shared entry point would make both harder to
  gate. Sibling ops, one shared state layout.

## 8. Now

**State (updated 2026-08-12):** **W1 is landed on `main`** at `47960a009` —
the three CPU host references (`vt::Mamba2ChunkScan`, `vt::Mamba2StateUpdate`,
`vt::RmsNormGatedGroup`) with their unit gates, after two fresh reviews (round 1
FAIL, round 2 PASS). `KERNEL-SSM-MAMBA` stays `INVENTORIED`: this is a host
reference, not generic Mamba support, and no lifecycle state moved, so it owes
no `STATUS.md` / `BENCHMARKS.md` projection. No performance claim is made.

**W2 (2026-08-13):** the CUDA arm for all three ops is implemented and gated —
see §8.3 for the equivalence contract it was written against, which supersedes
§6's "byte-compared to W1" exit criterion with a named reason (the two arms call
different libms, so a byte compare is not reachable; the primary gate is the
device output against the same double-precision reference at the same
upstream-ported tolerances). The §8.2 decode `SUBCASE` is CLOSED and re-proved.
`KERNEL-SSM-MAMBA` still stays `INVENTORIED`: this lands `src/`, `include/`,
`tests/` and this spec only, no lifecycle state moved, and no performance claim
is made.

**Owed before the row can move:** W3 (the `MambaSpec` producer for Mamba2
shapes), a fresh scoped review of W2, and the two residuals named in §8.3 —
routing the device-side `A < 0` and `state_indices` precondition checks through
the deferred error ring, and #547 (`ReferenceTierEligible` gating on
`UnifiedMemory()` where it needs `DeviceMemoryIsHostAddressable()`).

**Next action:** dispatch a fresh scoped review of W2, then W3.

### 8.1 W1 progress (host references landed, awaiting a fresh scoped review)

The three CPU host references and their unit gates are on
`row/KERNEL-SSM-MAMBA-SSD-W1`. `KERNEL-SSM-MAMBA` still stays `INVENTORIED` —
this lands `src/`, `include/` and `tests/` only, which owes no `STATUS.md` /
`BENCHMARKS.md` projection; the row moves when the device arm (W2) and the
`MambaSpec` producer (W3) close.

What landed: `vt::Mamba2ChunkScan` (the 5 stages in upstream order),
`vt::Mamba2StateUpdate` and `vt::RmsNormGatedGroup`, dispatched through the
normal `vt::` seam and gated by
`tests/vt/test_ops_mamba2_{ssd,state_update,gated_norm}.cpp`. The §4 extras both
hold: chunked == a sequential `double` recurrence, and chunk-size invariance
across {8,16,32,64,128}. `tp_world_size > 1` is refused naming
`extra_groups_for_head_shards`.

**Deviation from §3's port map:** the kernels are in `src/vt/cpu/cpu_ops.cpp`,
not a new `src/vt/cpu/cpu_mamba2_ssd.cpp`. A new library TU has to be listed in
the root `CMakeLists.txt`, which `check-doc-checkpoint` classifies as
`user_usage` and `landing_page`, so ANY new `src/vt/` file owes a
`docs/USAGE.md` update — which a kernel that no command, config key or C-ABI
entry point exposes has nothing true to write. Landing it beside the GDN and KDA
references it is a sibling of costs nothing and reds no gate. Whether that
classification should distinguish "a source file was added to the library" from
"how the project is used changed" is a policy question for the operator, not
something to settle by weakening a checker.

Two findings a reviewer should carry forward:

- **`state_dtype` is not cosmetic.** `_state_passing_fwd` stores the inter-chunk
  state at `state_dtype` and `_chunk_scan_fwd` reads that store back, so the knob
  moves `out`, not just `final_states`. The running state itself stays f32
  (ssd_state_passing.py:88-97 never re-reads its own store). Both are pinned by
  test (6) of the SSD suite, the second only after a mutation escaped the first
  version of it.
- **One equivalent mutant, deliberately kept.** Removing the `min(·, 0)` clamp on
  the intra-chunk decay does not fail any test, and cannot: with `A = -exp(A_log)
  < 0` and `dt >= 0`, `dA_cumsum` is non-increasing, so `dA_i - dA_j <= 0` for
  every `j <= i` and the clamp is never active. It is kept because it mirrors
  upstream verbatim and is a real guard outside that input contract. Recorded
  here rather than "fixed" with an out-of-contract test.

### 8.2 W1 repair (fresh review returned FAIL; `row/KERNEL-SSM-MAMBA-SSD-W1-FIX`)

A fresh scoped review of `9003dfad` returned **FAIL** with nine findings. The
port itself was confirmed faithful — independent reference, contract points, the
TP refusal, no `doctest::Approx`, f32 output arms all VERIFIED — and none of
those were re-done. What the repair changed, in the order the findings were
raised:

- **F1 (HIGH, memory-unsafe).** `RmsNormGatedGroupKernel` read the gated-norm
  `weight` through `Tensor::Ptr<float>()`, an unchecked `static_cast`, while the
  validator accepts f16/bf16. Upstream's `Mixer2RMSNormGated.weight` is
  `nn.Parameter(torch.ones(...))` (mamba_mixer2.py:91) at the MODEL dtype, i.e.
  bf16 for these checkpoints — so this was the normal call, not an exotic one:
  a 2x heap over-read plus garbage output (an all-ones bf16 weight, a
  mathematical no-op, shifted the output by `max|Δ| = 1.02523`). Fixed by
  reading it with `LoadF32`, as the sibling `RmsNormGatedKernel` already does.
  Requiring f32 was rejected: it would refuse the dtype vLLM actually passes.
- **F2 / F3, the two unpinned dtype claims.** A mutation that COMPOUNDED the
  running state into its own `state_dtype` store left all three suites green,
  and so did one that re-read the decode readout from the rounded cache. Both
  now carry an EXACT, tolerance-free pin, because a tolerance wide enough for
  the dtype is wide enough for the defect: the f32 recurrence does not depend on
  `state_dtype`, so `final_states(bf16)` must be bit-for-bit `bf16(final_states(f32))`;
  and `out` does not depend on the CACHE dtype at all, so the bf16-cache and
  f32-cache decode runs must be bit-identical. Each test also asserts that the
  rounding it relies on is live, so neither can degenerate into comparing two
  identical computations.
- **F4, the "equivalent mutant" that was not equivalent.** The argument that the
  `min(·, 0)` clamp can never fire is airtight given `A = -exp(A_log) < 0` and
  `dt >= 0` — but the op enforced NEITHER. Fed `A = +1.0` it returned a silently
  truncated recurrence. Both preconditions are now REFUSED: `A >= 0` in the
  kernels (`CheckMamba2ANegative`, where the data lives), `dt_min < 0` and an
  inverted `dt_limit` in the validator. §8.1's "one equivalent mutant,
  deliberately kept" is superseded by this: the clamp is equivalent *inside the
  enforced contract*, and both clamp sites now say so.
- **F5.** The `x.to(input_dtype)` cast at mamba_mixer2.py:149 was unpinned, and
  the test's OWN reference omitted it — it could not have seen the defect. The
  reference now takes an `input_dt`, and a bf16-activation / f32-out case pins
  that every output is bf16-representable. `1/(sqrt(var)+eps)` was also
  unpinned: added a near-zero-variance group, which is the input `eps` exists
  for.
- **F6.** The chunk-size invariance sweep ran `T = 128` over
  `{8,16,32,64,128}`, so at `chunk_size = 128` — Nemotron-3.5's shipped value —
  `nchunks == 1` and the arm exercised no state passing at all. `T` is now 300
  and the test ASSERTS `nchunks > 1` rather than trusting the shape.
- **F7.** `last_chunk_indices[b] == -1` (an empty sequence) was permitted and
  deviated silently; upstream's `states[last_chunk_indices]`
  (ssd_combined.py:154) negative-indexes another sequence's chunk there. That is
  an indexing quirk vLLM never exercises, so the op refuses.
- **F8.** Duplicate `state_indices` were a documented-but-unchecked precondition
  over a PARALLEL row dispatch. Now enforced; repeated NULL rows still allowed.
- **F9, the two over-wide buffers.** The gated-norm sum of squares was reduced in
  `double`; upstream and the sibling kernel reduce in f32, so it was NARROWED
  rather than excused. The `passed` working buffer stays f32 with an explicit
  reason (it holds only `state_dtype`-rounded values, so the width is not
  observable) plus a note that **W2 must not inherit either host-reference
  width**.

All but one of those mutations was re-applied after the repair and is now
caught; the store-side rounding pin from §8.1 was re-checked at the same time
and still reds. **Correction (round-2 review, 2026-08-12):** the claim
originally written here — that *every* one is caught — was wrong for the decode
half of F4. Dropping `CheckMamba2ANegative` at `cpu_ops.cpp:1877` leaves
`test_ops_mamba2_state_update` fully green, while the same mutation on its
chunk-scan twin at `:1633` reds. The guard itself is present, correct and
reachable (a direct probe refuses `A` = `+1.0`, `0.0`, `-0.0` naming `A_log`,
and accepts `-1e-30` and `-9.8e-45`), so this is a missing mutation-proof, not
a defect — ~~**owed:** an "A must be negative" `SUBCASE` on the state-update
refusal case mirroring `tests/vt/test_ops_mamba2_ssd.cpp:900`.~~ **CLOSED in W2**
(`tests/vt/test_ops_mamba2_state_update.cpp:658`). Re-proved here rather than
taken on report, as mutation **M9** of the W2 sweep: deleting
`CheckMamba2ANegative(A, "mamba2_state_update");` at `cpu_ops.cpp:1877` — the
call site is unique, the other two hits of that symbol being its definition at
`:1582` and the chunk-scan call at `:1633` — takes
`test_ops_mamba2_state_update -tc=mamba2 state update refuses the arms it does
not implement` from `1 passed | 0 failed`, `assertions: 11 | 11 passed`,
`Status: SUCCESS!` to `0 passed | 1 failed`, `assertions: 11 | 8 passed |
3 failed`, `Status: FAILURE!` — the three reds being `CHECK(threw)`,
`CHECK(msg.find("A_log") != npos)` and `CHECK_THROWS(...) did NOT throw at all`
at `:672`, `:673`, `:680`. The pristine binary was run under the identical
filter FIRST, so the filter is proved to select a non-zero assertion count
rather than nothing; `cpu_ops.cpp` was restored byte-for-byte and its md5
re-asserted at `9ed9eb980c239eca37ec7d92bfe0e766`.

One repo-wide test trap found while capturing the RED output, and worth carrying
to any doctest suite: **doctest 2.5.2 `INFO` prints a `const char*` VARIABLE as
`1`** — it binds the bool overload; only a string *literal* prints as text. Every
`ExpectClose` in these three suites had a `const char*` label, so a failure
reported `1: worst element ...` instead of naming the tensor. The labels are
`std::string` now.

Named residuals unchanged from §2: the CUDA arm (W2), `n_groups` TP sharding,
spec-decode temporal state, ReplaySSM and Mamba v1. One more, from the port
itself: the Triton dots downcast their tile inputs (`ssd_chunk_state.py:283-285`,
`ssd_chunk_scan.py:266-269`) where this host reference stays f32, so W2's
device-vs-host comparison is a tolerance comparison at the activation dtype, not
a byte compare.

### 8.3 W2 — the declared equivalence contract for the CUDA arm

This section is the spec copy of the contract the W2 implementer decided
**before** writing the kernel and recorded in `src/vt/cuda/cuda_mamba2_ssd.cuh`
and in all three test headers. Its original copy was a staged blob lost with the
worktree; it is re-authored here from the recovery commit `fcdb7d824`, unchanged.
It is a contract, not a tolerance budget: nothing in it was renegotiated to make
a run pass.

**1. f32 accumulation throughout; the upstream tile downcasts are deliberately
NOT mirrored.** Upstream downcasts its tiles before `tl.dot` — `b.to(x_ptr.
dtype.element_ty)` at `ssd_chunk_state.py:283-285`, `cb.to(...)` and
`prev_states.to(C_ptr.dtype.element_ty)` at `ssd_chunk_scan.py:266-269` and
`:359-363`. Those casts are the **input-precision requirement of `tl.dot`**, i.e.
of a tensor-core MMA, not a statement of the algorithm: every one of those tiles
is loaded with an explicit `.to(tl.float32)` and computed in f32 right up to the
instant it is fed to the MMA. These are scalar-FMA kernels with no MMA, so
mirroring the downcast would copy a constraint we do not have, and would be lossy
for nothing.

**2. This is not the "too wide" deviation §7 warns about, and the distinction is
checkable.** A token gate cannot catch a dtype that is too wide
([[token-gates-cannot-see-dequant-fallbacks]]), so the claim is made about the
**memory format**, which is byte-for-byte the host arm's: every load and store
goes through the operand's own declared dtype (`M2Load` / `M2Store`); `states`
and `CB` are f32 because upstream pins them there (`states_in_fp32=True`,
`ssd_combined.py:100-102`; `output_dtype=torch.float32`, `:124`); and the
inter-chunk `passed` buffer is allocated at **`state_dtype`**, *not* at the host
reference's f32 working width, which §8.2 F9 explicitly flagged as a width W2
must not inherit. No extra byte moves. Only the register precision of one product
differs, and it differs in the direction Triton itself takes wherever it is not
feeding an MMA.

**3. Accumulation ORDER is part of the port.** Except in the gated norm's group
reduction — which is a block reduction, and says so at the kernel — every
accumulation runs in ONE thread, over the SAME index range in the SAME direction
as the host reference. That is deliberate: it leaves the elementary functions as
the *only* admitted source of divergence, so the derived bound has exactly one
term to account for.

**4. A byte compare against the host arm is NOT reachable, and the downcasts are
not why.** The two arms call different libms — CUDA `expf` is documented at
≤ 2 ulp, glibc's at ≤ 0.5 — and the gated norm additionally reorders one
non-negative reduction. Everything else is identical by construction. §9's third
stop condition ("the device arm cannot reach the host reference byte-for-byte")
is therefore resolved as **not reachable for a named, non-defect reason**, and
the gap is kept open in the form below rather than closed by widening anything.

**5. The primary gate is therefore NOT device-vs-host.** It is the device output
against the **same double-precision sequential reference** the host arm is held
to, at the **same upstream-ported tolerances**, on the **same inputs** — e.g. atol
8e-3 / rtol 5e-3 from `test_mamba_ssm_ssd.py:210-213` on the driver shapes. Both
arms are asserted against it in the same test case, so a failure separates
cleanly: device-only means a device defect; both means the cited upstream
threshold does not cover this shape, which is a `NEEDS_DECISION`, not a wider
tolerance.

**6. The derived device-vs-host bar is `rtol(K) = 4·(K+2)·2⁻²⁴`**
(`ExpectDeviceMatchesHost` / `DerivedRtol`, `tests/vt/test_ops_mamba2_ssd.cpp`),
from 2.5 ulp of libm disagreement per decay factor through a product of at most
`K`, plus `(K-1)·u` of summation error. **No number was tuned and no tolerance
was widened**; `K` is the sequence length the comparison actually ran at.
Because a bar nobody audits is a false claim, every comparison logs the
**fraction of the budget actually used** through `MESSAGE` — not `INFO`, because
doctest prints `INFO` only on failure, so an `INFO` would have been invisible on
the green run that the claim rests on.

**Two deviations recorded with the arm, carried forward deliberately:**

- **Placement.** The kernels live in `src/vt/cuda/cuda_mamba2_ssd.cuh`, included
  by `cuda_gdn.cu`, not in a new `.cu`. Same reason as W1's `cpu_ops.cpp`
  placement (§8.1): a new library TU must be listed in the root `CMakeLists.txt`,
  which `check-doc-checkpoint` classifies `user_usage` + `landing_page`, so any
  new `src/vt/` file owes a `docs/USAGE.md` update that a kernel exposing no
  command, config key or C-ABI entry point has nothing true to write (#515).
- **The device arm does not re-check `A < 0` or `state_indices` distinctness.**
  Both operands are on-device; re-reading them costs a D2H plus a stream
  synchronise per call — the same host tax the GDN prefill path was rebuilt to
  remove — and makes the op uncapturable in a CUDA graph. This mirrors the policy
  `cuda_gdn.cu:8-13` already states for exactly this case. The kernels stay
  **memory safe** under a violation: an out-of-range `state_indices` slot writes
  nothing at all rather than out of bounds. **Owed, not implemented here:** route
  both through the deferred device error ring at `cuda_ops.cu:790-940`.

**Not fixed here, filed as #547.** The W2 RED run SIGSEGV'd on all three
binaries. GB10 reports `Backend::UnifiedMemory() == true`, so
`ReferenceTierEligible(kCUDA)` is true and, with no native kernel registered,
`GetOp` installs the CPU host kernel as a `vt-cpu-ref` provider over `cudaMalloc`
pointers — which `include/vt/backend.h` already says are not host-dereferenceable
on GB10. `op_provider.cpp:515-526` gates on `UnifiedMemory()` where it needs
`DeviceMemoryIsHostAddressable()`. That is shared-seam semantics across three
backends, so it takes its own row. Every CUDA case here calls
`RequireNativeCudaProvider`, so a device arm can never be gated by running the
host arm twice.

### 8.4 W2 evidence (gate host `promaxgb10-4ad8`, GB10 / sm_121a, 2026-08-13)

Build recipe, both arms: `cmake -G Ninja -DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=121a -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0
-DVLLM_CPP_TRITON=ON -DVLLM_CPP_BUILD_TESTS=ON`. The configure log was READ, not
assumed: `cutlass-nvfp4: ENABLED`, `cutlass-fp8: ENABLED`, `marlin-nvfp4:
ENABLED`, `fa2: ENABLED for [121a]`, `CUTLASS found at ~/cutlass-4.5.0` — an
absent CUTLASS silently falls back and the arm would not be the shipped one.
**0 warnings** in both build logs.

**Release, after the `origin/main` re-merge** — identical to the pre-merge
counts, so the merge moved nothing:

| suite | test cases | assertions | status |
|---|---|---|---|
| `test_ops_mamba2_ssd` | 11 / 11 passed | 2069 / 2069 | `SUCCESS!` |
| `test_ops_mamba2_state_update` | 10 / 10 passed | 5965 / 5965 | `SUCCESS!` |
| `test_ops_mamba2_gated_norm` | 12 / 12 passed | 3723 / 3723 | `SUCCESS!` |

**Debug arm** (`CMAKE_BUILD_TYPE=Debug`, so `NDEBUG` is OFF and every `assert`
in the tree is live; CXX `-g -O0`, CUDA `-g` and deliberately *not* `-G`, which
would disable device optimisation and change what was measured): the same
`11 / 2069`, `10 / 5965`, `12 / 3723`, all `SUCCESS!`, exit 0. This arm exists
because the gate build is `-O3 -DNDEBUG`, where an assert-abort defect stays
latent behind a green Release run.

**`compute-sanitizer`, 8 runs, all `ERROR SUMMARY: 0 errors` / `EXIT=0`:**
`memcheck` on the ssd optional-arms + dtype-knobs case, on the continuous-batch
`initial_states` subcase, and on both decode suites' `*CUDA arm*` cases;
`initcheck` on three; `synccheck` on the gated norm, which is the one kernel
with a block reduction and `__syncthreads`.

**Mutation sweep — 9 of 9 CAUGHT.** Each mutation patches one source file, is
rebuilt, and is run under a doctest `-tc` filter; the **pristine** binary is run
under the *identical* filter first, because a filter that selects no test case
makes doctest print `SUCCESS!` and an unverified filter would score a false
catch. Sources restored byte-for-byte after each, md5 re-asserted
(`cuda_mamba2_ssd.cuh` `cbb1f928f4b421bdea2e24476012eed2`, `cpu_ops.cpp`
`9ed9eb980c239eca37ec7d92bfe0e766`).

| # | mutation | control assertions | mutant |
|---|---|---|---|
| M1 | drop the inter-chunk state term | 27 `SUCCESS!` | `FAILURE!` |
| M2 | ignore `initial_states` in state passing | 297 `SUCCESS!` | `FAILURE!` |
| M3 | read `states[c]` for `states[c-1]` | 297 `SUCCESS!` | `FAILURE!` |
| M4 | drop the `D` skip connection | 570 `SUCCESS!` | `FAILURE!` |
| M5 | ignore `state_indices` (slot = row) | 1318 `SUCCESS!` | `FAILURE!` |
| M6 | treat the NULL row as slot 0 | 1318 `SUCCESS!` | `FAILURE!` |
| M7 | whole-row variance instead of per-group | 9 `SUCCESS!` | `FAILURE!` |
| M8 | sigmoid instead of silu | 9 `SUCCESS!` | `FAILURE!` |
| M9 | drop `CheckMamba2ANegative` on decode (§8.2) | 11 `SUCCESS!` | `FAILURE!` |

**Two mutations had to be REFORMULATED, and that is worth carrying.** The
obvious form of M1 (`if (!prev_zero)` → `if (false)`) and of M7 (passing
`1, hidden` for `n_groups, group_size`) do not COMPILE: the CUDA arm is built
`-Werror=all-warnings`, and nvcc raises `#550-D "prev_zero was set but never
used"` and `#177-D "group_size was declared but never referenced"` once the
mutation dead-codes the read. A mutation that will not build is not a caught
mutation and must not be scored as one. Both were rewritten to drop exactly the
same term while leaving every variable read — M1 multiplies the inter-chunk
product by `0.0f`, M7 passes `1, group_size * args.n_groups` (which *is*
`hidden`) — and both then failed as intended.

**The derived bar is audited, not asserted.** Across the 55 device-vs-host
comparisons in a green run, the worst one used **7.66%** of `rtol(K) =
4·(K+2)·2⁻²⁴`; the driver shapes used 0.32% and 0.18%. For contrast the same
`MESSAGE` line under mutant M3 reads `used 962173% of its derived budget`. So
the bound is neither tuned down to the observed error nor wide enough to hide a
defect.

## 9. Stop conditions

- The chunked scan cannot be made to match the sequential double-precision
  recurrence at any chunk size → stop, report `NEEDS_DECISION`, do not widen
  the tolerance to pass. A tolerance widened to make a scan agree with itself
  is the defect, not the gate.
- A required upstream contract point is ambiguous in source *and* the oracle
  cannot be run to settle it → `NEEDS_CONTEXT`; do not guess a mirrored
  behaviour ([mirror, never ask how a feature should behave] applies to product
  decisions, not to unread source — read the source first).
- The device arm cannot reach the host reference byte-for-byte → keep the gap
  open, name the next traceable hypothesis, and never record it as a ceiling.
