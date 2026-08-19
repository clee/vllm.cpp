# A2-B — NemotronH decodes a BATCH: the ordering contract, the mamba split, and the end of the `num_reqs <= 1` refusal

**Issue:** [#1395](https://github.com/mudler/vllm.cpp/issues/1395).
**Umbrella:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Predecessor:** [`nemotron-h-a2p-paged-forward.md`](nemotron-h-a2p-paged-forward.md)
— A2-P states at `:64`, `:68`, `:81` and `:146` exactly what it left here.
**Governing spec:** [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md) §1 `A2b`.
**Siblings:** [`nemotron-h-a2q1-fp8-mamba.md`](nemotron-h-a2q1-fp8-mamba.md),
[`nemotron-h-a2q2-nvfp4-moe-lmhead.md`](nemotron-h-a2q2-nvfp4-moe-lmhead.md).
**Base:** `origin/main` @ `5f68e60df22670a714f31d6362695b012b2598e2`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98`, `git rev-parse HEAD` verified while
writing this spec against the `parity-pin` block in
[`upstream-sync.md`](../upstream-sync.md), and `git remote -v` verified to be
`https://github.com/vllm-project/vllm` rather than a fork. Every `file:line`
below was re-derived in that checkout at that SHA; §Upstream chain records the
one anchor the filing issue got off by one.
**Lifecycle at this commit:** unchanged. A spec commit changes no lifecycle state
and owes no `STATUS`/`BENCHMARKS`/`NOW` write. §Records owed lists what the
implementing change owes.

**No product code is written by this spec, deliberately.** AGENTS.md permits the
split for "a large campaign benefits from agreement on the scope before
implementation waves start", and there is a second, concrete reason:
`row/…A2Q2B` is editing `nemotron_h_device.cpp` for the device `lm_head` at the
time of writing, and A2-B's surface is the same file.

---

## 0. What A2-B is, in one paragraph

`ForwardNemotronHForCausalLM` refuses every step carrying more than one request
— `src/vllm/model_executor/models/nemotron_h_registry.cpp:161-170`. A2-B lifts
that refusal. **The lift is much smaller than the refusal's wording suggests,
and this spec's first job is to say so precisely**, because A2-P did not merely
leave a placeholder: its §4.1 decision put the per-request indexing machinery in
at `num_reqs == 1`, and the runner has carried the four-way decode-first reorder
since M1.8. What is genuinely missing is narrower and sharper: **our
decode/prefill split implements the wrong one of upstream's two modes for this
architecture's backend**, nothing verifies the ordering contract the paged
forward silently assumes, and no gate in the tree has ever run this model with
two requests in one step.

A2-B is the **contract** unit. It is not a performance unit — but it is the unit
after which a throughput comparison against vLLM's production configuration
becomes *possible*, which is why #1395 files it as a blocker on every Nemotron
speed number rather than as a feature.

---

## Scope

### The G-SAFE clause, verbatim

`src/vllm/model_executor/models/nemotron_h_registry.cpp:161-170`:

```cpp
VT_CHECK(
    input.num_reqs <= 1,
    "Model architecture NemotronHForCausalLM: BATCHED decode is not ported "
    "(issue #810, .agents/specs/nemotron-h-a2p-paged-forward.md A2-B). ...");
```

| Clause | A2-B | Why |
|---|---|---|
| `input.num_reqs <= 1` | **CONSUMED — the clause is dropped** | this unit is that clause |
| `input.gdn_meta.num_spec_decodes == 0` (`nemotron_h_device.cpp:1216-1219`) | **STAYS, untouched** | the MTP head is #517 W5; a speculative row is not a batching row and A2-B must not absorb it |
| the `ModelAs<NemotronHLoadedModel>` checked downcast (`:174`) | **STAYS, untouched** | #775; unrelated to the request count |

> **The registry refusal is REPLACED, not simply deleted.** Dropping
> `num_reqs <= 1` removes the only thing standing between this forward and a
> batch it has never been run on. §Design D4 specifies what takes its place: a
> positive assertion, inside `BuildNemotronHPagedStep`, that the batch satisfies
> the decode-first ordering contract the forward already reads
> `gm.num_decodes` / `gm.num_prefills` as if it did. **A reviewer who finds the
> count clause removed and nothing asserted in its place returns FAIL.**

### In and out

| In A2-B | Out of A2-B |
|---|---|
| a mamba-flavoured decode/prefill split mirroring `mamba_attn.py:455-469`, including `treat_short_extends_as_decodes=False` and the single-token-prefill promotion | GDN's (Qwen3-Next's) existing `SplitDecodesAndPrefills` semantics, which are correct for `gdn_attn.py:213` and must stay byte-identical for that caller |
| a positive ordering assertion replacing the count refusal | any change to the runner's reorder, which already mirrors `utils.py:665` and is not this unit's to rewrite |
| the `is_prefilling` per-request signal that `split_decodes_and_prefills` needs under `treat_short_extends_as_decodes=False`, plumbed onto `v1::CommonAttentionMetadata` from data the runner already computes | `require_uniform` — **upstream's mamba backend never passes it** (§Upstream chain), so mirroring it here would be inventing a mode |
| a red-first multi-request gate through `ModelRegistry::Forward`, A/B against the same requests run singly | speculative rows (`num_spec_decodes > 0`), which stay refused — #517 W5 |
| the ported upstream tests for the split's modes, with their parameters and failure cases | the FP8 mamba arm (A2-Q1, [#940](https://github.com/mudler/vllm.cpp/issues/940), PR #1289) and the NVFP4 `lm_head` arm (A2-Q2b) |
| the CUDA-graph question answered by measurement (§Design D5) | writing a NemotronH decode graph driver, and any change to `GraphEligibleQueryLen` |
| a throughput-comparison *possibility* | any throughput, latency or memory number. **A2-B records none.** The measurement is the umbrella's next unit, and quoting one from this row is how a correctness unit acquires an unearned speed claim |
| mamba prefix caching for the multi-request case only insofar as it is refused by name | implementing `mamba_cache_mode == "all"` (upstream `mamba_mixer2.py` prefix-cache branch), which A2-P also left unported |

### What A2-B explicitly does NOT deliver

Stated so the implementer cannot silently widen scope, and so a reviewer has a
list to check against:

1. **No speed number, on any axis.** Not a ratio, not a token/s, not a "for
   reference" figure in the PR body.
2. **No speculative decode.** `num_spec_decodes > 0` still refuses by name.
3. **No prefix caching for the recurrent state.**
4. **No device `lm_head` and no FP8 mamba.** The host arms stay where A2-P left
   them; the `want`/`gathered` tail is already per-row general and A2-B touches
   it only if a gate proves otherwise.
5. **No parallelism inside the mamba per-request loop.** The loop stays serial
   host compute. Making it parallel is a performance change with its own
   correctness surface and it is not this unit.
6. **No new `require_uniform` mode**, and no change to `SplitDecodesAndPrefills`
   as GDN calls it.

---

## Upstream chain

Every line re-derived at `5559679229bc961848b121ccdeaa8fa5d79bec98`, and each
asserted **unique** with a `grep -n` over its own file rather than merely
present, because a wrong line number still holds plausible unrelated code.

| Concern | Upstream `file:line` | Verified |
|---|---|---|
| decode threshold for every mamba-family backend | `vllm/v1/attention/backends/mamba_attn.py:87` — `reorder_batch_threshold: int = 1` | **CONFIRMED, and unique** — `grep -n 'reorder_batch_threshold: int = 1'` returns exactly `87` |
| how the threshold is established | `mamba_attn.py:200` — `self._init_reorder_batch_threshold(1, self.use_spec_decode)` | **CONFIRMED, and unique** — the only `_init_reorder_batch_threshold` in the file |
| the split itself | `vllm/v1/attention/backends/utils.py:564` — `def split_decodes_and_prefills(` | **CONFIRMED** — `grep -n 'def split_decodes_and_prefills'` returns exactly `564` |
| the batch ordering contract | the same docstring, `utils.py:571-572` — "The batch is expected to be ordered as: `decode → short_extend → long_extend → prefill`" | **CONFIRMED.** Note the arrow is `→`, not `->` |
| per-request counts in the metadata | `mamba_attn.py:31` `num_prefills: int`, `:33` `num_decodes: int` | **CONFIRMED, both unique** at the top-level indent |
| the call site | `mamba_attn.py:464` | **CORRECTED.** #1395 cites `:463`. `:463` is the assignment `num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (`; the call `split_decodes_and_prefills(` is on `:464`, and `464` is the only call site in the file (`:22` is the import). Off by one, and the anchor to cite is `mamba_attn.py:464-468` because the argument list is what matters |
| **the argument this architecture actually passes** | `mamba_attn.py:467` — `treat_short_extends_as_decodes=False` | **CONFIRMED.** This is the single most important anchor in the unit |
| the single-token-prefill promotion, immediately before the call | `mamba_attn.py:455-462` | **CONFIRMED** — `single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)`, then `has_prior_state = seq_lens_cpu > 1`, then rows satisfying both are flipped OUT of `is_prefilling` before the split runs |
| what `treat_short_extends_as_decodes=False` costs | `utils.py:623-625` — `assert common_attn_metadata.is_prefilling is not None` then `is_prefill |= common_attn_metadata.is_prefilling` | **CONFIRMED** |
| the GDN backend, for contrast | `vllm/v1/attention/backends/gdn_attn.py:213` — `split_decodes_and_prefills(m, decode_threshold=1)`, both flags left at their defaults | **CONFIRMED** |
| the reorder | `utils.py:665` `def reorder_batch_to_split_decodes_and_prefills(`, driven from `vllm/v1/worker/gpu_model_runner.py:1127` | **CONFIRMED** |
| the model | `vllm/model_executor/models/nemotron_h.py` — `NemotronHMambaDecoderLayer:360`, `self.mixer = MambaMixer2(:373)`, `NemotronHAttention:409`, layer map `:532-534` | **CONFIRMED.** Nemotron-H is still in the OLD layout at this pin; it is not under `vllm/models/` |
| the mixer that consumes the split | `vllm/model_executor/layers/mamba/mamba_mixer2.py:751-767` (`num_prefills`, `has_prefill`, `has_decode`, then the decode-first `torch.split`) and `:808-812` (output split) | **CONFIRMED** |
| CUDA-graph capture is DECODE-ONLY for mamba | `mamba_attn.py:204-220` `build_for_cudagraph_capture`, asserting `m.max_query_len == 1 + self.num_spec_tokens`; `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH` at `:88` | **CONFIRMED** |

### ★ `require_uniform` — the issue's framing is half right, and the correction matters

#1395 and the dispatch both say `require_uniform` and
`treat_short_extends_as_decodes` "both change which requests count as decodes".
That is true of the **function**. It is not true of **this architecture's
caller**. A `grep -rn 'split_decodes_and_prefills(' vllm/ -A 6` at the pin
returns 13 call sites. `require_uniform=True` is passed by `flashinfer.py:1129`,
`mla/indexer.py:791`, `minimax_m3/…/indexer.py:287`,
`minimax_m3/common/sparse_attention.py:242`, `hpc_attn.py:186` and
`mla_attention.py:1952`. **`mamba_attn.py` passes it nowhere**, so it holds its
`False` default on every mamba step.

The spec therefore says what each does and what we do about it, as required:

| Mode | Upstream default | What `mamba_attn.py` passes | A2-B |
|---|---|---|---|
| `decode_threshold` | `1` | `decode_threshold=decode_threshold`, itself `reorder_batch_threshold` = `1` unless spec decode widens it (`:200`) | **mirror the value 1.** The spec-decode widening is out of scope because speculative rows stay refused |
| `require_uniform` | `False` | **not passed ⇒ `False`** | **mirror by NOT implementing it.** Adding a mode this architecture's backend cannot reach would be an invention with no oracle, and AGENTS.md's "mirror every applicable mode" turns on *applicable*. Recorded here so the absence is a decision with a reason, not an omission. Uniformity enters this architecture through the **CUDA-graph** path instead (`mamba_attn.py:204-220`), which §Design D5 handles |
| `treat_short_extends_as_decodes` | `True` | **`False`, explicitly (`:467`)** | **mirror it.** This is the divergence, and §Our baseline shows we currently implement `True` |

### What `treat_short_extends_as_decodes=False` means, concretely

A **short extend** is a request whose query length this step is `<= threshold`
but which is *still prefilling* — a chunked prefill whose final chunk happens to
be one token, or a request the scheduler clamped. Under `True` it is counted as a
decode. Under `False` it is counted as a prefill.

For a Mamba2 mixer that distinction is not bookkeeping. A decode row enters
`causal_conv1d_update` + `selective_state_update` reading and writing the same
state slot in place; a prefill row enters `causal_conv1d_fn` +
`mamba_chunk_scan_combined_varlen` with `has_initial_states_p` deciding whether
it continues. Misclassifying a still-prefilling row as a decode runs the
single-step recurrence over a row that has more than one token of context to
absorb, and — this is the part that matters for the gate — **it produces fluent
output, not an error.**

Upstream then softens the flag in one exact direction at `:455-462`: a row that
is prefilling, has query length exactly 1, **and** has `seq_len > 1` (so it does
have prior state) is promoted back to a decode, with the comment that `ReplaySSM`
handles it as a single-token flush. The two pieces are one behaviour and must be
ported together. Porting `treat_short_extends_as_decodes=False` without the
promotion is strictly wrong at the pin, and porting the promotion without the
flag is a no-op.

---

## Our baseline

Measured in this worktree at `origin/main` @ `5f68e60df`, by reading the tree
rather than by inheriting A2-P's description of it. **Three of the four things
A2-B was expected to build already exist**, and the one that does not is not the
one the issue points at.

### ★ The correction A2-P's `:81` owes

A2-P's scope table says the `num_reqs <= 1` clause stays because "nothing in
this unit … indexes per-request state by anything but slot 0". **That describes
A2-P's scope, and the landed implementation is strictly more general than its own
scope line.** A2-P's §4.1 made the opposite decision on purpose — "INDEX THROUGH
THE VECTORS EVEN AT ONE REQUEST … a forward that hardcodes slot 0 passes every
gate A2-P owns and then fails silently under A2-B" — and the code carries it.
The enumeration the dispatch asked for, site by site:

| Site | What it does today | What A2-B changes |
|---|---|---|
| `nemotron_h_device.cpp:1226-1247` — the `idx` / `init` build in `BuildNemotronHPagedStep` | loops `for r in [0,R)`, reads `gm.non_spec_state_indices_tensor[r]`, range-checks it against `state_slots`, sets `init[r] = 1` for `r < nd` and reads `gm.prefill_has_initial_state[r - nd]` otherwise | **nothing structural.** The `r < nd` branch is where the ordering contract becomes load-bearing, and D4's assertion is added HERE |
| `:1249-1256` — the `NemotronHPagedStep` upload | uploads `slot_mapping [T]`, `block_table [R, cols]`, `seq_lens [R]`, `query_start_loc [R+1]`, `state_idx [R]`, `state_has_initial [R]` | **nothing.** Already `R`-shaped end to end |
| `:1443-1457` `GatherNemotronHState` | allocates `[R, Cd, Kw-1]` and `[R, Hh, P, N]`, calls `vt::GdnStateGather(…, sdi.state_idx.t(), &hinit)` twice | **nothing.** `GdnStateGather` takes the index vector and the mask; both are extent `R` |
| `:1459-1471` `ScatterNemotronHState` | `vt::GdnStateScatter` writes only the rows named by `state_idx`; its own comment says this "keeps two concurrent sequences from overwriting each other once A2-B lifts the request count" | **nothing.** This is the property A2-B must now *prove*, and B-M3 is the mutation that proves it |
| `:1631-1706` — the mamba layer body | downloads `conv_all` / `ssm_all` at `[R, …]`, loops `for r in [0,R)`, slices `r*conv_row` and `r*ssm_row`, reads `[t0,t1)` from `gdn_meta.non_spec_query_start_loc`, calls `NemotronHMamba2Mixer` per request, writes back per request, re-uploads and scatters | **nothing structural.** Two real defects fall out at `R > 1` and are listed below |
| `:1637-1643`, `:1707-1714` — the `[NH-DIAG]` prints | `DiagL2(conv_all, 0, conv_row)` — request **0 only**, at every `R` | **fix.** A diagnostic that silently reports one request of a batch is the instrument that makes a batching defect invisible during triage. Either print per request or name the request in the format string |
| `:1477-1614` — the attention half and embedding | `NemotronHAttnBlockPaged` takes `sdi.block_table [R, cols]`, `sdi.seq_lens [R]`, `sdi.query_start_loc [R+1]` into `vt::PagedAttention`, which is varlen by construction; `pa.query_start_loc_host = meta.query_start_loc.data()` is the runner's own `R+1` vector | **nothing to write; something to PROVE.** See D1 |
| `:1765-1785` — the logits tail | `logits_indices` is a per-row index list and an empty one means "every row" | **nothing.** Already general |

**So the per-request state indexing is done, and the honest statement of this
unit is that A2-B *verifies and hardens* it rather than building it.** That
changes the shape of the work — most of the budget is gate, not code — and it
changes what a reviewer looks for.

Two real `R > 1` defects the audit did find in that body, both in the mamba
block:

- **D-B1, `init[r] = 1 for r < nd`.** Correct only if the first `nd` rows really
  are the decodes. Nothing in the forward checks it, and under
  `treat_short_extends_as_decodes=False` the set of rows that *are* decodes
  changes. D4 asserts it.
- **D-B2, the diagnostic.** Above.

### The runner already delivers an ordered batch

`src/vllm/v1/worker/gpu/runner.cpp:126-190`,
`reorder_batch_to_split_decodes_and_prefills`, ported from `utils.py:665`. It is
called **unconditionally** at `runner.cpp:1257`, before any metadata is built,
with the default `decode_threshold = 1` (`include/vllm/v1/worker/gpu/runner.h:103`).
It classifies four-way into `decode(0) → short_extend(1) → long_extend(2) →
prefill(3)` (`runner.cpp:147-162`) on exactly upstream's predicates:
`has_context = num_computed > 0`, `is_below = num_scheduled <= decode_threshold`,
`done_prefilling = num_computed >= num_prompt`.

**So the answer to "do we reorder, or require the runner to present an ordered
batch" is settled by what exists: the runner reorders, and A2-B requires the
ordered batch.** A2-B adds no reorder of its own. It adds the assertion that the
requirement held, which is what turns an assumption into a contract.

Note the runner's classifier already computes `done_prefilling` — the exact
signal `is_prefilling` is. It is computed and then discarded.

### ★ The one thing that is genuinely wrong: our split implements the other mode

`src/vllm/v1/attention/backends/gdn_attn.cpp:12-13`, verbatim:

```
// Ported from utils.py::split_decodes_and_prefills @ e24d1b24 (lines 564-633),
// T0 subset: require_uniform=False, treat_short_extends_as_decodes=True.
```

`SplitDecodesAndPrefills` (`gdn_attn.cpp:15-53`) has no `is_prefilling` term at
all: it takes the `max_query_len <= decode_threshold ⇒ all decodes` early return
(`:23-25`), then argmax's the first `query_len > threshold`. That is exactly
`treat_short_extends_as_decodes=True`.

That port is **correct for GDN** — `gdn_attn.py:213` passes the defaults — and it
is called for the GDN metadata at `gdn_attn.cpp:156` with
`decode_threshold = 1`. NemotronH's `GDNAttentionMetadata` comes off that same
builder. So today NemotronH's `num_decodes` / `num_prefills` are computed with
Qwen3-Next's flag settings, not with its own backend's.

At `num_reqs <= 1` the difference is unobservable: a single request is either the
whole decode side or the whole prefill side under either mode. **The refusal has
been hiding a real mirroring gap, not only an unbuilt feature.** That is the
finding that most changes how this unit should be reviewed.

`v1::CommonAttentionMetadata` (`include/vllm/v1/attention/backend.h:107-140`) has
`query_start_loc`, `seq_lens`, `num_computed_tokens_cpu`, `num_reqs`,
`num_actual_tokens`, `max_query_len`, `max_seq_len`, `block_table_tensor`,
`slot_mapping` — and **no `is_prefilling`**. `grep -rn is_prefilling src/ include/ tests/`
finds it only in `v1/engine/output_processor`, an unrelated per-request flag.

### The interlock test that exists

`tests/vllm/models/test_nemotron_h_paged_forward.cpp:1356-1390`, "NemotronH
paged: G-SAFE still refuses a BATCHED step by name", asserts the refusal
contains `NemotronHForCausalLM`, `BATCHED decode is not ported`, `A2-B` and
`#810`. **A2-B rewrites this case rather than deleting it** (§Gates G5).

The harness the multi-request gate needs is already there: `Fixture`,
`RunnerGreedy` (`:418`), `MakeNewReq`, `NewStep`, `DecodeStep` (`:400-414`, which
already takes a **vector** of ids), and every existing case constructs
`GPUModelRunner(..., /*max_num_reqs=*/2, ...)`.

---

## Port map

| Upstream | Local | Change |
|---|---|---|
| `utils.py:564-633` under `treat_short_extends_as_decodes=False` + `mamba_attn.py:455-462` | new, beside `SplitDecodesAndPrefills` in `src/vllm/v1/attention/backends/gdn_attn.cpp` | **ADD a second entry point**, do not re-flag the existing one. GDN's caller must stay byte-identical (§Risks R1). Name it for what it is — the mamba-family split — and cite `mamba_attn.py:464-468` in its header |
| `utils.py:623-625` `is_prefilling` | `v1::CommonAttentionMetadata` in `include/vllm/v1/attention/backend.h` | **ADD** `std::vector<int32_t> is_prefilling` (or `std::vector<uint8_t>`; mirror upstream's per-request bool). Populated in `runner.cpp` from the `done_prefilling` the reorder classifier already computes at `:147`. Empty ⇒ the mamba split must refuse rather than assume, because an empty vector read as "nothing is prefilling" is the silent-wrong-answer shape |
| `mamba_attn.py:87`, `:200` | the mamba split's `decode_threshold` argument | value `1`, with the anchor in the comment. No new constant: `gdn_attn.h:75` already defaults to 1 |
| `mamba_attn.py:31,33` → `mamba_mixer2.py:751-767` | `GDNAttentionMetadata::num_decodes` / `num_prefills`, consumed at `nemotron_h_device.cpp:1197-1198` | **REWIRE** the NemotronH path to the mamba split. Scope the rewire to this architecture; nothing that reaches GDN's own models may change |
| `nemotron_h_registry.cpp:161-170` | same | **REPLACE** per D4 |
| `mamba_attn.py:204-220` | — | **NO port.** D5 records the measurement instead |

---

## Design

### D1 — Per-request KV page indexing: what changes, and where

**Nothing changes. The obligation is a proof, not an edit.**

`NemotronHAttnBlockPaged` (`nemotron_h_device.cpp:1370-1400`) writes K/V with
`vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, sdi.slot_mapping.t())` — the
slot map is per **token** (`[T]`) and carries no request assumption — and reads
with `vt::PagedAttention(..., sdi.block_table.t(), sdi.seq_lens.t(),
sdi.query_start_loc.t(), pa)`, which is varlen over `[R, cols]`, `[R]` and
`[R+1]`. The four `VT_CHECK`s at `:1180-1191` already scale in `R`.

Two things a reviewer must confirm rather than assume, both because "the shapes
are right" is not "the kernel serves this population":

1. **The mixed-batch arm.** A batch with some rows at `query_len == 1` and some
   at `query_len > 1` must be served by `vt::PagedAttention` on **each backend
   the gate runs**, not only on the one that happens to dispatch a varlen
   launcher. `pa.causal` is a single flag for the whole batch, which is correct
   for both populations (causal masking is per query position), but that is an
   argument, and an argument is not a measurement.
2. **`pa.query_start_loc_host`** points at `meta.query_start_loc.data()`, the
   runner's own vector, which outlives the call and is `R+1` long. The
   host-resident grid-sizing path (`ops.h` `PagedAttentionArgs`) is the one that
   avoids a per-layer D2H sync; it must be exercised at `R > 1`, because a
   fallback to the D2H path would be correct and silent.

If either fails, that is a finding for the op, not a reason to widen A2-B. File
it and refuse by name.

### D2 — Per-request recurrent state

Enumerated in §Our baseline. Summary: `GatherNemotronHState`,
`ScatterNemotronHState`, the `idx`/`init` build and the mamba layer loop are all
already `R`-general; A2-B changes **D-B1** (the ordering assumption behind
`init[r] = 1 for r < nd`, handled by D4) and **D-B2** (the request-0 diagnostic).

The `state.has_initial = true`-always decision A2-P documented at
`nemotron_h_device.cpp:1408-1442` stays, and it is *more* important at `R > 1`,
not less: the gather's zeroing is what keeps a fresh request from reading the
previous tenant of a reused slot, and coupling the mixer's flag to the mask would
make that zeroing unobservable. B-M4 is the mutation that keeps it honest.

**One property is newly load-bearing and must be asserted directly rather than
inferred from tokens:** two requests in one step must land on **distinct** state
slots. `BuildNemotronHPagedStep` range-checks each `s` against `state_slots`
(`:1229-1231`) but never checks the values are distinct. Duplicate slots would
make two requests share a recurrent state — fluent output, wrong tokens. Add the
distinctness check where the range check is, and gate it (B-M6).

### D3 — Reordering: the runner delivers, A2-B requires and verifies

Settled by §Our baseline: `runner.cpp:1257` reorders every step,
unconditionally, on upstream's predicates. A2-B adds **no** reorder.

What A2-B adds is the mamba-flavoured split (Port map row 1) and the assertion
that the batch it is handed satisfies the ordering the split's argmax assumes.
Upstream's own docstring is the specification: "Assuming a reordered batch,
finds the boundary" (`utils.py:570`). Both our port and upstream's original
return `{0, num_reqs, 0, num_tokens}` the moment `query_lens[0] > threshold`
(`gdn_attn.cpp:31-33`, `utils.py:607-609`) — an *unordered* batch is therefore
not detected, it is silently reclassified as all-prefill.

### D4 — What replaces the G-SAFE clause

**Recommendation, and the implementer should not deviate without a
`NEEDS_DECISION`:**

- `nemotron_h_registry.cpp` — drop `input.num_reqs <= 1` entirely. Do **not**
  narrow it to a smaller count; a count is not the property.
- `BuildNemotronHPagedStep` (`nemotron_h_device.cpp:1193-1247`) — add, beside the
  existing `nd + np == R` check, a positive assertion of the contract the
  function's own comment already claims:
  - every one of the first `nd` requests has `query_start_loc[r+1] -
    query_start_loc[r] <= 1`, and no request after index `nd` has a query length
    `<= 1` **unless** the mamba split classified it as a prefill for the
    `is_prefilling` reason (which is the `treat_short_extends_as_decodes=False`
    case and is legitimate);
  - the `R` state slots are pairwise distinct (D2);
  - the message names the architecture and names #1395, per the same discipline
    A2-P's refusal follows.
- `num_spec_decodes == 0` (`:1216-1219`) is untouched and remains the speculative
  refusal.

Who owns the remainder: speculative rows → #517 W5 / #810; mamba prefix caching
→ #810, still unported and still unrefused by name, which is itself worth an
issue if the implementer finds no existing one.

### D5 — Interactions

| With | Interaction | Disposition |
|---|---|---|
| **A2-Q1** (FP8 mamba, [#940](https://github.com/mudler/vllm.cpp/issues/940), PR #1289 — **LANDING BUT HELD `DRAFT`**, not landed and not abandoned; see D6) | swaps the arm **inside** the per-request mamba loop; the loop, the gather and the scatter are unchanged by it. No seam conflict — but a **textual** conflict in `nemotron_h_device.cpp` is near-certain | order is free. Whoever lands second rebases and re-runs the focused gate on the merge result, not on either parent. `merge-tree` reporting clean is not the same as the merge building. **A2-B must not wait for it and must not assume it** — §Gates G6 says what A2-B measures on each side of that arm |
| **A2-Q2b** (device `lm_head`, in flight) | rewrites the `want`/`gathered` tail (`:1765-1785`). That tail is already per-row general | order is free; same rebase rule. A2-B must not pre-empt the tail |
| **CUDA graph capture** | `GraphEligibleQueryLen` (`src/vllm/v1/worker/gpu/cudagraph_dispatch.h:161`) admits `R > 1` uniform decode steps, and `runner.cpp:1510` calls it on the shared path for **whatever model the step routes to**. Lifting the refusal therefore newly admits NemotronH multi-request decode steps into the eligible population | **measure, do not assume.** NemotronH registers no decode graph driver, so the expectation is that the predicate names a length and nothing captures. That expectation is exactly the "absent hook looks like an armed instrument" shape. The gate reads `GraphDispatchStats` (`cudagraph_dispatch.h:187+`) on a multi-request run and **records the counters in the PR body**, whatever they say. If something does capture, A2-B refuses graph capture for this architecture by name and files the driver as owed — it does not write one |
| **`mamba_attn.py:204-220`** | upstream's mamba CG capture is decode-only and asserts `max_query_len == 1 + num_spec_tokens` | consistent with refusing; recorded so the refusal has an upstream anchor rather than being a local preference |
| **#1217** (`device_token_ids` has no per-model opt-in) | A2-P's `## Owed`. The decode-row splice is per row and its correctness at `R > 1` has never been observed | A2-B's multi-request gate is the first thing in the tree that can see it. If it diverges, that is #1217 evidence, filed and referenced — **not** silently repaired inside A2-B |


### D6 — ★ A2-Q1's device mamba arm is HELD, and it moves the ground A2-B's throughput axis stands on

Measured on `dgx:gpu0` (GB10, sm_121a), run `20260819T200231Z`, logs at
`/workspace/a2d1-discriminate/`. A three-leg discriminator settled two things,
and both belong in this spec because both change what A2-B may assume.

**Correctness — the arm loses exactly one token.** With A2-Q1's device mamba arm
ON the end-to-end token gate reads **95/96**; with it OFF, same binary, same box,
same prompts, it reads **96/96**. Four kernel counters prove leg 3 genuinely took
the host path rather than falling back silently, which is what makes the pair a
discriminator instead of two runs. **Only the last token of the longest prompt
differs.** PR #1289 is held `DRAFT` on that single token.

That divergence is **A2-Q1's to close, not A2-B's**, and A2-B must not absorb it.
It has one consequence here, and it is a gate-hygiene one: a 95/96 is exactly the
shape a batched gate can produce by accident, and an implementer who runs G1 on a
tree that carries the device arm will see a mismatch that is not theirs.
**Therefore every A2-B correctness leg runs with A2-Q1's device mamba arm OFF,
stated explicitly in the recipe, and any leg run with it on is reported as a
separate pair with the arm state named beside it.** An unlabelled 95/96 in an
A2-B PR body is not evidence about batching.

**Speed — the same arm is worth 6.64x per output token.** 10.319 s → 1.554 s,
which moves the ratio against vLLM from **718.1x to 108.2x**. A2-B records no
speed number (§Scope), so this is not a result A2-B reports. It is a **constraint
on how the throughput axis may be framed once it opens**, and it lands here
because #1395's whole argument is that A2-B is what makes that axis measurable.

Two things follow, and the second is the one that gets got wrong:

1. **The 718x baseline is dead. Do not quote it, and do not compare to it.** It
   was measured with the host mamba arm, and a number quoted often enough starts
   being treated as measured. Any A2-B-era framing that carries 718x is carrying
   a superseded figure.
2. **A per-output-token cost that a kernel arm cuts 6.64x is dominated by
   HOST-SIDE work, and that is precisely the term batching amortises
   differently.** Our mamba block is a serial per-request host loop
   (`nemotron_h_device.cpp:1653-1706`, and §Scope item 5 keeps it serial), so at
   `R` requests the host term scales with `R` while a batched device GEMM would
   not. **The consequence is a prediction, and it is falsifiable: A2-B's batching
   should improve tokens/s per STEP and should not improve — and may worsen —
   tokens/s per REQUEST, until the mamba arm is on the device.** That is R7
   restated with a mechanism.

So the honest statement of what A2-B's throughput gate measures, when the
umbrella opens that axis:

- it measures **with A2-Q1's device arm ON**, because AGENTS.md requires vLLM's
  production configuration as the denominator and it is not honest to hold our
  own numerator in a configuration a landed arm supersedes. Until #1289 clears
  its 95/96, that measurement's correctness precondition is unmet and the axis
  stays **PENDING #1289** — which is a result, recorded, not silence;
- it reports the **step** and **per-request** rates separately, because with a
  serial host mamba loop the two move in opposite directions and a single
  aggregate hides it;
- it never imputes a per-request number from an aggregate. The per-item record
  either exists or the claim is not made.

**A2-B still records no number itself.** D6 exists so that the row that does
cannot inherit a dead baseline or an unlabelled arm state from this one.


---

## Tests to port

Preserving parameters, modes, fixtures, tolerances, failure cases and the
revision anchor `5559679229bc961848b121ccdeaa8fa5d79bec98`. Document only an
unavoidable harness adaptation.

| Upstream | What it pins | Local |
|---|---|---|
| the `split_decodes_and_prefills` cases under `treat_short_extends_as_decodes=False` | that a short extend counts as a **prefill**, and that the `is_prefilling` assert (`utils.py:623`) fires when the signal is absent | extend `tests/vllm/v1/attention/test_gdn_metadata_builder.cpp`, whose `:224` already pins the ordering contract for the `True` mode. The `False` twin sits beside it, and the `True` cases must stay byte-identical — that is the regression guard for R1 |
| `mamba_attn.py:455-462` promotion | a prefilling row with `query_len == 1` **and** `seq_len > 1` is promoted back to a decode; one with `seq_len == 1` (no prior state) is **not** | a table-driven case over the four combinations of (`is_prefilling`, `seq_len > 1`). All four, not the happy pair |
| `mamba_attn.py:87` / `:200` | threshold 1 | assertion on the value the NemotronH path passes |
| the mamba mixer's decode-first splits (`mamba_mixer2.py:758-767`, `:808-812`) | decode rows occupy the **leading** token range | our equivalent is the `[t0,t1)` slice at `nemotron_h_device.cpp:1673-1680`; a case asserting a two-request step's per-request output lands at the right token offsets |

The upstream harness is Python and torch-tensor based; the adaptation is
host-vector based and is stated in each case's comment, as A2-P did for
`test_mamba_utils.py:2136`.

---

## Gates

**Correctness first. A2-B records no throughput, latency or memory number on any
axis** (§Scope, item 1).

### G0 — the red, first, and what "for the intended reason" means here

The RED is not the refusal. `ModelRegistry::Forward` on a two-request input
already throws today (`test_nemotron_h_paged_forward.cpp:1356`), and a test that
merely stops throwing has proven that a `VT_CHECK` was deleted.

> **The red-first test is G1 with the `num_reqs <= 1` clause already removed and
> nothing else changed.** It must fail with a **token mismatch** between the
> batched step and the same requests run singly — or, if it passes, the
> implementer reports that and the unit's shape changes, because a green G1 on
> an untouched forward is a claim that the split's mode never mattered, and that
> claim needs the §Our baseline finding rebutted rather than assumed away.

Capture that transcript — the `[doctest]` `test cases:` / `assertions:` /
`Status:` lines and the mismatching token indices — and put it in the PR body. A
test never seen failing has proven nothing.

### G1 — the batched gate. **More than one request in one step.**

Through `ModelRegistry::Forward`, from a real `GPUModelRunner`. Not
`NemotronHPagedForward`, not a fabricated `ModelForwardInput`.

Three comparisons, all required:

- **(a) against the same requests run singly.** Two prompts, `P` and `Q`. Arm A:
  one runner, both requests scheduled in the same step, prefill then `>= 3`
  decode steps. Arm B: two runners (or one runner used twice with reset state),
  each running one prompt alone. **Every token of `P` and every token of `Q`
  must match across the arms.** This is the comparison a single-request gate
  structurally cannot make, and it is the one that catches cross-request
  contamination, a mis-split and a duplicate state slot.
- **(b) against the pinned oracle**, `555967922`, same checkpoint, same prompts,
  same token counts, greedy. Recorded with the exact build and run recipe,
  revisions, model hash and contention state, per AGENTS.md.
- **(c) mixed populations, not only uniform decode.** At minimum: two decodes;
  one decode + one prefill; **one decode + one short extend** — the population
  that exists only because of `treat_short_extends_as_decodes=False`, and the
  one arm (a) with two plain decodes cannot reach.

`RunnerGreedy` (`:418`) is single-id; a batched sibling is needed. `DecodeStep`
(`:400-414`) already takes a vector of ids.

### G2 — the split, directly

Unit cases on the mamba split (§Tests to port), asserting the four-tuple against
upstream's values for each mode. Direct, because a token gate over a small
synthetic model can agree by luck on a classification error.

### G3 — the ordering contract

D4's assertion, gated: an out-of-order batch must **refuse by name**, not be
silently reclassified as all-prefill.

### G4 — CUDA-graph dispatch counters

`GraphDispatchStats` read on a multi-request run, values in the PR body
whichever way they fall (D5).

### G6 — the arm state is part of every recipe

Every correctness leg names, in the recipe and in the PR body, whether A2-Q1's
device mamba arm was ON or OFF. **The A2-B legs run with it OFF** (D6). A leg run
with it on is reported as a separate labelled pair. An unlabelled 95/96 is not
evidence about batching, and A2-B records no throughput number on any axis
regardless of the arm state.

### G5 — the rewritten interlock case

`test_nemotron_h_paged_forward.cpp:1356` is **rewritten, not deleted**: it now
asserts the speculative refusal and the ordering refusal by name, with `#1395`
and `#810` in the message. A reviewer who finds the case deleted returns FAIL.

### What each gate CANNOT see

| Gate | Blind to |
|---|---|
| any token gate | **a dtype that is too WIDE.** An f32 conv or SSM page is *more* precise: tokens match, goldens pass, the path moves twice the bytes. The memory format is compared against the oracle explicitly, read from the **running** engine's resolved config, not from source |
| any token gate | **a dequant fallback.** A silently dequantized NVFP4 expert produces correct tokens; only the memory format and the load accounting can see it |
| any token gate | **a dropped mechanism whose argmax is unchanged.** Hence G2's direct assertions |
| **G1 arm (a) with two plain decodes** | **the entire `treat_short_extends_as_decodes` difference.** Two decodes classify identically under both modes. This is why (c) is required and not optional |
| **any `num_reqs == 1` gate** | everything in this unit. Named so no A2-P result is quoted as A2-B evidence |
| G1 | whether the graph path changed. Hence G4 |
| a passing G1 alone | whether the code was **reached**. Hence B-M8 |

### Mutations

Applied **alone**, in a scratch copy, rebuilt, run, tree restored to the
**baseline sha** — the restore is the control that catches `shutil.copy2`
preserving mtime so ninja skips the rebuild.

| # | Mutation | Must RED |
|---|---|---|
| B-M1 | the mamba split's `treat_short_extends_as_decodes` behaviour flipped back to `True` | G1(c) and G2 |
| B-M2 | the `mamba_attn.py:455-462` promotion dropped | G2's promotion cases; report G1(c) either way — if G1(c) survives, say so, because that bounds what the token gate can see |
| B-M3 | `GdnStateScatter` widened to write **every** slot rather than only the named rows | G1(a) — this is the cross-request contamination the scatter's own comment claims to prevent |
| B-M4 | the fresh-request state zeroing dropped | G1(a) on the first step after a slot is reused. If it survives, the gate is blind to the loudest trap in the unit and the row owes a direct assertion on the zeroed rows |
| B-M5 | `init[r] = 1 for r < nd` changed to `init[r] = 1` for **all** `r` | G1(c). If it survives, the mixed-population arm is not reaching the prefill classification and the harness is not the one described |
| B-M6 | both requests pointed at the **same** state slot | D2's distinctness assertion. **G1 must also RED**; report as a pair — a distinctness check that reds while the tokens stay correct means the fixture is not actually sharing state |
| B-M7 | the ordering assertion removed and a deliberately unordered batch fed | G3 |
| B-M8 | the production call site deleted — `ForwardNemotronHForCausalLM`'s paged branch removed, host arm left in place | the focused gate must RED. A gate that stays green without the call site measures a class, not a capability (AGENTS.md §"Nothing lands dead"; method at [`reachability.md`](../reachability.md)) |
| B-M9 | the conv page dtype widened to f32 | the memory-format assertion. **The token gate must NOT red** — that asymmetry IS the demonstration that a token gate cannot see a too-wide dtype. Report as a pair, not as a failure |

**Report per mutation, all of it, every time:**

- the exact `[doctest]` `test cases:`, `assertions:` **and** `Status:` lines;
- a **non-zero case count**. `assertions: 0` is a skip wearing a pass, and the
  `assertions:` line can read `0 failed` while cases threw;
- `git diff --stat`, proving the edit applied. A mutation that never applied
  reads as a passing test;
- the compile exit code **and** error count. A mutation that fails to build reads
  as a passing test;
- a binary sha256 distinct from baseline.

**Never put a comma in a `TEST_CASE` name.** doctest's `-tc` splits filters on
commas, selects zero cases, prints `SUCCESS!` and exits 0. Assert the case count
moved; do not read the word `SUCCESS`.

**Never grep a run log for `\bok\b`.** ANSI colour codes defeat the word
boundary, and a green run has already read as `0 ok / 1 FAIL` in this tree.

### Reachability

The gate enters through `ModelRegistry::Forward` from a `GPUModelRunner` built on
a NemotronH `KVCacheConfig`, which is a production entry point. B-M8 is the
proof. Method: [`reachability.md`](../reachability.md).

### Gate hosts

Inherited from A2-P §5.7 and not re-derived here: `dgx.casa` (GB10, sm_121a) is
the primary host and is a **fleet device** — claim it with `rc run` / `rc hold`,
never `ssh` plus `flock`, because the fleet cannot see that mutex and a bypass
makes the box report free while somebody is on it. Thor (sm_110) is the portable
leg and is not a substitute for Marlin work
([#962](https://github.com/mudler/vllm.cpp/issues/962)). The local x86_64 box is
a development arm, not a gate host.

---

## Dependencies

| Depends on | State |
|---|---|
| A2-P, landed | **MET.** `5f68e60df` carries the paged forward and the narrowed G-SAFE clause |
| the runner's decode-first reorder | **MET.** `runner.cpp:126-190`, unconditional at `:1257` |
| `vt::GdnStateGather` / `GdnStateScatter` indexed forms | **MET.** Both already take the `[R]` index vector |
| `vt::PagedAttention` mixed-batch arm on each gate backend | **UNVERIFIED.** D1; verify before implementing, and file rather than widen if it fails |
| a NemotronH checkpoint on the gate host | inherited from A2-P §5.7; the driver is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| A2-Q1 (#940 / PR #1289), A2-Q2b | **NOT** dependencies for the correctness gate. Textual rebase only (D5). A2-Q1's device mamba arm is **held `DRAFT` on a 95/96** (D6) and every A2-B correctness leg runs with it OFF, named in the recipe |
| the throughput axis #1395 says A2-B unblocks | **PENDING [#1289](https://github.com/mudler/vllm.cpp/pull/1289).** Not A2-B's to measure, and D6 records why the axis cannot honestly open before that arm clears its one token |

---

## Work breakdown

Non-overlapping, and W1 is deliberately a read rather than an edit.

| W | Work | Output |
|---|---|---|
| W1 | verify D1 (the `vt::PagedAttention` mixed-batch arm) and re-derive the §Our baseline audit at the implementer's own base | a finding, in the PR body. If D1 fails, stop and return `NEEDS_DECISION` |
| W2 | the mamba split + the `is_prefilling` metadata field + the ported upstream cases (G2) | red-first, then green; GDN's `SplitDecodesAndPrefills` cases byte-identical |
| W3 | rewire the NemotronH path to the mamba split; D2's distinctness check; D-B2's diagnostic fix | |
| W4 | drop the registry clause, add D4's ordering assertion, rewrite the interlock case (G5) | G0's red captured first |
| W5 | the batched gate G1(a)(b)(c) + G4 counters | |
| W6 | mutations B-M1…B-M9, each with the full report | |
| W7 | records (§Records owed) | rides in the same PR |

---

## Risks and decisions

- **R1 — changing `SplitDecodesAndPrefills` in place would silently change
  Qwen3-Next.** It is shared, and its current semantics are correct for its
  current caller. **Decision: add a second entry point; leave the existing one
  and its cases byte-identical.** A reviewer diffs `gdn_attn.cpp` for exactly
  this.
- **R2 — an `is_prefilling` vector that is empty reads as "nothing is
  prefilling".** That is the whole `treat_short_extends_as_decodes=False`
  behaviour disabled, silently, on any path that forgets to populate it.
  **Decision: the mamba split refuses on an empty vector**, mirroring
  `utils.py:623`'s `assert … is not None`. Upstream asserts; so do we.
- **R3 — misclassification produces fluent wrong tokens, never an error.** It is
  the same failure class the G-SAFE clause was built for, one level down. G1(c)
  and B-M1 are the only things that can see it.
- **R4 — a token gate at `num_reqs == 1` will pass throughout.** Every existing
  NemotronH case is single-request. **None of them is evidence for A2-B**, and a
  PR body that quotes them as if they were should be rejected.
- **R5 — the CUDA-graph population changes as a side effect of deleting a
  refusal.** D5; G4 measures it rather than reasoning about it.
- **R6 — #1217's `device_token_ids` splice has never been observed at `R > 1`.**
  If G1 diverges only on the device path with a null-pointer CPU control passing,
  that is the #1217 signature and it is filed, not repaired here.
- **R7 — the serial per-request mamba loop makes a batched step slower per token
  than a single-request one.** Expected, and **not** a defect of this unit. A2-B
  records no speed number precisely so that this cannot be reported as a
  regression or as a win. D6 now gives this a measured mechanism rather than an
  expectation: the per-output-token cost is host-dominated (a device mamba arm
  cuts it 6.64x), so `R` serial host mixer calls are the term batching does not
  amortise.
- **R9 — a tree carrying A2-Q1's device mamba arm produces a 95/96 that is not
  A2-B's.** D6. Every A2-B correctness leg names the arm state in its recipe;
  an unlabelled near-miss is not evidence about batching.
- **R10 — the 718x figure is superseded and will be re-quoted anyway.** D6. It
  was measured with the host mamba arm and is 108.2x with the device arm. A
  number quoted often becomes treated as measured; grep its origin before
  carrying it.
- **R8 — `merge-tree` clean is not "the merge builds".** With A2-Q1 and A2-Q2b
  live on the same file, the focused gate runs on the merge result.

### Stop conditions

Stop and report rather than widening:

1. **D1 fails** — `vt::PagedAttention` does not serve a mixed batch on a gate
   backend. Return `NEEDS_DECISION`; do not write an attention kernel.
2. **G0's red does not appear** — the batched gate passes on an untouched
   forward. Report it with the transcript. Do not proceed as if it had failed,
   and do not weaken the gate to manufacture a red.
3. **A mutation cannot be made to red** after the harness has been checked
   against the four green-but-proves-nothing shapes (zero cases, comma filter,
   failed build, unapplied diff). Report the surviving mutation as an open gap.
4. **The oracle leg cannot run** on the gate host — record `PENDING` naming the
   external blocker. A pending result is a result; silence is not.
5. **Graph capture engages** for a multi-request NemotronH step. Refuse capture
   by name, file the driver as owed, and do not write one.
6. **A fix needs its own spec** — a shared-seam change or a checker-semantics
   change. AGENTS.md routes that to the normal row, spec and fresh-review path,
   not to this flow.

Attempt budgets control scheduling and never stop a correctable finding.

---

## Records owed on landing

The implementing change moves this row's lifecycle state, so it owes, in the
**same** change:

- `docs/STATUS.md`;
- `docs/BENCHMARKS.md` — **pending, failed or void is a result; silence is not.**
  A2-B records no speed number, and *that* is what BENCHMARKS records: the axis
  is now measurable and unmeasured, with #1395 naming why every prior Nemotron
  ratio was bounded;
- `docs/FEATURES.md` — the `NemotronHForCausalLM` row;
- this spec's `## Now`, and the row + checklist entry + rollup at
  `.agents/model-matrix.md:285` (`scripts/check-model-checklist.py` enforces the
  rollup);
- `docs/USAGE.md` if the reachable capability's checkpoint documentation changes;
  run `scripts/check-doc-checkpoint.py` rather than assuming either way;
- `scripts/runner-routing-allowlist.txt:26` — A2-B does **not** change its
  disposition (it is pending A2-Q2b's device `lm_head`), but the entry's prose
  must not be left claiming a single-request forward once it is not one.

`.agents/NOW.md` is authored at operator cadence and is not a per-row lifecycle
write.

**If the implementing change edits this spec's `## Gates` section to carry
runnable gate evidence — a named test binary with case and assertion counts and
an exit status — it moves this row into the runnable population and must re-pin
`RUNNABLE_BASELINE` in `scripts/check-gate-commands.py` in the SAME change.**
Not doing so reds `main` on `tests/scripts/test_check_gate_commands.py`; that has
already happened once, on [#1376](https://github.com/mudler/vllm.cpp/issues/1376).

---

## Now

**State at this commit: SPEC ONLY. No product code is written, no gate has run,
and no lifecycle state moves.**

What this spec established that was not known when #1395 was filed:

1. The pinned oracle's mamba backend passes
   `treat_short_extends_as_decodes=False` (`mamba_attn.py:467`) and never passes
   `require_uniform`. Our `SplitDecodesAndPrefills` implements the opposite flag.
   **That is a live mirroring gap the `num_reqs <= 1` refusal has been hiding**,
   and it is the substance of the unit.
2. A2-P's `:81` under-describes what A2-P landed. The per-request state indexing
   exists; A2-B verifies and hardens it rather than building it.
3. The runner already reorders decode-first, unconditionally.
4. `#1395`'s call-site anchor is `mamba_attn.py:464`, not `:463`.
5. A2-Q1's device mamba arm is **held `DRAFT` on a 95/96 GB10 divergence** and is
   worth **6.64x per output token**, which retires the 718x baseline and makes
   the host-side term the thing A2-B's batching does *not* amortise (D6).

## Owed

- The upstream-test port for the `treat_short_extends_as_decodes=False` modes is
  owed **by this row**, in the implementing change, not deferred.
- Mamba prefix caching (`mamba_cache_mode == "all"`) remains unported and, unlike
  batching and speculation, is **not refused by name** anywhere. If no issue
  tracks it when the implementer looks, file one and name this row as its owner.
- [#1217](https://github.com/mudler/vllm.cpp/issues/1217) —
  `ModelForwardInput::device_token_ids` has no per-model opt-in. A2-P's `## Owed`
  carries it; A2-B's G1 is the first gate that can observe it at `R > 1`.
- The throughput comparison #1395 names as blocked. A2-B unblocks it and does not
  perform it.

## Outcome

Not yet written. Per AGENTS.md this section is filled when the unit reaches
`DONE`: what was measured, what was rejected and why, and why each default has
its value — including, explicitly, the D1 result, the G4 graph-dispatch counters
whichever way they fell, and the disposition of every mutation that did not red.
