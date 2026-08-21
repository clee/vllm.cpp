# The Nemotron oracle golden cannot be regenerated, and now it says so

**Issue:** [#926](https://github.com/mudler/vllm.cpp/issues/926)
**Row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` ([model matrix](../model-matrix.md))
**Owed out of:** [#517](https://github.com/mudler/vllm.cpp/issues/517), which
committed the golden
**Blocks:** [#1289](https://github.com/mudler/vllm.cpp/pull/1289) is being scored
against this golden at 95/96

## 0. Scope (headline verdict)

`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json` is the reference
for every Nemotron-3.5-Lightning token claim in this tree. **Its capture
configuration is unrecoverable.** The recovery was attempted and failed on
evidence, not on effort, and §2 records the six independent places it was looked
for.

What this row does instead: it commits a generator, extends the golden's format
to carry the engine configuration, and makes the artifact **state which of two
things it is** — attributed, or unattributed-and-saying-so. The third state,
silence, was what #926 filed, and it is what this removes. The lost
configuration is not invented, and no gate is weakened to accommodate its
absence.

**In scope**

- `scripts/nemotron-h-oracle-capture.py` — capture, verify and validate.
- The `capture` provenance block, written truthfully into the shipped golden.
- The contract, gated in three places that cannot drift apart: the generator's
  `--check`, `tests/scripts/test_nemotron_h_oracle_capture.py`, and the C++
  consumer in `tests/vllm/models/test_nemotron_h_loader.cpp`.
- The A3 driver printing what it is being held to.

**Out of scope**

- Running the oracle. [#1431](https://github.com/mudler/vllm.cpp/issues/1431)
  owns that and five attempts have already been killed by the host-memory
  watchdog. This unit needs no GPU, which is why it could be done now.
- The index-29 top-2 margin ([#1388](https://github.com/mudler/vllm.cpp/issues/1388)).
- Deciding whether #1289's moved token is a defect.

## 1. What `oracle.json` recorded, and what it omitted

Seven top-level keys, as `af8170154` wrote them:

| Key | Value |
|---|---|
| `vllm` | `0.23.1rc1.dev1511+g555967922` |
| `transformers` | `5.14.1` |
| `flashinfer` | `0.6.15.post1` |
| `model` | `/mnt/nas_share/checkpoints/nemotron-3.5-lightning-30b-nvfp4` |
| `revision` | `29f2d1746d8f41e316523194b19018707749b1b1` |
| `sampling` | `{temperature: 0.0, max_tokens: 32}` |
| `golden` | 3 entries: `prompt`, `prompt_token_ids`, `token_ids`, `text` |

**Not one engine knob.** `enforce_eager`, `max_model_len`, `max_num_seqs`,
`max_num_batched_tokens`, `gpu_memory_utilization`, `num_gpu_blocks_override`,
`block_size`, the batch shape, `seed`, `ignore_eos`, the compile mode, the
cudagraph capture sizes, the attention backend and the MoE backend are all
absent. Every one of them can move a greedy argmax at a near-tie, because each
one changes batching, prefill chunking, paging or the kernel that runs — that
is, the reduction order.

For contrast, this tree already knows how to do this: the sibling golden
`tests/parity/goldens/gdn_ba_projection_bf16_sm121/oracle.json` records
`device`, `compute_capability`, `cuda_runtime`, `torch_version`, `dispatch`,
`implementation`, `repetitions` and `vllm_target_commit`. The Nemotron golden is
the outlier, not the norm.

## 2. The generator was never committed, and the configuration is gone

Six checks, each of which could have found it:

1. `git show --stat af8170154` adds exactly **three** files —
   `.agents/specs/nemotron-h-model.md`, this golden, and
   `tests/parity/hf_snapshot.h`. None is a generator.
2. `git log --all --oneline -- tests/parity/goldens/nemotron_35_lightning_greedy/`
   returns **exactly one commit** in the whole history. The golden has never
   been revised, re-attributed or regenerated.
3. `grep -rn nemotron_35_lightning_greedy scripts/` returns nothing, and a
   per-branch `git ls-tree` sweep for a Nemotron capture script finds none. The
   Nemotron scripts that do exist (`nemotron-h-a2q1-neartie-gap.py`,
   `nemotron-h-a2q1-dgx-gate.sh`, `nemotron-h-a2q2b-gpu-gate.sh`) are all later
   and none captures a golden. `git log --diff-filter=D -- scripts/` shows none
   was deleted either.
4. The capture ran from `$HOME/venvs/vllm-oracle-next` on `dgx.casa`
   (`af8170154`'s message and `nemotron-h-model.md` §5a). **`dgx.casa` was
   reimaged on 2026-08-14**, two days later.
5. Nothing of the run reached shared storage: a recursive grep for
   `MODEL_LOADED_OK` and `ORACLE_IDENTITY_OK` across `/mnt/nas_share/{rc,
   experiments,staging,...}` returns nothing, and the oldest `rc/` job directory
   is **2026-08-17** — the share post-dates the capture.
6. `af8170154`'s own message is the fullest surviving description of the run,
   and it names the venv, the identity assertions, the arch, the layer count and
   the block pattern. It names **no engine knob**.

The only timestamp that exists is the commit's author date,
`2026-08-12T22:37:57Z`, and the golden now records it as that rather than as a
run time.

**Verdict: unrecoverable.** Not "not found yet".

## 3. The lead that looked implicit and is a COMMON TERM

The logs of the rebuilt oracle show `kv_cache_dtype=fp8_e4m3` selected without
anyone asking for it, with `Checkpoint does not provide a q scaling factor.
Setting it to k_scale` (`kv_cache.py:134`) beside it. That is exactly the shape
of an unrecorded implicit choice that moves tokens — so it was checked rather
than assumed, and it is **not** a candidate difference.

The **checkpoint** carries it:

- `config.json` → `quantization_config.kv_cache_scheme` =
  `{"dynamic": false, "num_bits": 8, "type": "float"}`
- `hf_quant_config.json` → `quantization.kv_cache_quant_algo` = `"FP8"`

and at the pin, the default `kv_cache_dtype="auto"` is resolved from that
before `CacheConfig` is built:

- `vllm/engine/arg_utils.py:1916` — `resolved_cache_dtype =
  resolve_kv_cache_dtype_string(self.kv_cache_dtype, model_config)`, whose
  result is passed as `cache_dtype=` at `:1928`. This is the **unique** call
  site that sets it; the other two occurrences of the symbol at the pin are the
  import (`:115`) and a defensive comment (`attention.py:282`).
- `vllm/utils/torch_utils.py:374-392` — returns early unless `kv_cache_dtype ==
  "auto"`, then reads `hf_config.quantization_config`.
- `vllm/utils/torch_utils.py:310-362` — for `quant_method` starting `modelopt`,
  maps a `kv_cache_scheme` dict of exactly that shape to `"fp8"`.

So **every** unoverridden run of this checkpoint at this pin gets fp8_e4m3,
including the 2026-08-12 capture. Same on both sides is a measurement, and the
golden now records it under `capture.forced_by_checkpoint_or_device` together
with the MoE backend (MARLIN — a device without native FP4 takes the first
supported backend and at this pin no environment knob selects another), the
dtype and the quantization method.

**This narrows the unrecoverable set to the knobs a driver passes**, which is
also the set §1 lists. It does not recover any of them.

## 4. Two configurations, two answers, both repeatable

| Run | Configuration | Prompt 0 | Prompt 1 | Prompt 2 |
|---|---|---|---|---|
| 2026-08-18 `oracle_only.sh` attempt `a` | `max_model_len=512`, `max_num_seqs=8`, `gpu_memory_utilization=0.30`, `max_num_batched_tokens=512`, `enforce_eager=False`, `TokensPrompt`, one prompt per `generate()` | 32/32 | 32/32 | **26/32** |
| the #926 rebuild | `enforce_eager=True`, `gpu_memory_utilization=0.25`, `max_model_len=4096` | 32/32 | 32/32 | **29/32** |

The first ran its configuration **twice in one process** (`ORACLE_LEG 1`,
`ORACLE_LEG 2`) with identical results, `ORACLE TOKEN MATCH: 180/192`, log at
`/mnt/nas_share/rc/nhspeed/oracle.a.out` (the worker's `/workspace/nhspeed`).

That is **configuration sensitivity, not non-determinism**. The distinction
decides the gate form, and it decides it against weakening: AGENTS.md admits a
ratified distributional gate **only** where the oracle's own greedy decode is
non-deterministic, and here it is not. **A distributional gate is inadmissible
on this evidence.** What is licensed is re-deriving the golden under a named
configuration.

Note what the table also says: **prompt 2 has never been reproduced by anything
this repository can name.** Prompts 0 and 1 have been, twice.

## 5. Design

### 5.1 The contract

A Nemotron oracle golden is in exactly one of two states, and the file says
which:

- `capture.engine_config_recorded = true` — then `capture.engine.resolved`
  carries **every** key in `REQUIRED_ENGINE_KEYS`, `capture.batch.shape` says how
  the prompts were submitted, `capture.legs >= 2` and `capture.legs_agree` is
  true.
- `capture.engine_config_recorded = false` — then
  `capture.unrecoverable_reason` says why and `capture.issue` names the issue
  that owes the re-derivation, and `capture.engine` is null. "Unrecorded" and
  "here is the record" cannot both be true.

A null inside `resolved` is refused for every key but
`num_gpu_blocks_override`: **a value that could not be read is not a value that
was default.** That is the same rule AGENTS.md states for `.env` — a missing
value never becomes an assumption.

The shipped golden is in the second state. It is **kept**, because deleting
evidence to make a gate green is never the repair.

### 5.2 The generator

`scripts/nemotron-h-oracle-capture.py`, three modes:

| Mode | Needs | Does |
|---|---|---|
| `--check <golden>` | nothing | validates the contract. This is the CI gate. |
| `--verify <golden>` | the oracle | runs and compares, reporting a **configuration** difference before a token difference |
| `--capture --out <p>` | the oracle | runs and writes a golden that records its own configuration |

Four properties are deliberate:

1. **Identity is asserted, never assumed** — the pin substring `555967922` must
   be in `vllm.__version__` and the run aborts otherwise. `$HOME/venvs/
   vllm-oracle` on dgx has resolved to a 0.25.0 rollback that predates
   `NemotronHMoEDecoderLayer`, and a run through it fails in a way that reads as
   "the model is unsupported".
2. **The configuration is read BACK OUT of the built engine**, not echoed from
   the kwargs. `kv_cache_dtype`, the block size, the block count and the
   backends are chosen by vLLM, so what a driver passed is not what it ran.
3. **`--capture` refuses to write** a golden that fails its own contract, or
   whose legs disagree. A golden written from disagreeing legs records a coin
   flip.
4. **The body is under `if __name__ == "__main__":`** — vLLM v1 spawns
   EngineCore, the module re-imports, and an unguarded driver fails as a
   `multiprocessing` traceback naming neither vLLM nor the caller. The tell is
   the banner printing twice.

### 5.3 The named profile

`--profile nhspeed-a` is the 2026-08-18 configuration of §4: the only oracle
configuration on this checkpoint for which this repository has determinism
evidence, with its full resolved config readable at
`/mnt/nas_share/rc/nhspeed/oracle.a.out`, and with CUDA graphs ON because
`--enforce-eager` is never the denominator. **It is a name for a run that
happened. It is not a reconstruction of the lost one**, and nothing in this row
claims it is.

It is a token-golden configuration and **not** a speed denominator:
`max_num_batched_tokens=512` against the denominator's 8192 is a regime you
cannot tune down and keep a ratio through.

## 6. Gates

```sh
python3 scripts/nemotron-h-oracle-capture.py --check \
  tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json
python3 tests/scripts/test_nemotron_h_oracle_capture.py
ctest --test-dir build -R test_nemotron_h_loader --output-on-failure
```

All three run with no vLLM, no GPU and no checkpoint. That is the point: the
provenance defect is a records defect, and a gate for it must not need the
hardware whose absence caused it.

The contract exists in three copies — the generator's `REQUIRED_ENGINE_KEYS`,
the C++ consumer's `required` list, and this suite's `EXPECTED_ENGINE_KEYS`. The
suite asserts all three agree and **owns the expectation itself**, so a key
deleted from both production copies is still red.

## 7. Evidence

**Measured on** `mudler-ubuntu-box` (x86_64, 20 cores), in the worktree
`.claude/worktrees/row-926-golden-provenance` at base `5d548d003`, CPU-only
Release build (`-DVLLM_CPP_CUDA=OFF`), nothing overlaid. `libvllm 0.0.3 (ABI 23,
header 23)`. vLLM anchors read at `/home/mudler/_git/vllm` HEAD
`5559679229bc961848b121ccdeaa8fa5d79bec98`, remote `vllm-project/vllm`,
verified before citing.

### RED first

With the golden as `af8170154` left it, the suite collected 25 cases and
**3 were red**: `test_the_shipped_golden_satisfies_the_contract`,
`test_check_reads_the_shipped_golden` and
`test_the_shipped_golden_is_not_silently_attributed`, each reporting
`oracle.json: missing 'capture'`. After the provenance block: **26/26 OK**.

### Mutation proof — golden

Each mutation applied alone to the committed golden, the case rerun, the tree
restored and its sha256 re-asserted
(`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`).

| # | Mutation | Result |
|---|---|---|
| — | baseline | 33 assertions, 0 failed, SUCCESS |
| M1 | delete the whole `capture` block (the `af8170154` shape) | 20 assertions, **1 failed** |
| M2 | claim `engine_config_recorded: true` while `engine` stays null | 28 assertions, **1 failed** |
| M3 | blank `unrecoverable_reason` | 33 assertions, **1 failed** |
| M4 | point `issue` at something that is not a vllm.cpp issue | 33 assertions, **1 failed** |
| M5 | truncate one golden row below `max_tokens` | 33 assertions, **1 failed** |
| M6 | empty the `golden` array | 7 assertions, **1 failed** |

M5 and M6 are the anti-vacuity arms: a comparison over zero elements reports a
perfect score, so the width is asserted rather than trusted.

### Mutation proof — the checker

Deleting the unattributed branch's reason check from `check_golden` turns
`tests/scripts/test_nemotron_h_oracle_capture.py` red (26 run, 1 failed) and
restoring it returns 26/26 OK. `test_each_required_engine_key_is_load_bearing`
additionally drops each of the 20 engine keys in turn and asserts the contract
names the one it dropped.

### The driver states all three cases

`nemotron-h-gen --golden-info` on the shipped golden, on a copy with
`engine_config_recorded: true`, and on a copy with the block removed:

```
capture: engine configuration UNRECORDED — a token difference below is
         UNATTRIBUTABLE, not yet a defect; owed by .../issues/926
capture: engine configuration RECORDED
capture: the golden does not SAY whether its engine configuration was recorded
         (no capture block)
```

The same line prints beside `DIVERGENCE`, because a reader who sees `DIVERGENCE`
and stops reading is the reader it is for.

### A trap this hit

`REQUIRE_MESSAGE(..., "capture is missing '" << key << "'")` over a
`const char*` printed `capture is missing '1'`: doctest stringifies a bare
`char*` as a **bool**. The loop iterates `std::string` now, and the messages read
`capture is missing 'schema'`.

## 8. Risks

- **A named profile could be mistaken for the recovered one.** Mitigated by
  saying so in the generator, in the golden and in §5.3, and by the fact that
  `nhspeed-a` reproduces prompt 2 at 26/32 — it is visibly not the lost
  configuration.
- **Three copies of the key list can drift.** The suite asserts they agree and
  owns the expectation.
- **The re-derivation needs the oracle**, which #1431 blocks. Nothing here
  depends on it; the contract admits the unattributed state precisely so this
  work did not have to wait.

## 9. Stop conditions

- Do **not** relax the contract to admit a silent golden.
- Do **not** ratify a distributional gate for this row. §4 shows the oracle is
  deterministic at a fixed configuration, which is the condition AGENTS.md
  requires be **absent**.
- Do **not** write a reconstructed configuration into the golden. An invented
  provenance is worse than a stated absence.

## 10. Now

Landed: the generator, the contract, its three gates, the truthful provenance
block, and the driver line. The golden's tokens are **byte-for-byte unchanged**
— the diff against `af8170154` is 24 inserted lines and zero deletions.

## 11. Owed

- **The re-derivation, and it needs a decision.** Re-deriving the golden under
  `--profile nhspeed-a` would replace an unattributable reference with an
  attributable one and would change the reference #1289 is scored against — from
  32/32, 32/32, 32/32 to 32/32, 32/32, 26/32 on the oracle side. That is a
  change to a gate's reference and is the developer's call, not an implementer's.
  It is blocked on #1431 either way.
- **The index-29 top-2 margin** (#1388), also blocked on #1431. Note the
  ordering this row establishes: the margin measures how close the two
  candidates are; it does not tell you which configuration produced the
  reference. Both are needed and this one was cheaper.
- **The other goldens.** This contract is Nemotron-only by design; whether the
  rest of `tests/parity/goldens/` can name their capture configurations is
  unmeasured and is not claimed either way here.

## 12. Outcome

Recorded because the code does not say it:

- **The recovery was attempted and failed.** §2 is the negative result, and it
  is worth more than a plausible reconstruction would have been.
- **The `kv_cache_dtype` lead was closed by reading the checkpoint and the
  pinned source**, not by running anything. It looked like the best candidate
  and it is a common term.
- **The contract does not demand the configuration back.** Demanding it would
  have made this row wait for #1431 and would have made `main` red on an
  artifact nobody can currently fix. Demanding that the file *say which state it
  is in* costs nothing, is checkable today, and is the property that was
  actually missing.
- **A distributional gate was available and was rejected**, on the evidence in
  §4 rather than on preference.
