# Re-derive the Nemotron oracle golden under a NAMED configuration

**Issue:** [#1694](https://github.com/mudler/vllm.cpp/issues/1694)
**Parent defect:** [#926](https://github.com/mudler/vllm.cpp/issues/926), which
this row does NOT discharge -- see `## Owed`
**Row:** `GATE-NEMOTRON-GOLDEN-REDERIVE`
**Blocked by:** [#1431](https://github.com/mudler/vllm.cpp/issues/1431), the
oracle's engine start-up collapse on `dgx:gpu0`
**Predecessor:** [`nemotron-oracle-golden-provenance.md`](nemotron-oracle-golden-provenance.md),
which committed the generator and the contract and left this under its `## Owed`
**Index row:** `#1694` in [`issue-index.md`](../issue-index.md). #926 is NOT
appended again there: [PR #1432](https://github.com/mudler/vllm.cpp/pull/1432)
is already adding a #926 row, and two branches appending one issue is the
duplicate `check-agent-record.py` reports rather than a merge.
**Authority:** the developer explicitly ratified re-deriving this golden under a
named engine configuration (2026-08-22). A gate-reference change is a product
decision and it has been made. This row executes it; it does not re-argue it.

## 0. Headline, and the thing this row does NOT do

The committed golden
`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json` records the
model, the revision, the sampling parameters and the library versions and **not
one engine knob**, and its generator was never committed. That is #926. This row
produces a **second** golden, captured by the committed generator under the
named configuration `nhspeed-a`, which records the configuration it ran under
and can therefore be regenerated and attributed.

**Re-deriving does NOT tell us what [#1289](https://github.com/mudler/vllm.cpp/pull/1289)
scores.** #1289 is worth 7.28x on the warm basis per output token and is the
campaign's only measured speed win, and it is held DRAFT on a 95/96 against the
unattributable golden. What this row produces is a reference that can be
regenerated and attributed. **What #1289 scores against it is unknown and could
go either way**, and nothing in this row, its artifacts or its pull request may
be worded to imply otherwise.

Nor does this row **repoint** any gate. §5.3 says why, and names what repointing
would take.

## 1. The decisive measurement this row is built on

Three runs of this checkpoint exist under a recorded configuration, and they
disagree on exactly one prompt:

| Run | Configuration | Prompt 0 | Prompt 1 | Prompt 2 |
|---|---|---|---|---|
| 2026-08-18 `oracle_only.sh` attempt `a` (`nhspeed-a`), twice in one process | `mml=512, mns=8, gmu=0.30, mnbt=512, enforce_eager=False`, one `TokensPrompt` per `generate()` | 32/32 | 32/32 | **26/32** |
| the #926 rebuild | `enforce_eager=True, gmu=0.25, mml=4096` | 32/32 | 32/32 | **29/32**, diverging at index 29 |
| #1289's device Mamba arm (ours, not the oracle) | — | 32/32 | 32/32 | **31/32**, diverging at generation position 32 |

**Prompt 2 has never been reproduced by any nameable configuration.** Prompts 0
and 1 have been, twice, under two different recorded configurations. #1289's
single divergence sits at **generation position 32** -- the last token of the 32,
0-based index 31 -- which is **inside the tail that both other oracle-side runs
also fail**. That is the fact that makes the 95/96 unreadable: nobody can say
whether the moved token is a defect in our arm or a property of a configuration
nobody recorded.

**Where that position comes from, and how firm it is.**
[#1388](https://github.com/mudler/vllm.cpp/issues/1388), which filed the 95/96,
does **not** name a position -- its own next step is "read the `got:`/`exp:` ids
for the mismatching row". The position is named in
[PR #1432](https://github.com/mudler/vllm.cpp/pull/1432) §10.5, which is **open
and unmerged**, so it is cited here as a claim in flight rather than as something
`main` records. The #926 rebuild's index 29 comes from the same section. Neither
number is re-derived here, and this row does not depend on either: what it depends
on is that prompt 2's tail is where all three disagree, which every recorded run
shows on its own.

Two configurations, each internally repeatable, disagreeing on the same prompt is
**configuration sensitivity, not non-determinism**. AGENTS.md admits a ratified
distributional gate only where the oracle's greedy decode is non-deterministic,
so a distributional gate stays inadmissible here. What is licensed, and what the
developer ratified, is re-deriving under a name.

## 2. The configuration, by name, and why it is this one

**`nhspeed-a`** — `max_model_len=512`, `max_num_seqs=8`,
`gpu_memory_utilization=0.30`, `max_num_batched_tokens=512`,
`enforce_eager=False`, `num_gpu_blocks_override` unset. It is the sole entry in
the generator's `PROFILES` table (`scripts/nemotron-h-oracle-capture.py`,
symbol `PROFILES`).

Four properties, and each one is a measurement rather than a preference:

1. **It has a full resolved engine configuration on record.** The 2026-08-18 log
   is readable at `/mnt/nas_share/rc/nhspeed/oracle.a.out` (the worker's
   `/workspace/nhspeed`), and the driver that produced it is at
   `/mnt/nas_share/rc/nhspeed/oracle_only.sh`. "Whatever the script did" is not
   a name; this is a name because someone else can read what it resolved to.
2. **It is the only configuration on this checkpoint with determinism
   evidence** — two legs in one process, byte-identical, `ORACLE TOKEN MATCH:
   180/192`.
3. **It is the only configuration that has ever completed engine start-up on
   this box.** #1431's five failures include one that copied it, so this is a
   necessary condition and not a sufficient one.
4. **CUDA graphs stay ON.** `--enforce-eager` is never the denominator, and the
   #926 rebuild's configuration is eager.

It is a **token**-golden configuration and **not** a speed denominator.
`max_num_batched_tokens=512` against the denominator's 8192 is a regime you
cannot tune down and keep a ratio through.

**It is a name for a run that happened. It is not a reconstruction of the lost
one**, and the fact that it produces 26/32 on prompt 2 is what makes that
visible rather than what makes it suspect.

### 2a. The one deviation, named

`--capture` submits **one text prompt per `generate()` call`**; the 2026-08-18
run submitted **one pre-tokenized `TokensPrompt` per `generate()` call**. The
generator refuses `--capture --tokens-prompt` on purpose — `prompt_token_ids`
only come from a golden, and a capture that reads its prompts out of the artifact
it is replacing is circular.

The difference is confined to the tokenizer step and it is **checkable**: the
captured golden records `prompt_token_ids`, so if those are identical to the
committed golden's the engine received the identical token sequence and the
submission shape differed only in who tokenized. §7 records the comparison.
Either way the shape is written into `capture.batch.shape`, which is the field
that exists so this cannot be silent.

## 3. #1431, and why this is worth attempting now

Five previous attempts died in engine **start-up**, killed by a host-memory
watchdog before generating a single token. Minima against a 15000 MB floor:
12597 / 19433 / 19797 / 13941 / 14846 MB. Runs 2 and 3 used a **20000 MB** floor,
so their outcome at 15000 is inferred and not measured.

Already excluded by those runs, and **not re-derived here**:
`torch.compile` (run 5 hit the AOT cache, 0.30 s), CUDA graph capture (run 2 was
eager), and KV sizing (run 3 overrode the block count to 8, run 5 set an absolute
`kv_cache_memory_bytes`). What remained was the first forward.

What has changed: every one of those runs was on a box at load 100-150 with disk
near 100%. **The fleet is idle** — `dgx:gpu0`, `thor:gpu0` and `orin:gpu0` all
`ready`, no holder — and the box reports 2.4 TB free. That is a different
condition, and this row **re-establishes it by measurement** rather than assuming
it either way: §7 records `uptime`, `free -m`, `df`, the boot id and
`nvidia-smi --query-compute-apps` from inside the job.

**If it still cannot start, that is a legitimate result** and this row reports
#1431 confirmed under idle conditions, which is a stronger statement than the
existing evidence. **The watchdog floor is not lowered**, and no artifact of this
row may lower it: this box OOM-**reboots** rather than OOM-kills, and a reboot
takes down every other job on the fleet.

## 4. Scope

**In scope**

- One `rc` lease on `dgx:gpu0`, running the committed generator's `--capture`
  mode under `--profile nhspeed-a --legs 2`.
- The re-derived golden, committed **beside** the existing one.
- Extending the Python contract suite to hold **every** golden in the directory
  to the contract, read with a glob, so the new artifact is reached by a gate.
- The evidence, the comparison against the committed golden, and the verdict on
  whether #926 is discharged.

**Out of scope**

- Repointing `tests/vllm/models/test_nemotron_h_loader.cpp`, the A3 driver or
  #1289's score at any golden. §5.3.
- Root-causing #1431. This row measures whether it still fires; it does not fix
  it.
- The index-29 top-2 margin ([#1388](https://github.com/mudler/vllm.cpp/issues/1388)).
- Deciding whether #1289's moved token is a defect.

## 5. Design

### 5.1 The generator, not a hand-rolled driver

`scripts/nemotron-h-oracle-capture.py --capture`, already committed by the
predecessor row. It is used rather than replaced because it does three things a
hand-rolled driver would have to be trusted to do:

- **It asserts the pin's identity before anything else** and aborts otherwise
  (`assert_oracle_identity`). `$HOME/venvs/vllm-oracle` on this box has resolved
  to a 0.25.0 rollback that predates `NemotronHMoEDecoderLayer`, and a run
  through it fails in a way that reads as "the model is unsupported".
- **It reads the configuration back OUT of the built engine**
  (`read_resolved_config`), so what is written down is what vLLM ran and not what
  a driver passed. `kv_cache_dtype`, the block size, the block count and the
  backends are all chosen by vLLM.
- **It refuses to write** a golden that fails its own contract, or whose legs
  disagree. A golden written from disagreeing legs records a coin flip.

Two values it cannot read are supplied by the job and then **checked against the
run's own log**: `attention_backend` and `moe_backend`. At this pin
`VLLM_ATTENTION_BACKEND` and `VLLM_FUSED_MOE_BACKEND` do not exist — setting
either logs `Unknown vLLM environment variable` and changes nothing — and vLLM
does not expose the selection on the config, so the generator takes what the
caller says the startup log said. The job passes `FLASHINFER` and `MARLIN`, which
is what the 2026-08-18 run logged, and then greps **this** run's log for
`Using FLASHINFER attention backend` and `Using 'MARLIN' NvFp4 MoE backend`. A
zero count means the golden's backend fields are false and the artifact is not
committed.

### 5.2 The old golden is PRESERVED

`oracle.json` stays **byte-for-byte unchanged**, sha256
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`. The new one
lands beside it as `oracle.nhspeed-a.json`.

Three reasons, and the first is the binding one:

1. **Deleting evidence to make a record tidy is never the repair.** If the two
   disagree on prompt 2, **that difference is the finding**, and it is only
   legible while both files exist. Overwriting in place would leave a reader with
   one file, a diff in history, and no statement of which configuration produced
   which.
2. The committed golden is what every current consumer reads and what #1289's
   95/96 was scored against. Moving that reference silently, in the same change
   that produces the replacement, would make the score change and the reference
   change indistinguishable.
3. The contract already admits exactly two states and both files can state
   theirs: `oracle.json` says `engine_config_recorded=false` and argues why;
   `oracle.nhspeed-a.json` says `true` and carries all twenty resolved keys.

### 5.3 Nothing lands dead: what reaches the new golden

A golden nothing reads is dead. `tests/scripts/test_nemotron_h_oracle_capture.py`
pins `SHIPPED_GOLDEN` to `oracle.json` by name, so a second file in that
directory would be gated by nothing.

The repair is a **glob**, not a second hard-coded constant: the suite holds
**every** `*.json` under `tests/parity/goldens/nemotron_35_lightning_greedy/` to
`check_golden`, so this artifact and every future capture are reached without any
change writing a shared list. That is the record-shape rule AGENTS.md states —
one file per row, read with a glob — applied to goldens.

Proven by mutation rather than asserted: gut the new golden's `capture` block and
the suite must red **naming that file**. §7 records it.

**What is NOT reached, and is owed.** `test_nemotron_h_loader.cpp` and the A3
driver still read `oracle.json` only. Repointing them is a change to what the
token gate compares against, which needs its own fresh review and its own
operator gate run, and it should be taken **after** somebody has measured what
#1289 scores against the new reference. It is listed under `## Owed`.

## 6. Gates

Contract, no GPU, no vLLM, no checkpoint:

```sh
python3 scripts/nemotron-h-oracle-capture.py --check \
  tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json
python3 scripts/nemotron-h-oracle-capture.py --check \
  tests/parity/goldens/nemotron_35_lightning_greedy/oracle.nhspeed-a.json
python3 tests/scripts/test_nemotron_h_oracle_capture.py
scripts/agent-preflight.sh
```

The capture itself is gated by the generator: `--capture` runs `check_golden` on
the document it built and raises rather than writing when it fails.

## 7. Evidence

Filled in from the run. Every table names the tree it was measured on, including
any overlay, and every run records the immutable SHA, the exact command, the
environment and the exit status. `assertions: 0` is a skip wearing a pass, and a
completed-looking log directory can contain zero measurement, so the guard's own
log is read beside the job's.

## 8. Risks

- **The box does not start the engine.** Then the row reports #1431 confirmed
  under idle conditions and commits no golden. The floor is not lowered.
- **A named profile is mistaken for the recovered one.** Mitigated in §2, in the
  generator's own comments and in the golden's `capture.engine.profile`.
- **The re-derived golden disagrees with the committed one on prompt 2.** That is
  the expected outcome, not an error. It is reported as the finding.
- **A reader treats this row as scoring #1289.** Mitigated by saying it is not,
  in §0, in the commit body and in the pull request body.

## 9. Stop conditions

- Do **not** lower the watchdog floor below 15000 MB, for any reason.
- Do **not** `ssh` to a fleet device. `rc` is the only path, and the file mutex
  is retired for fleet devices.
- Do **not** overwrite `oracle.json`.
- Do **not** ratify a distributional gate.
- Do **not** write a reconstructed configuration into any golden. A capture that
  cannot read a key leaves it absent and the writer refuses.
- Do **not** report a token comparison as a verdict on #1289.

## 10. Now

Spec committed; the capture is pending the lease.

## 11. Owed

- **[#926](https://github.com/mudler/vllm.cpp/issues/926) is NOT discharged by
  this row, and stays open.** #926 is that the reference every Nemotron token
  claim is scored against cannot be regenerated or attributed. This row produces
  an attributable golden, but it does not move any consumer onto it: the C++
  token gate, the A3 driver and #1289's score still read `oracle.json`, which is
  still the unattributable file. #926 closes when the reference in USE is
  attributable, not when an attributable file exists somewhere in the tree.
- **Repointing the token gate.** `test_nemotron_h_loader.cpp` and the A3 driver
  still read `oracle.json`. Moving them to the attributed golden is its own row,
  and should follow a measurement of what #1289 scores against it.
- **What #1289 scores against the new reference** — unmeasured here, and named as
  unmeasured.
- **#1431's root cause.** Untouched by this row.
