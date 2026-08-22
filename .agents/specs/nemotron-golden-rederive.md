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
0-based index 31 -- on **the one prompt both other oracle-side runs also fail**.
WHERE inside the 32 tokens those two runs miss is not something `main` records,
so the overlap is established at prompt granularity and no finer. That is still
the fact that makes the 95/96 unreadable: nobody can say
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
on is that prompt 2 is the prompt all three disagree on, which every recorded run
shows on its own. It deliberately does not depend on WHERE in prompt 2 they
disagree, because only the in-flight #1432 numbers say that.

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

**A glob over the contract is not enough, and the fresh review of PR #1703
measured why.** `check_golden` asks whether a golden records the configuration
it was captured under. Nothing in that contract is tied to a model, a checkpoint
or a battery, so a contract-VALID file naming a completely different checkpoint,
carrying one entry instead of three and the prompt `"Write a haiku about ducks"`,
passed the glob and `--check` printed `0 problem(s)` over it. The suite now runs
`identity_problems(doc, reference)` beside the contract on every golden it finds:
`model`, `revision`, the prompt battery in ORDER, and `prompt_token_ids` must all
match `oracle.json`'s. The last of those is §2a's own check — it is the field
that says the engine received the identical token sequence despite `--capture`
submitting text where the 2026-08-18 run submitted `TokensPrompt`s — so a
difference there reds and gets read, rather than landing under a green. The guard
is reference-relative rather than a second table of literals, and the reference
is anchored by its own cases: the battery to `capture.PROMPTS`, the revision to
`capture.CHECKPOINT_REVISION`.

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

**Measured on** `mudler-ubuntu-box` (x86_64, 20 cores) in the worktree
`/home/mudler/_git/vllm.cpp-golden-rederive`, branch
`row/GATE-NEMOTRON-GOLDEN-REDERIVE`, base **pinned** at `2f2a709253c60599f`
after the second merge (`8540a27558813faef` before it), **nothing overlaid**.
The GPU side ran on `dgx:gpu0` (NVIDIA GB10, sm_121,
driver 580.173.02, aarch64) inside an `rc` lease, never over `ssh`, submitter
`claude/mudler-ubuntu-box/golden-rederive`. Job artifacts on the share under
`/mnt/nas_share/rc/golden-rederive/` (the worker sees `/workspace/golden-rederive`).

### 7.1 The contract gates, and the count beside every green

Measured at the review-repair commit, whose suite file is
sha256 `9a4407f77e73627bd5adfbc09b6002e2ab18f9a7e190567802a89e14b15f3af8` --
named by content rather than by commit id because the branch was rebuilt onto a
newly pinned base after the measurements and the commit id moved while the file
did not. The record commit after it touches neither the suite nor the golden, so
every figure below still reads at the branch head. They all MOVED with the
repair, so the pre-repair numbers this section used to carry -- 43 cases, and
`RANGE_COUNT=3` against `8540a27558813faef` -- are superseded rather than
re-stated. An evidence table has to name the tree it was measured on.

```
python3 scripts/nemotron-h-oracle-capture.py --check .../oracle.json   -> 0 problems, rc=0
python3 tests/scripts/test_nemotron_h_oracle_capture.py               -> 54 cases, OK
```

`scripts/agent-preflight.sh --fail-on-skip`: **zero gates skipped**, and
`commit-trailers` and `commit-style` both report `ok` against the pinned base
`2f2a709253c60599f` at **`RANGE_COUNT=7`**. `RANGE_COUNT=3` was 4 when it was
written -- `git rev-list --count 8540a2755..HEAD` read 4, not 3 -- and the
figure is now recounted at the head it is recorded on, which is the only place
a count means anything.

The count is not decoration. Both checkers were run against an EMPTY range
(`HEAD..HEAD`) and **returned rc=0** -- a gate that examined nothing prints the
same green as a gate that examined everything -- so the range width is reported
beside the result rather than inferred from it. The first preflight of this
branch is the other half of that lesson: it SKIPPED both gates because
`origin/main` had moved and was no longer an ancestor, and a skipped gate
reported nothing about this tree. The merge commit exists to make them run --
and it had to be made TWICE, because `origin/main` moved again while the branch
was in review and put the branch behind for the second time. `agent-preflight.sh`
takes its `TRAILER_BEHIND` arm on exactly that condition, so the merge is not
housekeeping: without it both trailer gates report nothing and the run still
looks green. `git merge-base --is-ancestor origin/main HEAD` is the check, and
it returns 0 at the head this section records. The base is PINNED by SHA
rather than by ref name, because `origin/main` moved a THIRD time mid-repair --
another worktree in this shared checkout fetched and the ref advanced from
`08c81a892` to `2f2a70925` under a merge already resolved against it.

**Three reds, and none of them is this row's.**

`test_cpu_x86_llamacpp_floor` fails on
`test_a_contended_leg_is_discarded_and_never_summarised`, reporting
`NO_QUIET_WINDOW`. That is
[#618](https://github.com/mudler/vllm.cpp/issues/618) by its exact case name --
the harness is load-dependent and this box was carrying other sessions' builds
at `load average: 60.87`, which is also why the earlier run of this branch saw a
second case of the same suite fail and this one did not.
`git diff --stat origin/main -- tests/scripts/test_cpu_x86_llamacpp_floor.py
scripts/cpu-x86-llamacpp-floor.sh` is **empty**, against a positive control on
this row's own files that is not, so the suite is byte-identical to `origin/main`.

`check-agent-record` and `test_agent_record` fail on the same one error:
`.agents/issue-index.md: issue #1649 listed twice`. Both `#1649` rows are on
`2f2a70925` already -- `git show origin/main:.agents/issue-index.md | grep -c
'^| \[#1649\]'` is `2`, added by `a7bb3130b` and `2f2a70925` -- and the three
rows this branch adds are `#1694`, `#1729` and `#1730`, which
`git diff origin/main -- .agents/issue-index.md` shows as the only additions.
`main` is red on this gate, and it is filed as
[#1731](https://github.com/mudler/vllm.cpp/issues/1731). The repair is not
available to this branch anyway: the file is append-only, so deleting a row is
the one edit its own rule forbids.

### 7.2 The re-derived golden is REACHED, proven by mutation

`SHIPPED_GOLDEN` named `oracle.json` alone, so a golden captured beside it would
have been gated by nothing. Every mutation below was applied alone, on the head
named in §7.1, and the whole table was **re-measured there**: the review repair
edited the very case M-B deletes, which silently disarms a mutation proof taken
before it.

| # | Mutation | Proof it applied | Result |
|---|---|---|---|
| — | baseline | — | **54 cases, `OK`** |
| M-A | a second golden in the directory carrying the `af8170154` shape (no `capture` block) | `git status --short` shows the untracked file | **1 failure**, and the subTest NAMES the file: `(golden='oracle.MUTANT.json')`, `missing 'capture'` |
| M-B | the same mutant golden, and the glob CASE deleted instead | `git diff --stat` **27 deletions**, `parse_rc=0` | **53 cases, `OK`** — nothing else in this tree holds it |
| M-G | a CONTRACT-VALID second golden of a DIFFERENT model: bogus `revision`, ONE entry, prompt `"Write a haiku about ducks"` | `git status --porcelain` shows the untracked file; `--check` reads it and prints `1 golden entry` | **before the repair: 54→43 cases `OK`, `--check` `0 problem(s)`.** After: **1 failure** naming all four differences |
| M-1 | M-G present, and the `identity_problems` CALL SITE deleted from the glob loop | `git diff --stat` 1 deletion, compiles | **54 cases, `OK`** — the call site is what makes the glob reach the guard |
| M-2 | M-G present, and `identity_problems` neutered to `return []` | `git diff --stat` 1 insertion, compiles | **7 failures**, all in `CaptureIdentityTests` |

M-B is the reachability question rather than the contract question. M-G is the
review finding that motivated the repair: it is what a glob gating provenance
SHAPE cannot see, and M-1 and M-2 are the two halves of its fix — that the guard
is REACHED from the glob, and that the guard itself is gated by cases rather than
by a mutation somebody has to remember to repeat.

**M-A also exhibits the `--check` labelling defect**, in its own evidence line:
the subTest says `oracle.MUTANT.json` and the contract message says
`oracle.json: missing 'capture'`, because the top-level `where` label is a string
literal. That is [#1729](https://github.com/mudler/vllm.cpp/issues/1729), listed
under `## Owed`; it is pre-existing and not repaired here.

Restored byte-for-byte after every mutation: suite sha256
`9a4407f77e73627bd5adfbc09b6002e2ab18f9a7e190567802a89e14b15f3af8` at the head
named in §7.1, `git status --porcelain` and `git diff` both clean over
`tests/`, and the committed golden's sha256 unchanged at
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`.

### 7.3 The capture shape was checked BEFORE spending a lease

A contract failure after a twelve-minute engine start costs a lease, so the
document `--capture` will build was constructed offline from the 2026-08-18
`nhspeed-a` log's own resolved values and run through `check_golden`:
**0 problems**. Dropping each of the twenty engine keys in turn, the contract
named the one dropped **every time** -- no key is decorative. In particular,
removing `attention_backend` or `moe_backend` is refused, which is why the job
passes both and then re-reads them out of the run's own log.

### 7.4 The BOX condition, measured twice, and it is not what the record assumed

`rc devices` reported `dgx:gpu0` `ready` and free before both submissions. It
says who holds the GPU. **It says nothing about what the host is doing**, and on
this box the two came apart:

| Lease | `uptime` load average | `MemAvailable` at start | Swap used |
|---|---|---|---|
| `20260822T150416Z` | **92.58, 135.52, 93.72** | **10464 MB** | 14163 MB |
| `20260822T152834Z` | **92.85, 85.02, 64.53** | **5031 MB** | 17354 MB |

`MemTotal` is 122502 MB and this job's own watchdog floor is 15000 MB, so **both
leases began BELOW the floor**. `/proc/pressure/memory` on the second read
`full avg60=83.64`. `nvidia-smi --query-compute-apps` was empty both times, so
the GPU was genuinely free and the pressure is host-side; `ps` inside the job
container sees only its own five PIDs, so whatever holds the other ~117 GB is
outside this job.

**This falsifies the premise the row was dispatched under.** The fleet was not
idle. The box sat in the same load band (100-150) that
[#1431](https://github.com/mudler/vllm.cpp/issues/1431) records for all five of
its failures, and other campaigns (`fp8-gate`, `ltx25-pixel-ab`) were queued on
it throughout.

The first lease was **killed rather than run**. A capture started at
`MemAvailable=10464` would have been shot by its own watchdog within a second
and the result would have read as `#1431 confirmed` -- a contended host wearing
the shape of a verdict about the oracle.

### 7.5 The quiet gate, and both of its arms

The second submission carries a precondition that measures the condition and
**exits 93 on timeout rather than proceeding**, because a wait-for-quiet loop
that times out and runs anyway while printing its success label is a failure this
repository has already had.

Thresholds are derived, not chosen. `MIN_AVAIL_MB=60000`: the one control
(2026-08-18 `nhspeed-a`) consumed 38 GB and the floor is 15 GB, so 53 GB is the
arithmetic minimum for that run to fit, and 60 GB is that with margin -- and it
is deliberately BELOW the control's own 90274 MB start, because a gate set at the
control's exact start might never open. `MAX_LOAD1=20`: `rc describe dgx:gpu0`
reports `cpus=20`, so `load1 <= cpus` is "not oversubscribed".

Falsified on five arms before it was trusted, by feeding the loop synthetic
`/proc` readings:

| Input | Expected | Result |
|---|---|---|
| avail 90000 MB, load1 1.50 | pass | `QUIET GATE PASSED`, rc=0 |
| avail 10464 MB, load1 92.58 (the real 15:04 condition) | refuse | `PRECONDITION_NOT_MET`, **rc=93** |
| avail 117417 MB, load1 135.52 (#1431's own condition) | refuse | `PRECONDITION_NOT_MET`, **rc=93** |
| avail 51528 MB, load1 2.00 | refuse | `PRECONDITION_NOT_MET`, **rc=93** |
| avail 60000 MB, load1 20.00 (both boundaries exactly) | pass | `QUIET GATE PASSED`, rc=0 |

The last arm is what stops it passing by refusing everything. The harness needed
one repair of its own first: its initial version extracted the loop from the
job script by string index and picked up the COMMENT rather than the code, and
all four arms then "failed" with a shell syntax error -- an instrument failure
that would have read as the gate working.


### 7.6 The capture did NOT run, and what that is and is not

`rc` job `e3b3d366-c515-402f-bfa7-803f7e760903`, run directory
`/mnt/nas_share/rc/golden-rederive/20260822T152834Z`, 2026-08-22.

Everything up to the engine succeeded:

```
TOOLKIT_RC=0   NVCC=/usr/local/cuda-13.0/bin/nvcc   CURAND=.../include/curand.h
TORCH_RC=0     VLLM_RC=0
vllm.__file__    = /tmp/oracle-venv/lib/python3.12/site-packages/vllm/__init__.py
vllm.__version__ = 0.1.dev1+g555967922
IDENTITY ASSERTED: 0.1.dev1+g555967922      IDENTITY_RC=0
```

So the oracle **is** the pin, it **is** the staged venv rather than a system
install, and the three toolchain traps that VOIDed earlier oracle jobs one header
at a time -- `Python.h`, `nvcc`, `curand.h` -- were all cleared.

Then the quiet gate refused, six samples, 100 s:

| elapsed | `MemAvailable` | `load1` |
|---|---|---|
| 0 s | 5340 MB | 0.80 |
| 20 s | 5338 MB | 0.57 |
| 40 s | 5345 MB | 0.41 |
| 60 s | 5344 MB | 0.29 |
| 80 s | 5341 MB | 0.29 |
| 100 s | 5335 MB | 0.29 |

**Read the two columns together, because that is the finding.** `load1` decays
from 0.80 to 0.29 -- the box becomes completely CPU-idle -- while `MemAvailable`
does not move at all: 5335-5345 MB, a 10 MB band, against a 122502 MB
`MemTotal`. Memory that does not come back when the work stops is not activity.
It is an allocation, and it was already there at 15:28:34 before this job built
anything, so it is not this job's.

Four supporting readings, each measured rather than inferred: `buff/cache` was
1316 MB and `Shmem` 45 MB, so it is **not reclaimable cache**; 17354 MB of swap
was in use, so the host had already been pushed to evict; `nvidia-smi
--query-compute-apps` was **empty**, so the GPU is genuinely free and the
pressure is host-side; and `ps -eo pid,rss,comm` inside the job container
returned **five PIDs with a 9.9 MB maximum**, so whatever holds the other ~117 GB
is outside anything this job can see.

The job was **killed at 15:49:59Z** rather than left to burn its remaining 23
minutes of quiet-wait: three jobs were queued behind it (`ltx25-pixel-ab`,
`gate-qwen38-27b-fp8-block`, `ltx25-fa2hd128`), the box went `busy` with the next
of them within 10 s, and six flat samples had already answered the question. The
run directory contains `apt.log`, `identity.log`, `job.log` and three pip logs,
and **no `oracle.nhspeed-a.json`** -- verified with `ls` and a `find` over the
whole share rather than assumed from the exit path.

**What this is NOT.** It is **not** #1431 confirmed, and it is **not** #1431
refuted. #1431 is a claim about **engine start-up** -- the host collapsing during
the first forward -- and **no engine was ever constructed here**. The gate that
stopped this run sits *before* `LLM(...)`. Reporting a contended host as "#1431
confirmed" is precisely the broken-instrument-as-verdict failure the gate exists
to prevent, and it is what the first lease of this row would have produced had it
been allowed to run.

**What this also is not: #1431's condition.** #1431's five runs each began with
**117417 MB** available on an otherwise clean box and fell during the forward.
This box began with **5031 MB** and never rose. Those are different walls, and
the second one was never reached today.

**The next traceable step, and why this row cannot take it.** The question is
which process holds ~117 GB on `dgx.casa` while the CPU is idle. Answering it
needs the **host** process table, which a job container cannot see. For a fleet
device that means either an `rc` job with host PID visibility or `rc hold` plus a
host shell -- and the standing rule is that `rc hold` + `ssh` is authorized only
for the `BENCH-QWEN38-27B-SOTA` campaign, not for this row. So it is reported to
the operator rather than taken here, and it is filed on #1431.

## 8. Risks

- **The box does not start the engine.** Realised, but not in the predicted
  shape: the engine was never constructed, so #1431 is neither confirmed nor
  refuted, and no golden is committed. The floor was not lowered.
- **A contended host gets reported as an oracle defect.** This was the live risk
  and the quiet gate is what stops it. Without the gate, this row's first lease
  would have produced a watchdog kill at `MemAvailable=10464` and it would have
  read as `#1431 confirmed`.
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

**The capture has not run, and no golden was re-derived.** The spec, the named
configuration, the committed generator's invocation, the reachability gate and
its mutation proof are landed. `oracle.json` is untouched, byte-for-byte, and
`#1289`'s score is unchanged because nothing it reads has moved.

The blocker is **not** the one this row was dispatched against. It is not
#1431's forward-pass collapse -- that was never reached. It is that `dgx.casa`
is carrying ~117 GB of host memory with an idle CPU, so the oracle cannot be
started at all, by this row or by any other. Named as a precise external
resource blocker on #1431, owner: the operator.

The capture is one `rc` job away once the host has memory: nothing else about
this row is unfinished, the recipe is staged at
`/mnt/nas_share/rc/golden-rederive/job.sh`, and its toolchain and identity legs
are already measured green.

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
- **The capture itself** -- #1694 stays open. Everything it needs is staged and
  gated; it needs one lease on a host with memory.
- **Who holds ~117 GB on `dgx.casa` while the CPU is idle.** Filed on #1431. It
  needs the host process table, which this row has no authority to read.
- **#1431's root cause.** Untouched by this row: its wall was never reached.
- **[#1710](https://github.com/mudler/vllm.cpp/issues/1710) -- `dgx:gpu0` begins
  a lease with 5 GB of 122 GB available and `rc` exposes no available-memory
  label, so `ready` cannot mean usable.** This is the blocker §10 describes, in
  its filed form: the capture is one `rc` job away, and the job cannot be
  usefully queued while a lease can be granted on a box with no memory. Owner:
  #1710. This row is BLOCKED on it and does not repair it -- the fix is a
  controller-side label, which this row has no authority over.
- **[#1729](https://github.com/mudler/vllm.cpp/issues/1729) -- `--check <path>`
  reports every top-level violation against the hardcoded name `oracle.json`.**
  Found by the fresh review of PR #1703 and visible in this spec's own §7.2
  M-A line. PRE-EXISTING: it predates this branch, and it becomes reader-visible
  the moment a second golden exists beside `oracle.json`, which is what this row
  produces. NOT fixed in flow: the repair threads the checked path into
  `check_golden` and owes its own regression case, which is a checker-semantics
  change rather than a one-line accuracy fix. Owner: #1729.
- **[#1730](https://github.com/mudler/vllm.cpp/issues/1730) -- this suite's only
  registration is one line of `ci.yml`, and `check-test-registration.py` stays
  at rc=0 when it is deleted.** The suite is also absent from
  `agent-preflight.sh`'s `SUITES` array. Found by the same review; PRE-EXISTING,
  since the suite was registered this way when it landed under #926. NOT fixed
  in flow: the repair edits a checker's required set and a shared runner array,
  which needs its own red-before evidence. Owner: #1730. Related: #408, #1509.
- **`sampling` is NOT part of the identity guard.** `identity_problems` holds a
  second golden to the same model, revision, prompt battery and tokenization as
  `oracle.json`, and deliberately not to the same `sampling` block:
  `check_golden` already ties every row's `token_ids` length to that file's own
  `sampling.max_tokens`, so a capture at a different depth is internally
  consistent and legible rather than silent. A future comparison that needs the
  depths equal should say so in its own row. Named here so the omission is a
  decision rather than an oversight.
