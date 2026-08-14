# One sanctioned way to take the GPU, and it must be able to prove it (2026-08-14)

**Kind:** tooling + gate · **Row:** `GATE-GPU-LOCK-WRAPPER` ·
**Issue:** [#587](https://github.com/mudler/vllm.cpp/issues/587) ·
**Affects:** `scripts/gpu-lock.sh`, `tests/scripts/test_gpu_lock.py`,
`scripts/agent-preflight.sh`, `.github/workflows/ci.yml`,
`.agents/environment.md`, `.agents/benchmarking.md`, `.agents/coordination.md`,
`.agents/roadmap_v1.md` (issue table only).

This spec is written before the repair it describes, and after the wrapper it
repairs. The wrapper landed on `row/GATE-GPU-LOCK-WRAPPER` (PR #596) with no
spec, no roadmap row and no issue-table entry; a fresh review returned FAIL on
ten findings. The spec is owed either way, so it is written here rather than
after the fact, and it records what the review measured.

## Gap

`#587`: GPU serialisation on `dgx.casa` ran through `flock`, but two different
lock *files* were in use. Two jobs holding different files run concurrently
while each believes it owns the box, `fuser $HOME/gpu.lock` shows an empty
waiter list throughout, and on GB10 — where `gpu_memory_utilization` reserves
HOST RAM — that is an OOM-reboot mechanism. It also voids any
contention-sensitive number silently.

`scripts/gpu-lock.sh` is the answer to the issue's own closing suggestion: a
wrapper that is the only sanctioned way to take the GPU, so the path stops being
a thing each script re-decides. Three things it shipped without:

1. **It could not prove it held the lock.** The nesting pass-through keyed on
   `[ "${VLLM_CPP_GPU_LOCK_HELD:-}" = "$LOCK_PATH" ]`, which proves the
   environment *names* a path and never that the lock is still *held*.
2. **It could not survive its own death.** No `trap`: SIGTERM left `ACQUIRE`
   with no `RELEASE`, and — measured, not inferred — left the wrapped job
   running orphaned with `flock -n` still reporting the lock HELD, because the
   job inherited the descriptor.
3. **It closed `#587` without closing it.** `.env` on this box sets
   `GPU_LOCK=/tmp/gpu`, and `$GPU_LOCK` outranks the canonical default, so an
   agent that loads `.env` exactly as `.env.example` documents resolves a
   different file from one that does not — through the wrapper.

## Design

### D1 — a pass-through is a proof, not a claim (F1)

`VLLM_CPP_GPU_LOCK_PID` was already exported at acquire and never read back. It
is now the discriminator: a pass-through is legitimate exactly when that PID is
a **live ancestor** of the invoking process, because only then is the descriptor
holding the lock demonstrably still open in a process we descend from. Anything
else — unset, non-numeric, dead, or a live non-ancestor — is refused `78`.

Walking the parent chain establishes liveness as a side effect: a dead ancestor
is reparented away and can never be found. `/proc/<pid>/stat` is read with
everything up to the *last* `)` discarded, because `comm` may contain spaces and
parentheses; `ps -o ppid=` is the fallback for hosts without `/proc`.

Refusing is the conservative direction and it is deliberate. A detached
descendant that genuinely inherited the descriptor is refused rather than passed
through; the refusal message names the escape
(`env -u VLLM_CPP_GPU_LOCK_HELD -u VLLM_CPP_GPU_LOCK_PID`), which makes the
wrapper take the lock for real. PID reuse could in principle make a stale PID
match a real ancestor; that requires the reused PID to land on the ancestor
chain, and the failure mode is the status quo rather than a new one.

### D2 — the canonical path stays `$HOME/gpu.lock` (F2)

**Operator decision, not relitigated here.** It is per-user, it is what most
callers take, and `/tmp` is cleared by the very reboot the lock exists to
survive, so the evidence of who held it does not outlive the incident.

One correction to the argument PR #596 shipped: `/tmp` is **sticky**, so a
non-owner *cannot* unlink or clobber another user's lock file. The real risks
are that any process on the box can take the flock, and that a file pre-created
by another user can lock its owner out of writing it. Stated accurately in the
script header and in `.agents/environment.md`.

`Closes #587` becomes `Refs #587`. The issue's Done-when is three clauses — one
canonical path, *every caller uses it*, and the discipline document names it
unambiguously — and only the first and third are satisfied by a wrapper. The
enumeration of what is still owed is in **Owed**, below.

### D3 — the wrapper's own death (F6) and the holder sidecar

The wrapped command moves to a background job with `wait`, because bash defers
trap handling until a *foreground* command finishes — which is exactly why the
shipped version could not answer its own SIGTERM. `<&0` is load-bearing: without
an explicit redirection a non-interactive shell reassigns an asynchronous list's
stdin to `/dev/null`, which would silently break every piped caller. `INT`,
`TERM` and `HUP` are forwarded to the job; an `EXIT` trap emits one real
`RELEASE` block on every path, guarded so a refusal or a timeout — which never
acquired — emits none.

The descriptor deliberately stays inherited by the job. If the wrapper is
`SIGKILL`ed the lock therefore remains held by the still-running job, which is
the safe direction: the GPU stays reserved rather than opening the door to a
second job.

On top of that, `<lock>.holder` records purpose, label, both PIDs, start time,
host and command while the lock is held, and is removed on release. It is
written only by a real `ACQUIRE`, so a nested pass-through neither rewrites nor
removes its ancestor's file. It supports the taxonomy that motivated it — a
four-hour queue where telling "live measurement" from "abandoned server" meant
`nvidia-smi` archaeology:

| observation | reading |
|---|---|
| sidecar + live PID, 0% GPU for hours | a ten-second conversation with its owner |
| sidecar + dead PIDs | a crashed hold, safe to break |
| no sidecar + lock held | a pre-wrapper holder, treat as opaque |

Both the wrapper PID and the job PID are recorded, because a `SIGKILL`ed wrapper
leaves its descendant holding the lock and a sidecar naming only the dead
wrapper would misclassify that as "safe to break". The `TIMEOUT` block reads the
sidecar back, so a blocked job is told who is ahead of it instead of being told
to go and look.

The sidecar is diagnostics and never a gate: an unwritable sidecar is stamped,
not refused, because the lock it would describe is already validated writable
and correctly held.

### D4 — the record opens before anything can refuse (F5)

`--record` appendability was checked *after* lock validation, so a lock-path
refusal never created the record file while a `TIMEOUT` reached it. The
commonest refusal therefore left no artefact, and stderr is what a dispatched
job discards. The check moves ahead of path resolution, so every outcome the
wrapper can produce reaches the record.

### D5 — fields, statuses and the silent fallback (F4, F8, F9)

`requested-lock` (verbatim) beside `lock-path` (resolved) is the field that makes
a `/tmp/gpu` divergence legible afterwards; it is now required by the field gate
and stamped on refusals too. `lock-source` names where the path came from, so
the fallback is never *silent* — `.env.example` ships `GPU_LOCK=` empty and
documents `set -a; . ./.env`, so an empty value is live on any host that copied
it, and falling back there is right while being unable to see it is not. This is
the repair that survives `.env.example` staying as it is: the *record* now says
which of the three sources supplied the path, so a divergence is visible even
where the prose has not caught up. An
explicit `--lock ''` is a usage error, not a request for the default: it used to
resolve `$HOME/gpu.lock`, which on a gate host is the file everyone else holds.
`EXIT_USAGE` is pinned by value, because a malformed command line and an
untakeable lock are the one distinction a dispatched job reads to decide whether
to retry.

### D6 — the suite runs (F10)

A test nothing runs is not a gate. `test_gpu_lock` joins the `SUITES` array in
`scripts/agent-preflight.sh` and gets its own step in `.github/workflows/ci.yml`.
It needs no GPU, no model and no network, and takes seconds.

## Risk

- The background-job change is the largest behavioural edit. Stdin passthrough
  and exit-status fidelity (`0`, `42`, `127`, `137`) are pinned by tests, and a
  mutation removes the `<&0` to prove the stdin check is real.
- Refusing an unverifiable pass-through could in principle refuse a legitimate
  detached descendant. That is the intended trade and the message names the
  escape.
- The `/tmp/gpu` callers are outside this change's authority. Until they are
  repointed, a run under the wrapper does **not** exclude an online-serving or
  gdn-packed-component run. Recorded in **Owed** and in the script header rather
  than left to be discovered.

## Tests

`tests/scripts/test_gpu_lock.py`, RED before the repair on every finding
(28 failures + 5 errors of 92 at `HEAD` of `row/GATE-GPU-LOCK-WRAPPER`).
Nineteen mutations, each breaking one guarantee in a scratch copy, all of which
must be caught:

| # | mutation | guards |
|---|---|---|
| M1 | the refusal returns instead of exiting | refuses, never falls back |
| M2 | `lock-path=` dropped | the path actually taken is recorded |
| M3 | `exit-code=` dropped | which thing killed the run |
| M4 | `exit "$RC"` → `exit 0` | `137` stays `137` |
| M5 | `EXIT_TIMEOUT=0` | timeout has its own status |
| M6 | `flock` → `true` | the lock is really taken |
| M7 | the wait becomes unbounded | bounded wait |
| M8 | the pass-through removed | nesting does not self-deadlock |
| M9 | the ancestor check → `true` | **F1** |
| M10 | the guard keys on presence, not path (reviewer's R7) | **F3** |
| M11 | `requested-lock=` dropped | **F4** |
| M12 | `lock-source=` dropped | **F8** |
| M13 | empty `--lock` falls back | **F8** |
| M14 | `EXIT_USAGE=2` → `78` | **F9** |
| M15 | the record block swapped back below validation | **F5** |
| M16 | the traps removed | **F6** |
| M17 | `} <&0 &` → `} &` | stdin survives the F6 repair |
| M18 | the sidecar never written | the sidecar |
| M19 | the sidecar never removed | the sidecar |

M15 *moves* the record block rather than deleting it: a deletion would redden
every record test and prove nothing about order, which is the whole finding.
M10 is the reviewer's own R7, which survived all three shipped nesting tests
because they asserted the stamp rather than the acquisition — this repository's
"a gate on a shared helper proves consistency, not correctness" trap.

## Gates

`python3 tests/scripts/test_gpu_lock.py`, then
`shellcheck -S warning scripts/gpu-lock.sh`, then
`scripts/agent-preflight.sh --staged`.

## Owed — every live `/tmp/gpu` reference (`#587` stays open)

Enumerated 2026-08-14 on `origin/main` merged into this branch. `.agents/`
historical evidence (`benchmark-record.md`, `parity-ledger.md`,
`model-matrix.md`, `coordination.md`'s campaign entries) records what *was* done
and is deliberately not rewritten.

| surface | occurrences | what it is |
|---|---|---|
| `scripts/dgx-online-serving.sh` | 3 (`exec 9>/tmp/gpu` at :1108) | live caller, takes the lock |
| `scripts/dgx-gdn-packed-component.sh` | 8 (`exec 9>/tmp/gpu` at :524, :543) | live caller, takes the lock |
| `scripts/opt-dgx-gate.sh` | 1 (:4) | names it in its header |
| `scripts/dgx-sglang-low-concurrency.sh` | 1 (:6) | names it in its header |
| `.github/workflows/triton-aot-sync.yml` | 2 (:79, :93) | **live CI**, `flock -w 7200 /tmp/gpu` |
| `.env` (untracked, this box) | 1 (:22) | `GPU_LOCK=/tmp/gpu` — should read `$HOME/gpu.lock`, or be emptied so the canonical default resolves |
| `.env.example` | :66-68 | still prescribes the raw `flock $GPU_LOCK -c '<command>'` and ships `GPU_LOCK=` empty. **NOT repaired here**, deliberately: `check-doc-checkpoint.py` classifies `.env.example` as `user_usage`, so any edit to it owes `docs/USAGE.md`, which this task's authority excludes. The two must land together — the same change should give `docs/USAGE.md` its first `scripts/gpu-lock.sh` entry, which it also lacks |
| `.agents/coordination.md` | 12, of which :99 is the live rule | said `${GPU_LOCK}` is `flock /tmp/gpu`; :99 repaired here, the other 11 are historical claim entries and stay |
| `.agents/specs/*.md` | 99 in 53 files | live gate instructions, e.g. `model-factory-registry.md:148`, `competitive-benchmarks.md:63`, `cuda-sglang-low-concurrency.md:333` |

`.env` is untracked developer configuration and outside this change's authority;
it is reported, not edited. The four scripts, the CI workflow and the 53 specs
are follow-up work under `#587`, which stays open until they are repointed.

## Stop conditions

- If the background-job change cannot preserve stdin, exit status or signal
  semantics on any pinned case, stop and report rather than relaxing a check.
- If refusing an unverifiable pass-through would redden a legitimate caller on
  `main`, stop and report rather than widening the guard back to naming a path.
- Never turn a red gate green by deleting an assertion or narrowing a scope.

## Outcome

Pending: the repair is implemented and gated here; the caller sweep in **Owed**
is not, and `#587` stays open for it. `Closes` was corrected to `Refs` for that
reason.

## Now

`GATE-GPU-LOCK-WRAPPER` is in review repair. No lifecycle state change is
claimed by this change: the wrapper is a tool and a gate, not a roadmap
capability, and it moves no row.
