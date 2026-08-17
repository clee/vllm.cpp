# GPU lease methodology: `rc` is the default where the host has it

Row: `ENV-GPU-LEASE-METHODOLOGY`.
Issues: [#1129](https://github.com/mudler/vllm.cpp/issues/1129),
[#1130](https://github.com/mudler/vllm.cpp/issues/1130).

## Scope

State one rule in the root policy file: where the host has
[resource-controller](https://github.com/mudler/resource-controller), whose
client is `rc`, claiming a device through it is the required path for GPU work,
and it replaces the `flock` file mutex as the default. Where the host does not
have `rc`, the file mutex remains the instruction.

In scope:

- `AGENTS.md` gains the rule and the link to the tool.
- `.agents/environment.md` gains the conditional in its how-to.

Out of scope:

- Copying the `leasing-a-gpu` skill into this repository. A copy of another
  document goes stale without saying so.
- Any checker change. This row changes no checker semantics, so it owes no
  red-before mutation under `## Changing the rules or a checker`.
- The oracle migration owed by #1129. The row that takes #1129 owns it.

## The defect, corrected against the tree

The brief for this edit said the root file's Commands section and its
GPU-adjacent prose point at the old mechanism. That is not what the tree holds.
Measured at `36381f346`, `AGENTS.md` matches `flock` zero times and names no
mutex, no lock path, and no `ssh` procedure at all.

The real defect is an absence, and it is the stronger argument. `AGENTS.md`
says of itself that it "contains the complete policy" and that files under
`.agents/` "cannot add or weaken a rule in this file". The lease requirement
lives only in `.agents/environment.md`, which is a task guide. Under the root
file's own terms it is therefore guidance and not a rule. A reader who follows
the root file alone is told nothing about how to reach a GPU.

## Why the rule is conditional

The developer stated the requirement as "when the host has it, it is the
default". Write it that way, because the hosts are not identical. A reader on a
box without `rc` still needs an instruction, and a reader on a box with `rc`
must not be able to argue for `ssh` plus `flock`.

## Why the bypass is not a style preference

Two mutexes that do not exclude each other are worse than one, and the cost is
measured. On 2026-08-17 one session took the file mutex over `ssh` while another
session held the same box through `rc`. Neither mutex excluded the other.
`.agents/specs/minimax-music3.md` §13.10 retains a whole speed axis as VOID
because of it, and `.agents/benchmark-record.md` records the window in which the
fleet reported `thor:gpu0` free while it was in use. That is the #777 failure
again, in which this repository carried two GPU mutexes and neither serialised
the other.

## What a lease can carry

The limit is now precise, measured on 2026-08-17 through two
`rc run -d dgx:gpu0` jobs and recorded in `.agents/environment.md`. A lease
carries bytes and not executables. The worker reads and writes the shared
`/workspace`, refuses direct execution from it because the mount pins
`file_mode=0664`, and runs staged content through `sh FILE`, through the dynamic
loader, or after a copy to `/tmp`. It cannot produce or fetch a runtime, because
it has no compiler, no downloader and no Python.

## Risks

The one checker this edit can break is `test_gpu_lock_one_truth` (#777), which
requires exactly one `**GPU mutex:**` bullet in `.agents/environment.md` and
requires that bullet to name both `${GPU_LOCK}` and `$HOME/gpu.lock`. The edit
adds no second mutex statement and does not touch that bullet, which already
reads "this runs INSIDE an `rc` lease, never instead of one". `AGENTS.md` is
outside that checker's scanned set, and the rule there spells the canonical
`${GPU_LOCK:-$HOME/gpu.lock}` so it can never read as a second truth.

If the conditional rule cannot be written without a second mutex statement, stop
and return `NEEDS_DECISION`. Do not weaken the checker.

## Gates

```sh
scripts/agent-preflight.sh
python3 -m unittest discover -s tests/scripts -p 'test_gpu_lock_one_truth.py'
```

`test_gpu_lock_one_truth` is the focused gate and stays green. The full preflight
is the row gate.

## Evidence

- Probe jobs `1cb56f84-62bf-4c90-b138-9bd4c3b0617a` and
  `c692d5a0-ec3d-4498-86e4-e86a2864e91a` on `dgx:gpu0`, 2026-08-17.
- `.agents/environment.md`, "The lease carries bytes, and the exec bit is a
  mount option".
- `.agents/specs/minimax-music3.md` §13.10 for the VOIDed speed axis.

## Stop conditions

- Stop if the edit needs a second mutex statement. Return `NEEDS_DECISION`.
- Stop if `test_gpu_lock_one_truth` goes red. Never widen it to pass.

## Owed

- #1129 stays open. No vLLM leg runs on `dgx.casa` by a lease-compliant path
  today, because nothing has staged a runtime on the NAS. Whether a relocated
  CUDA virtual environment starts inside a worker is UNMEASURED.

## Now

The rule is stated in `AGENTS.md` and the conditional is in
`.agents/environment.md`. The next step belongs to whoever takes #1129, which is
staging a runtime the lease can start.
