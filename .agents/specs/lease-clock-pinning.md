# ENV-LEASE-CLOCK-PINNING — the driver wants `CAP_SYS_ADMIN`, and the spread rule scores the deepest single sample

Issue: [#1354](https://github.com/mudler/vllm.cpp/issues/1354) — `nvidia-smi -lgc`
is refused inside an `rc` lease, and every clock-pinned figure in this repository
was taken over the retired host + `ssh` + `flock` path.
Row: `ENV-LEASE-CLOCK-PINNING` (proposed; the roadmap row is owed at claim, and
the work below is two changes, one to the fleet and one to
`tools/bench/gpu_clock_state.py`)
Amends: [`bench-assert-clock-state.md`](bench-assert-clock-state.md), whose
`MAX_WITHIN_RUN_SPREAD_PCT` this spec proposes to replace rather than widen.
Prior art: [#1265](https://github.com/mudler/vllm.cpp/issues/1265) — the same
class. A capability the records assume, which the current access path does not
provide.

## The defect, in two independent halves

**Half one is a missing Linux capability, and it is fleet-wide.** The refusal is
not about `uid`. The NVIDIA driver's administrator test is

    // kernel-open/nvidia/os-interface.c
    NvBool NV_API_CALL os_is_administrator(void) { return NV_IS_SUSER(); }

    // kernel-open/common/inc/nv-linux.h
    #define NV_IS_SUSER()                   capable(CAP_SYS_ADMIN)

so a privileged RM control — which is what `nvmlDeviceSetGpuLockedClocks`, and
therefore `nvidia-smi -lgc`, issues — is gated on `CAP_SYS_ADMIN` and never on
`uid == 0`. A container process can be root and still fail it, which is exactly
what the job reported:

```text
$ nvidia-smi -lgc 2190
The current user does not have permission to change clocks for GPU 0000000F:01:00.0.
LGC_RC=4
```

The worker's own manifest says why. In `infra-flux-kube`,
`manifests/dgx/rc-worker.yaml`, the `rc-worker` DaemonSet declares
`runtimeClassName: nvidia`, `NVIDIA_VISIBLE_DEVICES=all` and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, and the `worker` container — the
long-lived container that leased jobs run *inside* — **declares no
`securityContext` at all**. The only `securityContext` in the file sits on the
`workspace-perms` initContainer. With none declared the container gets the
default OCI capability set, which does not contain `CAP_SYS_ADMIN`.
`manifests/thor/rc-worker.yaml` and `manifests/orin/rc-worker.yaml` have the same
shape, so this is a property of the fleet and not of one box.

Three things this is **not**, each ruled out rather than assumed:

- **Not `NVIDIA_DRIVER_CAPABILITIES`.** That variable selects which userspace
  driver components the container runtime injects. `utility` is what supplies
  `nvidia-smi` and NVML, and NVML plainly works — every `--query-gpu` in the run
  succeeded and produced 155 to 246 samples per window. No value of that
  variable adds a Linux capability.
- **Not the device cgroup.** `/dev/nvidiactl` is reachable; the queries that
  travel the same node succeed.
- **Not the driver refusing a containerised caller as such.** The same driver on
  the same box accepted `-lgc 2190` from the host on 2026-08-15
  (`gpuClkMin 2190, gpuClkMax 2190`). Host root holds `CAP_SYS_ADMIN`; container
  root does not.

**One reading in the issue does not survive contact with the log.** The harness
ran `nvidia-smi -pm 1 2>&1 | tail -1` and got `All done.`, which looks like a
successful NVML write beside a refused one. It is not evidence: `job.log:8` of
the same run reports `NVIDIA GB10, 12.1, 580.173.02, Enabled, 3003 MHz` — read
**before** the `-pm` call — so persistence was already `Enabled` and the call was
a no-op. `nvidia-smi` also prints `All done.` after a per-GPU failure line, and
`| tail -1` discards that line. **No NVML write is known to succeed inside a
lease.**

**Half two is the instrument, and it is not caused by half one.**
`summarize_sm_clocks` computes

    spread_pct = (max - min) / median * 100

which is a range statistic over the raw samples. It scores the single deepest
excursion in the window and nothing else. Two consequences, both measured on the
2026-08-19 evidence at `/mnt/nas_share/rc/q38bf16/out/`:

**Its numerator only ever grows with window length.** `max - min` is
non-decreasing in `n` by construction, and only a moving median can offset it.
Recomputing
`spread_pct` over growing prefixes of the same leg (our c1 rep 2, n=155):

| prefix n | `spread_pct` (min/max) | p5–p95 band |
|---:|---:|---:|
| 30 | 6.79% | 2.33% |
| 60 | 14.10% | 0.93% |
| 90 | 16.43% | 0.44% |
| 120 | 26.36% | 0.00% |
| 155 | 26.36% | 0.00% |

The longer the sampler watches, the worse the window scores, while the
distribution it is describing gets *tighter*. That is the same perverse
incentive `MIN_BUSY_SAMPLES` was added to remove on the other side — the window
nobody watched outscoring the one that was — reintroduced by the statistic
itself. `MIN_BUSY_SAMPLES` bounds the failure from below; nothing bounds it from
above.

**It does not bound what it is read as bounding.** The nine 2026-08-19 windows:

| leg | n | min | p5 | median | p95 | max | `spread_pct` | p5–p95 band | samples <95% of median | longest run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours c1 r1 | 155 | 2177 | 2489 | 2489 | 2515 | 2515 | 13.58% | 1.04% | 5 | 1 |
| ours c1 r2 | 155 | 1859 | 2489 | 2489 | 2489 | 2515 | 26.36% | 0.00% | 5 | 1 |
| ours c1 r3 | 156 | 2158 | 2489 | 2489 | 2515 | 2515 | 14.34% | 1.04% | 4 | 1 |
| ours c8 r1 | 245 | 2138 | 2328 | 2515 | 2515 | 2515 | 14.99% | 7.43% | 28 | 5 |
| ours c8 r2 | 246 | 2177 | 2333 | 2515 | 2515 | 2515 | 13.44% | 7.24% | 29 | 5 |
| ours c8 r3 | 244 | 2190 | 2353 | 2515 | 2515 | 2515 | 12.92% | 6.44% | 29 | 5 |
| vLLM c1 r1 | 163 | 2262 | 2489 | 2489 | 2515 | 2515 | 10.16% | 1.04% | 2 | 1 |
| vLLM c1 r2 | 163 | 2080 | 2489 | 2489 | 2489 | 2515 | 17.48% | 0.00% | 4 | 1 |
| vLLM c1 r3 | 163 | 2054 | 2489 | 2489 | 2489 | 2515 | 18.52% | 0.00% | 4 | 1 |

At c1 the excursions are **single samples, isolated, and periodic**. Our c1 rep 2
dips at 46.6 s, 78.4 s, 107.0 s, 135.6 s and 164.0 s — spacings of 31.8, 28.6,
28.6 and 28.4 s against a median request E2EL of **28.591 s** in the same file.
The dip lands on the request boundary. It is a load transition, not a state the
window sat in. The first-third against last-third median moves **0.00%** in every
one of the nine windows: there is no drift to find.

The transfer bound follows from the mean, not the extremum. Time-weighted mean
SM clock against the median: −0.14%, −0.40%, −0.20% at c1 (ours) and +0.12%,
−0.25%, −0.31% (vLLM). The *entire* dip population is worth at most 0.40% of clock, so at
the tool's own physics ceiling of 1.0 percentage point of time per percentage
point of clock it can move a leg by at most 0.40%. The cross-arm offset on the
**mean** is **−0.101%**, against **0.000%** on the median. Measured throughput
agrees: 4.4068 / 4.4027 / 4.4040 tok/s across the three c1 legs, a range of
0.093%, with the ordering matching the mean-clock ordering (r1 > r3 > r2 on both
axes) and the magnitude coming in under the mean-clock prediction, as a partly
memory-bound decode should. `n = 3`, so that agreement corroborates and does not
prove.

**The proposed statistic is not vacuous, and this is the check that matters.**
The c8 windows carry a genuinely dispersed population — 28-29 of ~245 samples
below 95% of the median, in runs of up to 5 s, 9 to 10 separate runs — and their
p5–p95 band is 6.4% to 7.4%. A 5% band ceiling **still fails all three of them**.
The rule discriminates.

## Scope

**In scope.**

1. **The fleet change**, recommended and not performed here: grant
   `CAP_SYS_ADMIN` to the `worker` container of the `rc-worker` DaemonSet on
   `dgx`, `thor` and `orin`. This is a human decision about privilege on a shared
   host, argued under `## The fleet change` below.
2. **The instrument change**: replace `MAX_WITHIN_RUN_SPREAD_PCT` over
   `(max - min) / median` with a percentile band plus an explicit drift term, in
   `tools/bench/gpu_clock_state.py`, with the red-first tests below.
3. **The record reconciliation** the instrument change invalidates:
   `.agents/benchmarking.md` §"The clock is part of the measurement" and the
   argument block in [`bench-assert-clock-state.md`](bench-assert-clock-state.md).

**Out of scope.**

- Re-deriving the discarded Qwen3.8-27B c1 ratio. That is
  [#915](https://github.com/mudler/vllm.cpp/issues/915)'s to re-run once one of
  the two halves lands, and it needs fresh GPU time either way.
- `MAX_CROSS_ARM_OFFSET_PCT`, `MIN_BUSY_SAMPLES` and `MIN_BUSY_FRACTION`. They
  are untouched. The cross-arm rule passed perfectly on the very pairing this
  spec exists for, which is the reason to leave it alone.
- Any standing pin on a fleet box. `benchmarking.md` already forbids leaving a
  box pinned, and nothing here changes that.

## The fleet change

The minimal patch, on each of `manifests/{dgx,thor,orin}/rc-worker.yaml`, on the
`worker` container and not on the pod:

```yaml
        - name: worker
          image: ghcr.io/mudler/rc-worker:edge
          securityContext:
            capabilities:
              add: ["SYS_ADMIN"]
```

`privileged: true` also works and is strictly worse: it grants every capability,
disables seccomp and AppArmor, and relaxes the device cgroup, when one capability
is what the driver asks for.

**Say the cost plainly.** Leased jobs run inside this same long-lived container,
so this grants `CAP_SYS_ADMIN` to every job anybody submits through `rc`, not
only to the worker. `CAP_SYS_ADMIN` is close to root-on-the-host in practice.
Against that: the pod already runs as root, already mounts a `hostPath`, already
holds cluster RBAC that can scale Deployments across every namespace, and the
submitters are the same people who hold `ssh` on the box. The marginal exposure
is small, and it is still a decision for the fleet's owner rather than for this
row.

**It is worth taking, and the reason is measured.** Pinning demonstrably works on
this hardware once the capability is there. On 2026-08-15, from the host, a
requested 2190 delivered a **flat 2184 MHz** over n=861 retained busy samples —
min 2158, median 2184, max 2184, a `spread_pct` of **1.19%** — with
`clocks_event_reasons.active = 0x0` for the whole series and persistence mode
`Disabled`. So pinning both passes the existing ceiling and removes thermal
throttling entirely, and it does so *below* the boost point rather than by
fighting it.

Two consequences a reader must carry:

- **A pinned number is ~12% slower in absolute terms.** 2184 against a sampled
  median of 2489 is a 12.3% clock deficit. That is the price of comparability and
  it is only payable when both arms pay it. Never set a pinned absolute beside an
  unpinned one.
- **Persistence mode is not the lever.** The pinned, flat, unthrottled series ran
  with persistence `Disabled`; every throttling window of 2026-08-19 ran with it
  `Enabled`. Persistence governs whether the driver stays resident between
  clients, not DVFS under load. It is also not known to be settable from a lease,
  for the `-pm` reason above.

## Design

Three named thresholds replace one, and each answers a different question about
the window.

    MAX_WITHIN_RUN_BAND_PCT = 5.0     # (p95 - p5) / median * 100
    MAX_WITHIN_RUN_DRIFT_PCT = 2.0    # |median(last third) - median(first third)| / median * 100
    MAX_EXCURSION_MEAN_COST_PCT = 1.0 # |median - mean| / median * 100

- **The band** replaces the range. It answers "was this window one state", which
  is the job `bench-assert-clock-state.md` states for the spread rule, and it
  answers it without letting one sample decide. 5.0 is carried over unchanged and
  deliberately: the argument for that number in the prior spec is an argument
  about how far apart two *states* may be, and it transfers to the band intact.
  It accepts the only clean #543 capture (3.68% on the range, and no wider on the
  band), it rejects the two-state failure that rule exists for (~26% either way),
  and it rejects the c8 windows above at 6.4-7.4%.
- **The drift term** is what actually catches #543's within-boot failure — two
  probes eight minutes apart at 2398 and 1781 — in the case where the window
  straddles the transition and a band alone might not. 2.0 is twice the
  cross-arm offset ceiling, on the argument that a systematic move *inside* one
  arm may not exceed twice what is tolerated *between* arms. All nine 2026-08-19
  windows read 0.00%.
- **The mean-cost term** is the honest bound on what the excursions can transfer
  into a ratio, and it is the term that makes dropping the range statistic safe.
  A window whose mean sits within 1.0% of its median cannot move a
  clock-proportional kernel by more than 1.0%, by the same physics argument
  `MAX_CROSS_ARM_OFFSET_PCT` already rests on. It is an absolute value because
  an upward excursion transfers exactly as much as a downward one. The nine
  windows read 0.12% to 1.04%; the c8 legs sit at the ceiling, which is the
  correct place for them.

`spread_pct` is **kept in the record and stops being a gate term.** Removing it
would silently rewrite the meaning of every clock record already committed, and a
reader comparing an old record with a new one needs the old field to still be
there. It is reported, not asserted — the same demotion the prior spec applied to
the transfer coefficient.

### Why not simply restore pinning and leave the gate alone

Because the gate is wrong on its own terms, and pinning does not make it right.
The band and the drift term are what a pinned window would pass *for the right
reason* — one state, no drift — rather than by the accident of never dipping.
And two of the three fleet devices may not get the capability: `thor` and `orin`
are the same manifest shape, and a future box need not be. An instrument that can
only produce a verdict on a pinned box is an instrument that stops working the
next time the access path moves, which is the failure this row is named after.

## Risks

- **It looks like widening an assertion to turn a red green, and that is
  forbidden.** It has to be argued, not asserted. The defence is that the
  statistic changes rather than its threshold: the ceiling stays 5.0, two new
  terms are *added*, and the tests below include a case the current rule misses
  and the new rule catches. A change that only ever loosens has no such case. If
  the reviewer cannot mutate the new rule into failing on a two-state window,
  this section is wrong and the change must not land.
- **Percentiles over a small window are coarse.** At `MIN_BUSY_SAMPLES = 30` the
  p5 is the 2nd sorted sample and the p95 the 29th, so a 30-sample window can
  hide one excursion at each end. That is what the mean-cost term is for, and it
  is stated rather than hidden.
- **The three-term rule can pass a window with a sustained mid-window step of
  under 5%.** So could the old one. `MAX_CROSS_ARM_OFFSET_PCT` is what qualifies
  the ratio; this rule only establishes that each arm was one state.
- **`n = 3` throughput corroboration.** The agreement between mean-clock ordering
  and throughput ordering rests on three legs of one workload on one boot. It is
  offered as corroboration of a physics bound, never as the argument for it.
- **Granting `CAP_SYS_ADMIN` widens the blast radius of any `rc` submitter.**
  Named above; the decision is the fleet owner's.

## Tests

`tests/tools/test_gpu_clock_state.py`, red before the change in every case.

1. **The case the current rule misses.** A synthetic window of 200 samples that
   steps from 2400 to 2280 at the halfway point and never returns: a 5.0%
   sustained two-state window with **no** excursion beyond it. The current
   `spread_pct` reads 5.13% against a 5.0% ceiling and only barely fails; shift
   the step to 2300 (4.2%) and it **passes today**. The drift term reads 4.2% and
   fails. Red first on the second variant.
2. **The case the current rule fails wrongly.** Replay our c1 rep 2 window
   verbatim from committed fixture data: `spread_pct` 26.36%, band 0.00%, drift
   0.00%, mean cost 0.40%. Currently `reasons` is non-empty; after the change it
   is empty. This is the regression the row exists for and it must be pinned to
   real samples, not to a hand-built distribution.
3. **The case the new rule must still fail.** Replay our c8 rep 1 window
   verbatim: band 7.43%. `reasons` non-empty before and after, and the *message*
   must name the band rather than the range.
4. **Window length must not decide the verdict.** Assert that the band of a
   window equals the band of that window repeated twice, and that `spread_pct`
   of the same pair differs — an executable statement of the monotonicity defect,
   which fails today because nothing asserts it.
5. **`spread_pct` survives in the record.** Assert the field is still present and
   still equals `(max - min) / median * 100`, and that no gate expression reads
   it. The second half is the mutation: change the constant it used to be
   compared against and no test may move.
6. **Threshold mutation set**, matching the prior spec's discipline: each of the
   three constants moved in the passing direction must turn at least one case
   red, printing `git diff --stat` and any `compile_err` so a mutation that never
   applied cannot read as a pass.

## Gates

- `tests/tools/test_gpu_clock_state.py` green, with the count moving by the six
  cases above. A count that does not move is a failure, not a pass.
- Full CPU gate, since the helper is standard-library-only and runs with no GPU.
- No GPU gate is owed by the instrument half. The fleet half owes one: a single
  `rc run` on `dgx:gpu0` reporting `LGC_RC=0`, followed by one
  `gpu_clock_state.py` window showing the pin held. That is the acceptance test
  for the capability grant and it cannot be run before the grant.

## Stop conditions

- Stop and report `NEEDS_DECISION` if the fleet owner declines
  `CAP_SYS_ADMIN`. The instrument half still stands alone; the spec's framing
  does not.
- Stop if test case 1 cannot be made red before the change. That would mean the
  new rule catches nothing the old one missed, and the change is then a widening
  after all.
- Stop if the committed fixture for cases 2 and 3 cannot be taken from the
  2026-08-19 evidence. A hand-built distribution proves the arithmetic and not
  the claim.

## Evidence

Raw sample windows, twelve `*.samples.json` files, at
`/mnt/nas_share/rc/q38bf16/out/{bench-20260819T035148Z,vllm-20260819T073125Z,vllm-20260819T095758Z}/`.
Per-leg records beside them; `CLOCKS.txt` in the first directory folds them. The
refusal is at `job.log:65-66` of `bench-20260819T035148Z` and at `job.log:19-20`
of both `vllm-*` directories. The pre-`-pm` persistence reading is `job.log:8`.

Driver source quoted above: `NVIDIA/open-gpu-kernel-modules`,
`kernel-open/nvidia/os-interface.c` (`os_is_administrator`) and
`kernel-open/common/inc/nv-linux.h` (`NV_IS_SUSER`).

Worker manifests: `infra-flux-kube`, `manifests/{dgx,thor,orin}/rc-worker.yaml`,
at `7ce8c77`.

## Owed

- The roadmap row. This spec is committed first, per "Spec before code"; the row
  is opened at claim.
- [#1354](https://github.com/mudler/vllm.cpp/issues/1354) stays open until both
  halves land, because either alone leaves the other's defect in place.
- The re-run of the Qwen3.8-27B bf16 c1 pairing, owned by
  [#915](https://github.com/mudler/vllm.cpp/issues/915) and
  [#979](https://github.com/mudler/vllm.cpp/issues/979).

## Now

`PROPOSED`. Nothing is implemented. The diagnosis is complete and the two changes
are independent; neither has been made.
