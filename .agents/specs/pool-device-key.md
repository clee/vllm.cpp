# `POOL-DEVICE-KEY` — put the DEVICE in the `vllm::Pool()` free-list key

**Issue:** [#516](https://github.com/mudler/vllm.cpp/issues/516) (open).
**Row:** `POOL-DEVICE-KEY`. **Base:** `row/MODEL-DIFFUSION-LTX25` @ `aac24761`.
**Owning file:** this spec. **Status at write time:** spec committed before any
implementation, per AGENTS.md "Spec before code".

## 1. Scope

`include/vllm/model_executor/models/device_pool.h` — the shared, process-wide
caching device allocator every dense/MoE/diffusion forward draws its scratch
from — keys its free list by **byte size class only**. The device is not in the
key. A block allocated through backend A is handed to a `DBuf` running on
backend B whenever the two ask for the same size class in the same process.

In scope:

- `device_pool.h`: the key, the pool lifetime, the `Pool()`/`AuxPool()`/
  `ActivePool()` accessors, `Drain`, and the backend-less `Put` overload.
- `dense_device_glue.h` and `qwen3_5.cpp` `DBuf`: how a released pool block is
  handed to a `shared_ptr` (the ~28 copy-pasted deleters that name neither the
  pool nor the device).
- The two existing **per-caller workarounds** for this same fault, which the fix
  makes unnecessary and which must be REMOVED so the detector stays armed:
  `tests/vllm/models/test_ltx2_device.cpp` (`cpu_pool` scope, the SILENT-NaN
  direction) and `tests/vllm/models/test_deepseek_v2_forward.cpp`
  (`cuda_pool`/`cpu_pool`, the SIGSEGV direction).
- `DevicePoolPolicy` memoization in `dense_device_glue.h`/`qwen3_5.cpp`, which
  caches the FIRST device's residency policy in a function-local static and
  applies it to every later device — the same ambient-device assumption, one
  layer up.

Out of scope, explicitly:

- The size-class rounding itself. `VT_POOL_EXACT=1` (exact keying, reuse kept)
  was MEASURED still red, so over-allocation is not the fault and the class
  arithmetic is preserved byte-for-byte.
- `VT_POOL_BYPASS`, which stays a debugging lane and keeps its semantics.
- Any per-model change. If the fix makes a shipped model's gate red, STOP (§8).
- `ltx2_loader.*`, `ltx2_text_encoder.*`, the render path / engine wiring /
  server flag, and `ltx2_device.cpp` — concurrently owned by three other rows.
  This row needs no edit in any of them: `ltx2_device.cpp` reaches the pool only
  through `DBuf`.

## 2. The defect, as measured

One fault, two symptoms, selected by direction:

| direction | consumer | symptom |
|---|---|---|
| `cudaMalloc` block → CPU-backend forward | any CPU-backend case | **SIGSEGV** in `__memcpy_sve ← UploadStream ← PrepareStreamDev`, `compute-sanitizer` CLEAN (the fault is host-side) |
| CPU `aligned_alloc` block → CUDA forward | the shipped 21B DiT | **silent all-NaN output** |

Census of the silent direction: `video n=1024 nan=1024 inf=0 zero=0`,
`audio n=512 nan=512`, every element the identical `0x7fff0000` — the bf16
canonical quiet NaN widened by `WidenBf16`. A UNIFORM quiet NaN means a NaN was
COMPUTED and propagated; it is not plausible garbage read from a wrong pointer.

Three arms already discriminate the cause:

- `VT_POOL_BYPASS=1` (free list removed) → `test_ltx2_device` SUCCESS 13/13,
  6176 assertions.
- a per-case `DevicePool` via `ActivePoolScope` → SUCCESS 13/13, 6176.
- `VT_POOL_EXACT=1` (reuse kept, rounding removed) → still FAILURE.

So it is **cross-device reuse**, not over-allocation, and not the pool's
existence.

Why it stayed latent: it needs a bf16 host-backend device forward to run AFTER a
bf16 CUDA one in the same process. At f32 the two arms land in different size
classes and never trade blocks. The bug is old; only the ordering is new.

**Not established, and not required for the fix:** why a host `aligned_alloc`
block yields NaN on GB10 rather than merely running slowly through ATS. Unified
memory makes host pointers device-addressable, so the naive expectation is
correct-but-slow — yet the red forward is FASTER (0.230–0.252 s) than the green
solo one (0.594 s). Untested candidates: `aligned_alloc`'s 64-byte alignment vs
`cudaMalloc`'s 256 breaking a vectorised/TMA load, or kernels that require true
device memory. Recorded in §7 as an open question; the fix makes the ordering
unreachable either way.

## 3. Upstream anchors

vLLM does not have this bug because it never had this design: every allocation
record carries its own device, and every cache operation is device-scoped. Read
at the pinned oracle `555967922` (`$VLLM_SOURCE`) and at the local
`torch 2.11.0+cu130` headers:

- `vllm/device_allocator/__init__.py:12-14` — `HandleType` is documented
  `# py_device, py_size_or_aligned_size, py_ptr, py_handle`. The **device is
  field 0 of the handle**; the size is field 1. Ours keyed on field 1 alone.
- `vllm/device_allocator/cumem.py:200-219` — the free callback recovers the
  allocation from the pointer and reads the device back OUT of the handle
  (`device, size, d_mem, _ = data.handle`, `torch.cuda.synchronize(
  data.handle[0])`). A released block is re-associated with the device it came
  from, never with whoever asks next.
- `c10/cuda/CUDACachingAllocator.h:118-172` — the whole `CUDAAllocator`
  interface is parameterized by `c10::DeviceIndex device`
  (`getMemoryFraction`, `cacheInfo`, `getDeviceStats`, `releasePool`, …); the
  cache is per-device by construction.

Our `AuxPool()` comment already cites the STREAM half of the same invariant —
"two streams sharing one pool BREAKS" its reuse ordering, and torch answers that
with `record_stream`. The DEVICE half was never stated.

This row therefore mirrors upstream's **partitioning** (one cache per device),
not its stream tracking, which stays out of scope: our reuse ordering is
single-queue per pool and `AuxPool` remains the seam for the second stream.

`vllm/platforms/interface.py` `Platform` (our `platforms/interface.h`
`residency_policy()`) is likewise per-platform, which is why memoizing ONE
policy for the whole process (§1) is the same class of mistake.

## 4. Design

The device becomes structural, not a field someone must remember to pass.

**D1 — one `DevicePool` per device.** `DevicePool` gains a bound backend
(`explicit DevicePool(vt::Backend&)`). `Pool(vt::Backend& b)` and
`AuxPool(vt::Backend& b)` resolve a per-backend instance from a process-wide
table; the no-argument `Pool()` and `AuxPool()` are **removed**, so an
unqualified "the pool" can no longer be spelled. `vt::Backend*` is the device
identity: the registry hands out exactly one `Backend*` per `Device{type,index}`
(`vt::RegisterBackend(Device, Backend*)`, `kMaxDevicesPerType`), and
`GetBackend(type)` and `GetBackend(Device{type,0})` return the identical
pointer. Keying on the backend needs NO new virtual on `vt::Backend` and so
ripples into no backend implementation.

Lookup is a `std::mutex` + small vector, fronted by a thread-local
last-(backend,pool) memo, so the steady-state hot path is one pointer compare —
the pool exists to remove `cudaMalloc`, and must not pay a hash for it.

**D2 — the pool VERIFIES its device on every use.** `Get`, both `Put`s and
`Drain` keep their existing `vt::Backend&` parameter and now throw
`std::logic_error` when it is not the pool's own backend. This is a hard runtime
check, not `assert`: the SACRED builds are Release/NDEBUG, where an `assert`
would compile out and hand back the pre-fix behavior. It is the standing
detector for this defect class, and it is what makes an `ActivePoolScope`
pointed at another device's pool a loud refusal instead of a silent corruption.

**D3 — `ActivePool` resolves per device.** The thread-local becomes an
*override* defaulting to null; `ActivePool(vt::Backend& b)` returns the override
when set and `Pool(b)` otherwise. `ActivePoolScope` is unchanged in shape and
keeps serving the aux-stream case (`AuxPool(b)`), which is a stream distinction,
not a device one.

**D4 — `DBuf::ReleaseShared()` replaces ~28 hand-rolled deleters.** Every site
today is literally `alloc_bytes()`, then `Release()`, then a `std::shared_ptr`
whose deleter closes over the byte count alone and calls `Pool().Put(alloc, q)`.
That idiom names neither the device nor the pool, so
it also silently returns AUX-pool blocks to the MAIN pool — a second, live bug
in the same three lines. `ReleaseShared()` captures the buffer's OWN pool and
backend, so both are right by construction, and the backend-less
`Put(size_t, void*)` overload is removed with its last caller. Uncapped
retention is preserved for these cross-step buffers via a new
`Put(vt::Backend&, size_t, void*)`.

**D5 — the residency policy is memoized per device type**, not once per process.

**D6 — the two per-caller workarounds are removed.** They are the list of places
someone remembered; the fix is the property. Removing them is what proves it.

Not chosen, and why:

- *A composite `(Backend*, class)` key inside ONE pool.* Needs a
  `void*`→`Backend*` side map to serve the backend-less `Put`, i.e. a second
  hash operation on the hottest allocation path in the tree, to keep an overload
  that should not exist.
- *A `virtual Device device() const` on `vt::Backend`.* Reaches every backend
  implementation for information the caller already holds. Explicit stop
  condition for this row.
- *Per-caller `ActivePoolScope`.* Already rejected in #516 and re-rejected here:
  it is a list of remembered places, and the current red is exactly the siblings
  nobody scoped.

## 5. Tests

**T1 (RED-first, the row's own gate) — `tests/vllm/models/test_device_pool.cpp`,
new target `test_device_pool`.** Hardware-free: two distinguishable fake
`vt::Backend`s on the otherwise-unused `kXPU` slots, the technique
`test_backend_multidevice` / `test_reference_tier` already use. Cases:

1. **The defect.** Allocate on A, free, then allocate the same size class on B.
   The block B receives MUST NOT be the block A freed, and must come from B's
   own `Alloc`. RED before the fix.
2. **Reuse survives.** Get/Put/Get on ONE backend returns the identical pointer.
   Without this, "fixed" is indistinguishable from `VT_POOL_BYPASS` — the pool's
   whole reason to exist is reuse.
3. **Size-class rounding survives.** Two byte sizes in one class still trade one
   block, and `VT_POOL_EXACT` still separates them.
4. **`Drain(b)` is device-scoped.** Draining A frees A's blocks only; B's free
   list is untouched and no block is freed through the wrong backend.
5. **A cross-device `ActivePoolScope` is REFUSED** (throws, names both devices)
   rather than served.
6. **`ReleaseShared()` returns the block to its own pool and backend**, and an
   AUX-scoped buffer returns to the AUX pool, not the main one.

**T2 (end-to-end corroboration) — `test_ltx2_device --order-by=rand
--rand-seed=7`.** No checkpoint, no NAS, ~2 s. RED before the fix (exit 139,
`:533 FATAL ERROR: test case CRASHED: SIGSEGV`), and note the trap it carries:
**44 assertions, 0 failed, beside a SIGSEGV** — grep `Status:` and the CASE
count, never `assertions:` alone. Deterministic: default and `--order-by=name`
and seeds 1 and 7 are all red; the only green ordering is green by declaration
order, not by safety.

**T3 (blast radius).** 20+ test binaries mix a CUDA and a CPU backend in one
process. Enumerate them from the tree (not from memory) and run the full suite
before and after.

**T4 (#486 hypothesis).** `test_minimax_h3` SIGSEGVs on GB10 when two CUDA cases
share a process, `compute-sanitizer` clean. Run it under `VT_POOL_BYPASS=1` and
again after the fix. Green either way is a result and gets recorded; it does NOT
gate this row and the connection is not forced if it does not hold.

Mutation targets a reviewer should exercise: delete the device from the key
(T1.1 must fail); delete the D2 device check (T1.5 must fail); make
`ReleaseShared` use the main pool (T1.6 must fail); make `Drain` free every
bucket (T1.4 must fail).

## 6. Gates

Correctness only; this row claims no performance result.

- `test_device_pool` — new, must be RED before and GREEN after, with both
  outputs captured.
- `test_ltx2_device` — GREEN at `--rand-seed=7`, at `--order-by=name`, and at
  default order, **with no per-case pool scoping anywhere in the file**.
- Baselines that must not move (case/assertion counts):
  `test_ltx2` 29/1615 · `test_ltx2_vae` 16/1816 · `test_ltx2_text_encoder`
  17/3350 · `test_ltx2_pipeline` 35/2358 · `test_ltx2_loader` 20/2363 ·
  `test_ltx2_video` 17/170 · `test_ops_attention_cross` 9/32 ·
  `test_minimax_h3` 79/57395 (CPU) · `test_minimax_h3_video_fold` 6/137 ·
  `test_video_engine` 11/254 · `test_capi` FULL 55/505.
- Full `ctest` on the CPU host before and after, and on dgx (`-j 1`, GB10
  unified memory OOM-reboots the box under a parallel CUDA suite).
- Every doctest result reported as its `Status:` line AND its case count; the
  assertion count DROPS when cases throw and reads clean beside a crash.

Build discipline: never redirect build output to `/dev/null`; chain on the build
exit; clean-rebuild after a header change, because an incremental build masks
`-Werror` and `device_pool.h` is a header.

## 7. Risks

| # | risk | mitigation |
|---|---|---|
| R1 | `Backend*` is not device identity if one backend object is registered for two `Device` indices | The registry stores one pointer per `Device{type,index}`; asserted by `test_backend_multidevice`. Recorded as the assumption it is. |
| R2 | The per-backend lookup lands on the hottest allocation path | Thread-local last-(backend,pool) memo: one pointer compare in steady state. No hash, no lock, on the hit path. |
| R3 | D2's runtime check costs a branch per `Get` | One perfectly-predicted compare against a member; the alternative (`assert`) is compiled out of exactly the Release builds the gates run. |
| R4 | ~28 deleter sites across 25 model files is a wide diff | Every site is byte-identical today, and each becomes ONE line via `ReleaseShared()` — the diff SHRINKS the call sites and removes an idiom that can be got wrong. No `ltx2_*` file is touched. |
| R5 | Removing the two per-caller workarounds could red a suite for an unrelated reason | They are removed in the SAME change that makes them unnecessary; if either stays red, that is a finding to report, not a scope to re-apply (§8). |
| R6 | A pool is now created per backend, so a mixed-backend process holds two free lists | That is the point. Retention is bounded by each device's own peak scratch, and `Drain` is now device-correct where before it freed one device's blocks through another's backend. |
| R7 | Static-destruction order for a table of pools that print `VT_POOL_STATS` at exit | Pools are owned by a function-local static table and outlive every model object; the stats path is unchanged. |

## 8. Stop conditions

- Keying by device turns out to need an API change that ripples into every
  backend → STOP, report the design, write nothing.
- The fix makes a SHIPPED model's gate red → STOP and report; do not adjust the
  model.
- `NEEDS_CONTEXT` for missing binding context; `NEEDS_DECISION` for a material
  disagreement — never a silent scope change.
- Never weaken a bound, delete an assertion, or scope a caller away to reach
  green. In particular, the LTX-2.5 shipped case is the ONLY test exposing the
  SILENT direction: it does not get an `ActivePoolScope`.
- GPU work waits on `$HOME/gpu.lock` on dgx with a BOUNDED `flock -w`; on
  timeout, report and stop. Never kill a holder.

## 9. Evidence to record

The committed SHA of this spec (before implementation); the RED `test_device_pool`
and RED `test_ltx2_device --rand-seed=7`; the GREEN of both plus the LTX-2.5
device suite with no per-case scoping; the enumerated mixed-backend binaries and
the full suite before/after; the #486 result either way; every baseline as a
`Status:` line and case count; the exact `flock` lines, the wait, and `docker ps`
at both ends; `git log --oneline` and the final SHA.

## 10. Outcome

Landed. Spec `17532ea0b` (this file, committed before any implementation), RED
test `f4be8a4e2`, fix `1a2eb35ec`.

**Environment.** dgx.casa, GB10 sm_121a, CUDA 13.0.88, configured with all three
MANDATORY confirmations printed: `CUTLASS found at ~/cutlass-4.5.0; enabling
sm120a NVFP4 cutlass GEMM`, `FlashAttention-2 prefill/decode: ENABLED for
arch(es) [121a]`, `Triton AOT: vendored tree …/sm_121a matches triton_kernels/
(MANIFEST hashes OK)`. Both arms are CLEAN builds (`rm -rf build`), RED
1508/1508 exit 0 and GREEN 1508/1508 exit 0; the CPU host likewise clean,
1219/1219 exit 0, zero warnings under `-Werror`. `local-ai-worker` was stopped
before any GPU work and left DOWN. `mnt-nas_share.mount` had lost its boot race
after a reboot and was restarted with `sudo -n systemctl restart`; the shipped
DiT was then proven READABLE (21,025,119,068 bytes, first 16 bytes dumped)
before the opt-in case was allowed to run.

**RED, at `f4be8a4e2`.** Both directions, on the same binary:

| run | result |
|---|---|
| `test_device_pool` (CPU host) | 4 cases / 2 passed / 2 failed · 15 assertions / 5 failed · FAILURE · exit 1 |
| `test_ltx2_device --order-by=rand --rand-seed=7` | **exit 139**, `:533 FATAL ERROR: test case CRASHED: SIGSEGV` · 4 cases / 3 passed / 1 failed / 9 skipped · **44 assertions / 0 failed** |
| `--rand-seed=1` | exit 139, SIGSEGV at `:518` · 7 / 6 / 1 / 6 · 481 assertions / 0 failed |
| `--order-by=name` | exit 139, SIGSEGV at `:452` · 7 / 6 / 1 / 6 · 455 assertions / 0 failed |
| default order, `LTX2_SHIPPED_DIT` set (21B FP8) | **exit 1**, `:925 FATAL ERROR: REQUIRE( std::isfinite(v) )` · 13 / 12 / 1 · 4639 assertions / 1 failed — the SILENT direction |
| default order, fixture UNSET | SUCCESS 13/552 — the skip that impersonates a repair (§7.0(d)) |
| `test_minimax_h3` | exit 139, SIGSEGV at `:3974` · 38 / 36 / 2 / 41 · 42,724 assertions |
| `test_minimax_h3` + `VT_POOL_BYPASS=1` | SUCCESS 79 / 79 · 451,993 assertions · exit 0 |

**GREEN, at `1a2eb35ec`** (`b4618b8c7` before an amend that added the
doc-gate argument to the message; the tree is identical), with NO per-case pool scoping anywhere (both
workarounds deleted):

| run | result |
|---|---|
| `test_device_pool` | 8 / 8 · 26 assertions · SUCCESS (also under `--order-by=rand` seeds 1 and 7) |
| `test_ltx2_device` seed 7 / seed 1 / by-name / default | 13 / 13 · 552 assertions · SUCCESS · exit 0, all four |
| `test_ltx2_device` + shipped 21B DiT, default AND seed 7 | **13 / 13 · 6176 assertions · SUCCESS · exit 0** |
| `test_minimax_h3` (pool ON) | **79 / 79 · 451,993 assertions · SUCCESS · exit 0** |
| `test_deepseek_v2_forward` | 11 / 11 · 1558 assertions · SUCCESS |

**#486 is this bug.** Asked as a question, not assumed: at the RED commit
`test_minimax_h3` SIGSEGVs with the pool on and is 79/79 with `VT_POOL_BYPASS=1`
— the bypass lane changes nothing but the free list. At the fixed commit it is
79/79 with the pool ON. The hypothesis is confirmed by measurement in both
directions.

**ENOSPC re-verification.** The local box hit 100% disk during this session
(operator note). Every build log here was grepped for `No space left` — zero hits
in the local RED build, the local clean AFTER build, the dgx RED build, the dgx
GREEN build and the dgx AFTER `ctest` — and the local gates were then re-run
CHAINED to their build in one command, with `ninja` reporting "no work to do"
first, so no result here comes from a stale binary left behind by a died build.

**Baselines, CPU host, full `ctest`: 402/402 passed, exit 0.** Every named
baseline byte-for-byte where it was: `test_ltx2` 29/1615 · `test_ltx2_vae`
16/1816 · `test_ltx2_text_encoder` 17/3350 · `test_ltx2_pipeline` 35/2358 ·
`test_ltx2_loader` 20/2363 · `test_ltx2_video` 17/170 ·
`test_ops_attention_cross` 9/32 · `test_minimax_h3` 79/57395 ·
`test_minimax_h3_video_fold` 6/137 · `test_video_engine` 11/254 · `test_capi`
55/505.

**Full CUDA `ctest -j 1` on dgx, AFTER: 98% passed, 9 failed out of 437.** The
nine, with their exact signatures:

| test | signature |
|---|---|
| `test_serve_low_tools` | the Python bench-tooling suite (no C++, no GPU) |
| `test_linear_method` | `:246 CHECK( after == before + 1 )` → `0 == 1` — the `fused_gate_up` counter did not move, i.e. the fused Marlin gate-up path FELL BACK; its numeric arm passed at `bitexact=12288/12288 max_abs=0`, which is exactly what a fallback to the split path looks like |
| `test_ops_gdn` | `:728 CHECK( bad == 0 )` → `2609 == 0`, a GDN kernel numeric check |
| `test_capi` | **SEGFAULT** at `:482` "capi: custom logits processor forces the generated token (ABI v8)" — 4 cases / 3 passed / 1 failed / 51 skipped, 47 assertions / 0 failed |
| `test_glm4_moe_lite_paged_engine`, `test_qwen3_apc_e2e`, `test_minicpm3_paged_engine`, `test_internlm2_paged_engine`, `test_llama_paged_engine` | checkpoint-gated paged-engine suites |

**These nine are UNATTRIBUTED, and that is stated rather than glossed.** The dgx
full-suite BEFORE arm was NOT run: the box was at 99–100% disk with a single 31 GB
build tree, so the RED and GREEN trees could not coexist, and the GPU lock was
shared with three other agents (one bounded `flock -w 2700` wait expired without
acquiring). Without that arm, "pre-existing" would be an inference, and this row
does not report inferences as measurements.

What IS measured and bears on them: the same binaries are GREEN on the CPU host
in a 402/402 full run — including `test_capi` at its recorded 55/505 and
`test_linear_method` — and every pool-adjacent gate on dgx is green
(`test_ltx2_device` 13/13·6176 in four orderings, `test_minimax_h3` 79/79,
`test_deepseek_v2_forward` 11/11, `test_device_pool` 8/8). None of the nine
signatures is a cross-device scratch symptom: two are a dispatch counter and a
kernel numeric check that this diff does not reach, five are checkpoint-gated,
one is Python.

**`test_capi` IS A TIMING FLAKE, and that one is measured, not inferred.** It
reproduced on the CPU host, where this row's change is CPU-only and every pool
gate is green: a first full `ctest -j 1` had it at its recorded 55/505 SUCCESS,
a second full run on the SAME binary (ninja: "no work to do") failed it, and
standalone it is **55/55 · 505 assertions · SUCCESS**, then `--repeat
until-fail:8` passed **8 of 8** with per-run wall times of 0.78, 1.02, 1.65,
2.00, 4.21, 12.56, 228.94 and 339.88 s. A test whose duration spans three orders
of magnitude on a contended box is timing-sensitive, which is what
`.agents/environment.md` already records for `test_capi`. Non-deterministic on a
host where the pool change cannot produce it ⇒ not this row's.

**The exact next step, for whoever picks up the remaining eight.** Re-run them
standalone on this GREEN tree and again on a pre-fix tree; the cheap
discriminator that needs no second build is `VT_POOL_BYPASS=1`, which removes the
free list entirely — a failure that survives bypass cannot be a pooling failure.
The rerun script is `~/work/pool-device-key/dgx_rerun.sh`; it was written and
shipped but never ran, because two consecutive bounded `flock -w 2700` waits
expired without acquiring the shared GPU lock (three other agents were rendering
on it). Reported and stopped, per the lock protocol, rather than camping.

**Blast radius, MEASURED not guessed.** `VT_POOL_STATS=1` makes every pool print
one line naming its backend at exit, so the full suite counts pools per binary
rather than grepping for device names. On the CPU host **42 of 402 test binaries
instantiate a `DevicePool` at all**, and exactly one — `test_device_pool` itself,
with its 11 fakes — instantiates more than one, which is the expected answer for
a host with a single backend. The grep-level upper bound is 51 test sources that
name both `kCUDA` and `kCPU`.

**What was rejected.** A composite `(Backend*, class)` key inside one pool: it
needs a `void*`→`Backend*` side map to serve the backend-less `Put`, i.e. a
second hash operation on the hottest allocation path, to keep an overload that
should not exist. A `virtual Device device() const` on `vt::Backend`: reaches
every backend implementation for information the caller already holds, and was
an explicit stop condition. Per-caller `ActivePoolScope`: a list of remembered
places, which is what the current red already disproved.

**Why the defaults are what they are.** The device check is a runtime `throw`
rather than an `assert` because the gate builds are Release/NDEBUG, where an
assert compiles out and hands back exactly the pre-fix behaviour. The pool
lookup is a thread-local last-(backend,pool) memo rather than a hash because a
`DBuf` resolves its pool on every construction and the pool exists to avoid a
synchronizing `cudaMalloc`. Size-class rounding is untouched: `VT_POOL_EXACT=1`
was measured still red, so it was never the fault.

**Not established, and left open.** Why a host `aligned_alloc` block yields a
uniform quiet NaN on GB10 rather than running correct-but-slow through ATS. The
fix makes the ordering unreachable, so the question is now academic for this row,
but it is not answered and is not claimed to be.

**Related, NOT fixed here.** `MoeAuxStreamFor` (`qwen3_5.cpp`) caches its aux
stream on `d.q.device.index` alone, so a CPU device 0 and a CUDA device 0 collide
in the key. It is unreachable today because the only call site is gated on
`Backend::SupportsAuxStream()`, which no host backend answers true to — so it is
a latent trap of this same family rather than a live defect, and widening this
diff to it was not worth the review surface. Recorded here so the next reader
finds it.

**Owed, and outside this row's granted scope** (this spec was the only record
surface granted): the `#516` line in the issue table of `.agents/roadmap_v1.md`,
and the `porting-inventory.md` §L8 note at line 1455 which still reads "the
shared `DevicePool` is DEVICE-BLIND … repairing it is owed as its own row" and
should now point at this spec.
