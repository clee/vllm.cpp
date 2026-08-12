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
