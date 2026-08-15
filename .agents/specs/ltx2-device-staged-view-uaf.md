# The LTX-2 device test read a weight view out of a temporary that had already freed it

Row: `FIX-LTX2-DEVICE-UAF-904`
Issue: [#904](https://github.com/mudler/vllm.cpp/issues/904)
Owning row: `ROAD-V1-LTX25`
Baseline: `origin/main` @ `5a0ffe9e3`

## 1. Scope

Make `sanitize-cpu (address,undefined)` and `sanitize-cpu (thread)` green on
`main` by giving the second staged `Ltx2DitDeviceWeights` in one LTX-2 device
test case a name, so the `vt::Tensor` view copied out of it does not outlive the
storage that view points into.

**Out of scope, deliberately:** any change to `vt::cpu::Threadpool`, to
`ParallelForRows`, or to the ownership contract between a queue and the buffers
an in-flight op reads. §2 explains why: the measured evidence does not implicate
any of them, and #904's title does. Also out of scope: a checker or a type
change that would refuse a borrowed `vt::Tensor` outliving its owner. That is a
real footgun and it is recorded under `## Owed` rather than done here.

## 2. What the defect is, and the correction #904 owes

`.agents/issue-index.md` and the title of #904 both state the cause as:

> `~Ltx2DitDeviceWeights` frees the staged DiT buffers on the main thread
> (`ltx2_device.cpp:1088`) while a `vt::cpu` threadpool worker is still reading
> one inside `AddKernel` (`cpu_layernorm.cpp:33`), so the staged weights'
> lifetime is not joined to the in-flight parallel op that reads them.

**That is the wrong cause.** It would send the next reader to redesign where a
join belongs — the op dispatch, the queue, or the destructor — and none of those
is implicated. The defect is a dangling view into a destroyed temporary, in the
test, on one line.

`tests/vllm/models/test_ltx2_device.cpp`, the case *"ltx2 device: an f32
keyframes embedding under a bf16 stream is REFUSED"*, at `5a0ffe9e3`:

```cpp
723  staged.weights.keyframes_abs_pos_embedding =
724      Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kBF16)
725          .weights.keyframes_abs_pos_embedding;
726  CHECK(RefusalMessage([&] {
727          (void)Ltx2DitForwardDevice(q, p, staged.weights, &m.video, &m.audio, vt::DType::kBF16);
728        }).empty());
```

`Ltx2StageDitWeightsToDevice` returns an `Ltx2DitDeviceWeights` that owns each
staged buffer through a `shared_ptr<void>` whose deleter calls `vt::cpu::Free`
(`src/vllm/model_executor/models/ltx2_device.cpp:1085-1088`). A `vt::Tensor` is
a **borrowed view** — `view.data` is a raw pointer into that storage. Line 723
copies the view out; the temporary the view borrows from is destroyed at the end
of the full-expression, which is line 725; every staged buffer is freed there.
The forward at line 727 then reads through the dangling pointer.

### The evidence that distinguishes the two causes

ASan attributes both the allocation and the free to **line 724**, and the read
to the call at 726-727:

```
freed by thread T0 here:
    #1  FreeAligned64                        src/vt/cpu/cpu_backend.cpp:28
    #3  operator()                           src/vllm/model_executor/models/ltx2_device.cpp:1088
   #14  vllm::Ltx2DitDeviceWeights::~Ltx2DitDeviceWeights()
   #15  DOCTEST_ANON_FUNC_76                 tests/vllm/models/test_ltx2_device.cpp:724

previously allocated by thread T0 here:
    #1  AllocAligned64                       src/vt/cpu/cpu_backend.cpp:20
    #3  vllm::Ltx2StageDitWeightsToDevice    src/vllm/model_executor/models/ltx2_device.cpp:1085
    #4  DOCTEST_ANON_FUNC_76                 tests/vllm/models/test_ltx2_device.cpp:724
```

A teardown-races-an-in-flight-op defect would attribute the free to the case's
closing brace at line 729, where the *named* `staged` dies. It attributes to
724, before the read at 726. The named `staged` is never the freed object.

Which thread performs the read is not the signal, and reading it as one is how
#904 reached the threadpool. `ParallelForRows` dispatches rows to workers and
blocks until they finish, so a worker frame under a synchronous `vt::Add` is
ordinary. #904 captured the read on worker `T2`; CI run `31885935312` captured
the same defect with the whole read chain inline on `T0`
(`Threadpool::Run` → `ParallelForRows` → `AddKernel`). Same address, same
`LoadF32At` frame, same free site — because the thread was never the variable.

`Ltx2StageDitWeightsToDevice` has 14 other call sites, all in this same test
file. Line 724 is the only one that does not bind the result to a named object,
which is also why exactly one case aborts.

## 3. The change

Bind the second staging to a named `const` local that outlives the forward:

```cpp
const Ltx2DitDeviceWeights restaged =
    Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kBF16);
staged.weights.keyframes_abs_pos_embedding =
    restaged.weights.keyframes_abs_pos_embedding;
```

This keeps the case's intent exactly as written: the positive control still runs
a **fresh** staging rather than the object the case already had, so it cannot be
satisfied by a view that was never replaced. Saving the original bf16 view
before line 714 overwrites it would be one line shorter and would weaken the
control that way, which is why it was rejected — see §6.

## 4. Tests and evidence

The test *is* the change, so there is no new case to add; the red is the
existing case under the existing lane. Both runs use the `sanitize-cpu
(address,undefined)` job's own configuration, on this box, with `setarch -R`
(without it the binary SIGSEGVs with no output on this host, which reads exactly
like a crash in the code under test).

```sh
cmake -S . -B build-sanitize -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_CUDA=OFF \
      -DVLLM_CPP_SANITIZE='address,undefined'
cmake --build build-sanitize --target test_ltx2_device
UBSAN_OPTIONS=print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 \
VT_POOL_BYPASS=1 setarch -R ./build-sanitize/tests/test_ltx2_device
```

| | result |
|---|---|
| RED, at `5a0ffe9e3` unmodified | `rc=1`, `AddressSanitizer: heap-use-after-free` at `cpu_layernorm.cpp:33 in LoadF32At`, address `0x50a000038c40` — the same address CI run `31885935312` reports |
| GREEN, with §3 applied | `rc=0`, `18 passed / 0 failed`, `assertions: 546 passed / 0 failed`, `Status: SUCCESS!` |

### The case is not vacuous after the fix

A test that stops aborting can stop asserting. Mutated the guarantee the case
exists for — `ltx2_device.cpp:832`, `keyframes_embedding->dtype == out.x->t()
.dtype` inverted to `!=`, so the production refusal fires on the *matching* bf16
view the positive control restores:

```
compile_rc=0
git diff --stat: src/vllm/model_executor/models/ltx2_device.cpp | 2 +-
run_rc=1
TEST CASE:  ltx2 device: an f32 keyframes embedding under a bf16 stream is REFUSED
[doctest] test cases:  18 | 16 passed | 2 failed | 0 skipped
[doctest] assertions: 528 | 526 passed | 2 failed
[doctest] Status: FAILURE!
```

The mutation compiled, applied to exactly one line, and the named case failed.
The tree was restored byte-for-byte afterwards (`git diff --stat` shows only the
test file).

## 5. Risks

**Low.** One test-local lifetime change; no production file is touched, so no
shipped behaviour can move. The residual risk is that another sanitizer red
hides behind this abort — ASan stops the process at the first report, so the
rest of the binary's cases never ran in the RED. Mitigated by running the full
`build-sanitize` ctest suite, not just this target.

## 6. Rejected alternatives

- **Save the pre-overwrite bf16 view and restore it.** One line shorter. It
  makes the positive control assert over the same view the case already proved
  was accepted at line 700, so a staging path that produced a stale or aliased
  view on a second call would still pass. The fresh restage is the stronger
  control and is what the author wrote.
- **Join the threadpool in `~Ltx2DitDeviceWeights`.** This is what #904's title
  asks for. It would make this case pass while leaving the dangling view in
  place, and it would add a synchronisation cost to every teardown for a defect
  that is not there. §2 gives the evidence.
- **Make `vt::Tensor` own or reference-count its storage.** A borrowed view is
  the deliberate design across the whole of `vt`; changing it for this is a
  project-wide decision that no red requires. Recorded under `## Owed`.

## 7. Cost

One test file, 8 lines added and 2 removed, including the comment that says why
`restaged` has to be named.

## 8. Now

`main` is red on `sanitize-cpu` on both arms because of this case. With §3
applied the lane's remaining known reds are the ones tracked elsewhere; this row
claims only its own.

## Owed

- A guard against a borrowed `vt::Tensor` outliving the object that owns its
  storage. Nothing in the tree refuses it, and it took ASan on a lane that has
  been cancelled on every `main` run for weeks to find this one instance. Not
  filed as a separate issue by this row, because scoping it is a `vt` design
  question rather than a defect report, and the row that takes it will need to
  decide between a type change, a checker, and a convention.

#904's stated cause is not owed anything — it is corrected in §2 and in the
pull request body, so the correction carries its evidence and its date in Git
rather than in an edited issue.
