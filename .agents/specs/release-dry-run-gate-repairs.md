# Merged-SHA release dry-run gate repairs

Identity: `ENG-RELEASE-WINDOWS`

Issues:
[#499](https://github.com/mudler/vllm.cpp/issues/499) and
[#500](https://github.com/mudler/vllm.cpp/issues/500), with post-merge follow-up
[#512](https://github.com/mudler/vllm.cpp/issues/512) and
[#514](https://github.com/mudler/vllm.cpp/issues/514), hosted-contract follow-up
[#525](https://github.com/mudler/vllm.cpp/issues/525), native socket-runtime
follow-up [#537](https://github.com/mudler/vllm.cpp/issues/537), and strict
Vulkan-test follow-up [#540](https://github.com/mudler/vllm.cpp/issues/540)

Parent specifications:
[release-binary-matrix.md](release-binary-matrix.md) and
[windows-binary-release.md](windows-binary-release.md)

Related native compiler contract:
[windows-msvc-strict-build.md](windows-msvc-strict-build.md)

Status: `ACTIVE`. The developer approved this bounded design on 2026-08-12. It
starts from exact main commit `e1087a8812c9b7d96fca5a813981f378fcace638`
after PR #446 merged. The #537 implementation, a fresh review, the operator
gate, and a new non-publishing ten-tuple dry run remain required.

## Scope

Repair the two independent failures exposed by the first exact-merged-SHA
release dry run after PR #446:

1. make the version compiled into every release `vllm-server` equal the exact
   release identity supplied by the release plan, including `-pre.1`, while
   preserving the existing backend suffix; and
2. make `tests/vt/test_cpu_isa_x86.cpp` own the standard stream definition its
   doctest `std::string_view` diagnostics instantiate under MSVC.

Keep CMake's numeric `project(vllm_cpp VERSION 0.0.3)` identity, archive names,
manifests, `VERSION` records, C ABI version, backend suffix policy, release
matrix, `/W4 /WX`, and the strict extracted-archive version validator
unchanged. Do not change production ISA selection, suppress a warning, alter a
release tuple, publish a tag or release, or fold an independent dry-run failure
into these issues.

## Hosted baseline and root causes

Manual workflow run
[`31607273683`](https://github.com/mudler/vllm.cpp/actions/runs/31607273683)
executed against exact merged SHA
`03ed25a5745ccbbfba7ce9e0f5a1c4a5251b7c30` with release identity
`0.0.3-pre.1`.

Issue #499 affected six jobs after their Linux or macOS build and packaging
work had completed:

| Tuple/job | Job ID | Observed gate |
|---|---:|---|
| Linux arm64 CPU | `94152936141` | extracted version mismatch |
| macOS arm64 Metal | `94152936210` | extracted version mismatch |
| Linux x86_64 Vulkan | `94152936237` | extracted version mismatch |
| Linux x86_64 CPU | `94152936310` | extracted version mismatch |
| Linux x86_64 musl CPU | `94152936346` | extracted version mismatch |
| macOS arm64 MLX | `94152936489` | extracted version mismatch |

`CMakeLists.txt:12` declares only numeric project version `0.0.3` and
`CMakeLists.txt:689` configures `include/vllm/version.h.in`. That template
exports only the numeric `PROJECT_VERSION_{MAJOR,MINOR,PATCH}` components, and
`src/vllm/version.cpp:5-12` reconstructs `0.0.3` before applying the current
backend suffix. The release builders require `VERSION`, but their CMake
configure calls do not pass it: `scripts/build-cpu-release.sh:26-40`,
`scripts/build-linux-accelerator-release.sh:32-48`,
`scripts/build-macos-release.sh:29-43`, and
`scripts/build-windows-release.ps1:180-198`. They use `VERSION` later for the
archive and metadata, so the same archive contains manifest/`VERSION` identity
`0.0.3-pre.1` and a server that reports `0.0.3`.

This is not a validator defect. `scripts/validate-release-archive.py:662-671`,
`:740-749`, and `:782-790` correctly require the extracted executable's output
to contain the manifest's exact version and the `VERSION` record's exact C ABI.
That comparison remains fail-closed.

Issue #500 affected native Windows CPU job `94152936312` and Vulkan job
`94152936445`. Both failed while compiling
`tests/vt/test_cpu_isa_x86.cpp:112` through doctest's binary-expression
stringification. The operands are `std::string_view`; the translation unit
includes `<string>` but not `<ostream>`. Under this MSVC include closure,
`std::basic_ostream` is only forward-declared, so the library's
`operator<<(basic_ostream&, string_view)` instantiation produces incomplete
type, missing `iostate`, and cascading C2146/C2027/C2065 diagnostics. The
production x86 ISA code compiled far enough to reach this test-only boundary.

## Design

### Exact compiled build version

Add a CMake cache string named `VLLM_CPP_BUILD_VERSION`. Its default is the
complete numeric `${PROJECT_VERSION}`, so developer and non-release builds keep
reporting `0.0.3` without a new required argument. Generate that string into
`include/vllm/version.h` and make `vllm::Version()` begin with the generated
build version instead of reconstructing the three numeric components.

Preserve the existing backend suffix exactly after that identity. For example,
the default CPU build reports `0.0.3`, a prerelease CPU artifact reports
`0.0.3-pre.1`, and a prerelease CUDA artifact reports
`0.0.3-pre.1+cuda`. Do not derive the prerelease from the tag inside CMake and
do not change the numeric `PROJECT_VERSION`, shared-library `VERSION`/`SOVERSION`,
or C ABI version.

All four release builder families pass their already-required `VERSION` value
to CMake as `-DVLLM_CPP_BUILD_VERSION=...`. The workflow remains responsible
for deriving the single release identity from `release/release-version.json`
and supplying it to each builder. CMake must reject an empty explicit build
version rather than silently reverting to the project version.

The generated value is build metadata, not a second release authority. The
release plan, manifest, archive name, `VERSION` record, executable output, and
post-publication audit must still agree, and the extracted-archive validator
continues enforcing that agreement.

### Explicit stream ownership in the MSVC test

Add the standard `<ostream>` definition directly to
`tests/vt/test_cpu_isa_x86.cpp`. This is a test translation-unit header-hygiene
repair: it makes doctest's diagnostic formatting valid without relying on a
transitive include. Do not change the compared `std::string_view` values,
doctest expressions, production x86 ISA code, compile flags, warning policy, or
standard-library formatting behavior.

## RED-first tests and mutation evidence

Implementation starts by capturing failures against the pinned baseline:

1. Extend the version test contract so a default configuration requires exact
   `0.0.3` plus the existing backend suffix, while a separate configure with
   `-DVLLM_CPP_BUILD_VERSION=0.0.3-pre.1` requires exact `0.0.3-pre.1` plus that
   same suffix. The prerelease arm must fail on the current numeric-only
   `Version()` implementation.
2. Extend `tests/scripts/test_release_pipeline.py` over the real four builder
   scripts. It must require each configure invocation to forward its existing
   `VERSION` input through `VLLM_CPP_BUILD_VERSION`; deleting the argument,
   substituting a literal, using `PROJECT_VERSION`, or omitting one builder
   must fail.
3. Preserve or extend the archive-validation mutation proving that a server
   output of `0.0.3` is rejected when manifest and `VERSION` declare
   `0.0.3-pre.1`. No skip-version path may be introduced into a release job.
4. Extend the Windows portability suite over the real
   `tests/vt/test_cpu_isa_x86.cpp` closure. It must require the explicit
   standard stream definition used by the `std::string_view` doctest
   expressions; removing it must fail the structural test and the native MSVC
   compile. Comments, literals, or unrelated transitive includes are not valid
   substitutes.

Every mutation runs in a scratch copy and restores the candidate tree
byte-for-byte. A structural Linux result does not substitute for executing the
native MSVC compiler.

## Gates

Focused local gates are:

1. the default and prerelease-configured `test_version` cases, including the
   current backend-suffix arm;
2. `python3 -m unittest tests.scripts.test_release_pipeline -v` and the focused
   release archive/metadata validator tests;
3. `python3 -m unittest tests.scripts.test_check_windows_portability -v` plus
   the direct portability checker; and
4. clean CPU configure/build and `test_cpu_isa_x86` execution on the available
   local host, followed by full unstaged, staged, and post-commit
   `scripts/agent-preflight.sh`.

Hosted acceptance requires native `windows-2022` CPU and Vulkan Release builds
to compile `test_cpu_isa_x86` under unchanged `/W4 /WX`, then execute their
existing focused runtime/ISA and archive gates. The six affected Linux,
macOS, and musl jobs must extract the archive and accept the exact
`0.0.3-pre.1` server identity. CUDA jobs must preserve their existing
`+cuda` suffix after the same prerelease identity.

After merge, rerun the complete ten-tuple non-publishing workflow at the exact
merged SHA. Every required tuple, aggregate handoff, and verify job must be
green, and each archive/manifest/`VERSION`/executable identity must agree.
Manual dispatch must continue skipping attest and publish. Only that immutable
dry run can authorize the already-planned developer-controlled
`v0.0.3-pre.1` tag flow; the tag run must rebuild, attest, publish exactly 32
assets, and pass the existing authenticated post-publication audit before any
binary-release claim advances.

## Risks and stop conditions

- A build-version string can accidentally become an independent release
  authority or be lost through one platform's quoting. The single declaration
  and all four builder mutations must keep the release graph closed.
- Replacing numeric component macros outright could break an internal consumer.
  Preserve them unless their removal is separately proved and reviewed; only
  `Version()` needs the full generated identity.
- Backend suffixes can be dropped or duplicated when the base string changes.
  Exact CPU and CUDA expectations are required.
- A transitive stream include may make a local compiler green while MSVC stays
  red. The test TU must own `<ostream>`, and native compilation remains binding.
- Stop with `NEEDS_DECISION` if #500 requires any production ISA, behavior,
  warning-level, or warning-suppression change.
- Stop with `NEEDS_CONTEXT` if a rerun exposes a direct diagnostic outside
  these two root causes. File and specify an independent issue rather than
  weakening a gate or silently expanding this repair.
- Do not tag or publish if any of the ten dry-run tuples, aggregate handoff,
  verify job, exact version checks, or Windows native gates is not green on the
  same merged SHA. A partial matrix is not release evidence.

## Written-spec self-review

- Scope is limited to exact compiled release identity and one test-TU stream
  definition; release topology and production ISA behavior do not change.
- The CMake default, release override, backend suffix, four builder inputs,
  strict validator, and native compiler boundary are explicit.
- RED tests and mutations fail for the two observed defects before code changes.
- Local, hosted, dry-run, tag-run, and post-publication evidence are separated;
  none is inferred from another.

## Outcome

Implemented both bounded dry-run repairs without changing release topology,
production ISA behavior, warning policy, or validator strictness.

- `VLLM_CPP_BUILD_VERSION` is now a non-empty CMake cache string defaulting to
  `PROJECT_VERSION`. The generated version header retains the numeric component
  macros and also carries that exact identity; `Version()` appends the existing
  `+cuda` qualifier to it unchanged.
- The CPU, Linux accelerator, macOS, and Windows release builders each forward
  their already-required `VERSION` value to CMake exactly once.
- `tests/vt/test_cpu_isa_x86.cpp` now owns the standard `<ostream>` definition
  required by its doctest diagnostics.

RED evidence on the pinned candidate preceded implementation. The builder
contract reported zero `VLLM_CPP_BUILD_VERSION` arguments for all four builder
families, the portability contract reported no active `<ostream>` include, and
a separately configured `0.0.3-pre.1` `test_version` failed because the binary
reported `0.0.3`. The retained strict-validator mutation rejects a numeric-only
server output when the manifest and `VERSION` declare `0.0.3-pre.1`.

Focused green evidence:

- release pipeline suite: 41 tests passed;
- Windows portability suite: 71 tests passed, followed by the direct checker;
- archive and platform metadata suites: 43 tests passed;
- an explicit empty build-version configure was rejected; and
- clean CPU builds passed the default and prerelease version tests. The clean
  prerelease build also passed all 8,242 x86 ISA assertions. Server smoke output
  was exactly `vllm.cpp 0.0.3 c-abi=17` by default and
  `vllm.cpp 0.0.3-pre.1 c-abi=17` with the release override.

The two hosted CUDA failures from dry run `31607273683` provide additional
root-cause evidence: jobs `94152936151` and `94152936356` both completed their
build and package phases and failed only because the extracted binary reported
the numeric project version. They did not expose a third repair within this
row's scope.

Native Windows compilation, the complete ten-tuple non-publishing workflow,
and the tag-run publication/audit remain post-merge acceptance gates. No tag or
release is authorized by the local evidence alone.

### Follow-up outcome and repair contract: issue #512

PR #508 merged as `2bc4be070a3883f0f7115682469a289f42d86d1a`.
Exact-SHA dry run
[`31625581156`](https://github.com/mudler/vllm.cpp/actions/runs/31625581156)
proved the prior fixes far enough for Windows CPU to compile
`test_cpu_isa_x86.cpp` successfully. Job `94211117810` then failed before its
first no-argument test executed:

```text
Cannot bind argument to parameter 'Arguments' because it is an empty array.
```

`scripts/build-windows-release.ps1:20-26` declares `Invoke-Checked` with a
mandatory `string[] Arguments` parameter. The script deliberately passes
`@()` when executing tests that take no arguments at lines 226-231 and 264-266.
PowerShell parameter binding rejects that explicit empty collection before the
helper invokes the executable. The build, strict MSVC warning gate, and
`test_cpu_isa_x86` link all succeeded; this is not a recurrence of #500.

The #512 repair is limited to making `Invoke-Checked` accept an explicitly empty
argument array while preserving argument splatting and the non-zero exit-status
failure contract. It must not remove `Invoke-Checked`, bypass any test, add a
dummy argument, weaken the Windows release gate, change a release tuple, or
publish a tag.

RED-first evidence must execute the real PowerShell helper contract with both
zero and nonzero argument counts. Before the fix, the zero-argument arm must
fail with the observed parameter-binding error. After the fix, it must prove
that the target runs exactly once with zero arguments, that nonempty arguments
arrive unchanged, and that a nonzero child exit remains rejected. Mutating the
helper back to a mandatory non-empty array must make the focused contract red.

Focused acceptance is the PowerShell contract test plus
`tests.scripts.test_release_windows_metadata`,
`tests.scripts.test_release_pipeline`, the direct Windows portability and
release-binary checkers, and the full repository preflight. Hosted acceptance
requires both Windows CPU and Vulkan jobs in a new exact-merged-SHA ten-tuple
dry run to execute, package, and validate their archives. All other required
tuples, aggregate `build`, and `verify` must also pass on that same SHA before
`v0.0.3-pre.1` is tagged. The tag run must then pass all 15 required jobs and
the authenticated post-publication audit over exactly 32 assets.

Issue #514 is a separate Windows Vulkan compile defect from the same dry run.
Job `94211117906` compiled and linked `test_cpu_isa_x86`, then MSVC rejected
`setenv` at `tests/vt/test_backend_cross_device.cpp:1004,1048` and `unsetenv`
at line 1050 with C3861. Those POSIX-only calls set, then restore,
`VT_FUSED_TIER` around the cross-device fused-chain test. Linux accepts them;
MSVC exposes `_putenv_s` instead.

The #514 repair is limited to a test-local cross-platform environment seam.
On Windows it must use checked `_putenv_s`, with an empty value removing the
variable. On POSIX it must preserve checked `setenv(..., 1)` and `unsetenv`.
The fused-chain case must still execute tiers 0 and 1, assert the selected tier,
and restore the caller's prior environment state. Do not disable the case,
change fused-chain production behavior, add a global compatibility macro, or
weaken `/W4` or any release gate.

RED-first structural coverage must reject the three live unguarded POSIX calls
and require both platform arms in the real translation unit. Mutation must
remove or bypass the Windows arm and make the focused suite fail. Focused green
is the relevant Windows portability suite and direct checker, a clean local
CPU compile/execution of `test_backend_cross_device`, and the full preflight.
Hosted acceptance remains the same exact-merged-SHA dry run: Windows Vulkan
must compile, execute, package, and validate before any tag is authorized.

The #514 implementation keeps the environment change local to the cross-device
test. A single `SetTestEnvironment` helper uses checked `_putenv_s` on Windows,
passing an empty value for removal, and retains checked `setenv(..., 1)` and
`unsetenv` on POSIX. The fused-chain case still selects tiers 0 and 1, asserts
the active tier, and restores either the saved value or the prior absence.

RED-first structural coverage failed on the pinned candidate because the helper
and Windows arm were absent. After the repair, the 72-case Windows portability
suite and direct checker passed. A scratch mutation deleting the `_putenv_s`
call failed the focused contract, and the candidate files were restored
byte-for-byte. A clean Release CPU configuration compiled and executed
`test_backend_cross_device`: 19 cases and 6 assertions passed. Native MSVC and
the exact-merged-SHA ten-tuple dry run remain hosted acceptance gates.

Fresh review of immutable implementation `e0b17eb9` found that the compiled
version test derived its default expectation from the same cache value under
test. Mutating the cache default from `${PROJECT_VERSION}` to `9.9.9` therefore
changed both subject and oracle and left the focused suite green. The review
loop adds an independent release-contract assertion over the real CMake cache
declaration. With the `9.9.9` mutation applied, that assertion failed exactly
with `9.9.9 != ${PROJECT_VERSION}`; after restoration it and the complete
42-test release pipeline suite passed. Production CMake and runtime code remain
unchanged by this follow-up.

Issue #512 implementation makes the existing mandatory `Arguments` parameter
explicitly accept an empty collection; argument splatting and the checked
nonzero exit path are unchanged. The Windows `-ContractTest` path now invokes a
temporary recording target once with zero arguments, invokes it with three
literal nonempty arguments and checks each value, and requires an exit-23 child
to be rejected. Before the production edit, the new focused contract failed on
the missing `AllowEmptyCollection` declaration; the hosted exact-SHA failure
above is the matching real-PowerShell RED execution. The local host has no
PowerShell runtime, so execution of the new live process contract remains a
required native Windows PR gate rather than being inferred from local Python.
After the fix, Windows metadata passed 8/8, release pipeline passed 42/42, and
both direct Windows portability and release-binary checkers passed. Removing
`AllowEmptyCollection` made the focused contract red again, and both changed
files were restored to their exact pre-mutation SHA-256 values.

Fresh review of combined candidate `0a70da1d` found two test-strength gaps.
Adding an unconditional `return` as the first statement of
`Invoke-CheckedContractTests` left all eight Windows metadata tests green even
though the hosted `-ContractTest` path would silently skip its runtime proofs.
Changing both fused-tier restoration calls to target
`VT_FUSED_TIER_WRONG` likewise left the focused portability test and direct
checker green. The test-only repair now binds the three exact `Invoke-Checked`
calls and their ordered literal outcomes, rejects an executable early control
exit in the contract-test function, and pins both restoration branches to the
exact `VT_FUSED_TIER` key with `saved.c_str()` or `nullptr` respectively.

After the repair, the early-return mutation and separate wrong-key mutations
in the prior-present and prior-absent restoration branches each failed their
focused test. The existing `AllowEmptyCollection`-removal and `_putenv_s`-
removal mutations also remained red. Production files were restored to their
pre-mutation SHA-256 values. The combined Windows metadata, release pipeline,
and Windows portability suites passed 122/122, followed by both direct Windows
portability and release-binary checkers. Native PowerShell execution and MSVC
compilation remain the required external Windows PR gates. The full unstaged
repository preflight passed every gate except one transient tempfile failure in
`test_check_release_binary_contract`; an immediate isolated retry of that exact
suite passed all 30 tests.

### Hosted contract follow-up: issue #525

PR #524 candidate `a4d61bbddbc1a0aa744aea72c1cff3c4ed165a72`
executed the new #512 contract on both hosted Windows lanes. CPU job
`94242642198` and Vulkan job `94242642222` both accepted the explicit empty
argument array, then failed at the exact non-empty record comparison with
`nonempty arguments did not arrive unchanged`. Both stopped in
`build-windows-release.ps1 -ContractTest` before configuration. The shared
diagnostic isolates the failure to the contract harness's temporary `.cmd`
argument recorder; it does not reopen the original empty-array binding defect
or establish a product build failure.

The #525 repair replaces only the batch recorder with a native PowerShell
target whose parameter declaration accepts all remaining arguments and writes
a structured exact record. The real `Invoke-Checked` function must still invoke
that target for both the explicit empty array and the three literals `alpha`,
`two words`, and `--flag=value`. The exit-23 target and rejection assertion
remain live. Do not weaken an equality, normalize values, remove a behavior,
or infer a hosted pass from structural Linux coverage.

RED-first coverage must reject the current `.cmd` recorder and require the
PowerShell recorder's remaining-arguments binding plus a structured exact
empty/non-empty record. Mutations that remove the remaining-arguments binding,
drop `two words`, bypass the real helper, or accept the nonzero child must make
the focused contract red. Focused green remains Windows metadata, release
pipeline, both direct Windows checkers, and full preflight. Hosted acceptance
requires both native Windows PR jobs to execute the complete contract and
continue into their MSVC build/runtime/archive gates.

After #525 is integrated, current `origin/main` must be merged into the task
branch before the operator gate and plain push so the PR-size checker receives
an ancestor base. The HTTP 503 downloading glslang in job `94242642545` is an
external retry condition, not authorization to change the freshness gate. No
tag is authorized until PR #524 merges and a new exact-merged-SHA dry run has
all ten tuples plus aggregate handoff and verify green.

The #525 implementation replaces the batch argument recorder with a temporary
PowerShell script. Its `ValueFromRemainingArguments` string-array parameter
defaults to an empty array, and it serializes the exact count and argument array
as compact JSON. The unchanged real `Invoke-Checked` helper executes that script
once with `@()` and once with `alpha`, `two words`, and `--flag=value`; the
contract parses each record and compares its exact count and values. The
separate exit-23 batch target and rejection assertion remain unchanged.

RED-first coverage rejected the prior `record-arguments.cmd` implementation.
After the repair, mutations removing the remaining-arguments binding, dropping
`two words`, bypassing `Invoke-Checked`, or disabling the nonzero rejection each
failed the focused metadata contract, and the script was restored byte-for-byte.
The Windows metadata and release pipeline suites passed 50/50, followed by the
direct Windows portability and release-binary checkers and the full unstaged
repository preflight. The local host has no PowerShell runtime, so native
execution remains a required hosted Windows gate.

### Native socket-runtime follow-up: issue #537

PR #524 candidate `d47d8408ab4dc2639e47ddfc7c7a997fa45d9981`
executed the complete #525 PowerShell contract successfully on hosted Windows.
CPU job
[`94265239385`](https://github.com/mudler/vllm.cpp/actions/runs/31641616323/job/94265239385)
then built `test_openai_api_server.exe` under unchanged `/W4 /WX`, served the
final request in the first real-socket smoke test, and terminated with signed
status `-1073740791` (`0xC0000409`) before doctest printed an assertion or
summary.

The failure has two independently grounded causes. First, vendored
cpp-httplib v0.49.0 ends `process_and_close_socket` with an immediate
shutdown/close. Upstream commit `8e702d3837b2164765ca1d98cb6d180ae4711e70`
records that on Windows this can send an abortive RST when bytes are still in
flight, making a fully written response appear to the client as a failed read.
Its accepted repair half-closes the write side, drains at most 100 ms or 1 MiB,
then performs final shutdown/close. Second, each local real-socket test owns a
raw joinable `std::thread`, contains aborting `REQUIRE` assertions after that
thread starts, and stops/joins only on the normal path. When a Windows client
read is reset, doctest unwinds through the joinable thread and `std::terminate`
turns the useful assertion into the observed process-wide fast-fail. Upstream
commit `ae8356d86eabfd3ad4a969b55266fb3ecc2aa834` documents and repairs that exact
test-lifetime pattern with scoped teardown.

The #537 implementation is limited to both complete repairs. Backport the
upstream accepted-socket drain helper and its single production call site with
the upstream 100 ms and 1 MiB bounds unchanged. Add one test-local scoped
server-thread owner and route every real-socket OpenAI test through it so
server stop and thread join happen both normally and while an assertion
unwinds. Preserve every existing request, status, body, concurrency, capacity,
route-gating, and shutdown assertion. Do not skip the Windows runtime target,
weaken `REQUIRE`, catch and discard a test failure, lengthen a timeout, detach a
thread, change release topology, or vendor unrelated upstream cpp-httplib
changes.

RED-first evidence must demonstrate both defects against the pinned candidate.
A focused behavioral test must make the accepted-socket peer leave unread
request data and prove the old immediate close presents as a reset/failure on
Windows while the bounded drain preserves the completed response. A separate
test must intentionally fail after a server thread starts and prove scoped
teardown lets doctest report the assertion instead of terminating the process.
Structural coverage over the real vendored header and socket tests supplements,
but does not replace, those native behaviors. Mutations that restore the old
immediate accepted-socket close, remove either drain bound, or remove the scoped
stop/join path must make the focused gates red, with the tree restored
byte-for-byte afterwards.

Focused green requires the OpenAI API server test, the cpp-httplib regression
test, the Windows release metadata/pipeline contracts, both direct Windows
checkers, and full unstaged/staged/post-commit preflight. Hosted acceptance
requires both Windows CPU and Vulkan lanes to execute the full OpenAI API test,
all other PR jobs to pass at one immutable head, and a fresh reviewer to mutate
both guarantees. After merge, the exact merged SHA must pass all ten release
tuples plus aggregate build and verify before `v0.0.3-pre.1` is tagged. The tag
run must then pass all 15 required jobs and the authenticated audit over exactly
32 assets.

Stop with `NEEDS_DECISION` if the bounded upstream backport changes the public
HTTP API, broadens beyond accepted-socket close, or requires any release-gate
waiver. Stop with `NEEDS_CONTEXT` if Vulkan reports a different native failure;
file and specify that defect separately rather than folding it into #537.

### Strict Vulkan-test follow-up: issue #540

The same immutable PR #524 candidate reached a different boundary in hosted
Windows Vulkan job
[`94265239433`](https://github.com/mudler/vllm.cpp/actions/runs/31641616323/job/94265239433).
The PowerShell contract and prior Windows portability fixes passed; MSVC then
rejected `tests/vt/test_backend_cross_device.cpp:525-529` under unchanged
`/W4 /WX`. The unbound-flash-layout CPU-oracle block redeclares `cpu`, `cq`,
`cd`, `ck`, `cv`, and `cslots`, shadowing names at lines 462-466 in the
enclosing `ReshapeAndCache` test. MSVC C4456 diagnoses all six. Git history
grounds the collision: `2c86f79ec` added the enclosing oracle and `822b3a2e15`
later added the nested oracle with the same short names.

The #540 repair is test-only. Rename exactly those six inner declarations to
role-specific unbound-layout names and update only their uses in that nested
oracle. Preserve the data, types, tensor shapes and strides, CPU operation,
device loop, memcmp assertions, `/W4 /WX`, build targets, release topology, and
all production files. Do not suppress C4456, relax the warning gate, introduce
a compiler conditional, remove the nested oracle, or conflate this diagnostic
with #537.

RED-first evidence is the hosted MSVC diagnostic above plus a focused local
contract over the real `ReshapeAndCache` test that rejects each of the six
shadowing inner declarations. Each independent mutation restoring one old name
must fail the focused contract, and the source must be restored byte-for-byte.
Focused green requires that contract, the direct Windows portability checker,
a clean local CPU compile/execution of `test_backend_cross_device`, and full
unstaged/staged/post-commit preflight. Hosted acceptance requires both native
Windows lanes to compile their complete unchanged targets; Vulkan must continue
through runtime, package, and archive validation.

After both #537 and #540 pass fresh review and operator gates, PR #524 must be
plain-pushed at one exact SHA and all required PR checks must pass. The merged
SHA must then pass the complete ten-tuple non-publishing workflow before the
prerelease tag. Tag-run publication and the authenticated exactly-32-asset
audit remain mandatory. Stop with `NEEDS_DECISION` if a production edit or
warning-policy change is required; otherwise this is a bounded test hygiene
repair.

#### #537 implementation outcome

The implementation backports only cpp-httplib
`8e702d3837b2164765ca1d98cb6d180ae4711e70`: accepted plain HTTP sockets now
half-close writes, drain for at most 100 ms or 1 MiB, and then perform the
existing final shutdown and close. The helper remains internal to the vendored
header and has exactly one production call site. No public HTTP API, TLS close
path, timeout, or release topology changed.

All 12 real-socket cases in `test_openai_api_server.cpp` now use one test-local
scoped owner that stops the server and joins its thread on normal return and
assertion unwind. A deliberately failing doctest fixture must exit with
doctest's normal status 1, print the named assertion, and emit its failure
summary; a subprocess harness rejects termination or any other exit status.
The accepted-socket regression leaves 64 KiB of unread request data after a
complete close-delimited request and requires the complete response followed by
an orderly EOF.

The focused transport tests, the full OpenAI API server test, Windows release
metadata, both direct Windows checkers, and the 74-test portability suite pass
locally. Mutations restoring immediate close, changing either drain bound,
removing scoped stop, and removing scoped join each made the focused gates red;
the last two respectively timed out and reproduced `SIGABRT`. The mutated files
were restored byte-for-byte. Linux also passes the socket behavior, but the
native reset RED and final acceptance remain correctly pending the hosted
Windows CPU and Vulkan lanes; this local result is not substituted for them.

#### #537 hosted falsification and next diagnostic

Hosted CPU job
[`94287249909`](https://github.com/mudler/vllm.cpp/actions/runs/31648432555/job/94287249909)
and Vulkan job
[`94287249981`](https://github.com/mudler/vllm.cpp/actions/runs/31648432555/job/94287249981)
executed candidate `f52b547d44439e7cfb005f5697142b755ff65586` with both
repairs above and still terminated with `0xC0000409`. The log and doctest's
source-order execution place the failure in teardown of
`api_server: socket smoke — real HTTP requests over an ephemeral port`: its
final chat response completes, but doctest never prints that test's duration or
summary. Therefore the accepted-socket reset plus unscoped server thread was a
real defect, but it was not the complete root cause of this native fast-fail.
That earlier causal claim is rejected; its implementation remains required by
its focused regressions.

The next hosted probe must preserve the full unchanged test and release gate,
add phase evidence around destruction of the `httplib::Client`, scoped server
thread, and `ServerHarness`, and install a diagnostic `std::terminate` marker in
the test process. It must run the isolated socket-smoke case first with doctest
success/duration output and then retain the normal full-suite invocation if the
probe survives. The probe may only add diagnostics; it must not skip an
assertion, detach a thread, catch a failure, alter production lifetime, or be
accepted as the fix. Use the first missing teardown marker plus the terminate
marker to identify one owner, then remove the probe and write a RED regression
for that exact lifetime defect before implementing a repair. If the isolated
case passes but the full process fails, bisect test-case prefixes in fresh
processes to prove the cross-test state dependency rather than guessing.

Candidate `f06c77fe4af502a9934e525112008e1a02bdc1ff` executed that probe in
[run `31688115193`](https://github.com/mudler/vllm.cpp/actions/runs/31688115193).
CPU job `94408881944` and Vulkan job `94408881974` produced the same boundary:
the isolated socket case passed with all six destruction markers, while the
full 54-case process produced those same six markers and then fast-failed with
exact signed status `-1073740791` (`0xC0000409`) before the doctest summary.
Neither job emitted the diagnostic `std::terminate` marker. This rules out the
three instrumented owners as the final failing boundary and establishes a
source-order cross-test dependency as the next hypothesis.

The developer approved one diagnostic-only adaptive prefix bisect. It lists the
54 doctest cases in file order, executes every probed `--first=1 --last=N`
prefix in a fresh process, requires the full prefix to reproduce only the exact
native fast-fail, and requires the first case as a known-good short prefix. It
then binary-searches the smallest failing `N`, confirms `N-1` succeeds and `N`
fast-fails in new processes, and runs only case `N` with matching first/last
bounds to distinguish an isolated defect from cumulative contamination. The
probe emits one stable diagnostic containing `N`, the listed test name, all
three confirmation statuses, and the dependency classification. The existing
unfiltered full-suite invocation remains unchanged and still runs afterwards.
Any other probe status stops the release job rather than being classified.

Hosted run `31728014706` resolved that boundary to source-order case 47/54,
`api_server: an explicit-cpu device-selected engine serves /v1/completions`.
Both its prefix and its isolated invocation terminate with the exact native
fast-fail status, while the confirmed predecessor succeeds. The defect is
therefore isolated to that test or the production path it exercises, not a
cross-test lifetime dependency.

The next diagnostic is limited to that isolated process. Preserve and print
its already-captured output, and add flushed phase witnesses around construction
of `LoadedEngine`, construction of the async serving stack, completion dispatch,
response validation, and scope teardown. Do not change timing, add sleeps,
weaken assertions, or accept this instrumentation as the repair. The hosted
Windows CPU/Vulkan result must identify the last completed phase. Then remove
the diagnostic and add the smallest RED regression for the actual owner before
changing production code. If the phase evidence does not distinguish an owner,
stop and extend the spec rather than guessing.

Hosted CPU run `31732971268` printed only
`OPENAI_EXPLICIT_CPU_PHASE: before-loaded-engine` in both the isolated probe and
the unchanged full-suite invocation, then terminated with `0xC0000409`. It did
not print `after-loaded-engine`. This excludes the async serving stack,
completion dispatch, response validation, and teardown, but it does not yet
distinguish the three expressions in the construction statement: synthetic
weight creation, tokenizer-fixture creation, and the `LoadedEngine` constructor
itself. C++ does not impose a useful ordering between those argument
evaluations, so their absence cannot identify which one terminated.

The final diagnostic split keeps the same test and values but materializes the
synthetic weights and tokenizer into named locals, with flushed witnesses before
and after each factory and before and after `LoadedEngine` construction. It
must move those locals into the constructor so ownership matches the original
by-value call. No production source, timing, assertion, environment, or engine
parameter may change. Hosted CPU and Vulkan must agree on the last completed
factory/construction phase. If both factories complete and construction still
fast-fails, stop and spec constructor-internal attribution rather than guessing;
otherwise remove every prefix/phase diagnostic and write the smallest RED
regression for the failing factory before repairing it.

### Current-main portability regression: issue #645

Current `main` at `cefacd2d00cb9b4776331cd213116773cd97f811` added LTX2
sources containing the non-standard `M_PI` macro. The existing real-tree
Windows portability regression is RED and names exactly these three files:

- `src/vllm/model_executor/models/ltx2.cpp`
- `src/vllm/model_executor/models/ltx2_video_vae.cpp`
- `src/vllm/model_executor/models/ltx2_audio_vae.cpp`

The repair is limited to replacing those uses with a standard C++ constant
whose type and value preserve the current expressions. Do not define feature
macros, add a checker exemption, weaken the real-tree scan, or alter LTX2
algorithms. The existing failing real-tree test is the RED regression; focused
green requires that test, the complete Windows portability suite, the combined
release script suite, and a clean CPU build covering the affected translation
units. Hosted MSVC CPU and Vulkan builds remain binding.

Fresh review must mutate at least one repaired expression back to `M_PI` and
must perturb the replacement constant enough to prove an LTX2 numerical test
detects it. Stop with `NEEDS_DECISION` if preserving the current value requires
an algorithmic or tolerance change rather than a constant substitution.

#### #645 implementation outcome

The regression was the non-standard macro itself: `ltx2.cpp` consumed `M_PI`,
while both VAE files carried fallback definitions even though the video VAE did
not consume the macro. The C++20 production expressions now use
`std::numbers::pi_v<double>`, which preserves the previous double type and
rounded value, and the unused video fallback is gone. No algorithm or tolerance
changed.

Before the repair, the real-tree portability regression failed with exactly the
three specified files. Afterwards that regression and all 76 Windows
portability tests pass; the combined Windows metadata, release-pipeline, and
portability suite passes 130 tests. A CPU Release build compiled all three
affected translation units and linked `test_ltx2` and `test_ltx2_vae`; their
upstream-golden numerical gates pass 30/30 cases with 1627 assertions and 36/36
cases with 3039 assertions, respectively.

### Current-main VideoEngine portability regression: issue #648

The configured direct Windows checker on current `main` reports three sites in
`src/vllm/multimodal/video_engine.cpp`: the unconditional `<sys/stat.h>` include
and the `::stat` calls in `IsDir` and `Exists`. These landed with the generalized
video seam at `cefacd2d00cb9b4776331cd213116773cd97f811` and cannot reach the
native MSVC release build.

Replace that POSIX dependency with C++20 `std::filesystem` queries using the
non-throwing `std::error_code` overloads. `IsDir` remains true only for a
directory; `Exists` remains true for any existing filesystem entry; missing or
uninspectable paths remain false. Do not add a Windows-only branch, guard the
POSIX include, exempt the file from the checker, or change family resolution.

RED evidence is the configured direct checker naming all three sites. Focused
behavior must cover an existing directory, a regular file, and a missing path
through the public VideoEngine resolution surface; add only the smallest test
needed if existing coverage cannot prove each classification. Green requires
the direct checker, the complete portability and combined release suites, a
clean build of the affected translation unit, and the VideoEngine tests.

Fresh review must restore the POSIX implementation to prove the direct checker
is red and mutate directory classification to an existence-only query to prove
behavior coverage rejects a regular file. Stop with `NEEDS_DECISION` if the
portable implementation changes an observable resolution result or error.

#### #648 implementation outcome

The regression came from the generalized VideoEngine seam using POSIX `stat`
for two private classification helpers even though C++20 filesystem support is
already part of the project baseline. The direct portability checker failed at
the unconditional `<sys/stat.h>` include and both `::stat` calls before the
repair. `IsDir` now uses the non-throwing `std::filesystem::is_directory`
overload and `Exists` uses the corresponding `exists` overload; both inspect an
`std::error_code`, so missing and uninspectable paths still return false rather
than throwing.

The focused public-surface case distinguishes a directory without a shard
index, a regular non-checkpoint file, and a missing path through
`ReadVideoCheckpointTensorNames`. This pins the existing three refusal reasons
and makes an existence-only directory predicate misclassify the regular file.
After the repair the direct checker passed, the VideoEngine target built and
its 12 cases / 260 assertions passed, and the combined Windows metadata,
release-pipeline, and portability suite passed all 130 tests. No family
resolution rule or public error changed.

Fresh mutation review of the adaptive probe found three false-green contract
gaps: the listing fixture did not distinguish file order from name order, the
unexpected-status branches at an intermediate midpoint and isolated probe were
not executed, and the emitted diagnostic field names were not pinned. The
repaired live PowerShell contract uses a deliberately name-order-inverted
source-order listing, injects status 7 independently at the midpoint and
isolated boundaries and requires both invocations to throw, and captures the
real host output for an exact diagnostic-schema comparison. Mutating the
listing call to name order, accepting either unexpected status, or renaming
`predecessor_status` now makes the contract red. The restored candidate passes
the 12-test Windows metadata suite, the combined 130-test metadata/pipeline/
portability suite, the live PowerShell contract, and all four direct Windows
release checkers.

### Cross-platform PowerShell contract follow-up: issue #599

Fresh review of diagnostic candidate `5a845c28a7a9afe5addf94771e59a71cecd31e81`
ran the required direct Windows portability checker on a Linux host with
PowerShell Core installed. `Invoke-CheckedContractTests` created `fail.cmd` and
invoked that path directly. Non-Windows PowerShell handed the batch file to
`gio` instead of a Windows command interpreter; `gio` reported the unusable
path but returned success, so the contract failed at its own guard with
`nonzero child exit was accepted`. The repository preflight did not substitute
for this direct checker and its green result is not evidence for this boundary.

The #599 repair is contract-test only. Replace the platform-specific failing
child fixture with a script directly executable by both Windows PowerShell and
PowerShell Core on non-Windows hosts. Preserve the real-process invocation,
exact exit status 23, zero- and non-empty-argument forwarding proofs, production
`Invoke-Checked` behavior, native release gate, and release topology. Do not
mock this boundary, special-case the checker host, skip the nonzero child, or
accept a desktop-opener status.

RED-first evidence is the direct checker failure above. Focused coverage must
pin the portable fixture and its exact exit 23, then run the actual
`-ContractTest` path under installed PowerShell Core. Mutating the fixture back
to `.cmd`, changing its exit status, or bypassing the real child invocation must
make the focused gate red. Focused green requires Windows release metadata and
pipeline tests, the complete Windows portability unit suite, the direct Windows
portability and release-binary checkers, and full unstaged/staged/post-commit
preflight. Hosted CPU and Vulkan execution remains required because local
PowerShell Core does not prove Windows process semantics.

Stop with `NEEDS_DECISION` if the repair would change production invocation or
the native release gate rather than only its contract fixture.

#### #599 implementation outcome

The failure was confined to the contract fixture: PowerShell Core 7.6.4 on
Linux dispatched the temporary `fail.cmd` through `gio`, whose success status
made the real `Invoke-Checked` helper accept the supposed failure child. The
fixture is now a temporary PowerShell script containing only `exit 23`; both
Windows PowerShell and cross-platform PowerShell execute that script directly.
Production `Invoke-Checked`, its argument splatting, the release gate, and the
workflow topology are unchanged.

Before the repair, the focused metadata contract failed on `fail.cmd` and the
live `-ContractTest` failed with `nonzero child exit was accepted`. After the
repair, the live PowerShell contract, all nine Windows metadata tests, all 42
release-pipeline tests, all 76 Windows-portability tests, and both direct
Windows checkers passed. Independently restoring the batch fixture, changing
the child to exit 24, and bypassing the real child invocation each made both
the focused structural and live-process tests fail; restoration returned the
script and test to SHA-256 values `bc98e85ec59b076e04c490b23d00c41b9a8fe57277d2ba051f4fc0d3f65c569e`
and `cb81cc1b9914306081b0861929e907cf6d50310b8a6dde7f4d27aeae40700df8`.
Native Windows CPU and Vulkan acceptance remains pending the hosted PR jobs.

Fresh review of `bb3a4ef409d372b7f22698b8b530b4cb7f953cb9` disproved the
outcome's claim that direct invocation of the portable `.ps1` fixture preserved
a real-process boundary: the parent and invoked fixture both recorded PID
`3526837`. Direct `.ps1` invocation creates another PowerShell scope in the
same host process, so the earlier process-boundary claim is rejected.

The follow-up resolves the running host through
`[System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName`, which
is available to Windows PowerShell 5.1 as well as PowerShell Core, and passes
that executable plus `-NoProfile -NonInteractive -File` through the unchanged
`Invoke-Checked` helper for every fixture. The failing fixture records its PID
before exact exit 23, and the live contract rejects equality with the parent
PID. Before the repair that assertion failed with `failure target did not
execute in a child process`; after it, the live contract, all 9 Windows metadata
tests, all 42 release-pipeline tests, all 76 Windows-portability tests, and the
four direct Windows release checkers pass. Mutating the failure launch back to
direct `.ps1`, changing exit 23 to exit 24, or bypassing `Invoke-Checked` makes
both structural and live-process coverage red. Native hosted Windows remains
the final authority for Windows process semantics.

Fresh review of `8dfddfed2b27f6830768acbe32803c1bb7399459` found that the
external live Python test still trusted the contract's in-script PID guard.
Deleting that guard left the live method green, so the claimed child-process
evidence was not independently checked outside the script. The strengthened
contract emits one stable diagnostic containing only the parent and failing
child integer PIDs. The live Python method parses both values, requires each to
be positive, and independently rejects equality; it also pins the exact runtime
equality guard so removing either layer is red.

Before the test change, deleting the runtime guard reproduced the false green.
The strengthened test was then RED because the diagnostic did not yet exist.
After adding the diagnostic only to `Invoke-CheckedContractTests`, the direct
PowerShell contract and live Python method pass with distinct PIDs. Removing
the runtime guard, launching the failing `.ps1` directly, changing exit 23 to
exit 24, and bypassing `Invoke-Checked` each made the focused live method red;
each mutation was restored before the next. All 9 Windows metadata tests, all
42 release-pipeline tests, all 76 Windows-portability tests, and the four direct
Windows release checkers pass. Production `Invoke-Checked`, the release gate,
and release topology remain unchanged; native hosted Windows is still binding.

#### #540 implementation outcome

Implementation evidence: the focused contract was red with all six old
declarations present (12 subtest failures), then green after the nested locals
were renamed to `unbound_cpu`, `unbound_queue`, `unbound_device`, `unbound_k`,
`unbound_v`, and `unbound_slots`. Restoring each old declaration independently
made its named contract checks red, with both source files restored to their
recorded hashes afterwards. The 73-test Windows portability suite, direct
Windows portability checker, and a clean CPU-only Release build and execution
of `test_backend_cross_device` (19 cases, 6 assertions) passed. Native MSVC
acceptance remains pending the hosted Windows CPU and Vulkan lanes.

Fresh review found that the negative checks recognized only six exact
declaration spellings. The equivalent comma declarator
`const Device unbound_device{...}, cd(DeviceType::kCPU, 0);` therefore passed
the old contract. The follow-up contract derives declarator names within the
exact unbound oracle while respecting comma, brace, and parenthesis nesting.
That bypass and independent mutations for all six forbidden names now fail;
the restored tree passes all 74 Windows portability tests and the direct
checker.
