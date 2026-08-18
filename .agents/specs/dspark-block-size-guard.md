# The DSpark block floor reaches no production caller (`SPEC-DSPARK-BLOCK-SIZE-GUARD`)

| Field | Value |
|---|---|
| Row | `SPEC-DSPARK-BLOCK-SIZE-GUARD` (engine-matrix, Speculative decoding) |
| Issue | [#1225](https://github.com/mudler/vllm.cpp/issues/1225) |
| Scope | Make the landed `k >= block` floor in `SpeculativeConfig::ResolveDspark` reachable from the loader, so a speculative length below a DSpark draft's block is refused instead of silently drafting a structurally wrong block. Three parts: (A) read the draft `config.json` for `n_predict` and the block key; (B) pass them at both `ResolveDspark` call sites instead of `std::nullopt`; (C) a red-first test that enters through `LoadedEngine::ResolveSpecConfig`. **Excluded:** the DSpark speculator itself and its block layout (landed under `SPEC-DSPARK`), draft architecture classification and `IsDsparkDraft` (owned by `SPEC-DSPARK-QWEN3-ROUTING`, [#1193](https://github.com/mudler/vllm.cpp/issues/1193), in flight), the DeepSeek-V4 DSpark runtime (hardware-blocked), and the GPU run gate that would exhibit the garbling (§6, owed). |
| Upstream chain | At the pin `555967922`: `vllm/config/speculative.py:945-961` (the Gemma4 `block_size` → `n_predict` normalization) → `:973-988` (the `n_predict` default and the divisibility rule) → `:990-994` (k required) → `:1003-1027` (the `k >= dspark_block_size` hard error and its wording). Beyond the pin: PR [vllm#52197](https://github.com/vllm-project/vllm/pull/52197) at `7075ddac`, whose two hunks are architecture routing only and leave every line above unchanged. |
| Our baseline | `include/vllm/config/speculative.h:156-188` (`ResolveDspark`, the floor at `:179-185`). Call sites `src/vllm/entrypoints/model_loader.cpp:881-883` and `:1675-1677`, both passing `std::nullopt` twice. The draft config is already read at `:471-474` inside `LoadDsparkDraft`. Tests: `tests/vllm/config/test_speculative_dspark.cpp:99-107`, which call `ResolveDspark` directly. |
| Port map | §3. |
| Tests to port | §5. |
| Gates | §6. |
| Dependencies | Landed: `SPEC-DSPARK` W1-W8 ([dspark-spec-decode.md](dspark-spec-decode.md), `ACTIVE`). In flight and adjacent: `SPEC-DSPARK-QWEN3-ROUTING` ([dspark-qwen3-routing.md](dspark-qwen3-routing.md)), which edits the same `ResolveSpecConfig` branch for classification. §7 R5 records the seam between them. |
| Work breakdown | §4. |
| Risks/decisions | §7. |
| Pin policy | Mirror the pin, plus one recorded divergence argued in §2 and repeated in the commit body. |
| Role / claim | fresh implementer, branch `row/DSPARK-BLOCK-SIZE-GUARD` |
| Base | `65d6cdaed3e20e9bc70b4f9374fccafefefa7bd0` (origin/main, 2026-08-18) |
| Parity pin | vLLM `555967922` (0.26.0.dev0) at `$VLLM_SOURCE` |

## 0. Verdict

Two findings, both verified in a clean worktree at the base above. The second
one changes what the fix has to do, so it is stated in full rather than folded
into the design.

**1. The floor is unreachable.** `ResolveDspark` implements upstream's hard
error at `include/vllm/config/speculative.h:179-185`, and both production call
sites pass `std::nullopt` for `n_predict` and for `dspark_block_size`:

```
src/vllm/entrypoints/model_loader.cpp:881-883   LoadedEngine::ResolveSpecConfig
src/vllm/entrypoints/model_loader.cpp:1675-1677 the draft load inside FromModelDir
```

Every other reference to those parameters is in
`tests/vllm/config/test_speculative_dspark.cpp`, which supplies the values by
hand. This is the unpassed-parameter shape of `.agents/reachability.md`: the
argument exists, the test drives it, and no user can arrive at it.

The consequence is silent. Our draft step is sized only by `k` —
`DsparkBlockLayout::num_speculative_steps` is `N (k)`
(`include/vllm/v1/worker/gpu/spec_decode/dspark/speculator.h:56`) — and nothing
under `src/vllm/v1/worker/gpu/spec_decode/dspark/` or in the loader reads the
draft's block key. No weight is shaped by the block, so a short `k` trips no
`VT_CHECK`. It drafts a structurally wrong block and the tokens keep flowing.

**2. A literal port of the floor would still not fire.** Upstream reads
`getattr(hf_config, "dspark_block_size", None)` (`:1011-1015`). A grep of the
whole pinned checkout finds that identifier in `speculative.py` and in no other
file, so no vLLM config class defines it and it can only arrive from a draft
`config.json`. Both published Qwen3 DSpark drafts were read live on 2026-08-18:

| Draft | `architectures` | `model_type` | block key | `n_predict` | `dspark_block_size` |
|---|---|---|---|---|---|
| `deepseek-ai/dspark_qwen3_4b_block7`, upstream's own e2e draft | `["Qwen3DSparkModel"]` | `qwen3` | `block_size: 7` | absent | absent |
| `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153be924f17ce4bf62726954eeaa4a73e854` | `["DSparkDraftModel"]` | `qwen3` | `block_size: 7` | absent | absent |

Upstream's `block_size` → `n_predict` normalization at `:945-961` is guarded by
`"Gemma4DSparkModel" in architectures`, so it does not reach either row. #52197
rewrites the second row's `architectures` to `Qwen3DSparkModel` and changes
nothing else.

Trace `k=6` against either draft through the pinned `__post_init__` and the
result is the same on both sides of #52197: `n_predict` is `None`, so `:973-988`
does nothing; `dspark_block_size` is `None`, so `:1003-1027` does nothing. **The
value is accepted.** Upstream's floor is, in practice, a DeepSeek-V4 guard, and
DeepSeek-V4 DSpark is the one lane we do not implement.

So the literal port is not a fix. It would land a branch keyed on a field no
supported checkpoint sets, which is the unselected-branch shape of
`.agents/reachability.md`, and it would leave the silent path exactly as it is.

## 1. Scope

In: the two call sites, the config read that feeds them, and the test that
enters through the loader.

Out: everything in the `Scope` row's exclusion list. In particular this row does
not touch `IsDsparkDraft`, `ResolveDsparkArchitecture`, or the method
classification in `ResolveSpecConfig`. Those belong to
`SPEC-DSPARK-QWEN3-ROUTING`, which is in flight against the same branch of the
same function.

## 2. The divergence, and why it is argued rather than smuggled

`AGENTS.md` says to mirror vLLM wherever vLLM defines the behavior, and it also
says that nothing lands dead. §0 finding 2 puts those two in direct tension: the
faithful port is dead on arrival for every checkpoint we ship, and the shipped
lane keeps a silent-wrong-output path.

**Decision: port the pin exactly, and add one fallback.** The floor is
`dspark_block_size` when the draft config carries it, and `block_size` when it
does not. Everything else — the threshold, the comparison, the error wording,
the `n_predict` default and divisibility rules — is upstream's.

The reasons, in the order they carry weight:

1. **The intent is upstream's own.** The comment at `speculative.py:1004-1010`
   says to require `num_speculative_tokens` to be at least the block size, and
   gives 5 or 7 as the examples. For a Qwen3 draft the block size is spelled
   `block_size`, and it is 7 on both published drafts. We are supplying the
   value upstream's own sentence names, not choosing a different rule.
2. **Upstream already treats `block_size` as the block depth.** The Gemma4
   normalization at `:945-961` maps exactly that key onto `n_predict` for a
   self-contained draft. The Qwen3 drafts are self-contained in the same sense.
3. **Silent wrong output is the worst failure class we have.** A token gate does
   not necessarily catch it, because the drafter still produces tokens and the
   target still verifies them.
4. **It is the minimum that changes nothing already working.** Mapping
   `block_size` onto the *floor* rather than onto `n_predict` leaves every
   currently-accepted configuration byte-identical: `k >= 7` behaves exactly as
   today, `k` stays required, and the divisibility rule is untouched. Only
   `k < block` changes, from silent acceptance to a refusal.

This is one tracked exception under `AGENTS.md` `## Changing the rules or a
checker`: argued in the commit body, visible in the diff, and rejectable by the
reviewer. It is not a waiver and it is not a registry entry.

The honest residual is that upstream accepts `k=6` here and we will refuse it.
That is a deliberate stricter-than-upstream refusal on a path upstream leaves
unguarded, and §6 G3 owes the oracle run that shows which side is right about
the output.

## 3. Port map

| Upstream, at the pin | Lands in | Shape |
|---|---|---|
| `speculative.py:973-979`, the `n_predict` read and default | `src/vllm/entrypoints/model_loader.cpp`, a new file-local `ReadDsparkDraftKeys` | Read `n_predict` from the draft `config.json` when present. |
| `speculative.py:945-961`, the Gemma4 normalization | the same helper | When `n_predict` is absent and `architectures` contains `Gemma4DSparkModel`, take `block_size` as `n_predict`. Guarded by the architecture name exactly as upstream guards it. |
| `speculative.py:1011-1015`, the floor read | the same helper | Read `dspark_block_size` when present. **Divergence (§2):** fall back to `block_size` when it is not. |
| `speculative.py:1003-1027`, the hard error | `include/vllm/config/speculative.h:179-185` | Already landed and already worded from upstream. Not re-implemented; only reached. |
| `speculative.py:990-994`, k required | `model_loader.cpp:877-880` | The existing early throw fires before `ResolveDspark` sees `n_predict`, which would make the threaded `n_predict` default unreachable. Narrow it to fire only when the draft carries no `n_predict` either, and let `ResolveDspark` raise otherwise. |

`ReadDsparkDraftKeys` resolves the draft directory with the existing
`ResolveDflashDraftDir` (`model_loader.cpp:279`) and returns both values empty
when no `config.json` is there, so `ResolveSpecConfig` keeps working on a path
that `LoadDsparkDraft` will later reject with its own message. It applies
`TranslateSpeculatorsDsparkConfig` first, exactly as `LoadDsparkDraft` does, so
both config layouts are read through the same normalized shape.

This is the second read of the draft config, alongside the one at `:471-474`.
`dspark-qwen3-routing.md` §3 already weighs a second read of a five-layer draft
config and records it as not a measured cost. Hoisting the read would mean
restructuring `FromModelDir`, which collides with the in-flight row for no
correctness gain.

## 4. Work breakdown

| Step | Content | Gateable on this host |
|---|---|---|
| W1 | The red test: a draft config carrying `dspark_block_size` and one carrying only `block_size`, each with `k` below it, through `LoadedEngine::ResolveSpecConfig`. Capture both reds. | yes, CPU |
| W2 | `ReadDsparkDraftKeys` plus the two call sites plus the narrowed early throw. Focused green. | yes, CPU |
| W3 | The reachability mutation of §6: restore `std::nullopt` at `:881-883` in a scratch copy and confirm the focused gate goes red. | yes, CPU |
| W4 | The GPU run gate of §6 G3. | needs a GPU lease |

W1-W3 need no GPU. W4 does, and it does not block W1-W3.

## 5. Tests to port

Upstream has no unit test for this resolution. It is covered end to end by
`tests/v1/e2e/spec_decode/test_spec_decode.py::test_dspark_correctness_and_acceptance_rate`,
which needs a GPU and two checkpoints. The existing
`tests/vllm/config/test_speculative_dspark.cpp:99-107` already pins the floor at
the function; it stays, because it localizes a failure, and it is not the proof.

New file `tests/vllm/entrypoints/test_dspark_block_size_guard.cpp`, entering at
`LoadedEngine::ResolveSpecConfig`:

| Case | Today | After |
|---|---|---|
| Draft with `dspark_block_size: 7`, `k = 6` | accepted | throws, naming 7 and 6 |
| Draft with `block_size: 7` and no `dspark_block_size`, `k = 6` | accepted | throws, naming 7 and 6 |
| Either draft, `k = 7` | accepted | accepted, `num_speculative_tokens == 7` |
| Either draft, `k = 14` | accepted | accepted |
| Gemma4 draft with `block_size: 7`, no `k` | throws "requires num_speculative_tokens" | accepted, `k` defaults to 7 (`:973-979`) |
| Draft path that does not exist, `k = 7` | accepted | accepted, unchanged |

The last row is the regression guard for §3's filesystem-independence note.

## 6. Gates

| Gate | Content | State |
|---|---|---|
| G1, focused | `test_dspark_block_size_guard` and `test_speculative_dspark` green, both reds captured first. | required, CPU |
| G2, reachability | Restore `std::nullopt, std::nullopt` at `:881-883` in a scratch copy. `test_dspark_block_size_guard` must go red. A green here is the finding, not a pass. | required, CPU |
| G3, run | On `Qwen3.8-27B` + `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153b`, decode at `k = 6` against the pinned oracle at the same `k`, greedy, same prompts. This is the gate that shows whether a short `k` garbles, and therefore whether §2's stricter-than-upstream refusal is right. | **owed**, needs a GPU lease |
| G4, spec-off | Decode with speculation off byte-identical before and after. The change touches only the dspark branch, so a difference is a defect in the change. | owed |

G3 is the one that settles §2. Until it runs, the divergence rests on upstream's
own comment and on the absence of any block-shaped weight in our draft path, not
on a measurement of our output. The row does not claim it.

## 6a. G1 and G2 RESULTS (2026-08-18)

CPU-only build, `-DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo`, at base
`65d6cdaed`. No GPU on this host, so G3 and G4 stay owed.

**G1 red, before the fix.** `test_dspark_block_size_guard` compiled cleanly
(`compile_rc=0`, so this is a real result and not a build failure reading as a
pass) and reported 8 cases with 4 passed and 4 failed, 17 assertions with 14 passed and 3
failed, `Status: FAILURE!`. The four failures were the intended ones:

```
TEST CASE:  the loader refuses k below the draft's dspark_block_size
  ERROR: CHECK_THROWS_AS( LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
         std::invalid_argument ) did NOT throw at all!
TEST CASE:  the loader refuses k below the draft's block_size
  ERROR: CHECK_THROWS_AS( ... ) did NOT throw at all!
TEST CASE:  the refusal names the block and the k that was asked for
  FATAL ERROR: expected a refusal for k=6 against a block-7 DSpark draft
TEST CASE:  a Gemma4 draft's block_size defaults k
  ERROR: test case THREW exception: speculative-config: method "dspark" requires
         num_speculative_tokens (a DSpark draft config carries no n_predict)
```

The first two are the hole itself: a `k` below the block resolved without
complaint on both the upstream-keyed and the published-checkpoint config shape.

**G1 green, after the fix.** 8 cases with 8 passed and 0 failed, 22 assertions
with 22 passed and 0 failed, `Status: SUCCESS!`. The assertion count rises from 17 to 22
because the three previously-throwing cases now reach the checks past the
refusal, which is the shape a real green has here rather than a muted one.

No regression in the neighbours, each rebuilt and rerun at the same base:

| Suite | Result |
|---|---|
| `test_speculative_dspark` | 9 cases, 9 passed; 30 assertions, 30 passed |
| `test_loaded_engine_dense` | 19 cases, 19 passed; 87 assertions, 87 passed |
| `test_speculative_unknown_keys` | 9 cases, 9 passed; 63 assertions, 63 passed |
| `test_qwen3_dspark_config` | 8 cases, 8 passed; 25 assertions, 25 passed |

**G2, the reachability mutation.** `keys.n_predict, keys.block_floor` restored to
`std::nullopt, std::nullopt` at the `ResolveSpecConfig` call site only. The
mutation compiled (`compile_rc=0`) and `git diff --stat` confirmed it applied, so
neither of the two ways a mutation reads as a false pass is in play. The focused
gate went back to 8 cases with 4 passed and 4 failed, `Status: FAILURE!`. The test
therefore enters through the production call site rather than measuring the
class. The file was restored and its sha256 checked equal to the pre-mutation
value (`181250a7c130db41...`), and the suite returned to 8/8.


## 7. Risks and decisions

**R1. The floor could be wrong about our implementation.** Upstream says a short
`k` garbles for DeepSeek-V4 DSpark. Nobody here has shown it garbles for a Qwen3
draft in *our* speculator. The refusal is therefore conservative: it turns a
possibly-wrong output into a named refusal. If G3 shows a short `k` is merely
worse and not wrong, the right follow-up is to relax the fallback to a warning,
not to delete the threading. Recorded so the next reader does not re-derive it.

**R2. The `block_size` fallback could mask a future upstream key.** If a later
pin adds `dspark_block_size` to the Qwen3 drafts, the explicit key still wins,
because the fallback only applies when it is absent. The precedence order is the
mitigation.

**R3. Reading the config in `ResolveSpecConfig` adds a filesystem touch to a
function that had none.** Mitigated by returning empty on a missing
`config.json` rather than throwing, and pinned by the last row of §5.

**R4. Narrowing the early throw changes an error path.** Today a Gemma4 draft
with `block_size` and no `k` is refused; upstream defaults `k`. After the change
`ResolveDspark` raises the equivalent error when there is genuinely no
`n_predict`, so the native Qwen3 case keeps today's message. This is a mirror
repair, and it is required for the threaded `n_predict` to be reachable at all.

**R5. Collision with `SPEC-DSPARK-QWEN3-ROUTING`.** That row edits the same
`ResolveSpecConfig` dspark branch to add classification. The two changes are
semantically independent — it owns `IsDsparkDraft` and the architecture
normalization, this row owns the `ResolveDspark` arguments — but they will
conflict textually. The mitigation is shape: this row's footprint in that branch
is one helper call and the two argument lists, with the helper defined elsewhere
in the file. Whichever lands second resolves by taking the other's classification
lines whole and re-applying its own argument lines. Both rows read the same
`config.json`, so a later cleanup can hoist one read; neither row should do it.

## 8. Owed

- [#1225](https://github.com/mudler/vllm.cpp/issues/1225) — this row. Open,
  owned here.
- G3 and G4 of §6, both needing a GPU lease. G3 is the measurement that decides
  whether §2's divergence should stay a refusal or become a warning.
- `docs/USAGE.md` owes the `RadixArk/Qwen3.8-27B-DSpark` weight pin (repo,
  revision `85ef153be924f17ce4bf62726954eeaa4a73e854`, sha256
  `9d26d5e637551c244d543c67c790bd0947f360e005c569e5851a185ffe692786`). It is
  owed by `SPEC-DSPARK-QWEN3-ROUTING` W4, which reads the same checkpoint, and
  is recorded here so it is not written twice.
- If `SPEC-DSPARK-QWEN3-ROUTING` lands a named refusal for the DeepSeek-V4
  shape, the `dspark_block_size` half of this row's read becomes unreachable in
  practice, and only the `block_size` fallback carries. That does not make the
  read wrong, and it is the reason §2 argues the fallback rather than relying on
  the upstream key.

## 9. Stop conditions

- Stop and report if closing the hole turns out to require touching
  `IsDsparkDraft` or the method classification. Those belong to the in-flight
  row and a silent overlap is worse than a round trip.
- Stop if the fix cannot be built or exercised without CUDA. It can: every
  gate in §6 except G3 and G4 is CPU.
- Do not claim G3. A GPU run gate is owed until it runs.

## Now

`ACTIVE`, not `DONE`. The spec is committed,
[#1225](https://github.com/mudler/vllm.cpp/issues/1225) is open, and W1-W3 have
landed: the floor is threaded at both call sites, the red was captured before the
fix, and the reachability mutation is recorded in §6a. W4 is owed and needs a GPU
lease, so the row cannot reach `DONE` here.

The divergence of §2 is decided and recorded rather than deferred, and G3 is the
measurement that can overturn it. Until G3 runs, the claim this row makes is that
a `k` below the draft's block is REFUSED, not that a `k` below the block would
have garbled: the second is upstream's statement and our structural reading of
the draft path, not our measurement.
