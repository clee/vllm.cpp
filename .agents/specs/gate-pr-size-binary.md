# GATE-PR-SIZE-BINARY — retire the fail-closed binary guard

**Row:** `GATE-PR-SIZE-BINARY`
**Issue:** [#615](https://github.com/mudler/vllm.cpp/issues/615)
**Base:** `origin/main` `7572b0f4e`
**Status:** ACTIVE, 2026-08-13

## 1. Scope

One behavioural change to `scripts/check-pr-size.py`: remove the error raised
for a changed path that git reports as binary.

**In scope.** The `change.lines is None` branch in `change_errors`, the header
sentence that advertises it, the `SITE_ASSET` comment that contradicts it, and
the two cases in `tests/scripts/test_check_pr_size.py` that pin it.

**Out of scope.** Explicit path classification, the checker-evidence contract,
the role checks, and the retired line budget. None of them changes. This is not
a size rule and it does not reopen one — the per-class budgets were retired on
2026-08-10 by developer decision and stay retired.

## 2. Anchors

Local, not upstream — this is a project governance checker with no vLLM
counterpart.

| What | Where |
|---|---|
| The guard | `scripts/check-pr-size.py:480-481` |
| Its advertisement | `scripts/check-pr-size.py:56-57` |
| The classifier that already protects us | `scripts/check-pr-size.py` `classify_path`, raises `ValueError` on any unclassified path |
| The `asset` class the guard contradicts | `scripts/check-pr-size.py:160-165` (`SITE_ASSET`) |
| The guard's landing commit | `450a1b696`, 2026-08-10 |
| The golden precedent it post-dates | `971d55063`, 2026-08-09 |
| Blocked work | [#431](https://github.com/mudler/vllm.cpp/pull/431) |

## 3. Design

`change_errors` currently short-circuits on binaries before any class-specific
rule runs:

```python
if change.lines is None:
    errors.append(f"binary change {change.path!r} is not reviewable as text")
    continue
```

Delete the branch. Everything downstream already tolerates `lines is None` —
the checker-evidence contract tests `evidence_change.lines is None` explicitly
rather than assuming an int, so a binary simply cannot serve as mutation
evidence, which remains correct.

Classification runs *before* this branch and is unchanged, so the ordering after
the edit is: classify (raise on unknown) → class-specific rules. An unclassified
binary is still refused, by the classifier, with the message that names the real
defect — an unclassified path — instead of one that names an unfixable property
of the file.

**Why the guard is not load-bearing.** Its stated job is that a binary "is not
reviewable as text". True, and irrelevant: nothing else in this checker reviews
text either. It classifies paths and enforces an evidence contract. The property
that keeps an unreviewable blob out of the tree is that it must first earn a
class, and that check is the one being kept.

**Why not an exemption list instead.** An allowlist of blessed binary paths is a
shared must-write surface — every golden-bearing PR would edit it, which is
precisely the lock AGENTS.md forbids. Classification already partitions these
paths by *where they live*, which is the derived-at-read-time shape.

## 4. Risks and decisions

| Risk | Assessment |
|---|---|
| A large unreviewed binary lands in a product path | The classifier still refuses any path without a class, and `product` requires arriving on a PR. A binary in a classified location was always intended to be legal — see the `SITE_ASSET` comment. |
| This reads as weakening a gate to go green | It is a deliberate retirement, argued in the commit message per the no-waiver-registry rule, not a repair of a red run. No PR of mine is unblocked by it; the beneficiaries are #431 and future golden work. |
| Goldens become unreviewable in practice | Unchanged by this edit — they are unreviewable as text either way. Golden provenance is enforced by the parity gates and the oracle-identity requirements, which is where it belongs. |
| The retirement is silently reversed later | The RED-first test in §5 asserts the new behaviour directly, so a reintroduction turns it red. |

## 5. Tests

RED-first, in `tests/scripts/test_check_pr_size.py`:

1. `test_a_classified_binary_is_accepted` — a binary at a classified path
   (`tests/parity/goldens/.../our_ids.npy`, and a `website/static/` asset)
   produces **no** error. **RED before the change** for the intended reason:
   the guard fires.
2. `test_an_unclassified_binary_is_still_refused` — a binary at an unclassified
   path still errors, and the error names classification, not binaryness. This
   is the guard rail that keeps the retirement scoped. Green both before and
   after (the classifier raises first), so it is a regression pin, not evidence.
3. Rewrite `test_retiring_the_budget_did_not_retire_the_other_contracts` so its
   binary clause asserts the *classified* binary passes while the *unclassified*
   one fails, keeping the other two contracts pinned exactly as they are.
4. Delete `test_binary_changes_fail_closed_instead_of_becoming_free`, which
   states the retired rule and cannot survive it.

## 6. Gates

- `python3 -m pytest tests/scripts/test_check_pr_size.py` green, with case 1
  shown RED on the unmodified checker first.
- `python3 scripts/check-pr-size.py --base <base> --head <head>` classifies this
  PR's own change without error.
- `scripts/agent-preflight.sh --staged` clean.
- The checker-evidence contract must be satisfied *by this very PR*: it changes
  a `governance_checker`, so it must ship executable mutation evidence in
  `tests/scripts/test_check_pr_size.py`. It does.

## 7. Evidence

**RED before**, on the unmodified checker, for the intended reason — all four
subtests of `test_a_classified_binary_is_accepted` die on the guard:

```
AssertionError: Lists differ:
  ["binary change 'website/static/fonts/sora-700.woff2' is not reviewable as text"] != []
4 failed, 43 passed, 119 subtests passed
```

`test_an_unclassified_binary_is_still_refused` was already green here, as §5
predicted — classification runs first, so it is a rail and not the evidence.

**GREEN after:** `43 passed, 123 subtests passed`. Wider governance suite
(`test_check_pr_size` + `test_agent_record`): `92 passed, 125 subtests`.

**The retirement does what it is for**, checked directly against a
golden-bearing change:

```
golden-bearing PR errors -> NONE (was: 2 refusals)
unclassified binary      -> ["unclassified repository path 'junk/blob.bin'"]
```

**The checker accepts its own diff**, satisfying the evidence contract it
enforces on `governance_checker` paths:
`check-pr-size.py --base 7572b0f4e --head <head>` → `OK`, exit 0.
`check-commit-trailers.py` → `OK: commit trailer contract`, exit 0.

**Stop condition §8 checked, not assumed.** The full `tests/scripts/` suite is
byte-identical before and after: `8 failed, 20 passed, 2 skipped` on the
modified worktree *and* on unmodified `main` `7572b0f4e`, across
`test_gen_vulkan_spirv`, `test_mlx_system_headers` and `test_now_render`. All
pre-existing; this change adds no failure. `test_cpu_kernel_bench.py` fails
collection on unmodified main too (it wants a built benchmark binary).
`test_cpu_x86_llamacpp_floor` and `test_now_render` are order-dependent under a
loaded parallel run and pass in isolation on both trees.

## 8. Stop conditions

- If removing the branch turns any other case in the suite red for a reason not
  named in §5, stop — that is a load-bearing use of the guard this spec did not
  find, and the design in §3 is wrong.
- If the checker cannot classify its own diff after the edit, stop.
- If a reviewer judges that classification alone does not carry the protection,
  stop and escalate rather than widening the change.
