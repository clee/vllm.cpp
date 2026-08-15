# Qwen3.8-27B (bf16): the token gate and the speed axes

**Row:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`
(`.agents/model-matrix.md`) — `Qwen/Qwen3.8-27B` declares
`Qwen3_5ForConditionalGeneration`, the architecture that row owns.
**Issue:** [#915](https://github.com/mudler/vllm.cpp/issues/915)
**Related:** [#821](https://github.com/mudler/vllm.cpp/issues/821) owns the
NVFP4 / Q4_K_M arms of the same checkpoint; [#910](https://github.com/mudler/vllm.cpp/issues/910)
owns the tie-break divergence this gate ran into three times.
**Lifecycle:** `DONE`
**Owner:** unassigned

## Scope

`Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, bf16, text
path. Two axes, in order: a greedy token gate against the pinned oracle, then —
only if that gate is clean — throughput, latency and memory against vLLM's
production configuration.

Out of scope: the NVFP4 and Q4_K_M arms (#821), the vision path, any fix for
#910, and advancing the parity pin.

## Why this row needs no port

[`porting-inventory.md`](../porting-inventory.md) deviation 17 records the
finding this spec measures against: `Qwen/Qwen3.8-27B` "is the already-gated
Qwen3.6-27B shape retrained — `config.json` differs in exactly one key
(`transformers_version`) and the safetensors tensor-name set is identical (1199
names, zero difference either direction)". So no loader, forward or registry
change is owed, and none was made. What was owed is evidence, and the same
entry says so: "its own token-exact gate is OWED and unrun".

## Design of the adjudication

Greedy, 7 prompts x 16 tokens, both arms on the same prompts and token counts.
The oracle capture is deterministic across 3 repeats (`deterministic: true`,
`multi_member_cells: 0`), so a strict per-prompt comparison is well-posed.

**Only the first divergence per prompt is adjudicable.** After it the two arms
are conditioned on different prefixes, so any later position compares two
different conditionings rather than one disagreement. Three prompts diverge, so
there are three numbers — not thirty-four. A position count over the whole grid
is not a quality score and is not recorded as one.

At each first divergence the two arms agree on every preceding token, so both
are conditioned on a BYTE-IDENTICAL prefix. Feeding that prefix to the pinned
oracle and reading its **fp32** next-token distribution measures exactly one
thing: how far apart the oracle itself holds its own choice and ours.

## Risks

- **A bf16 instrument cannot resolve a bf16 tie.** A `transformers` CPU probe
  reported all three as exact ties, and every runner-up gap it printed was an
  exact multiple of 0.125 — one bf16 ULP in that exponent range. It could not
  have reported anything else, so it is a secondary cross-check with that
  limitation stated, never the answer. vLLM computes logprobs from fp32 and can
  separate pairs bf16 collapses.
- **A re-decode probe assumes the prefix it needs.** Reading the distribution
  at step `d` of a fresh greedy decode is only valid if that decode reproduced
  the captured prefix. Mitigated by a second, teacher-forced probe that feeds
  the prefix as token IDs and asserts both the echoed prefix and that the
  oracle's top-1 equals the captured token.
- **A stale binary measures a tree that does not exist.** Mitigated by
  rebuilding at `origin/main` and re-running both axes on that binary.
- **A degraded build voids every number.** Mitigated by asserting the
  configure-log fast-path lines (CUTLASS, FA2, Triton AOT `sm_121a`) and
  aborting the build otherwise.

## Tests

This row changes no `src/`, `include/` or `tests/` file, so it ports no test.
`git diff origin/main..HEAD` over those three paths is empty and that emptiness
is the claim. What it adds is evidence, listed under Evidence required.

## Gates

- Token gate: every first-divergence position within `kNearTieMnats = 500` of
  the oracle's teacher-forced argmax, on the pinned oracle, or the row fails.
- Speed: recorded only if the token gate is clean. vLLM's **production**
  configuration is the denominator — never `--enforce-eager`. Same client
  (`vllm bench serve`) drives both arms.
- Both under `$HOME/gpu.lock`, clocks pinned, contention recorded per leg.

## Evidence required

- Oracle identity asserted (`vllm.__version__`, `flashinfer`) with ABORT on
  mismatch, plus a `Python.h` precondition so a missing header aborts loudly
  rather than dying inside Triton's JIT.
- Per divergence: the oracle top-2 gap in millinats, our token's rank in the
  oracle top-20, and a verdict against the band.
- The build recipe, revisions, checkpoint size, binary md5, boot id, SM clock
  and the contention actually observed.

## Stop conditions

- If any divergence is out of band, STOP: report it as a real divergence, run
  no benchmark on that checkpoint, and let no record imply it is
  baseline-ready. Do not attempt a fix.
- If the box is not quiet at a leg boundary, wait or drop the leg. A number
  taken under load is worse than no number.

## Now

`DONE` for the token axis and for the c1/c4/c8 online-serving axes recorded in
[BENCHMARKS](../../docs/BENCHMARKS.md). See `## Outcome`.

## Outcome

To be completed with the measured result.
