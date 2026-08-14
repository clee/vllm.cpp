# The diffusion lane's device seam — the sibling that never adopted it, and the gate that cannot see it

Row: `LTX25-DEVICE-SEAM-SIBLING`
Issues: [#659](https://github.com/mudler/vllm.cpp/issues/659), [#660](https://github.com/mudler/vllm.cpp/issues/660)
Campaign: [#644](https://github.com/mudler/vllm.cpp/issues/644)
Base: `11cc1d5896b480a1b652db9249319242053aca93`

Both defects were found by a **peer session's reviewer** while reviewing the
main-red repair for landing, not by this campaign. They are recorded here
because they are in this lane and this campaign owns them.

## 0. What is wrong today

`11cc1d589` routed LTX-2.5's device question through the platform seam. It fixed
the lane it was aimed at and left two things standing.

**(a) #659 — the seam was adopted, its companion guard was not.**
`src/vllm/multimodal/ltx2_video.cpp:549-562` now asks two questions:

```cpp
const vt::DeviceType accelerator = vllm::platforms::CurrentPlatform().device_type();
if (accelerator == vt::DeviceType::kCPU ||
    vt::TryGetBackend(accelerator) == nullptr) { Fail(...); }
```

That is "is there an accelerator, and is a backend registered for it". The
precedent it cites, `SelectQueueForModel`, asks a third question —
`src/vllm/entrypoints/model_loader.cpp:97`:

```cpp
(architecture.empty() || plat.supports_model_architecture(architecture))
```

`supports_model_architecture` exists precisely so a **partial** backend can
decline by name. Two platforms override it and both declare a short list:
`src/vllm/platforms/metal.cpp:70` and `src/vllm/platforms/tenstorrent.cpp:55`.
On such a build a `device = 1` LTX-2.5 load **was refused by name** and is now
accepted, so the failure moves from a refusal that says which piece is missing
into a kernel bind that says nothing. CUDA is unaffected, which is exactly why
this is invisible on the box that runs the gates.

The refusal that *is* there argues, correctly, that serving the CPU forward
behind an accelerator handle "would make every later timing and every 'it ran on
the GPU' claim false". A partial backend that binds and dies has the same
property one level down: it is a device claim the build cannot honour.

**(b) #660 — the gate that certified (a) is a token grep, and the sibling lane
spells its way past it.** `scripts/check-device-leakage.py:78` is

```python
RE_KCUDA = re.compile(r"\bkCUDA\b")
```

`src/vllm/multimodal/minimax_h3_video.cpp:221-226 @ 11cc1d589` — anchored on the
base SHA, because this row is what removes it and an unanchored line number here
would point at a blank line the moment it lands — never writes that token:

```cpp
vt::DeviceType MiniMaxH3VideoDeviceType(int32_t device) {
  if (device != 0 && device != 1) { throw ...; }
  return static_cast<vt::DeviceType>(device);
}
```

It hardcodes the ABI's `0/1` into the enum by integer cast and scores **zero**
in the `kcuda` bucket. The same defect, in the sibling diffusion lane, under a
different spelling.

**The sharpest form of it:** `tests/vllm/models/test_minimax_h3_video_fold.cpp:162`
asserts `MiniMaxH3VideoDeviceType(1) == vt::DeviceType::kCUDA`. The *test* spells
the token honestly and is counted; the *source* launders it and is not. The gate
therefore reads the confession and misses the act.

This is the shape the project already has a name for — an instrument that cannot
report its own blind spot returns a pass. "`kcuda` 2 → 0" is a true statement
about the token and a weaker statement about the property than it appears to be.

## 1. Scope

**In.**

1. `minimax_h3_video.cpp` resolves its device through the platform seam rather
   than by integer cast, mirroring what `ltx2_video.cpp` now does. The public
   `MiniMaxH3VideoDeviceType(int32_t)` contract in
   `include/vllm/multimodal/minimax_h3_video.h:63` is preserved — `0` is CPU,
   anything else is refused or resolved, never cast.
2. `ltx2_video.cpp` asks `supports_model_architecture` alongside the two
   questions it already asks, and refuses **by name**, naming the platform and
   the architecture, when a registered backend declines the model.
3. `check-device-leakage.py` gains a bucket that sees an integer cast to
   `vt::DeviceType`. The bucket must be **derived from the property**, not from
   the one spelling we happen to have found — see §3.

**Out.**

- Any change to what CUDA does. This row must be a no-op on a CUDA build, and
  the gate for that is the existing suite passing unchanged on this box.
- Widening or rewriting `device-leakage-baseline.json` beyond the new bucket's
  own entries. **The baseline is a ratchet**: a peer measured that
  `--write-baseline` refuses to raise it (`REFUSING to write a HIGHER baseline
  (32 → 35)`) but that **hand-editing the JSON to a higher number makes the
  checker PASS**. So the only real defence is that the diff is visible and
  reviewed. Any baseline line this row touches is called out in the PR body,
  with its reason, and the reviewer checks the file by md5 against the base for
  every key this row does not claim.
- The H3 video engine's model behaviour. This is a device-resolution change.

## 2. Upstream anchors

There is no upstream for this: vLLM has no LTX-2.5 or MiniMax-H3 video engine,
and the device seam is ours. The mirror source is therefore **our own
`SelectQueueForModel`** (`src/vllm/entrypoints/model_loader.cpp:59-104`), which
is the shape every other model path already uses, and
`include/vllm/platforms/interface.h:263`, which defines the capability question.
Mirroring an internal seam is what "route it through the shared surface" means;
a second, parallel device resolution in the diffusion lane is exactly the
hand-rolled path AGENTS.md forbids.

## 3. Design

**The capability question, at the same site as the existing two.** One added
clause, one refusal message that names the platform, the architecture and the
fact that the backend declined — not a shape error. It must be possible to read
the refusal and know that the build is partial rather than broken.

**The sibling's resolution.** `static_cast<vt::DeviceType>(device)` is replaced
by an explicit mapping: `0` → `kCPU`, non-zero → the platform's
`device_type()`, refused when that is `kCPU` or its backend is absent or it
declines the architecture. That makes the two diffusion lanes answer the device
question the same way, which is the point of the seam.

**The gate's new bucket is the hard part, and it is where this row can go
wrong.** A bucket that greps `static_cast<vt::DeviceType>` closes the one site
we found and nothing else — the same defect as the token grep, one spelling
later. The bucket is defined by the property: *an integer literal or integer
variable becoming a `vt::DeviceType` without passing through the platform
seam*. Whatever the implementation, the acceptance test is adversarial and
stated up front:

> The bucket must go RED for **at least three spellings** of the same defect
> that are not the one already in the tree — e.g. a C-style cast, a
> `vt::DeviceType(x)` functional cast, and an assignment through an
> intermediate `int`. If it only catches the one we knew about, it is a
> regression test wearing a gate's clothes, and the row says so rather than
> claiming coverage.

If the property cannot be expressed at the granularity a text checker allows,
that is a finding to record, not to paper over: the row then states the residual
blind spot in the checker's own message, because **a checker's message is the
authority on what it enforces**.

## 4. Tests

RED-first for each of the three.

1. A build with a platform that declines the architecture must refuse the LTX-2.5
   `device = 1` load **by name**. The existing test fixture pattern for a
   partial backend is `metal.cpp` / `tenstorrent.cpp`; if neither is
   constructible in the CPU test build, the test injects a stub platform rather
   than skipping — a skipped test here is the whole finding.
2. `MiniMaxH3VideoDeviceType` keeps its contract: `0` → `kCPU`, `-1` and `2`
   throw (`test_minimax_h3_video_fold.cpp:161-164` already assert this and must
   stay green **unchanged**), and `1` resolves through the seam rather than by
   cast. The new assertion is that on a CPU-only build `1` is **refused**, which
   the cast could never do.
3. The checker's three-spelling adversarial test above, each spelling asserted
   RED individually, not as a batch.

**Mutations that must be run and recorded:** revert each of the three changes
independently and confirm the corresponding test goes RED; and confirm the
existing suite is byte-identical in count on CUDA-absent builds, since a changed
count is RED even when it reads green.

## 5. Risks

- **The capability guard could refuse a load that works today.** `metal.cpp` and
  `tenstorrent.cpp` are the only overriders, so the blast radius is those two
  builds — but if either currently *runs* a diffusion model despite a short
  list, this row breaks it. Check before assuming; if it does run, the finding
  is in the list, not in the guard.
- **The new bucket could red other lanes.** An integer-to-`DeviceType` cast
  elsewhere in the tree is either the same defect (fix it or record it) or a
  legitimate deserialization boundary (which needs a stated, per-entry reason,
  never a blanket directory exemption).
- **Baseline churn.** See §1 Out.

## 6. Stop conditions

- If the capability guard turns out to refuse a currently-working configuration,
  stop and return `NEEDS_DECISION` rather than either shipping the refusal or
  dropping the guard.
- If the new bucket cannot reach three independent spellings, stop and report
  the residual blind spot; do not ship a one-spelling bucket described as
  coverage.
- `dgx.casa` is not required for any of this. Nothing here is a GPU measurement.

## Findings from implementation

**The brief and this spec disagreed on one line, and this spec won.** The
dispatch brief said `test_minimax_h3_video_fold.cpp:161-164` must stay green
*unchanged*, but §4.2 above enumerates what "the contract" means — `0 → kCPU`,
`-1` and `2` throw — and separately requires that on a CPU-only build `1` is
**refused**. Line 162 asserted `MiniMaxH3VideoDeviceType(1) == kCUDA`, which is
precisely the cast's answer and cannot survive the change. It is now
build-conditional and asserts BOTH arms: `== accelerator` where one is
registered, refused-by-name where none is. 161/163/164 are untouched.

**`test_minimax_h3_video_fold.cpp`'s CUDA-load case registered a BACKEND and no
PLATFORM.** It could, because the cast never asked whether the build had an
accelerator. It does now, so the fixture supplies both halves. That is the
defect being visible rather than a harness concession: a build with a CUDA
backend registered and no CUDA platform is not a build that runs on CUDA.

**The architecture key is the FAMILY string** (`ltx-2.5`, `minimax-h3`), not an
HF `architectures[0]` class name. The diffusion lanes are reached through
`LoadVideoEngine`/`VideoModelParams::family` and never read an `architectures`
entry, so the family slug is the only stable identifier they have. It does mean
`supports_model_architecture`'s key space now mixes HF class names
(`OPTForCausalLM`) with family slugs; the two cannot collide, and the refusal
names the string the user actually typed. Flagged for the reviewer as the one
judgement call in the row.

**`metal.cpp` and `tenstorrent.cpp` run no diffusion model** — searched in their
own vocabulary over `src/vllm/platforms/{metal,tenstorrent}.cpp` and
`src/vt/{metal,tenstorrent}/` with `OPTForCausalLM` as a positive control in the
same command: the control hit three times, `ltx|minimax|h3|diffus|video` hit
zero. So §5's stop condition is not triggered: the guard refuses nothing that
works today.

**The `dev_cast` bucket produced one false positive on the real tree**, and it is
the interesting kind. `kv_connector.h:225`'s
`supports_worker_transfer_on(vt::DeviceType /*device*/) const` strips to
`(vt::DeviceType )` followed by `const` — textually a C-style cast. Two
discriminators fixed it (the `(` must not be glued to an identifier; the `)` must
not be followed by a declarator suffix) and M29 pins both, together with the
proof that the discriminators did not cost the real detection.

**Residual blind spots are recorded in the checker's own docstring**, per §3's
instruction: type aliases, macros and template parameters that resolve to
`DeviceType`, `bit_cast`/`memcpy`/union punning, conversions inside the unscanned
`src/vt/` leg, and the fact that nothing type-checks the operand. The bucket
flags every cast *to* `DeviceType` and relies on `DSR-ALLOW` for the legitimate
ones.

## Findings from review (PR #671, head `094ac9e4`)

**The bucket missed the purest form of the defect, and the test that ruled that
out could not see it.** `(vt::DeviceType)d` scored 1; `(vt::DeviceType)1` scored
**0**, because the C-style alternative's trailing lookahead admitted only
`[A-Za-z_(]` and `1` is a digit. Naming a device by its literal enum value is
precisely what this bucket exists to police. Worse, M29's "the discriminators did
not cost the real detection" assertion used an *identifier* operand, so the test
could not detect the gap it was written to rule out — a guard that certifies
itself, which is the disease this row exists to fix, reproduced inside the row's
own instrument. M29 now pins the literal and signed-literal operands first.

**Four more plain spellings read GREEN in scanned files** and are closed rather
than documented, each measured at zero new hits over `src/vllm` + `include/vllm`:
global-scope qualification `static_cast<::vt::DeviceType>`, the
elaborated-type-specifier `static_cast<enum vt::DeviceType>`, direct-list-init in
a *declaration* (`vt::DeviceType dt{raw}` — M23 caught only the unnamed
temporary), and pointer punning `*reinterpret_cast<vt::DeviceType*>(&raw)`.

**The pointer form is what makes three of the four cast keywords real.**
`reinterpret_cast`, `const_cast` and `dynamic_cast` to a scoped enum are
ill-formed — compile-checked, all three rejected, with `static_cast` as a live
control that compiles — so before this change those keywords could only ever
match code that does not build, creating an appearance of coverage the pattern
did not have. Matching a pointer target makes them live.

**The declaration form takes `{` and never `(`, and that costs nothing.**
`vt::DeviceType dt(raw)` is ill-formed (no implicit int→scoped-enum conversion),
so excluding it loses no real spelling, whereas admitting `(` matched every
function *definition* whose return type is `DeviceType` — measured, 3 false
positives including `MiniMaxH3VideoDeviceType` and `ResolveExplicitDeviceType`.
M34 pins that negative.

**The fold test asked two of the three questions and so red on the build class
#659 exists to serve.** `test_minimax_h3_video_fold.cpp`'s `have_accelerator`
predicate tested only `device_type() != kCPU && TryGetBackend() != nullptr`. On
Metal or Tenstorrent both are true, so the test took the `== accelerator` arm
while the source *correctly* refused, and the refusal surfaced as an uncaught
exception — a false RED, invisible on the CPU and CUDA boxes that run the gates,
which is this row's own thesis about #659 turned against its own test. The
predicate is now three-way and asserts *which* refusal, since a right refusal for
a wrong reason is a wrong diagnosis that reads as a right one.

**The decline CONSEQUENCE inverts the cited precedent, deliberately.** The PR
described the change as mirroring `model_loader.cpp:97`, and it mirrors that
site's *question* while inverting its *answer*. `:97`'s capability test lives on
the `kAuto` path, whose response to a decline is to fall through to `:103` and
**serve on CPU**; `metal.cpp:65-69` states that policy in as many words ("falls
back to the CPU reference … and runs correctly, just slowly — which is strictly
better than dying inside a kernel bind"). Both diffusion lanes instead **throw**.
That is correct, but it is the *explicit-device* path's polarity, not the
`kAuto` path's: `device = 1` is an explicit accelerator request, and
`model_loader.cpp:71-72` already says of that path "an explicit accelerator whose
queue cannot be created must FAIL the load loudly, never silently serve on CPU" —
the same argument `ltx2_video.cpp:545-548` makes for refusing rather than serving
the CPU forward behind an accelerator handle. So the lanes mirror the capability
question from one path and the failure polarity from the other, and both halves
are the seam's own.

**Residual, not this row's to fix:** `vulkan.cpp` does not override
`supports_model_architecture`, so it inherits `interface.h:263`'s default `true`
and a partial Vulkan build still binds and dies inside a kernel. Only
`metal.cpp:70` and `tenstorrent.cpp:55` narrow the claim.

## Findings from review round 2 (PR #671, head `074ef1420`)

**F5 — the row's own thesis came back for its instrument a THIRD time, and this
one was in the checker's MESSAGE.** M33 had made the pointer target real, and the
docstring then said the bucket is "anchored on the TARGET TYPE, not on the
operand and not on one cast keyword", and listed `std::bit_cast` as unreachable
"because no spelling of the target type appears at the conversion site". Nine
spellings that DO write the target type at the conversion site scored **zero**.
AGENTS.md makes a checker's own message the authority on what it enforces, so an
over-claiming message is a defect in the gate, not a wording nit — and it is #660
exactly, one sigil later.

Each spelling was **compile-verified legal** (`g++ -std=c++20 -Wall -Wextra`,
exit 0) before being called a miss, because a "miss" that does not compile is
not a miss; and each measured **0 before / 1 after**, individually:

| spelling | why it slipped |
|---|---|
| `reinterpret_cast<vt::DeviceType&>(raw)` | reference target; the standard *defines* it as M33's pointer pun, and `\*?` admits a star but not an ampersand |
| `static_cast<vt::DeviceType const>(d)` | east const |
| `static_cast<vt::DeviceType const&>(t)` | east const + reference — the form that compiles with **no** warning, so it survives a `-Werror` build where the plain east-const prvalue trips `-Wignored-qualifiers` |
| `(vt::DeviceType const)d` | east const, C-style |
| `(vt::DeviceType)*cursor` | the trailing class admitted a sign but not a dereference — **and this is the wire-decode spelling the docstring itself names as the expected `DSR-ALLOW` case**, so the one site the bucket predicted meeting was the one it could not see |
| `(vt::DeviceType)~mask`, `(vt::DeviceType)!flag` | the same hole |
| `reinterpret_cast<vt::DeviceType**>(p)` | `\*?` is one star |
| `std::bit_cast<vt::DeviceType>(raw)` | the docstring's own reason was false for it: it spells the target in full |

Three edits close them: the named-cast target takes cv-qualifiers on **either**
side of the name and a **run** of `*`/`&`; the C-style trailing class admits
`*~!`; `bit_cast` joins the cast keywords.

**What was NOT closed, and why the docstring now says so per-entry.** `&` is
deliberately absent from the C-style trailing class: `) &` and `) &&` are
ref-qualifiers on a member declarator, which is precisely the false positive the
trailing guard exists to reject, and the guard cannot tell
`void note (vt::DeviceType*) &` from a pun. So the **C-style** pointer pun
`*(vt::DeviceType*)&x` stays blind, and is named as blind with the reason that is
true *of it* — the named-cast spelling of the same pun is caught, which is what
makes it a narrow gap rather than the class. The old blind-spot list shared one
reason across three entries and that reason was false for one of them; each entry
now carries its own.

**Measured, not asserted.** Over **740** files in `src/vllm` + `include/vllm`,
with `\bDeviceType\b` = **162** matches as a positive control **in the same
command**: **0 new, 0 lost**. Shipped hits 1, widened hits 1 — the same
allowlisted `platform.cpp:85` registry-walk inverse. `dev_cast` stays 0, `total`
stays 32, `scripts/device-leakage-baseline.json` is untouched by this round.

**M35-M40**, one mutant per newly-closed spelling, each sub-spelling asserted
**individually** (the M29 shape), because a mutant that only exercises the case
the pattern was tuned for is a guard that certifies itself. RED-before is real
rather than narrated: with the pattern reverted in-process to its pre-repair
value, M35-M39 all go RED while M20, M33 and M40 stay GREEN — so a blanket
failure cannot pass for five findings. M40 is the negative the widening could
have cost: pointer- and reference-returning declarations, ref- and
rvalue-ref-qualified members **with a space before the paren**, volatile
members, and `std::vector<vt::DeviceType*>`.

**Anchor drift, re-derived at the merge.** `9f2b9bb9a` moved
`tenstorrent.cpp`'s `supports_model_architecture` from `:52` to `:55`; the four
citations on this branch (`ltx2_video.cpp`, `test_diffusion_device_seam.cpp`, and
§0 and the residual note above) are corrected. `metal.cpp:70`,
`interface.h:263` and `model_loader.cpp:97` were re-derived on the merged tree
and all three still hold. Every `path:NN` anchor in this row's files was
resolved mechanically; the only remaining unresolvable ones are upstream Python
paths and two `examples/*/main.cpp` citations that carry their own `@ fc636c76`
and so are provenance rather than drift. The `minimax_h3_video.cpp:221-226`
citation in §0 is now anchored on the base SHA for the same reason.

## Now

`READY`, implemented and awaiting a fresh review on
`row/LTX25-DEVICE-SEAM-SIBLING` (PR #671). All three changes plus the F5 repair
are on the branch with their RED, GREEN and mutation evidence in the PR body;
next is a fresh reviewer — not the implementer — on the immutable head, then the
operator's own gate rerun.

The row stays `READY` in `roadmap_v1.md` deliberately. A lifecycle move to
`ACTIVE` owes `docs/STATUS.md` and `docs/BENCHMARKS.md` in the same change
(scripts/check-doc-checkpoint.py), and those are projections of what the project
CLAIMS — which this row does not change until it lands. Writing them from an
unmerged PR would also put two shared files under a lock for the length of a
review. The operator moves the state, and writes those two surfaces, when it
merges.
