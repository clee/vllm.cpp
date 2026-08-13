# LTX-2.5 — tiled + streaming Conv VAE video decode

**Row:** `LTX25-TILED-DECODE` (row 2 of the LTX-2.5 full port).
**Issue:** [#644](https://github.com/mudler/vllm.cpp/issues/644).
**Branch:** `row/LTX25-TILED-DECODE`.
**Upstream pin:** Lightricks `LTX-2` @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
(the SHA `scripts/gen-ltx2-vae-goldens.py` and `test_ltx2_vae.cpp` already pin).
**Owner of the parent spec:** `.agents/specs/ltx-2-5.md` is operator-owned; this row
does not edit it.
**Scope boundary:** this row owns the **VAE DECODER** path only. `row/LTX25-PROMPT-ADALN`
owns the DiT loader / module refusals; `row/LTX25-IMAGE-COND` owns the VAE **encoder**
keys and image conditioning.

---

## 0. The premise this row was dispatched on, and why it is refuted

The dispatch brief said: *448x256/25f stops; MemAvailable falls 73.0 → 13.8 GiB in 24
seconds on the decode side of the last drain; upstream's DEFAULT decode is tiled and
streaming and we call the untiled path, so port `tiled_decode`.*

The brief also required the attribution to be measured before building. It was, and
**the causal half of the premise does not hold.** Two independent results, both recorded
here before any implementation, because acting on an unverified premise is the failure
mode this campaign has been punished for repeatedly:

**(a) Upstream's own default tiling is a NO-OP at 448x256/25f.** Executed, not read:
`scripts/probe_ltx2_tiling_layout.py` runs `TileSizeConfig.from_long_side(long_side=768/64,
frames=80/24)` — the exact auto layout `ltx_pipelines/utils/helpers.py:62-63,80-88`
builds for a Conv VAE — through `TileSizeConfig.to_splitters` (`tiling.py:797-834`) and
`ConvVideoDecoder._prepare_tiles` (`conv_video_decoder.py:359-381`) at the pinned SHA:

| request | latent (T,H,W) | resolved tile cfg | tiles | temporal groups |
|---|---|---|---|---|
| 128x128/9f | 2, 4, 4 | h 768/64, w 768/64, f 80/24 | **1** | **1** |
| 320x192/25f | 4, 6, 10 | h 448/64, w 768/64, f 80/24 | **1** | **1** |
| **448x256/25f** | **4, 8, 14** | h 448/64, w 768/64, f 80/24 | **1** | **1** |
| 896x512/25f | 4, 16, 28 | h 448/64, w 768/64, f 80/24 | 4 | 1 |
| 768x768/73f | 10, 24, 24 | h 768/64, w 768/64, f 80/24 | **1** | **1** |
| **768x768/81f** | **11, 24, 24** | h 768/64, w 768/64, f 80/24 | **2** | **2** |
| 1024x576/97f | 13, 18, 32 | h 448/64, w 768/64, f 80/24 | 8 | 2 |
| 1280x704/121f | 16, 22, 40 | h 416/64, w 768/64, f 80/24 | 8 | 2 |
| 1920x1088/241f | 31, 34, 60 | h 448/64, w 768/64, f 80/24 | 36 | 4 |

`split_by_size` returns a single interval when `dim <= size` (`tiling.py:199-200`) and
`split_temporal_causal` short-circuits identically (`tiling.py:239-240`). At 448x256 the
latent is 8x14 against a 14x24 grid tile, and 4 latent frames against a 10-latent-frame
temporal tile. So upstream, on its own defaults, calls `self.forward` **once on the whole
volume** at this size — exactly what `ltx2_video.cpp:1477-1479` does. Running tiled
decode there would allocate one extra full-size pixel buffer and decode identically.

The temporal axis does not start chunking until **81 frames**; the spatial axes do not
start tiling until the long side exceeds 768 px (or the short side exceeds its
aspect-coupled tile). "The win at our resolution is TEMPORAL chunking" is therefore not
true at 25 frames.

> **Corrected 2026-08-13 (review of PR #656).** This paragraph, §2 and §"What this row
> therefore does" all said **121 frames**, and 121 is wrong. `latent_t = (frames - 1) / 8
> + 1`, and `split_temporal_causal` short-circuits only while `latent_t <= 10`
> (`tiling.py:239-240`), so the split first happens at `latent_t = 11`, i.e. **81 frames**.
> Executed at the pinned SHA over `range(1, 137, 8)` on the 1024x576 AUTO layout, the
> interval count goes 1 -> 2 exactly at 81. The row's own golden already said so —
> `kLtx2AutoCases` carries `768x768/81f -> t_intervals = 2, chunks = 2` — and
> `docs/FEATURES.md` said 81; only the prose was wrong.
>
> **Root cause, recorded because it is the reusable part.**
> `scripts/probe_ltx2_tiling_layout.py` swept 9, 25, 25, 25, **121**, 241. It never
> sampled a frame count between 25 and 121, so it stepped straight over its own binding
> point and the number it happened to land on was written down as the threshold. The
> sweep now walks the temporal axis in single latent-frame steps across the boundary and
> asserts where the transition is, so skipping it again is not possible.
>
> **What the correction costs, stated plainly.** 81..120 frames is the tiled regime, and
> `docs/USAGE.md` records the LTX-2.5 recipe default as **1024x1536 at 121 frames** — an
> ordinary request is inside it. Proven on the shipped checkpoint at the AUTO layout,
> 64x64/81f (the same AUTO layout the golden's 768x768/81f row resolves, at a size a
> full-precision reference decode can finish), latent 11,2,2 -> 2 tiles / 2 groups:
>
> ```
> [equiv] tiles=2 groups=2 -> untiled [3,81,64,64]  streamed [3,81,64,64] chunks=2
> [equiv] max|diff| = 0.71614238619804382   non-bit-identical floats = 985849 / 995328
> ```
>
> 99.05% of the pixels move, by up to 0.716 on a signal of scale ~1. That is upstream's
> behaviour and not a defect, but the ONE-TILE CONTROL's safety argument — "routing
> through the streaming entry point is safe at every size the AUTO layout does not tile" —
> **covers below 81 frames and does not cover 81..120**.

**(b) The decoder's own buffers at that shape are ~2 orders of magnitude too small.**
The shipped ladder is read from the checkpoint's own `__metadata__["config"]`
(`/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors`):

```
decoder_blocks = res_x(4), compress_space(m2), res_x(6), compress_time(m2),
                 res_x(4),  compress_all(m1),  res_x(2), compress_all(m2), res_x(2)
patch_size 4, decoder_base_channels 128, norm_layer pixel_norm,
causal_decoder false, timestep_conditioning false, spatial_padding_mode zeros
```

`_decoder_bottleneck_channels` (`conv_video_decoder.py:43-58`) gives 128 x 8 = 1024, so
the reversed walk for a `[128, 4, 8, 14]` latent is:

| step | C, T, H, W | activation | largest `CausalConv3d` pad buffer |
|---|---|---|---|
| conv_in | 1024, 4, 8, 14 | 1.8 MB | 0.5 MB |
| res_x x2 | 1024, 4, 8, 14 | 1.8 MB | 3.9 MB |
| compress_all m2 | 512, 7, 16, 28 | 6.4 MB | 7.3 MB |
| res_x x2 | 512, 7, 16, 28 | 6.4 MB | 10 MB |
| compress_all m1 | 512, 13, 32, 56 | 47.7 MB | 51 MB |
| res_x x4 | 512, 13, 32, 56 | 47.7 MB | 60 MB |
| compress_time m2 | 256, 25, 32, 56 | 45.9 MB | 51 MB |
| res_x x6 | 256, 25, 32, 56 | 45.9 MB | 55 MB |
| compress_space m2 | 128, 25, 64, 112 | 92 MB | 92 MB |
| res_x x4 | 128, 25, 64, 112 | 92 MB | 104 MB |
| conv_out + unpatchify | 3, 25, 256, 448 | 34 MB | 104 MB |

A handful of live volumes, peak on the order of **hundreds of MB**, against a reported
**60 GiB**. The whole pixel tensor is 32.8 MiB; streaming it away cannot recover 60 GiB.

The measured peak-RSS number for this path is in §5. **If the memory is not in the decode
buffers, that is the finding**, and §5 says where it actually is.

## 1. What this row therefore does

The structural port is still correct and still owed — `ltx2_video_vae.cpp:17-18` records
`tiled_decode` as out of phase L4 and owed, and the sizes where it *does* bind
(896x512 and up, 81 frames and up) are inside this project's v1 scope. So this row
ports the mechanism, and states honestly which size it is a no-op at.

What changes:

1. `Ltx2TileSizeConfig` / `Ltx2DimensionSizeConfig` and the split + mapping algebra —
   `tiling.py:13-49, 174-272, 369-379, 494-571, 619-841`.
2. `Ltx2ConvVideoDecodeTiled`, the streaming decode — `conv_video_decoder.py:383-484`
   plus `_accumulate_temporal_group_into_buffer` (`:508-557`), expressed as a **callback
   per temporal chunk** rather than a Python generator, so the C++ caller gets the same
   "yield and drop" peak-memory property.
3. `Ltx2ConvVideoDecodeAuto` — the `AUTO_TILING` layout
   (`ltx_pipelines/utils/helpers.py:59-88`, `TileSizeConfig.from_long_side`
   `tiling.py:754-795`), so the pipeline gets upstream's default rather than a
   vllm.cpp-invented one.
4. `ltx2_video.cpp` routes decode through the streaming entry point and writes each PPM
   frame as its chunk arrives, mirroring `ti2vid_two_stages.py:369-376` where the
   consumer streams chunks straight into the muxer.

What this row does **not** do:

- It does not port `memory_efficient_decode.py:1-43` (workspace buffers, in-place chunked
  Conv3d, free-before-conv). Measured against §5's attribution it is not the binding
  lever, it is a second, independent rewrite of `CausalConv3d`, and it belongs with the
  bf16/NVFP4 production arm phase L6 owes. Recorded as owed here, not silently dropped.
- It does not touch the encoder tiling (`prepare_tiles_for_encoding`,
  `map_*_interval_to_latent`, `video_vae.py:549-618`) — that is `row/LTX25-IMAGE-COND`'s
  surface.
- It does not change `Ltx2ConvVideoDecode`'s numerics. The untiled path stays the gated
  reference and the tiled path is held to it.

## 2. The three things that fail silently

* **Ramps must partition unity, or the untiled-equivalence gate is vacuous.**
  `masks_are_complementary` (`tiling.py:438-472`) is checked per axis over the *unique*
  out-slices on that axis — a cartesian product over tiles would multi-count the same 1-D
  interval and never sum to 1. When it holds, no denominator buffer is allocated at all.
  When it does not, upstream falls back to `compute_summed_weights` (`:475-491`), forced
  to CPU float32 precisely so a mask cannot place a multi-GB `[F,H,W]` tensor on the
  device. Both arms are ported; the complementary arm is the one that runs on the shipped
  defaults, and a test pins the *fallback* arm too so it cannot rot.
* **The masks are separable 1-D, never a dense N-D mask.** `scale_by_masks_1d`
  (`tiling.py:423-435`) multiplies by one reshaped 1-D mask per axis. `Tile.blend_mask`
  (`:403-420`) exists and materializes the dense product — it is a debugging property,
  and using it in the accumulate loop would allocate a second tensor the size of the tile.
  This port has no dense-mask path.
* **The temporal mapping is not the spatial mapping.** `map_temporal_slice`
  (`video_vae.py:549-555`) maps `[begin, end)` to `[begin*s, 1 + (end-1)*s)` and passes
  `left_starts_from_0=True`; `map_spatial_slice` (`:586-592`) maps to `[begin*s, end*s)`
  with `left_starts_from_0=False`. The `+1` is the first frame the causal decoder keeps,
  and the ramp flag changes where the linear ramp starts by one sample
  (`tiling.py:38-43`). Swapping them shifts every temporal chunk by a frame and puts a
  zero-weight sample at the start of every spatial tile.

## 3. Dtype

Unchanged from `ltx2_video_vae.cpp:22-45`: this is the CPU reference arm and every buffer
is f32. Upstream's tiled buffer inherits `latent.dtype` (`conv_video_decoder.py:427-431`)
and the masks are `float32` so a bf16/fp16 tile promotes (`tiling.py:425`); when phase L6
lands the checkpoint-dtype arm, the buffer follows the latent and the masks stay f32.
Recorded so the production arm does not have to rediscover it. No golden here can catch a
too-wide dtype — the generator casts every upstream parameter to f32, so the oracle runs
f32 too and the comparison is vacuous by construction.

## 4. Tests, and what each one would catch

Goldens are generated by `scripts/gen-ltx2-tiling-goldens.py`, which imports and runs the
pinned upstream exactly as the existing `gen-ltx2-*-goldens.py` scripts do, asserts
`ltx_core.__file__` lives under `--ltx2`, and emits the resolved upstream SHA as
`kLtx2TilingUpstreamRevision` for the C++ suite to pin.

1. **Interval algebra** — `split_by_size`, `split_temporal_causal`, `split_by_count`,
   `_grow_last_tile_to_min`, `_validate_tile_intervals`, and both mappers, over a size
   sweep that includes the `dim <= size` short-circuit, the min-tile growth path, and the
   causal `start-1 / left_ramp+1` shift. Catches an off-by-one in the ramp or the shift,
   which no tensor gate at a no-op size can see.
2. **Trapezoidal mask values**, both `left_starts_from_0` arms, against upstream tensors.
   Catches the endpoint convention (`linspace[:-1][1:]` vs `linspace[1:-1]`).
3. **`masks_are_complementary` verdict** on a layout that partitions unity and one that
   does not, plus `compute_summed_weights` values on the second. Catches a
   denominator-free accumulation applied to a layout that needs a denominator — which
   would look like a dim seam, not a crash.
4. **Tiled-vs-untiled equivalence** at a fixture small enough to force real tiling
   (§4.1). This is the correctness gate.
5. **A blend mutation** — the ramp replaced by a hard cut — must RED (4). Stated up front
   because an overlap-blend bug shows up as a seam, which a `max|diff|` gate sees and a
   shape check never will.
6. **Streaming equals batch**: the concatenation of the yielded chunks equals the full
   tensor, and the callback is invoked once per temporal group with the frame counts
   `group_tiles_by_temporal_slice` predicts.

### 4.1 The equivalence bound — MEASURED, and it does not exist

This section originally planned to derive a bound on `max|tiled - untiled|`. That was
written before the sweep. **The sweep refutes it**, and the correction is recorded here
rather than quietly replaced.

`scripts/probe_ltx2_tiled_vs_untiled.py` and `scripts/probe_ltx2_tiled_vs_untiled_shipped_ladder.py` run upstream's own
`ConvVideoDecoder` at reduced dimensions across tile sizes, latent extents, both
causality polarities, and both with and without a `res_x_y` block:

| arm | latent | tile (f/ovl, s/ovl) | chunks | max&#124;tiled − untiled&#124; | ÷ output range |
|---|---|---|---|---|---|
| causal | 5,4,4 | **no split** | 1 | **0** | **0** |
| causal | 5,4,4 | 12/4, 16/8 | 2 | 1.54113 | 1.16 |
| causal | 9,12,12 | 16/4, 24/8 | 3 | 1.72940 | 0.98 |
| causal | 9,12,12 | 24/8, 48/16 | 2 | 1.41765 | 0.80 |
| non-causal | 9,12,12 | 24/8, 48/16 | 2 | 1.18317 | 0.68 |

The gap is the size of the signal and does **not** converge as the tile grows, on either
polarity, with or without the global `res_x_y` norm. The reason is structural: the
decoder's receptive field measured in LATENT units is wider than the overlap the layout
uses — the shipped AUTO layout blends a 768 px tile with a **64 px** overlap on a 32x
grid, which is TWO latent cells. Upstream accepts that seam and blends it. There is no
bound to state and inventing one would be fabricating a gate.

So the gate set is:

* **A.** our tiled output == **upstream's tiled** output, to the same `5e-6` band every
  other LTX-2.5 golden uses. This is the correctness gate.
* **B.** our untiled output == upstream's untiled output, same band — otherwise A proves
  nothing about the tiling.
* **B'. The ONE-TILE CONTROL, whose bound is ZERO and is measured.** A tiling config
  whose splits all short-circuit produces one tile, and `tiled_decode` then reproduces
  `forward` bit for bit: upstream's own value is exactly `0` on every arm swept, emitted
  as a golden rather than assumed, and the port is held to `== 0.0`. This is the property
  that makes routing the pipeline through the streaming entry point safe at every size
  where the AUTO layout does not tile — which today is every resolution this project has
  run.
* **C.** the tiled-vs-untiled gap is **upstream's own number**, held to the golden band,
  so a port that blends differently moves A and C together.

The blend mutation is what proves A and C bind; its numbers are in `## Outcome`.

## 5. The memory attribution — measured

`scripts/probe_ltx2_decode_memory.cpp` loads the real shipped conv VAE
(`ltx-2.5-video-vae-conv-bf16.safetensors`, 1.45 GB bf16) through `Ltx2LoadVaeWeights` +
`Ltx2ParseConvVideoDecoderConfig` and calls `Ltx2ConvVideoDecode` at the exact
448x256/25f latent. It measures two different things on purpose:

* a 50 ms sampler thread reading `/proc/self/status` `VmRSS`/`VmHWM` and `/proc/meminfo`
  `MemAvailable` — the SYSTEM pressure, which is what the failing report sampled;
* a replacement of the global `operator new`/`delete` in the final link, so every
  `std::vector` inside `libvllm.a` routes through an exact live/peak byte counter — the
  DECODER's own allocations, which RSS overstates because glibc retains freed arenas.

It touches no product code, so it cannot perturb what it measures.

**Validation of the analytic model first**, because the full-size run takes ~30 CPU
minutes: at a 64x64/9f latent the model predicts the largest single allocation is the
`CausalConv3d` pad buffer of the last `res_x` block, `128 x 11 x 18 x 18` f32 =
**1.74 MiB**. Measured: `HEAP largest single alloc = 1.74 MiB`, `HEAP peak live =
5.13 MiB`, decode-attributed RSS 6.13 MiB. The model is exact, so its 448x256/25f
prediction (peak on the order of 350 MB, largest single allocation 99.3 MiB) is
trustworthy — and the full-size run confirms it in `## Outcome`.

## 6. Gates

* `ninja` clean-configure + build, `BUILD_EXIT` checked separately from the run, log
  grepped for `No space left` / `BFD assertion`, `df -h /` recorded.
* `ctest -N` denominator recorded and asserted against the full `ctest` run.
* Focused: `test_ltx2_vae`, `test_ltx2_tiling`, `test_ltx2_video`, `test_ltx2_pipeline`,
  with **case and assertion counts**, not "passed" — a changed count is RED even when
  green, and a doctest binary prints `0 failed` while throwing.
* RED-before captured for every new golden; the blend mutation captured RED and restored
  byte-for-byte.

## 7. Stop conditions

* If the equivalence bound cannot be derived from measurement — i.e. the hard-cut seam is
  not separated from the blended difference by a clear margin — the gate is not
  established and the row stops with the numbers rather than widening the band.
* If routing the pipeline through the streaming entry point would change the untiled
  numerics at the shipped sizes, stop and return `NEEDS_DECISION`.

## Now

`DONE` — the mechanism is ported, gated against executed upstream, and routed through the
pipeline. The FAIL review of PR #656 is answered in `## Outcome`; both blocking findings
are closed with reproduced RED/GREEN evidence. Three axes stay open and are named there:
the 60 GiB is **not** attributed (it is not in the decode), the reference decoder's
~30 CPU-minute 448x256/25f decode is a separate newly measured problem, and a tiled
decode over a NOISE-DRAWING config has no gate.

## Outcome

### What was measured, and what it refutes

**1. Upstream's own AUTO tiling is a NO-OP at 448x256/25f.** Executed at the pin, emitted
as a golden (`kLtx2AutoCases`) and asserted in `test_ltx2_tiling`:

| request | latent (T,H,W) | resolved cfg | t/h/w intervals | chunks |
|---|---|---|---|---|
| 128x128/9f | 2,4,4 | h 768/64, w 768/64, f 80/24 | 1 / 1 / 1 | 1 |
| 320x192/25f | 4,6,10 | h 448/64, w 768/64, f 80/24 | 1 / 1 / 1 | 1 |
| **448x256/25f** | **4,8,14** | h 448/64, w 768/64, f 80/24 | **1 / 1 / 1** | **1** |
| 896x512/25f | 4,16,28 | h 448/64, w 768/64, f 80/24 | 1 / 2 / 2 | 1 |
| 1280x704/121f | 16,22,40 | h 416/64, w 768/64, f 80/24 | 2 / 2 / 2 | 3 |
| 1920x1088/241f | 31,34,60 | h 448/64, w 768/64, f 80/24 | 4 / 3 / 3 | 5 |
| **768x768/81f** | **11,24,24** | h 768/64, w 768/64, f 80/24 | **2 / 1 / 1** | **2** |
| 1024x576/97f | 13,18,32 | h 448/64, w 768/64, f 80/24 | 2 / 2 / 2 | 2 |

Tiling first binds at **896x512** spatially and **81 frames** temporally — see the
correction box in §0, which records why the prose here said 121 and the golden said 81.
The dispatching premise — "we call the untiled path where upstream tiles" — is false at
the size that failed.

**2. The decode's own memory is 170x too small to be the 60 GiB.** Real shipped conv VAE,
real 448x256/25f latent decoded to completion (`[3, 25, 256, 448]`), exact `operator new`
accounting plus RSS sampling:

| | 64x64/9f (model validation) | **448x256/25f** |
|---|---|---|
| largest single allocation | 1.74 MiB (predicted 1.74) | **99.20 MiB** (predicted 99.3) |
| heap peak live, exact | 5.13 MiB | **361.72 MiB** |
| decode-attributed RSS | 6.13 MiB | **362.23 MiB** |
| process `VmHWM` | 2339 MiB | 2695 MiB (weights alone are 2335) |
| decode wall | 25.27 s | 2681.02 s |

The analytic ladder model predicted the largest allocation to three significant figures at
both scales, so the attribution is not an artefact of one run. **`361.72 MiB` against a
reported `60 GiB` is a factor of 170.** Flat process RSS at 4.9 GB while `MemAvailable`
fell 60 GiB is itself self-consistent with the decode NOT being the consumer: whatever
took that memory was not this process's anonymous heap.

**The 60 GiB is NOT attributed by this row and the axis stays open.** The next traceable
hypotheses, in order: which PID the RSS sampler actually read and at what interval; any
other process on that box during the window; and — if it was the same process — a CUDA or
`mmap` allocation, neither of which lands in `VmRSS` the way a `std::vector` does. None of
that is recoverable from the report as written, and this row will not guess at it.

*(The probe's own `MemAvailable` delta of 11.1 GiB is NOT the decode: a full 405-target
build and a 416-test `ctest -j 6` ran on the same box during the window. That is exactly
the confound the exact heap counter exists to remove, and it is why `361.72 MiB` is the
number quoted and `11.1 GiB` is not.)*

**3. A NEW problem, found while measuring.** `Ltx2ConvVideoDecode` at 448x256/25f took
**2681 s — 44.7 minutes** of single-threaded double-precision convolution, about 3.5 TMAC
at the 1.9 GMAC/s the 64x64/9f run measured. The reference decoder is not a shipping
decode path and phase L6 already owes the production-dtype arm; this quantifies how far
away it is. It is a separate row, not this one — but it also means the reported "24
seconds" of memory fall cannot have been a completed decode.

**4. Tiled decode is NOT an approximation of untiled decode** — §4.1, swept. The
one-tile control IS exact, on both causality arms, ours and upstream's: `max|diff| == 0`.

### The gate

**The `ctest -N` denominator is 424, not the 416 first recorded.** 416 was the ninja edge
count of `ninja test_ltx2_tiling test_ltx2_vae test_ltx2_video`, whose last line is
`[416/416] Linking CXX executable tests/test_ltx2_video` — a build number read as a test
number, so "full ctest 416/416" described a run that never happened at that denominator.
The configure line that produces 424, recorded because a denominator without one is not
reproducible:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ctest --test-dir build -N | tail -1     # Total Tests: 424
```

Build: clean `rm -rf build` + reconfigure + `cmake --build build -j 4`, `BUILD_EXIT`
captured separately from the run, log grepped for `No space left` / `BFD assertion`
(count 0), `df -h /` recorded at 93% before and 89% after.

Focused, with COUNTS (a changed count is RED even when green):

| suite | cases | assertions |
|---|---|---|
| `test_ltx2_tiling` | 10 / 10 | 907 / 907 |
| `test_ltx2_vae` | 36 / 36 | 3039 / 3039 |
| `test_ltx2_video` | 31 / 31 | 673 / 673 |
| `test_ltx2_pipeline` | 37 / 37 | 2382 / 2382 |

Full: `ctest --test-dir build --output-on-failure` -> **424/424 tests passed, 0 failed**,
`CTEST_EXIT=0`, 222.72 s, 424 lines of `N/424 Test` in the log — the run's own
denominator asserted against the `ctest -N` above. Two tests report `Skipped` by
their own guards (`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`).

One re-run was needed and is recorded rather than hidden: an earlier pass aborted at
96/424 with `test_muse_glimmer_text` throwing
`safetensors: empty file in /tmp/muse_glimmer_text_0.safetensors`. That fixture path is a
FIXED name in shared `/tmp`, and six agents were compiling on the box at 98% disk; run
serially the suite is 24/24, 528/528, SUCCESS. Not charged to this diff, which touches
no muse_glimmer path — but the fixed `/tmp` name is real shared-state flakiness and is
worth its own issue.

### The untiled-mapper mutation — RED, then restored (F2)

`map_t` and `map_s`'s `{1.0f}` broadcast masks set to `{0.0f}` in
`src/vllm/model_executor/models/ltx2_tiling.cpp`, which multiplies the whole decoded
volume by zero — a black clip.

* **BEFORE this change** (reviewed head): `test_ltx2_tiling` 9 cases / 9 passed,
  830 assertions / 830 passed, `Status: SUCCESS!`, exit 0; `test_ltx2_video` 30/30,
  502/502, SUCCESS. The finding reproduces exactly.
* **AFTER**: 10 cases -> **7 passed / 3 failed**, 907 assertions -> **5 failed**, exit 1.
  Three of the failures are the section-3b mask assertion, once per axis; the fourth is
  `untiled-spatial control max|diff| vs untiled = 2.31736` against upstream's 0.
* restored with `git checkout`, `git diff` empty, rebuilt: 10/10, 907/907, SUCCESS.

### The render mutations — RED, then restored (F6, F7)

* `chunk.first_frame + f` -> `f` (per-chunk frame numbering) in `ltx2_video.cpp`:
  the new multi-chunk case goes 171 assertions -> **35 failed**, exit 1 — the second
  chunk overwrites the first and the clip's tail is missing.
* the stale-frame cleanup loop disabled: same case -> **72 failed**, exit 1, on
  `!stale.good()` for every frame a shorter re-render left behind.
* both restored; `test_ltx2_video` 31/31, 673/673, SUCCESS.

### The blend mutation — RED, then restored

`Ltx2TrapezoidalMask1d`'s linear ramp replaced by a hard cut at the ramp midpoint, in a
scratch copy of `ltx2_tiling.cpp`:

* **RED**: 9 cases -> 3 passed / **6 failed**; 830 assertions -> **39 failed**. The case
  and assertion COUNTS were unchanged, which is the point — only the verdicts moved.
* the tiled output moved from `<= 5e-6` to **max|diff| vs upstream = 1.11411** (causal)
  and **1.16359** (non-causal): five orders of magnitude past the band, and ~half the
  output's own range.
* complementarity also broke, and the mapper masks with it — the mutation is caught in
  four independent places, not one.
* restored and verified byte-for-byte: `md5 e734ee2e3503e10ea111305a33574293` before and
  after, `git diff` empty, rebuild GREEN 9/9, 830/830.

### The review of PR #656 — FAIL, and what closed each finding

The reviewer confirmed all three refutations independently, regenerated the tiling
goldens byte-for-byte, reproduced the memory probe at three scales and verified
`Ltx2VideoDecodeStreaming` bit-identical to the untiled path on the real checkpoint.
None of that is re-litigated. Two blocking findings and six minor ones were raised.

**F1 (blocking) — the temporal binding point is 81 frames, not 121.** Corrected at all
five sites (`ltx2_tiling.h`, `ltx2_video.cpp`, `docs/USAGE.md`, and §0/§1/§2 here);
`docs/FEATURES.md` and the golden already said 81. Root cause recorded in the §0
correction box: `scripts/probe_ltx2_tiling_layout.py` swept 9, 25, 25, 25, 121, 241 and
never sampled between 25 and 121, so it stepped over its own binding point. The probe now
WALKS the temporal axis one latent frame at a time and **asserts** the transition is at
81, so skipping it again fails instead of publishing a number. The 81..120 window is
stated as a user-visible consequence rather than implied, because
`docs/USAGE.md` records the recipe default as 1024x1536 at 121 frames.

Reproduced on the shipped conv VAE by `scripts/probe_ltx2_tiled_equivalence.cpp`
(new, committed with its compile line):

```
[equiv] 64x64, latent 11,2,2  AUTO h=768/64 w=768/64 f=80/24
[equiv] tiles=2 groups=2  ->  untiled [3,81,64,64]  streamed [3,81,64,64] chunks=2
[equiv] max|diff| = 0.71614238619804382   non-bit-identical floats = 985849 / 995328
[equiv] untiled |out|max = 0.75126725435256958
```

99.05% of pixels move, by up to 95% of the output's own range. Upstream's behaviour,
mirrored — but the one-tile control's safety argument covers **below 81 frames only**.

**F2 (blocking) — the `!IsTiled()` mapper branches were shipped and ungated.**
Reproduced first: with `map_t`/`map_s`'s `{1.0f}` broadcast masks mutated to `{0.0f}` —
which multiplies the entire decoded volume by zero, i.e. renders a black clip —
`test_ltx2_tiling` reported 9/9 cases, 830/830 assertions, SUCCESS and `test_ltx2_video`
30/30, 502/502, SUCCESS.

Measured rather than assumed: swept over all eight (frames, height, width) x
(tiled, untiled) combinations against upstream at the pinned SHA,

| frames | spatial | upstream `tiled_decode` |
|---|---|---|
| tiled | any combination | runs; `max|diff|` vs `forward` == **0.0** |
| UNTILED | every combination | **TypeError** at `conv_video_decoder.py:424` |

because `DEFAULT_MAPPING_OPERATION` hands it `slice(0, None)` and :424 subtracts that
`None` stop. So the two halves are not symmetric and are closed differently:

* the SPATIAL half is gated end to end — new `UNTILED_SPATIAL` arm in the generator
  (`kLtx2TileDec*UpstreamUntiledSpatialVsUntiled` / `*UntiledSpatialChunkCount`) and a
  `(B'')` control in `RunDecodeArm` on both causality arms;
* the TEMPORAL half is **refused by name** in `Ltx2ConvVideoDecodeTiled`, mirroring
  upstream's own failure rather than inventing a concrete stop upstream never computes.
  The golden `kLtx2TileDec*UpstreamUntiledFramesRaises` records that upstream raises, so
  the refusal is mirrored and not local policy;
* the mapper output itself is pinned by new goldens section 3b
  (`kLtx2UntiledMap*`, upstream's `create_tiles` executed on an all-untiled config).

The same black-out mutation now goes **RED**: 10 cases -> 7 passed / **3 failed**,
907 assertions -> **5 failed**, `untiled-spatial control max|diff| vs untiled = 2.31736`
against upstream's 0. Restored byte-for-byte (`git diff` empty) and re-verified GREEN.

**F3 — the recorded `ctest -N` did not reproduce.** It does not: 416 was the ninja edge
count from `ninja test_ltx2_tiling test_ltx2_vae test_ltx2_video`, whose last line is
`[416/416] Linking CXX executable tests/test_ltx2_video`. The real denominator is in
"The gate" below, with the configure line that produced it.

**F4** `ltx2_video_vae.cpp:17-18` no longer records tiled decode as owed; the encoder
half still is. **F5** the shared `Ltx2NoiseStream*` across tiles is documented at
`Ltx2ConvVideoDecodeTiled` and recorded under "What is owed" — it mirrors upstream
(`self.forward` per tile with one generator) and is inert on the shipped checkpoint.
**F6** a multi-chunk render case now drives `chunk.first_frame + f` through the PPM
writer at 81 frames. **F7** a previous render's frame tail is deleted before a new render
writes. **F8** the "never materialized" claim is bounded to the tiled case at both
anchors. **F9** the compile line for both probes is recorded in their headers.
**F10** the fixture's `res_x_y` block is disclosed in the test header.

### What is owed

* **A gate over a NOISE-DRAWING tiled decode.** `Ltx2ConvVideoDecodeTiled` calls
  `Ltx2ConvVideoDecode` once per tile with the SAME `Ltx2NoiseStream*`, exactly as
  upstream calls `self.forward` per tile with one generator — so on a checkpoint with
  `timestep_conditioning = true` (`ltx2_video_vae.h:175` defaults it true) or a block with
  `inject_noise` (`:100`), the draw count and order differ between the tiled and untiled
  paths. That is upstream's behaviour, mirrored, not a local divergence; it is inert on
  the shipped `ltx-2.5-video-vae-conv` checkpoint (`timestep_conditioning: false`, no
  `inject_noise` block), and the golden fixtures disable it for exactly that reason. What
  is missing is a gate over a config that DOES draw.
* **A multi-chunk render at a SHIPPED resolution.** The new `test_ltx2_video` case chunks
  at 81 frames on the reduced fixture, which is the right shape but not the right size.
  On the real checkpoint even the 64x64/81f equivalence probe above took over ten minutes
  of single-threaded f32 reference decode, so a gated multi-chunk render at a shipped
  resolution belongs with phase L6's production-dtype arm, not here.
* **The 60 GiB attribution.** Open, with the next hypothesis named above.
* **The reference decoder's throughput.** Newly quantified (item 3); belongs with phase
  L6's production-dtype arm.
* **`memory_efficient_decode.py:1-43`** (workspace buffers, in-place chunked Conv3d,
  free-before-conv). Deliberately not ported: it is a second, independent rewrite of
  `CausalConv3d`, and the measurement says peak decode memory is ~362 MiB, so it is not
  the binding lever here.
* **One fixture header for the LTX-2.5 deterministic stream.** `test_ltx2_tiling.cpp`
  duplicates ~120 lines of `test_ltx2_vae.cpp`'s PRNG and param-role helpers rather than
  hoisting them, because two other rows are editing that file concurrently and a
  relocation is the one merge shape this protocol has been bitten by. Owed once those
  land.
