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
`_scratch/probe_tiling.py` runs `TileSizeConfig.from_long_side(long_side=768/64,
frames=80/24)` — the exact auto layout `ltx_pipelines/utils/helpers.py:62-63,80-88`
builds for a Conv VAE — through `TileSizeConfig.to_splitters` (`tiling.py:797-834`) and
`ConvVideoDecoder._prepare_tiles` (`conv_video_decoder.py:359-381`) at the pinned SHA:

| request | latent (T,H,W) | resolved tile cfg | tiles | temporal groups |
|---|---|---|---|---|
| 128x128/9f | 2, 4, 4 | h 768/64, w 768/64, f 80/24 | **1** | **1** |
| 320x192/25f | 4, 6, 10 | h 448/64, w 768/64, f 80/24 | **1** | **1** |
| **448x256/25f** | **4, 8, 14** | h 448/64, w 768/64, f 80/24 | **1** | **1** |
| 896x512/25f | 4, 16, 28 | h 448/64, w 768/64, f 80/24 | 4 | 1 |
| 1280x704/121f | 16, 22, 40 | h 416/64, w 768/64, f 80/24 | 8 | 2 |
| 1920x1088/241f | 31, 34, 60 | h 448/64, w 768/64, f 80/24 | 36 | 4 |

`split_by_size` returns a single interval when `dim <= size` (`tiling.py:199-200`) and
`split_temporal_causal` short-circuits identically (`tiling.py:239-240`). At 448x256 the
latent is 8x14 against a 14x24 grid tile, and 4 latent frames against a 10-latent-frame
temporal tile. So upstream, on its own defaults, calls `self.forward` **once on the whole
volume** at this size — exactly what `ltx2_video.cpp:1477-1479` does. Running tiled
decode there would allocate one extra full-size pixel buffer and decode identically.

The temporal axis does not start chunking until **121 frames**; the spatial axes do not
start tiling until the long side exceeds 768 px (or the short side exceeds its
aspect-coupled tile). "The win at our resolution is TEMPORAL chunking" is therefore not
true at 25 frames.

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
(896x512 and up, 121 frames and up) are inside this project's v1 scope. So this row
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

### 4.1 The equivalence bound

Tiled and untiled decode are **not** bit-identical, and a round number would be a
fabricated bound. The difference is exactly: each output sample is `sum_i m_i * y_i`
where `sum_i m_i == 1` and every `y_i` comes from a decode of a *different* latent crop.
Inside a tile's interior the crop's receptive field is complete and `y_i` is the untiled
value up to f32 accumulation order; near a tile edge it is not, and the ramp is what
attenuates it. So the bound is derived, per fixture, from a measured decomposition:

* run untiled, run tiled, record `max|diff|`;
* re-run tiled with every mask forced to a one-hot (no blending, hard cut at the tile
  centre) and record `max|diff|` again — the *seam* magnitude;
* the accepted band is set from the blended number with the same headroom the existing
  VAE goldens use, and the test asserts the hard-cut number is **larger by at least an
  order of magnitude**, so the band cannot be satisfied by a broken blend.

The measured numbers and the resulting band go in `## Outcome`, not here.

## 5. The memory attribution — measured

Recorded in `## Outcome` after the run. The measurement is `_scratch/decode_mem_probe`,
which loads the real shipped conv VAE (`ltx-2.5-video-vae-conv-bf16.safetensors`, 1.45 GB
bf16) through `Ltx2LoadVaeWeights` + `Ltx2ParseConvVideoDecoderConfig` and calls
`Ltx2ConvVideoDecode` at the exact 448x256/25f latent, while a 50 ms sampler thread reads
`/proc/self/status` `VmRSS`/`VmHWM` and `/proc/meminfo` `MemAvailable`. It touches no
product code, so it cannot perturb what it measures.

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

`ACTIVE`.

## Outcome

To be filled on completion: the measured memory attribution, the derived equivalence
bound, the mutation RED/GREEN, and the largest resolution that completes.
