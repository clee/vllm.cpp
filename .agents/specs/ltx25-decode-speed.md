# LTX25-DECODE-SPEED — where an LTX-2.5 render's hours go, and the ranked levers

Row: `LTX25-DECODE-SPEED`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)).
Issue: [#1006](https://github.com/mudler/vllm.cpp/issues/1006).

This row produces **a finding and a ranked lever list. It ships no product code.**
Each lever it names becomes its own row, with its own issue, spec and fresh
review. The reason for that split is in §8.

## Now

`SPIKE`. The source half is complete and is recorded below. The runtime half is
recorded in §1.3 with its own state; where it is unfinished the axis says so
rather than borrowing the source half's confidence.

## 0. What was asked, and what was actually wrong with the question

The dispatch asked why LTX-2.5 rendering is slow here, and named the video VAE
decode as the suspect on the strength of two records:

> *"Expect minutes, not seconds: most of a 320x192/25f render is spent
> single-threaded in the host VAE decode at 0% GPU."* — `docs/USAGE.md:873-874`
> at `332aed738`

> `Ltx2ConvVideoDecode` at 448x256/25f took **2681.02 s**, and the decode's own
> exact heap peak at that size is **361.72 MiB** —
> [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` items 2 and 3.

Both survive checking, and §1.1 records where each number came from. The
**framing** did not survive: this row was dispatched believing that the
un-ported `memory_efficient_decode.py` was the leading candidate for a missing
~59 GiB. It is not, and it cannot be — §4 shows the arithmetic that closes that
hypothesis rather than leaving it open. The 59 GiB and the 2681 s are **two
unrelated defects** that happen to be adjacent in the log, and treating them as
one problem is what kept both open.

## 1. Provenance of every number this row rests on

AGENTS.md's rule that a number quoted often becomes treated as measured applies
to this row's own inputs. Each is traced to its origin before it ranks anything.

### 1.1 Numbers that survive

| Number | Origin | Status |
|---|---|---|
| decode wall 2681.02 s @ 448x256/25f | [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 2, exact-heap-accounted probe on the shipped conv VAE | MEASURED |
| decode heap peak 361.72 MiB, largest single allocation 99.20 MiB | same table, exact `operator new` accounting; the analytic ladder predicted the largest allocation to 3 s.f. at two scales | MEASURED |
| auto tiling resolves to 1 tile / 1 chunk at 448x256/25f | same spec, `kLtx2AutoCases` golden, executed at the pin and asserted in `test_ltx2_tiling` | MEASURED |
| ~7.25 TFLOP of dense 3x3x3 convolution in one 448x256/25f decode | derived this row from the LTX-2.5 conv VAE config read out of the checkpoint header, over 42 convs | COMPUTED |
| ~3.5 TMAC for the same decode | [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 3, derived independently | COMPUTED |

The last two are the same quantity derived twice by two agents from two
starting points (3.5 TMAC = 7.0 TFLOP against 7.25 TFLOP, a 3.6% spread from
rounding the block ladder). Two independent derivations agreeing is the reason
this row is willing to rank a lever on a computed number.

### 1.2 A number that does NOT survive as stated

`docs/USAGE.md:873-874` says a 448x256/25f render *"loses about 59 GB in 24
seconds inside the decode"*. The **59 GiB** is real and is visible in every
render log. **"Inside the decode" is not established by anything in the tree**,
and [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 3 already
said so in as many words — *"the reported '24 seconds' of memory fall cannot
have been a completed decode"* — because the decode at that size takes 2681 s,
not 24. The doc sentence attributes to the decode a fall the same repository had
already shown the decode cannot have caused. §4 replaces the attribution.

### 1.3 What this row measured itself

A probe was queued on `dgx.casa` behind `$HOME/gpu.lock` (never jumped; it
waited out a `llama-imatrix` holder and its own 85 GiB sustained-headroom
guard). It samples GPU utilization, GPU clock, `MemAvailable`, the render
process's `VmRSS` / `Anonymous` / `utime` / `Threads`, the CUDA compute-app
footprint, and the written frame count onto **one clock**, so the phases can be
cut against each other rather than inferred.

Host: `kairos-17dd`, GB10, driver `580.173.02`, `clocks.max.sm` 3003 MHz,
persistence mode **Disabled**, boot id `03717c9d-63c8-4652-a8fe-a63d012c5718`,
20 cores. Build `0e1bee42f`, CUDA on, arch `121a`, run in `vllmcpp-build:gb10`.

Rung 1 is 448x256/25f with a watchdog that kills the render below 40 GiB
`MemAvailable` — the diagnostic is the fall, not a finished render. Rung 2 is
320x192/25f, which completes.

**Per the benchmarking guide, nothing here is a throughput ratio.** There is no
denominator (§7), so these are one-sided phase attributions of our own engine.
They say where our time and bytes go. They do not say what the gap to a
reference is, and this row does not claim one.

### 1.4 Evidence that already existed and had not been read as evidence

The completed 320x192/**49f** render of 2026-08-15
(`~/work/ltx25-e2e/render8-console.log` on `dgx.casa`) sampled `MemAvailable`
and 1-minute load average every 2 minutes for its whole 8598 s. Nobody had used
it to attribute a phase. It does:

* `LOCK_ACQUIRED 18:12:52Z avail=115 GiB`; frames written 20:29; audio at 20:36;
  `EXIT=127 ELAPSED=8598s` (127 is `ffmpeg` absent from the image — the render
  itself completed, 49 frames on disk).
* `MemAvailable` falls 115 -> 43 GiB over the first ~10 minutes and **never
  returns**. That is ~72 GiB acquired during load and held for the whole run.
* From 18:22 to 20:34 — **~2 h 07 m of a 2 h 23 m render, about 89% of its
  wall** — `MemAvailable` is flat at 43-46 GiB and the **1-minute load average
  sits at 1.0-1.3 on a 20-core box**, with two brief excursions (8.66, 12.36) at
  what are evidently phase boundaries.

A sustained load of ~1.1 for two hours is one runnable thread. That is a
measured, pre-existing observation of the single-threading, independent of this
row's probe and of `docs/USAGE.md`.

**What load alone cannot decide** is whether that one thread is computing on the
host or blocking on a GPU. Both look like load 1.0. Separating them is exactly
what rung 2's GPU-utilization column is for, and it is why this row queued for
the GPU rather than quoting the doc.

## 2. What the oracles actually do

Six checkouts, each verified at the SHA the dispatch named. `git status
--porcelain` was clean on five; vLLM had 6 deleted files (`build_rust.sh`,
`requirements/build/*.txt`) and its `vllm/` source tree is clean.

| Oracle | SHA | Implements LTX-2.5? |
|---|---|---|
| Lightricks LTX-2 | `fd4ded7f2` | **yes — the reference implementation** |
| `diffusers` | `3a2f35d4e` | **yes — both LTX-2.5 decode arms** |
| SGLang | `f63458b5b` | no — LTX-2 and LTX-2.3 only |
| vLLM-Omni | `a4ea67a21` | no — recipes stop at 2.3 |
| vLLM | `555967922` | no — no LTX, no diffusion VAE decode at all |
| SGLang-Omni | `748a0b437` | no — no LTX, no video VAE, no video model |

Two of those cells contradict what this campaign has been recording, and both
contradictions matter. They are §2.2 and §2.6.

### 2.1 Lightricks LTX-2 — the primary reference

Paths are relative to `packages/ltx-core/src/ltx_core/` unless stated.

**Device: GPU, decided at build time, not at call time.** The decoder is
constructed directly onto a device —
`packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1139`
(`decoder = self._decoder_builder.build(device=self._device, dtype=build_dtype).eval()`),
`loader/single_gpu_model_builder.py:267-288` with `:273` defaulting the device
to CUDA, and `devices.py:29-39` resolving CUDA -> MPS -> CPU. There is no
`.cuda()` on the decode path because placement already happened.
*Positive control for that negative:* `grep -rn "\.cuda()" packages/*/src`
returns 4 hits, all in quantization weight prep
(`quantization/blockwise/_impl.py:129,297`,
`ltx_kernels/blockwise/linear.py:193,263`), and `grep -rn "to(device="` returns
90 — the search reaches the files.

**Ops: plain `torch.nn.Conv3d`.** `CausalConv3d` at
`model/video_vae/convolution.py:266`, its `nn.Conv3d` built at `:292-302` with
`padding=(0, H//2, W//2)`, and the **single conv call site for the entire
decoder** at `convolution.py:312`. Temporal padding is done by hand at
`:306-307` (causal) / `:309-311` (non-causal). No Triton, no CUDA extension —
`packages/ltx-kernels/src/ltx_kernels/vae/` is entirely fused neighborhood
attention for the *diffusion* decoder.

**Tiled or whole-tensor: both exist, and at our size upstream runs whole-tensor.**
`decode_video` (`model/video_vae/conv_video_decoder.py:486-506`) selects
`tiled_decode` (`:383`) when a tiling config is present, else one whole-tensor
`self(latent)` at `:504-506`. Pipelines default to `AUTO_TILING`
(`ltx_pipelines/distilled.py:197`, `:352`; `tiling.py:859`), which resolves
through `ltx_pipelines/utils/helpers.py:119-146` -> `:75-116` to the conv-VAE
defaults at `helpers.py:62-63`: long-side tile 768 / overlap 64, frames tile 80
/ overlap 24. With `VIDEO_SCALE_FACTORS = (8, 32, 32)` (`types.py:33`), a
448x256/25f request is 4 latent frames and a 768/448-px spatial envelope — under
every threshold, so `split_by_size` returns the untiled interval
(`tiling.py:199-200`) and `split_temporal_causal` short-circuits identically
(`tiling.py:239-240`). **One tile, one chunk.**

That is the same resolution our own `kLtx2AutoCases` golden records
([`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 1). Upstream
and this port agree that tiling is inert at the failing size, so **tiling is not
a lever for it** on either side.

**Dtype: bf16 end to end, no autocast, no f32 promotion.** `distilled.py:109`
sets `self.dtype = torch.bfloat16`; `blocks.py:1136-1139` casts the latent and
builds at that dtype; the decoder follows its own weights at
`conv_video_decoder.py:282-284` (`output_dtype = sample.dtype`;
`weights_dtype = next(self.parameters()).dtype`; `sample = sample.to(weights_dtype)`).
The only f32 override in the family is HDR
(`ltx_pipelines/utils/media_io/color_config.py:64-66`). `PixelNorm.forward`
(`model/common/normalization.py:32-40`) computes `mean(x**2)` **in the
activation dtype** — it does not promote.
*Positive control:* `grep -rn "autocast" packages/` returns 11 hits, all in
`audio_vae/vocoder.py:525,589-608`, `blocks.py:1184-1185`, and two comments in
the DiffVAE transformer. Zero in any conv-decoder file.

**Memory format: `channels_last_3d` (NDHWC), for weights AND activations.**
`memory_efficient_decode.py:617-627` (`CHANNELS_LAST_3D_WEIGHTS`) stores every
5-D conv weight as `channels_last_3d` at load; `:655-656` and every workspace
allocation (`:167`, `:226`, `:389`, `:442`, `:492`) pass `memory_format=`. The
plain `forward` is NCDHW. This selects a different cuDNN 3-D convolution kernel
family, and it is precisely the class of difference AGENTS.md says a token gate
cannot see.

**`memory_efficient_decode.py`, read in full (683 lines).** Its docstring
enumerates four optimizations at `:1-20`, and the module is **on by default** —
`memory_efficient: bool = True` at `blocks.py:1059`, its sole caller
`blocks.py:1090-1095` gated only on `memory_efficient and not diffusion_vae`,
and no pipeline passes `False`.

1. **Workspace buffers** (`:4-7`): one `[B, C, T+2, H, W]` tensor per block
   holds real data in `[1:-1]` with replicate padding written into `[0]` and
   `[-1]` (`_pad_workspace_temporal`, `:108-114`). This replaces
   `CausalConv3d`'s per-call `repeat` + `torch.concatenate`
   (`convolution.py:306-311`), which allocates a fresh `T+2` tensor **per conv**.
2. **In-place temporally-chunked Conv3d**, non-causal only (`:122-204`), writing
   back over `workspace[:, :C_out, 1:-1]`; aliasing is made safe by chunking
   with 1-frame boundary save/restore (`:159-169`, `:190-204`). Chunk size is
   searched `16 -> 3` by `_find_temporal_split_size` (`:91-105`) and engages only
   when `total_frames > 16` (`:151-156`).
3. **In-place norm / affine / SiLU** (`_pixel_norm_inplace` `:256-259`,
   `F.silu(..., inplace=True)` at `:318`, `:339`, `:519`).
4. **Free-before-conv** (`_causal_pad_free_and_conv` `:234-248`, with `del x` at
   `:245`), so input and output never coexist — `:15-16`.

Dispatch is `_memory_efficient_forward` (`:541-609`), per-block at `:592-607`.

**Streaming: a generator, at temporal-tile granularity, with no host offload.**
`tiled_decode` yields per temporal group and keeps at most the current buffer
plus the previous chunk alive (`conv_video_decoder.py:466-471`, rotate at
`:474-476`, drain at `:479-484`); the pipeline hands the iterator straight to
the encoder (`blocks.py:1140-1144`, `distilled.py:315`,
`ltx_pipelines/utils/media_io/encode.py:120-121`, `:175-178`). At 25 frames
there is one group and one yield, so streaming buys nothing at our size.
*Positive control for "no VAE offload":* `grep -rn "offload"` over
`packages/ltx-core/src packages/ltx-pipelines/src` returns **82** hits — the term
is live, `--offload {none,cpu,disk}` is documented at
`ltx_pipelines/docs/installation.md:90` — and filtering those 82 for `vae|decod`
returns **zero**. Offload is transformer-weights-only upstream.

**The LTX-2.5 conv VAE config, read from the shipped checkpoint header** (range
read of `ltx-2.5-video-vae-conv-bf16.safetensors`; the repo ships no copy — a
`grep -rn "decoder_blocks"` returns only code references at
`model_configurator.py:87` and `conv_video_decoder.py:43,52,166,181,192,207,222`):
`dims 3`, `latent_channels 128`, `patch_size 4`, `norm_layer pixel_norm`,
`decoder_base_channels 128`, **`causal_decoder FALSE`**,
**`timestep_conditioning FALSE`**, `spatial_padding_mode "zeros"`, blocks
`res_x(4), compress_space(m2), res_x(6), compress_time(m2), res_x(4),
compress_all(m1), res_x(2), compress_all(m2), res_x(2)`.

Two consequences follow from `model_configurator.py:90-91`: the **non-causal**
symmetric-pad branch (`convolution.py:309-311`) is the one that runs, and there
is **no decode noise injection, no `decode_timestep`, no AdaLN** in the shipped
LTX-2.5 conv arm at all.

**LTX-2.5 ships two video VAEs, selected from checkpoint metadata, and the
diffusion one is the recommended download** — `README.md:80-81`,
`model_configurator.py:18`, `:26-34`, `:242-250`. This port implements the conv
arm and refuses the diffusion arm by name. That refusal is correct and is not
this row's business, but every comparison below is about the **conv** arm and
says so, because the two have different ops, different tiling defaults and
different memory profiles.

### 2.2 `diffusers` — implements LTX-2.5, and this campaign has been recording otherwise

This is the first of the two contradictions, and it is the one that changes the
denominator answer.

`diffusers` at `3a2f35d4e` implements **both** LTX-2.5 video decode arms:

* the convolutional arm, `AutoencoderKLLTX2Video`
  (`src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py:1025`, registered
  at `autoencoders/__init__.py:16`), and
* **the LTX-2.5 diffusion decoder**, `LTX2VideoDiffusionDecoderModel`
  (`ltx2_diffusion_decoder.py:700`, registered at `__init__.py:31`), whose
  docstring at `ltx2_diffusion_decoder.py:702` reads *"The LTX-2 diffusion video
  decoder, introduced in LTX-2.5."*, with its own pipeline
  `LTX2VideoDiffusionDecodePipeline` (`pipeline_ltx2_diffusion_decode.py:27`,
  docstring at `:29`).

Further 2.5-specific text: `ltx2_diffusion_decoder.py:477`, `transformer_ltx2.py:1118`
(`(LTX-2.5.1+)`), and `pipelines/ltx2/utils.py:272` naming LTX-2.5's Gemma-4
text encoder.

The two arms share a latent space, stated at `ltx2_diffusion_decoder.py:704-706`:
*"latents are interchangeable between the convolutional decoder and this one."*

**Conv-arm behaviour, for comparison with ours:** tiling flags default OFF —
`use_tiling = False` at `autoencoder_kl_ltx2.py:1169`,
`use_framewise_decoding = False` at `:1174`; tile minima 512/512/16 and strides
448/448/8 at `:1183-1190`. Decode dispatch is `_decode` at `:1266-1289`: framewise
at `:1278`, spatial tiling at `:1281` (requires width or height **above** the
512-px minimum), otherwise **one whole-tensor `self.decoder(z, temb, causal=causal)`
at `:1284`**. At 448x256 neither branch is taken even with tiling on. Dtype is the
single inherited pipeline dtype, bf16 in the shipped example
(`pipeline_ltx2.py:74`), with the cast running latents **to** the VAE
(`pipeline_ltx2.py:1642-1643`) and **no f32 promotion anywhere in the decoder** —
`PerChannelRMSNorm.forward` computes in the activation dtype
(`autoencoder_kl_ltx2.py:50-59`, mean at `:56`, sqrt at `:58`).
*Positive control:* grepping `\.float()|autocast|float32|torch\.float` across both
LTX autoencoder files returns exactly two hits, both the same `timestep_scale_multiplier`
scalar parameter (`autoencoder_kl_ltx2.py:967`, `autoencoder_kl_ltx.py:984`), while
the same grep hits `.float()` at `image_processor.py:203`.

The temporal chunking loop exists (`_temporal_tiled_decode` at
`autoencoder_kl_ltx2.py:1497`, loop at `:1510`) but **is dead for LTX by
default**: `enable_tiling` (`:1192`, setting `use_tiling` at `:1218`) never sets
`use_framewise_decoding`. *Positive control:* `use_framewise_decoding` has 15
hits repo-wide and three **other** VAEs do set it True
(`autoencoder_kl_hunyuan_video.py:712`, `autoencoder_kl_mochi.py:834`,
`autoencoder_kl_magvit.py:799`) — neither LTX VAE ever does.

**The one quantified memory statement in any oracle** is diffusers' own, and it
is about the *diffusion* arm at a size an order of magnitude past ours —
`ltx2_diffusion_decoder.py:319-323`: *"at 121 frames and 512x768 that is 3 x
5.67 GiB, which by itself dominates decode memory"*. Also
`ltx2_diffusion_decoder.py:297-300`, on why a neighborhood-attention mask must
never be materialized (*"a 69x64x96 stage needs 167 GiB"*), and `:768-770` /
`:797-799` on which stages tiling covers.

*Not found in diffusers:* any `memory_efficient_decode` analogue, any CPU or
threaded decode path, and any pipeline-level `enable_vae_tiling()`.
*Positive controls, respectively:* `grep -rn 'memory_efficient'` returns many
hits, all `memory_efficient_attention` (`attention.py:208`,
`attention_processor.py:361`, `:1967`); `grep -n 'def enable_'
pipeline_utils.py` returns `enable_model_cpu_offload:1195`,
`enable_sequential_cpu_offload:1313`, `enable_group_offload:1380`,
`enable_attention_slicing:2076`, `enable_freeu:2296`. diffusers assumes the
decoder runs on the accelerator; **it has no host decode to compare our 2681 s
against.**

### 2.3 SGLang — a full LTX-2/2.3 lane, but no 2.5

SGLang ships a first-class LTX video-generation lane that runs in its own GPU
CI. Registry `python/sglang/multimodal_gen/registry.py:631-640` (LTX-2, HF path
`"Lightricks/LTX-2"`) and `:641-648` (LTX-2.3); pipelines
`runtime/pipelines/ltx_2_pipeline.py:324`, `:526`, `:916`, exported at `:926`;
GPU CI cases at `test/server/gpu_cases.py:408-412`, `:714-718`, `:741-746`.

**LTX-2.5 is absent.** `ltx-2\.5|ltx_2_5|ltx2\.5|LTX25|LTX 2\.5` returns **0**
hits repo-wide. *Positive control, identical tool and path:* the same regex
family for 2.3 returns **113** hits under `python/sglang/multimodal_gen/`.

Its decode is worth recording as a design reference even so. GPU-resident —
`stages/model_specific_stages/ltx_2/decoding_av.py:71` moves latents to
`get_local_torch_device()`, all under `torch.autocast` at `:80-84`. bf16 —
`configs/pipeline_configs/ltx_2.py:189` sets `vae_precision = "bf16"`,
overriding the base default `fp32` at `configs/pipeline_configs/base.py:206`.
Tiling **on** by default (`base.py:207`) and enabled at `decoding_av.py:86-87`,
but the spatial gate at `runtime/models/vaes/ltx_2_vae.py:1908-1910` needs a
latent extent above `512 // 32 = 16` and a 448x256 latent grid is 14 x 8 — so
**SGLang also takes the single whole-tensor pass at our size**. It carries two
levers it does not use for LTX: `_temporal_tiled_decode`
(`ltx_2_vae.py:2206`), reachable at 25 frames if `use_framewise_decoding`
(`:1759`) were flipped, and a streaming `decode_chunk` with per-conv cache
(`AutoencoderKLCausalLTX2Video`, `:2287`, `:2304`) wired only into the SANA-WM
stages. It also shards the decode spatially across ranks
(`ltx_2_vae.py:1742-1745`, `:1913-1925`, `layers/parallel_conv.py:667`).

A caution for any future row: `.agents/oracles/sglang.md:33-40` records
`gateable = yes` on the strength of an **LLM serving** run. That says nothing
about whether `sglang.multimodal_gen` builds and runs a video model here. A row
that reuses the mark for the diffusion lane would be asserting a state it has
not checked.

### 2.4 vLLM-Omni — 2.3 verified as the ceiling, and the adapter disqualified

The dispatch's two claims were checked exactly and **both hold**, with one line
number to correct.

`_PIPELINE_RECIPES` is declared at `ltx2_recipes.py:161` with its entries at
**`:162-166`** and closing brace at `:167`; the version axis takes only `"2"`
and `"2.3"`, and `:170-175` raises for anything else. The dispatch's
`:162-166` is right for the entries; cite `:161-167` for the whole dict. The
component table matches — `ltx2_components.py:105-111`, raising at `:114-119`.

**No LTX-2.5 anywhere in the repo.** *Positive controls:* `LTX-2` hits abound
(`ltx2_transformer.py:171,360,539,556,585,709,790,961-962`) and `grep -rn "2\.3"`
over the LTX paths returns **41**.

Worth recording because it is a silent-wrong-answer shape: version detection
`detect_ltx_model_version` (`ltx2_components.py:141-169`) is binary — `"2.3"`
(`:150,160,164,166`) or `"2"` (`:169`) — and the fallback only logs
(`:168`). **An LTX-2.5 checkpoint on the native path is silently treated as
LTX-2**, failing later at weight load rather than being refused by name.

`DiffusersAdapterPipeline` (`pipeline_diffusers_adapter.py:54`) carries
`supports_request_batch = False` at **`:68`** and
`supports_step_execution: bool = False` at **`:69`** — both confirmed. What they
disable: all four step-execution hooks raise `NotImplementedError` at
`:153-175`, so `DiffusionEngine._resolve_execution_mode`
(`diffusion_engine.py:193-211`) cannot pick step mode; and with request batching
off it **raises unless `max_num_seqs == 1`** (`:204-210`). `forward()` at
`:181-189` is a single black-box `self._pipeline(**kwargs)` over a stock
`diffusers.DiffusionPipeline` (`:116`), with CFG parallel, sequence parallel
and caching all refused up front (`:195-232`). The campaign spec's
disqualification is upheld: this is a reference-degraded serial eager wrapper,
the exact analogue of benchmarking vLLM with `--enforce-eager`.

For the 2.0-2.3 generations vLLM-Omni's decode is GPU-resident
(`interface.py:92` states *"VAE(s) (always on GPU)"*; the offload backends pull
them back at `offloader/sequential_backend.py:234-239` and
`layerwise_backend.py:301-306`), bf16 at one pipeline dtype
(`ltx2_components.py:325`, `data.py:970-975`), whole-tensor by default
(`vae_use_tiling` False at `data.py:697-698`), and its tiling is **spatial
only** — every tile carries the full frame range
(`distributed/autoencoders/autoencoder_kl_ltx2.py:102`, `:137`), unlike its WAN
VAE which does chunk over time (`autoencoder_kl_wan.py:121`). The decoder math
is diffusers' (`autoencoder_kl_ltx2.py:7`, `:76`, `:152`); vLLM-Omni's own
contribution is the distributed tile executor. It never calls Lightricks
`ltx_core` — *positive control:* `Lightricks` returns hits
(`ltx2_transformer.py:1`, `scheduling_flow_match_euler_discrete.py:275`) while
`ltx_core|ltx_video` returns zero.

Its maintainers' own memory note, `recipes/LTX/LTX-2.md:315-318`: LTX-2 one-stage
*"peaked at about 73.5 GiB on one H200 141GB"*; LTX-2.3 *"Start on a 96GB-class
GPU or use CPU/layerwise offload."*

### 2.5 vLLM — implements nothing here, so the primary-reference rule does not bind

`grep -ril "ltx"` over the whole vLLM tree returns **zero**. *Positive control:*
the same machinery finds `Qwen3ForCausalLM` at
`vllm/model_executor/models/registry.py:196`, with 48 `Qwen` hits in that one
file. The registry has no image- or video-generation section at all. The newer
`vllm/models/` package holds only `deepseek_v32`, `deepseek_v4`, `inkling`,
`minimax_m3`.

vLLM contains VAE code but no diffusion decode: `CheersVAEDecoder`
(`models/cheers.py:223-281`) is 2-D `nn.Conv2d` only and feeds SigLIP in the
*understanding* path (`cheers.py:677-696`); `CheersVAEModel` is encoder-only
(`:284-285`); `bagel.py:546-560` and `cosmos3.py:59` only route VAE weights at
load. The one "diffusion" registry entry,
`"DiffusionGemmaForBlockDiffusion"` (`registry.py:400-402`), is a text
block-diffusion LLM.

So under AGENTS.md's *"where it implements nothing"* branch, LTX-2.5 legitimately
falls to a secondary oracle.

### 2.6 SGLang-Omni — nothing, and the campaign should stop expecting otherwise

This is the second contradiction: SGLang-Omni was carried as a live candidate
for this model class, and it is not one.

`ltx|lightricks` over the whole tree returns **4 lines, all false positives** —
base64 fragments inside `benchmarks/tts_serving/voice_upload_fixtures.py:85`,
`:105`, `:124`, `:148`. *Positive control:* the same grep over SGLang proper
returns 1885 lines across 50+ files, and within sglang-omni `Conv1d` returns 57
hits, so the tree is greppable and the pattern is right.

Its registry (`sglang_omni/models/registry.py:98`, `:136`) scans
`sglang_omni/models/`, whose complete contents are ASR, TTS, music and omni-chat
architectures. Its only VAEs are 1-D audio (`minimax_music3/dav.py:2`, `:114`);
its only `Conv3d` is a VLM patch embedding on the *input* side, which both
models replace with a Linear for speed
(`ming_omni/components/vision_encoder.py:104,122,171,342`,
`qwen3_omni/components/image_encoder.py:26,31`).

No LTX, no video VAE, no diffusion video pipeline, at any generation.

## 3. What our decode actually is

Anchors are in this tree at `332aed738`.

**It is the CPU reference arm, and it says so.** The header block at
`include/vllm/model_executor/models/ltx2_video_vae.h:47-54` and the file block
at `src/vllm/model_executor/models/ltx2_video_vae.cpp:25-49` both record that
every buffer is f32 because this is a reference arm, that upstream instead runs
the checkpoint dtype, and — `ltx2_video_vae.cpp:46-49` — that **"PHASE L6 OWES
THE PRODUCTION ARM... this file is a correctness reference, not the shipping
path, and no memory or throughput number should be taken from it."**

That annotation is honest and predates this row. What it does not say, and what
this row establishes, is that **the correctness reference is what production
runs today**: `src/vllm/multimodal/ltx2_video.cpp:3258` calls
`Ltx2VideoDecodeStreaming` on the render path, which reaches
`Ltx2ConvVideoDecode` through `ltx2_video_vae_tiled.cpp:113` and `:369`. There
is no second arm to fall back to. A file that disclaims its own throughput
number is nonetheless the file every render executes.

**The convolution is a seven-deep scalar loop nest accumulating in `double`.**
`ltx2_video_vae.cpp:161-185`: the output loops at `:161-164`, the accumulator
declared `double acc` at **`:165`**, and the multiply-accumulate at **`:170-176`**
casting **both** operands with `static_cast<double>` before the FMA, stored back
through `static_cast<float>` at `:181`.

This is not confined to the conv. `double acc` occurs at **8 sites** in that one
file (`:165`, `:201`, `:303`, `:312`, `:546`, `:570`, `:579`, `:916`) and
`static_cast<double>` at **29**. The 1x1x1 convolution `Linear3d` — which is a
plain GEMM, `[out_ch, in_ch] x [in_ch, N]` — is a scalar f64 loop at
`:190-209`, accumulator at `:201`. The attention block is three more (`:546`,
`:570`, `:579`).

No oracle does this. Lightricks computes in bf16 with no autocast and no
promotion (§2.1); diffusers the same, including inside its norms (§2.2); SGLang
sets bf16 explicitly over an fp32 base default (§2.3). **f64 accumulation on the
model path appears in no reference at all.**

**It routes through no shared op and no threadpool.** `ParallelForRows`
(`src/vt/cpu/cpu_threadpool.cpp:413`) is synchronous, and 10+ CPU kernels use it
— `cpu_conv2d.cpp:78`, `cpu_conv1d_depthwise.cpp:72`, `cpu_layernorm.cpp:53`,
`cpu_paged_attn.cpp:185`, `cpu_quant_gemm.cpp:191`, `cpu_attn_relpos.cpp:89`,
`cpu_ops.cpp:28` among them. **Zero of them are in the video VAE decode.**
Grepping `ParallelForRows|Threadpool|std::thread|omp|vt::` across
`ltx2_video_vae.cpp`, `ltx2_video_vae_tiled.cpp` and `ltx2_tiling.cpp` matches
only the substrings `complementary` and `compress`. That grep is the positive
control for its own negative: it returns >0 rows, and every row is a false
match, which is a different and stronger statement than a grep returning
nothing.

The CUDA table that does exist for LTX-2.5 —
`include/vllm/model_executor/models/ltx2_kernels.h` and
`src/vt/cuda/cuda_ltx2.cu` — is the **DiT** device-forward glue
(`vt::OpId::kLtx2`), seven ops covering AdaLN, modulate, add-gated, gate-heads,
RoPE, output-modulate and SiLU. It contains no convolution and nothing the VAE
decode reaches. **The video VAE decode has no device arm at all.**

**Memory format is NCDHW f32** (`Volume::At` at `ltx2_video_vae.cpp:73-75`
indexes `((c * t + ti) * h + hi) * w + wi`), against upstream's NDHWC bf16 fast
path (§2.1). That is 2x the bytes per element and a different kernel family, and
it is exactly the difference AGENTS.md says a token gate structurally cannot
report.

## 4. The 60 GiB — attributed away from the decode, and where it actually is

**The decode is excluded, twice over, by two independent methods.**

1. *Measured.* Exact `operator new` accounting over a real 448x256/25f decode to
   completion gives a heap peak of **361.72 MiB**
   ([`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome` item 2) — a
   factor of **170** below 60 GiB. Process RSS stayed flat at 4.9 GB while
   `MemAvailable` fell.
2. *Computed, as an upper bound.* Summing **every** intermediate the LTX-2.5
   conv decoder ever produces at 448x256/25f and assuming **nothing is ever
   freed**, in f32:

   | stage | GiB |
   |---|---|
   | `res_x(4)` @ 512, 13x32x56 | 2.010 |
   | `res_x(6)` @ 256, 25x32x56 | 2.861 |
   | `res_x(4)` @ 128, 25x64x112 | 3.814 |
   | everything else | 0.964 |
   | **total** | **9.649** |

   That is the pathological ceiling for the entire conv decode with no frees at
   all, and it is still **6x short of 59 GiB**. The realistic upstream peak is
   ~95 MiB bf16 on the workspace path, ~200-300 MiB on the plain path — which
   makes our measured 361.72 MiB the right order for a correct f32 NCDHW port
   rather than evidence of a leak.

**So `memory_efficient_decode.py` cannot be the missing 59 GiB, and this row
closes that hypothesis rather than carrying it.** The dispatch named it as the
obvious candidate; the arithmetic above is what a guess would have skipped.
Porting it remains worth doing for byte traffic and for the NDHWC memory format
it carries (§6 lever 4) — it is simply not a memory-attribution lever at this
size.

**Where the bytes actually are.** The render's own documentation already
accounts for ~68 GiB before a single decode instruction runs —
`docs/USAGE.md:862-864` at `332aed738`: *"Staging the 21.00B FP8 transformer
costs about 44 GB on a 119 GB GB10, and `--encoder` adds the text tower on top
of that — roughly 24 GB of host bf16 that stays resident, because a prompt
arrives per request."*

The completed 49-frame render (§1.4) is consistent with exactly that and with
nothing else: `MemAvailable` falls 115 -> 43 GiB, a **72 GiB** acquisition, in
the first ten minutes — during load, before any decode — and then **does not
move for two hours** while one thread computes. Weights that are staged and held
do not show as a decode allocation, do not appear in the decode's `operator new`
accounting, and are exactly the shape of *"flat process RSS while MemAvailable
fell"*.

**The hypothesis this row carries forward** is therefore: *the 59 GiB is model
residency — the staged DiT plus the resident text tower — held across a decode
that needs neither, and the 448x256/25f failure is that residency plus the
decode's own footprint crossing the 119 GiB unified pool, not a decode
allocation.*

**It is stated as a hypothesis, not a finding, and this row does not close it.**
What settles it is the rung-1 probe of §1.3: `MemAvailable`, the render
process's `VmRSS` and `Anonymous`, and the CUDA compute-app footprint sampled on
one clock across the load/denoise/decode boundary. If the fall lands in
`Anonymous` during load and the level then holds flat into the decode, the
hypothesis is confirmed and the lever is releasing the DiT and the text tower
before the decode (§6 lever 5). If the fall lands at the decode boundary
instead, the hypothesis is refuted and the next one is a CUDA or `mmap` mapping,
which is where [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome`
item 2 already pointed. Either outcome is a result; neither is a guess.

**Two things this row will not do.** It will not restate `docs/USAGE.md:873-874`'s
"inside the decode" as though it were measured (§1.2), and it will not declare
the 448x256 ceiling a limit — per AGENTS.md, an apparent ceiling is an
unresolved implementation difference, and the next traceable hypothesis is named
above.

## 5. Why the decode is single-threaded and on the host

Three separate answers, because they are three separate defects and only the
first is about threads.

**Could it use `ParallelForRows`? Yes, trivially.** The op is embarrassingly
parallel over `oc` and over `ti`: `ltx2_video_vae.cpp:161-164` is a perfectly
nested loop whose body writes one output element and reads only `padded` and
`weight`. `ParallelForRows(CurrentThreadpool(), out_channels, ...)` is the same
shape `cpu_conv2d.cpp:78` already uses for 2-D convolution in this tree. Nothing
structural prevents it.

**Does it? No, and nothing in the file was ever wired to a threadpool** (§3,
with the false-match positive control). This is not a tuning gap. The decode was
written as a scalar reference and shipped as production.

**Is there a device path that is simply not wired? No — there is no device path
to wire.** This is the sharper finding, and it is why "add threads" is the wrong
instruction. `vt::OpId::kLtx2` covers the DiT only (§3). There is no `vt::Conv3d`
op reaching this decoder, no CUDA kernel for it, and no CPU `vt::` op either.
Wiring is not the missing step; the arm does not exist.

**How much of the render this accounts for.** ~89% of a completed 320x192/49f
render's wall is a flat-memory phase at load ~1.1 on 20 cores (§1.4). Whether
that phase is host compute or GPU-blocking is what rung 2 decides, and the
answer is recorded there rather than asserted here.

**Why 2681 s is not mysterious once the arithmetic is done.** One 448x256/25f
decode is **~7.25 TFLOP** of dense 3x3x3 convolution across 42 conv calls
(§1.1). 7.25 TFLOP in 2681 s is ~2.7 GFLOP/s sustained — an entirely ordinary
figure for a naive direct 3-D convolution on one core, and one made worse by
f64 operands, which halve the achievable NEON lane width against f32 before any
blocking or SIMD is considered. **The decode is not slow because of an
algorithmic difference from upstream. It is slow because 7.25 TFLOP is being
executed one scalar f64 FMA at a time on one of twenty cores.**

## 6. The ranked levers

Ranked by expected magnitude over effort. Every estimate names its reasoning and
says plainly when it is speculative. **No estimate here is a measured speedup**,
because measuring one requires an arm that does not exist yet; they are
magnitude arguments from arithmetic and from what the oracles run.

| # | Lever | Expected magnitude | Reasoning | Upstream anchor | Size | What would prove it |
|---|---|---|---|---|---|---|
| 1 | [#1007](https://github.com/mudler/vllm.cpp/issues/1007) **Give the video VAE decode a device arm.** It has none; production runs the CPU reference. | The dominant term. 7.25 TFLOP that upstream runs on an accelerator in a fraction of a second | §5 arithmetic; every oracle is GPU-resident | `blocks.py:1139` + `single_gpu_model_builder.py:273`; `decoding_av.py:71`; `interface.py:92` | **large** — a new `vt::` conv3d op plus CUDA/CPU arms, mirroring `cuda_ltx2.cu`'s seam | end-to-end wall at 320x192/25f, same seed, same frames, byte-compared pixels against the f32 reference |
| 2 | [#1008](https://github.com/mudler/vllm.cpp/issues/1008) **Drop f64 accumulation to the checkpoint dtype, and NCDHW to NDHWC.** 8 `double acc` sites, 29 `static_cast<double>`, f32 buffers. | Large on the host arm; on a device arm it decides which cuDNN family runs | f64 appears in no oracle; NDHWC is upstream's default-on fast path | `conv_video_decoder.py:282-284`; `normalization.py:32-40`; `memory_efficient_decode.py:617-627`, `:655-656` | **small-to-medium**, and it is the cheapest large win available today | per-stage byte counters plus wall, against the existing goldens — the goldens cannot see this, so it needs its own instrument |
| 3 | [#1009](https://github.com/mudler/vllm.cpp/issues/1009) **Route the decode through `ParallelForRows`.** Exists, synchronous, used by 10+ CPU kernels, unused here. | Bounded by core count; 20 on GB10 | §5; `cpu_conv2d.cpp:78` is the same shape | none — this is a local seam, not an upstream mirror | **small** | wall at fixed thread counts, plus `max\|diff\| == 0` against the serial arm |
| 4 | [#1011](https://github.com/mudler/vllm.cpp/issues/1011) **Port `memory_efficient_decode.py`.** Workspace reuse, in-place norm/SiLU, free-before-conv, temporal conv chunking, NDHWC. On by DEFAULT upstream. | Byte traffic, not the 60 GiB — §4 closes that | `blocks.py:1059` default True; the four optimizations at `:1-20` | `memory_efficient_decode.py:1-20`, `:91-105`, `:122-204`, `:234-248`, `:541-609`, `:617-627` | **medium** — it is a second independent rewrite of `CausalConv3d` | peak-heap counter at a fixed size, against the current 361.72 MiB |
| 5 | [#1014](https://github.com/mudler/vllm.cpp/issues/1014) **Release the DiT and text tower before the decode.** ~68 GiB documented as staged and resident; ~72 GiB observed acquired and held. | Decides whether 448x256/25f completes at all | `docs/USAGE.md:862-864`; §1.4 trace | upstream offloads transformer weights (`installation.md:90`) and never the VAE | **medium** | rung 1 of §1.3 — and this lever is **conditional on that probe confirming §4's hypothesis** |
| 6 | [#1010](https://github.com/mudler/vllm.cpp/issues/1010) **Emit phase timings and peak memory from the render path.** A 2.5-hour render wrote **one** line to `run.log`. | No speedup. It is the precondition for measuring any of 1-5 | — | — | **small** | its own output |
| 7 | [#655](https://github.com/mudler/vllm.cpp/issues/655) + [#1012](https://github.com/mudler/vllm.cpp/issues/1012) **Register `ltx_core` as an oracle and install it on the gate host.** | No speedup. It is the precondition for any *ratio* (§7) | AGENTS.md oracle table; issue #655 | — | **small-to-medium** | a recorded pin plus a gateability measurement |

**Honest about lever 1's size.** It is ranked first on magnitude and is by far
the largest change. Levers [#1008](https://github.com/mudler/vllm.cpp/issues/1008) and [#1009](https://github.com/mudler/vllm.cpp/issues/1009) are cheap, compose with each other, and pay
on the host arm that exists today, so a sensible order is 6, 2, 3, then 1, with
5 gated on the probe. That is a recommendation, not a finding, and the row that
takes lever 1 should re-derive the order against whatever the probe returns.

**Speculative, and labelled so.** The *magnitudes* of levers 2 and 3 are
arithmetic (dtype width, core count) and are as solid as arithmetic gets. The
*composition* of 2 with 3 is not: a threaded f32 SIMD arm may become
memory-bound where the scalar f64 arm was ALU-bound, and the combined figure
could fall well short of the product. Nothing here predicts it, and the row that
implements them must measure the composition rather than multiply the parts.

**No ceiling is declared anywhere in this table.** Where a magnitude is unknown
it says so.

## 7. The denominator

**Today there is none, and this row will not manufacture one.**

Ruled out, each for a reason established in §2:

* **vLLM** implements no LTX and no diffusion VAE decode (§2.5). Under AGENTS.md
  this is the *"implements nothing"* branch, not a failure.
* **vLLM-Omni** stops at 2.3 (§2.4), and its only route to a 2.5 checkpoint is
  `DiffusersAdapterPipeline`, which is `supports_step_execution = False`
  (`:69`) and `supports_request_batch = False` (`:68`), forced to
  `max_num_seqs=1`, with CFG parallel, sequence parallel and caching all
  refused. That is a reference-degraded configuration and is disqualified by the
  same rule that forbids `--enforce-eager`. Worse for a *correctness* gate, its
  native path would silently mislabel a 2.5 checkpoint as LTX-2
  (`ltx2_components.py:168-169`).
* **SGLang** implements 2.0 and 2.3 but not 2.5 (§2.3, 0 hits against a 113-hit
  control).
* **SGLang-Omni** implements nothing in this class (§2.6).

That leaves two candidates, and the choice between them is a real decision this
row does not have the authority to make alone:

* **Lightricks `ltx_core`** is the reference implementation and *is already the
  oracle every LTX-2.5 gate in this repository executes against*. It is in **no**
  `.agents/oracles/` file and **not** in the AGENTS.md table. That is issue
  **#655**, and it means a stack of existing LTX-2.5 correctness gates run
  against an oracle the policy does not admit — a protocol violation no checker
  can see, because no checker knows the oracle exists.
* **`diffusers`** is **already in the AGENTS.md table, already has
  `.agents/oracles/diffusers.md`, and — §2.2 — actually implements LTX-2.5,
  both decode arms.** This campaign has been recording that no registered oracle
  carries 2.5, and that record is wrong.

**Does #655 block measurement here? Split the question, because the answer
differs by axis.**

* For a **throughput ratio** — yes, it blocks, and it is not the only blocker.
  A ratio needs a reference running the same workload in a production
  configuration. Beyond registration, `ltx_core` **is not installed on the gate
  host at all**: `dgx.casa` carries one venv,
  `~/venvs/vllm-oracle-pin-555967922`, and a search for `ltx_core` under `$HOME`
  returns nothing. A same-tool both-sides profile of LTX-2.5 is therefore not
  possible today on any oracle. That is a hard, citable blocker on every ratio
  this campaign has left `PENDING`.
* For **levers 1-4** — no, it does not block, and waiting on it would be an
  error. Those levers are justified by *upstream source* (every reference runs
  this decode on an accelerator, in checkpoint dtype, with no f64 anywhere) and
  by *our own* one-sided phase attribution. An 89%-of-wall single-threaded host
  phase is a defect against our own engine's structure; it does not need a
  denominator to be worth removing. AGENTS.md requires the denominator before a
  **parity claim**, and this row makes none.

**The recommendation, offered as a decision to be taken and not as a finding:**
register `ltx_core` per #655 *and* record that `diffusers` covers LTX-2.5, then
measure gateability of each on `dgx.casa` before any ratio is quoted. `diffusers`
is the lower-friction path — already admitted, already pinned — and `ltx_core`
is the higher-fidelity one and the one the existing correctness gates already
depend on. They are not alternatives; #655 has to be closed either way, because
the gates that already ran against `ltx_core` do not become admissible by
choosing a different oracle for a future one.

## 8. Gates, scope and stop conditions

**Scope.** This row ships a spec and issues. It writes no product code, changes
no checker, and makes no lifecycle transition, so it owes no `docs/STATUS.md`,
`docs/BENCHMARKS.md` or `docs/FEATURES.md` edit under AGENTS.md's projection
table. The `docs/USAGE.md:873-874` correction that §1.2 identifies is **owed and
deliberately not taken here** — it belongs with the row whose probe settles §4,
so the doc changes once, to something measured, rather than twice.

**Why the levers are separate rows.** Each of 1-5 needs a red-first test, an
independent fresh review, and its own upstream-anchored spec. Bundling them
would put a new CUDA op, a dtype change no golden can see, a threading change,
and a memory-lifetime change behind one review — and the second of those is
precisely the class AGENTS.md says a correctness gate cannot report on.

**Gate for any lever that follows.** Correctness first: pixels byte-compared
against the current f32 reference decode at a fixed seed and size, `max|diff|`
recorded, before any wall-clock number is accepted. A decode that is faster and
different is not a win, and the goldens as they stand **cannot** catch a dtype
that is merely too wide — the generator casts every upstream parameter to f32,
so the oracle itself runs f32 and the comparison is vacuous by construction
(`ltx2_video_vae.cpp:41-44`). Lever 2 must therefore ship its own instrument.

**Stop conditions.**

* Stop and report `NEEDS_DECISION` if the probe shows the 59 GiB is **not**
  model residency. Lever 5 dissolves and §4's next hypothesis takes over.
* Stop and report if the GPU lock never frees. The source half of this row
  stands on its own and is marked as such; an unmeasured axis that says so beats
  a guess.
* Do not implement any lever from this row. It has no implementation authority
  and no fresh review.
* Do not quote a ratio against any oracle until #655 is closed and gateability
  is measured on the gate host.

## Owed

Every issue this row filed is owned here. None is fixed in this flow, because
this row has no implementation authority and no fresh review (§8).

| Issue | Lever | State |
|---|---|---|
| [#1006](https://github.com/mudler/vllm.cpp/issues/1006) | this row: the investigation and this spec | closed by this row landing |
| [#1007](https://github.com/mudler/vllm.cpp/issues/1007) | 1 — the video VAE decode has no device arm | owed |
| [#1008](https://github.com/mudler/vllm.cpp/issues/1008) | 2 — f64 accumulation, f32 NCDHW against upstream bf16 NDHWC | owed |
| [#1009](https://github.com/mudler/vllm.cpp/issues/1009) | 3 — `ParallelForRows` unused by the decode | owed |
| [#1010](https://github.com/mudler/vllm.cpp/issues/1010) | 6 — one log line per 2.5-hour render | owed, and it should land first |
| [#1011](https://github.com/mudler/vllm.cpp/issues/1011) | 4 — `memory_efficient_decode.py`, re-ranked | owed |
| [#1012](https://github.com/mudler/vllm.cpp/issues/1012) | record: `diffusers` implements LTX-2.5 | owed |
| [#1014](https://github.com/mudler/vllm.cpp/issues/1014) | 5 — the 59 GiB, decode excluded, one hypothesis named | owed |

* **The 60 GiB attribution.** Carried forward from
  [`ltx25-tiled-decode.md`](ltx25-tiled-decode.md) `## Outcome`, now with the
  decode excluded by two independent methods (§4) and one named hypothesis plus
  the exact measurement that settles it.
* **`docs/USAGE.md:873-874`'s "inside the decode".** Not established (§1.2).
  Owed to the row that closes §4.
* **A throughput number for LTX-2.5 on any axis.** `docs/BENCHMARKS.md` carries
  one LTX line, under `## Open gaps`. It stays there, and §7 says why.
* **`memory_efficient_decode.py`.** Still unported; re-ranked by §4 as byte
  traffic and memory format rather than as the 60 GiB.
* **Gateability of `ltx_core` and of `diffusers` for LTX-2.5 on `dgx.casa`.**
  Neither is installed there today.
