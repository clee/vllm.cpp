// LTX-2.5 CONV VIDEO VAE — the convolutional video decoder, and the explicit
// refusal of the diffusion one.
//
// LTX-2.5 ships TWO video decoders behind one checkpoint field
// (`config.vae._class_name`, video_vae/model_configurator.py:18-34):
//
//   "CausalVideoAutoencoder" -> ConvVideoDecoder   — ported here
//   anything else            -> NADiffusionDecoder — NOT ported, REFUSED BY NAME
//
// The diffusion decoder is a neighborhood-attention model with its own row. Per
// .agents/specs/ltx-2-5.md section 0 item 2 it is refused with a message naming
// the missing piece and NEVER silently downgraded to the conv decoder — a
// downgrade would return a lower-quality render as if it were the requested one,
// which no gate in this project can detect.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/
//   OURS                          <-  UPSTREAM
//   Ltx2ConvVideoDecode           <-  model/video_vae/conv_video_decoder.py:263-357
//   (block construction)          <-  model/video_vae/conv_video_decoder.py:61-143
//   (ResnetBlock3D)               <-  model/video_vae/resnet.py:12-186
//   (UNetMidBlock3D)              <-  model/video_vae/resnet.py:189-277
//   (CausalConv3d)                <-  model/video_vae/convolution.py:266-317
//   (DepthToSpaceUpsample)        <-  model/video_vae/sampling.py:68-123
//   (AttnBlock3D / _RMSNorm2D)    <-  model/video_vae/attention.py:11-69
//   (unpatchify)                  <-  model/video_vae/ops.py:35-60
//   (per-channel statistics)      <-  model/video_vae/ops.py:63-84
//   (PixArt timestep embedding)   <-  model/transformer/timestep_embedding.py:6-141
//   Ltx2ParseVideoDecoderKind     <-  model/video_vae/model_configurator.py:18-34
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * TEMPORAL PADDING IS A REPLICATED FIRST FRAME, NOT ZEROS. CausalConv3d
//    prepends `k_t - 1` copies of frame 0 (convolution.py:306-307). MiniMax-H3's
//    causal Conv3d zero-pads instead, so the two are NOT interchangeable even
//    though both put every temporal pad on the LEFT.
//  * TWO DIFFERENT PixelNorm EPSILONS. video_vae/resnet.py:46 and
//    conv_video_decoder.py:243 construct `PixelNorm()` with its DEFAULT eps of
//    1e-8, while the audio VAE reaches PixelNorm through
//    `build_normalization_layer`, which passes 1e-6 (normalization.py:58). Using
//    one value for both is a silent, tiny, everywhere-bias.
//  * DEPTH-TO-SPACE UNPACKS `(c p1 p2 p3)`, AND THE TEMPORAL STRIDE DROPS THE
//    FIRST FRAME (sampling.py:112-120). Getting the channel order wrong shuffles
//    pixels inside every 2x2 block; keeping the first frame shifts the whole clip.
//  * `unpatchify` DECOMPOSES CHANNELS AS `(c p r q)` WITH `h` TAKING q AND `w`
//    TAKING r (ops.py:50-58) — r and q are NOT interchangeable.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// Every buffer this header names is f32, because this is the CPU REFERENCE arm.
// Upstream runs the decoder in the CHECKPOINT's dtype instead
// (`sample.to(weights_dtype)` in, `sample.to(output_dtype)` out —
// conv_video_decoder.py:283-286, 355-356), and it has none of the float32 pin the
// audio tower carries. The bf16/NVFP4 arm that inherits the checkpoint dtype is
// owed by phase L6; see ltx2_video_vae.cpp for why no gate here can catch a dtype
// that is merely too WIDE.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"  // Ltx2VaeWeights

namespace vllm {

// video_vae/enums.py:4-6.
enum class Ltx2NormLayer { kGroupNorm, kPixelNorm };

// video_vae/enums.py:16-20. Only the modes a decoder can actually select are
// listed; `make_conv_nd` forwards the rest to torch, which this port refuses.
enum class Ltx2PaddingMode { kZeros, kReflect, kReplicate };

// Which decoder a checkpoint asks for.
enum class Ltx2VideoDecoderKind { kConv, kDiffusion };

// `_RMSNorm2D` is `F.normalize(x, dim=1) * (sqrt(C) * gamma)` (attention.py:11-30),
// so the denominator floor is torch's `F.normalize` DEFAULT eps of 1e-12 — an L2
// normalize, not a mean-square RMS, and not this project's usual rms_norm epsilon.
// Named so it can be pinned: mutation proves 1e-12 -> 0.0 leaves every golden
// green, because the reduced-dimension activations are O(1) and the floor never
// binds. It still decides whether an all-zero channel vector divides or produces
// NaN.
inline constexpr double kLtx2RmsNorm2dEps = 1e-12;

// `config.vae._class_name` -> the decoder kind, mirroring
// `_vae_class_name_from_metadata` + `VideoDecoderConfigurator.from_metadata`
// (video_vae/model_configurator.py:18-34, 242-250): the conv decoder is selected
// by the exact string "CausalVideoAutoencoder" and by an ABSENT field (upstream's
// default); anything else is the diffusion decoder.
Ltx2VideoDecoderKind Ltx2ParseVideoDecoderKind(const std::string& vae_class_name);

// One entry in `decoder_blocks`. `multiplier` 0 means "the upstream default for
// this block kind" — 2 for `res_x_y` (conv_video_decoder.py:98), 1 for every
// `compress_*` (conv_video_decoder.py:111, 120, 129).
struct Ltx2VideoDecoderBlock {
  std::string name;
  int64_t num_layers = 1;
  int64_t multiplier = 0;
  bool inject_noise = false;
  bool residual = false;
};

// ─── THE INVISIBLE-CONSTANT CLASS ────────────────────────────────────────────
// An HONEST LIMIT of these goldens, and it is a CLASS, not one instance. Any
// epsilon or floor that exists to stabilize a division is, by construction,
// invisible to a reduced-dimension parity gate: the deterministic stream produces
// O(1) activations, the term it guards never binds, and the tensor comparison
// therefore accepts any value at all — including 0.0, and including one 100x off.
// MEASURED, by mutating each in turn with EVERY golden staying green:
//
//   Ltx2ConvVideoDecoderConfig::norm_eps   1e-6 -> 1e-4   green
//   Ltx2ConvVideoDecoderConfig::pixel_norm_eps 1e-8 -> 1e-6 green
//   kLtx2BweMelLogClamp                    1e-5 -> 1e-8   green
//   kMiniMaxH3SnakeEps                     1e-9 -> 0.0    green
//   kLtx2RmsNorm2dEps                      1e-12 -> 0.0   green
//
// So every member of the class is held by a SOURCE-ANCHORED CONSTANT ASSERTION in
// tests/vllm/models/test_ltx2_vae.cpp, cited to the upstream line that sets it,
// rather than by the tensor comparison — and a constant that is added later and
// left unpinned is a new hole, not a covered one. The BWE clamp additionally gets
// a golden whose input SATURATES it, because that is the one the reduced-dimension
// stream can be pushed into reaching and the one real silence reaches in
// production.
struct Ltx2ConvVideoDecoderConfig {
  // Defaults mirror `_build_conv_video_decoder`
  // (video_vae/model_configurator.py:81-94).
  int64_t in_channels = 128;
  int64_t out_channels = 3;
  // In CHECKPOINT (encoder) order. The decoder walks it REVERSED, exactly as
  // conv_video_decoder.py:222 does.
  std::vector<Ltx2VideoDecoderBlock> decoder_blocks;
  int64_t patch_size = 4;
  Ltx2NormLayer norm_layer = Ltx2NormLayer::kPixelNorm;
  bool causal = false;
  bool timestep_conditioning = true;
  Ltx2PaddingMode spatial_padding_mode = Ltx2PaddingMode::kReflect;
  int64_t base_channels = 128;
  int64_t norm_num_groups = 32;
  double decode_noise_scale = 0.025;
  double decode_timestep = 0.05;
  // The GroupNorm arm's eps, and the one `res_x_y`'s shortcut norm3 uses.
  // `ResnetBlock3D.__init__` declares `eps: float = 1e-6` (video_vae/resnet.py:31)
  // and hands it to every nn.GroupNorm it builds (resnet.py:44, 65, 94);
  // `UNetMidBlock3D` carries the same value as `resnet_eps` (resnet.py:216).
  double norm_eps = 1e-6;
  // `PixelNorm()`'s DEFAULT (normalization.py:22), reached bare from
  // video_vae/resnet.py:46 and conv_video_decoder.py:243 — NOT the 1e-6 the audio
  // VAE gets through build_normalization_layer.
  double pixel_norm_eps = 1e-8;
  std::string prefix;
};

// The deterministic source for every `torch.randn` upstream draws, consumed in
// CALL ORDER. That is precisely the guarantee an upstream `torch.Generator`
// gives, and it is what makes the decoder's noise injection reproducible on both
// sides. A null stream means "no noise is available", which is an ERROR whenever
// the config asks for noise rather than a silent zero fill.
class Ltx2NoiseStream {
 public:
  virtual ~Ltx2NoiseStream() = default;
  virtual std::vector<float> Draw(int64_t count) = 0;
};

// A (channels, frames, height, width) clip in [-1, 1]-ish pixel space (upstream
// maps it to [0, 1] outside the decoder, conv_video_decoder.py:497-499).
struct Ltx2VideoFrames {
  int64_t channels = 0;
  int64_t frames = 0;
  int64_t height = 0;
  int64_t width = 0;
  std::vector<float> data;
};

// ConvVideoDecoder.forward at batch 1. `latent` is
// [latent_channels, latent_t, latent_h, latent_w], channel-major.
//
// `timestep` overrides `decode_timestep` when non-null (the decoder's own
// default is used otherwise, conv_video_decoder.py:304-305). `noise` must be
// non-null whenever `timestep_conditioning` is set or any block sets
// `inject_noise`.
Ltx2VideoFrames Ltx2ConvVideoDecode(const Ltx2ConvVideoDecoderConfig& config,
                                    const Ltx2VaeWeights& weights,
                                    const std::vector<float>& latent, int64_t latent_channels,
                                    int64_t latent_t, int64_t latent_h, int64_t latent_w,
                                    Ltx2NoiseStream* noise, const double* timestep = nullptr);

// The seam a caller reaches for when it holds a checkpoint rather than a decided
// kind. `kConv` forwards to Ltx2ConvVideoDecode; `kDiffusion` THROWS, naming
// NADiffusionDecoder and its missing neighborhood-attention kernel. It never
// falls back.
Ltx2VideoFrames Ltx2VideoDecode(Ltx2VideoDecoderKind kind,
                                const Ltx2ConvVideoDecoderConfig& config,
                                const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                                int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                                int64_t latent_w, Ltx2NoiseStream* noise,
                                const double* timestep = nullptr);

}  // namespace vllm
