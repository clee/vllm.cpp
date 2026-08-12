// LTX-2.5 AUDIO VAE — the spectrogram decoder (audio_vae.py) and the BigVGAN
// vocoder that turns its output into samples (vocoder.py), ported 1:1 from the
// upstream `ltx_core` modules and gated against them by
// scripts/gen-ltx2-vae-goldens.py, which EXECUTES those modules at reduced
// dimensions on CPU.
//
// Everything here is f32 and batch-1, matching the oracle: the upstream BWE path
// deliberately forces the whole vocoder chain to run in float32 regardless of the
// weight dtype, because bf16 accumulation through ~108 sequential convolutions
// degrades its spectral metrics by 40-90% (vocoder.py:585-595). The audio
// decoder is small enough that the same choice costs nothing, and it keeps one
// numeric contract across the two stages.
//
// ─── SCOPE, so nothing is discovered later ───────────────────────────────────
// This is the DECODE direction only: AudioDecoder + Vocoder (+ the BWE chain).
// The ANALYSIS half (`AudioEncoder`, audio_vae.py:60-246, and the mel front-end
// `AudioProcessor`, ops.py:8-55) is what a REFERENCE AUDIO would need, and it is
// NOT ported here — it is owed, and the same is true of the video VAE's encoder
// (see ltx2_video_vae.cpp).
#include "vllm/model_executor/models/ltx2_audio_vae.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_h3.h"
#include "vt/dtype.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vllm {

const std::vector<float>& Ltx2VaeWeights::Get(const std::string& name) const {
  const auto it = tensors.find(name);
  VT_CHECK(it != tensors.end(), "ltx2 vae: missing parameter " + name);
  return it->second;
}

namespace {

// ---------------------------------------------------------------------------
// Elementwise activations
// ---------------------------------------------------------------------------

void Silu(std::vector<float>& x) {
  for (float& v : x) v = static_cast<float>(v / (1.0 + std::exp(-static_cast<double>(v))));
}

void LeakyRelu(std::vector<float>& x, double slope) {
  for (float& v : x) {
    if (v < 0.0f) v = static_cast<float>(static_cast<double>(v) * slope);
  }
}

// ---------------------------------------------------------------------------
// 2-D primitives. Feature maps are [C, H, W] with H = TIME and W = mel bins,
// which is upstream's (batch, channels, time, frequency) at batch 1.
// ---------------------------------------------------------------------------

struct Conv2dSpec {
  int64_t in_channels = 0, out_channels = 0;
  int64_t h = 0, w = 0;
  int64_t kh = 3, kw = 3;
  // F.pad order is (left, right, top, bottom) — left/right on W, top/bottom on H
  // (causal_conv_2d.py:38-47). Padding is CONSTANT ZERO, F.pad's default.
  int64_t pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0;
  int64_t stride_h = 1, stride_w = 1;
  int64_t dil_h = 1, dil_w = 1;
};

std::vector<float> Conv2d(const std::vector<float>& in, const Conv2dSpec& spec,
                          const std::vector<float>& weight, const std::vector<float>* bias,
                          int64_t* out_h, int64_t* out_w) {
  const int64_t ci = spec.in_channels, co = spec.out_channels;
  VT_CHECK(static_cast<int64_t>(in.size()) == ci * spec.h * spec.w,
           "ltx2 conv2d: input size does not match [C, H, W]");
  VT_CHECK(static_cast<int64_t>(weight.size()) == co * ci * spec.kh * spec.kw,
           "ltx2 conv2d: weight size does not match the kernel");

  const int64_t ph = spec.h + spec.pad_top + spec.pad_bottom;
  const int64_t pw = spec.w + spec.pad_left + spec.pad_right;
  std::vector<float> padded(static_cast<size_t>(ci * ph * pw), 0.0f);
  for (int64_t c = 0; c < ci; ++c) {
    for (int64_t y = 0; y < spec.h; ++y) {
      for (int64_t x = 0; x < spec.w; ++x) {
        padded[static_cast<size_t>((c * ph + y + spec.pad_top) * pw + x + spec.pad_left)] =
            in[static_cast<size_t>((c * spec.h + y) * spec.w + x)];
      }
    }
  }

  const int64_t eff_h = spec.dil_h * (spec.kh - 1) + 1;
  const int64_t eff_w = spec.dil_w * (spec.kw - 1) + 1;
  const int64_t oh = (ph - eff_h) / spec.stride_h + 1;
  const int64_t ow = (pw - eff_w) / spec.stride_w + 1;
  VT_CHECK(oh > 0 && ow > 0, "ltx2 conv2d: empty output");

  std::vector<float> out(static_cast<size_t>(co * oh * ow));
  for (int64_t oc = 0; oc < co; ++oc) {
    for (int64_t y = 0; y < oh; ++y) {
      for (int64_t x = 0; x < ow; ++x) {
        double acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0;
        for (int64_t ic = 0; ic < ci; ++ic) {
          for (int64_t a = 0; a < spec.kh; ++a) {
            const int64_t sy = y * spec.stride_h + a * spec.dil_h;
            for (int64_t b = 0; b < spec.kw; ++b) {
              const int64_t sx = x * spec.stride_w + b * spec.dil_w;
              acc += static_cast<double>(padded[static_cast<size_t>((ic * ph + sy) * pw + sx)]) *
                     static_cast<double>(
                         weight[static_cast<size_t>(((oc * ci + ic) * spec.kh + a) * spec.kw + b)]);
            }
          }
        }
        out[static_cast<size_t>((oc * oh + y) * ow + x)] = static_cast<float>(acc);
      }
    }
  }
  *out_h = oh;
  *out_w = ow;
  return out;
}

// CausalConv2d's padding rule (causal_conv_2d.py:34-47). `kHeight` is the shipped
// default and puts EVERY temporal pad on top, so a frame never sees the future.
void ApplyCausalPadding(Conv2dSpec& spec, Ltx2CausalityAxis axis) {
  const int64_t pad_h = (spec.kh - 1) * spec.dil_h;
  const int64_t pad_w = (spec.kw - 1) * spec.dil_w;
  switch (axis) {
    case Ltx2CausalityAxis::kNone:
      spec.pad_left = pad_w / 2;
      spec.pad_right = pad_w - pad_w / 2;
      spec.pad_top = pad_h / 2;
      spec.pad_bottom = pad_h - pad_h / 2;
      break;
    case Ltx2CausalityAxis::kWidth:
    case Ltx2CausalityAxis::kWidthCompatibility:
      spec.pad_left = pad_w;
      spec.pad_right = 0;
      spec.pad_top = pad_h / 2;
      spec.pad_bottom = pad_h - pad_h / 2;
      break;
    case Ltx2CausalityAxis::kHeight:
      spec.pad_left = pad_w / 2;
      spec.pad_right = pad_w - pad_w / 2;
      spec.pad_top = pad_h;
      spec.pad_bottom = 0;
      break;
  }
}

// PixelNorm (normalization.py:14-40) over the CHANNEL axis, per (h, w) location.
// Reached through build_normalization_layer, which passes eps=1e-6 — a different
// value from the video VAE's bare PixelNorm() default of 1e-8.
void PixelNorm(std::vector<float>& x, int64_t channels, int64_t spatial, double eps) {
  for (int64_t i = 0; i < spatial; ++i) {
    double mean_sq = 0.0;
    for (int64_t c = 0; c < channels; ++c) {
      const double v = x[static_cast<size_t>(c * spatial + i)];
      mean_sq += v * v;
    }
    mean_sq /= static_cast<double>(channels);
    const double inv = 1.0 / std::sqrt(mean_sq + eps);
    for (int64_t c = 0; c < channels; ++c) {
      x[static_cast<size_t>(c * spatial + i)] =
          static_cast<float>(x[static_cast<size_t>(c * spatial + i)] * inv);
    }
  }
}

// The audio VAE's normalization switch (normalization.py:43-59). The GroupNorm arm
// reuses the shared 3-D-shaped implementation: its statistics span the group's
// channels and every non-channel element, which is exactly torch's rule at any rank.
void ApplyNorm(const Ltx2AudioDecoderConfig& config, std::vector<float>& x, int64_t channels,
               int64_t spatial, const Ltx2VaeWeights& weights, const std::string& prefix) {
  if (config.norm_type == Ltx2NormType::kPixel) {
    PixelNorm(x, channels, spatial, config.pixel_norm_eps);
    return;
  }
  MiniMaxH3GroupNorm3d(x, channels, spatial, config.num_groups, weights.Get(prefix + ".weight"),
                       weights.Get(prefix + ".bias"), config.norm_eps);
}

// ---------------------------------------------------------------------------
// 1-D primitives. Signals are [C, T].
//
// These are NOT reimplemented here. LTX-2.5's vocoder and MiniMax-H3's are the
// same BigVGAN lineage, so Conv1d, ConvTranspose1d, the replicate/zero pad, the
// Snake(Beta) nonlinearity and the alias-free `Activation1d` are ONE
// implementation, published from minimax_h3.h and gated by BOTH suites. See that
// header: a second copy of the alias-free trim geometry is the duplicate that
// goes wrong quietly, because each copy keeps its own green gate while the two
// audio VAEs drift apart.
// ---------------------------------------------------------------------------

// get_padding (vocoder.py:15-16) and torch's `padding="same"` split, which the
// legacy ResBlock1 arm uses (resnet.py:22). Both are symmetric for odd kernels.
int64_t GetPadding(int64_t kernel_size, int64_t dilation) {
  return (kernel_size * dilation - dilation) / 2;
}

// ---------------------------------------------------------------------------
// AudioDecoder pieces
// ---------------------------------------------------------------------------

struct AudioMap {
  std::vector<float> data;
  int64_t channels = 0, h = 0, w = 0;
};

AudioMap CausalConv(const AudioMap& in, const Ltx2AudioDecoderConfig& config, int64_t out_channels,
                    int64_t kernel, const Ltx2VaeWeights& weights, const std::string& prefix) {
  Conv2dSpec spec;
  spec.in_channels = in.channels;
  spec.out_channels = out_channels;
  spec.h = in.h;
  spec.w = in.w;
  spec.kh = spec.kw = kernel;
  ApplyCausalPadding(spec, config.causality_axis);
  AudioMap out;
  out.channels = out_channels;
  out.data = Conv2d(in.data, spec, weights.Get(prefix + ".conv.weight"),
                    &weights.Get(prefix + ".conv.bias"), &out.h, &out.w);
  return out;
}

// ResnetBlock (resnet.py:155-176) with temb always None: the decoder builds it
// with temb_channels=0, so no temb_proj exists at all.
AudioMap ResnetBlock(const AudioMap& x, const Ltx2AudioDecoderConfig& config, int64_t out_channels,
                     const Ltx2VaeWeights& weights, const std::string& prefix) {
  VT_CHECK(config.causality_axis == Ltx2CausalityAxis::kNone ||
               config.norm_type != Ltx2NormType::kGroup,
           "ltx2 audio vae: causal ResnetBlock with GroupNorm is not supported (resnet.py:130-131)");
  const int64_t spatial = x.h * x.w;

  AudioMap h = x;
  ApplyNorm(config, h.data, h.channels, spatial, weights, prefix + ".norm1");
  Silu(h.data);
  h = CausalConv(h, config, out_channels, 3, weights, prefix + ".conv1");

  ApplyNorm(config, h.data, h.channels, h.h * h.w, weights, prefix + ".norm2");
  Silu(h.data);
  h = CausalConv(h, config, out_channels, 3, weights, prefix + ".conv2");

  AudioMap residual = x;
  if (x.channels != out_channels) {
    // nin_shortcut is a 1x1 causal conv, so its padding is zero on every side.
    residual = CausalConv(x, config, out_channels, 1, weights, prefix + ".nin_shortcut");
  }
  VT_CHECK(residual.data.size() == h.data.size(),
           "ltx2 audio vae: resnet residual and main-branch shapes must match");
  for (size_t i = 0; i < h.data.size(); ++i) h.data[i] += residual.data[i];
  return h;
}

// AttnBlock (attention.py:31-55): full self-attention over every (time, mel)
// location. It is NOT causal — upstream applies it to the whole map.
AudioMap AttnBlock(const AudioMap& x, const Ltx2AudioDecoderConfig& config,
                   const Ltx2VaeWeights& weights, const std::string& prefix) {
  const int64_t c = x.channels;
  const int64_t n = x.h * x.w;
  std::vector<float> normed = x.data;
  ApplyNorm(config, normed, c, n, weights, prefix + ".norm");

  auto project = [&](const char* leaf) {
    Conv2dSpec spec;
    spec.in_channels = c;
    spec.out_channels = c;
    spec.h = x.h;
    spec.w = x.w;
    spec.kh = spec.kw = 1;
    int64_t oh = 0, ow = 0;
    return Conv2d(normed, spec, weights.Get(prefix + "." + leaf + ".weight"),
                  &weights.Get(prefix + "." + leaf + ".bias"), &oh, &ow);
  };
  const std::vector<float> q = project("q");
  const std::vector<float> k = project("k");
  const std::vector<float> v = project("v");

  const double scale = std::pow(static_cast<double>(c), -0.5);
  std::vector<float> attended(static_cast<size_t>(c * n), 0.0f);
  std::vector<double> scores(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    double max_score = -std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < n; ++j) {
      double dot = 0.0;
      for (int64_t ch = 0; ch < c; ++ch) {
        dot += static_cast<double>(q[static_cast<size_t>(ch * n + i)]) *
               static_cast<double>(k[static_cast<size_t>(ch * n + j)]);
      }
      scores[static_cast<size_t>(j)] = dot * scale;
      max_score = std::max(max_score, scores[static_cast<size_t>(j)]);
    }
    double sum = 0.0;
    for (int64_t j = 0; j < n; ++j) {
      scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - max_score);
      sum += scores[static_cast<size_t>(j)];
    }
    for (int64_t ch = 0; ch < c; ++ch) {
      double acc = 0.0;
      for (int64_t j = 0; j < n; ++j) {
        acc += scores[static_cast<size_t>(j)] *
               static_cast<double>(v[static_cast<size_t>(ch * n + j)]);
      }
      attended[static_cast<size_t>(ch * n + i)] = static_cast<float>(acc / sum);
    }
  }

  Conv2dSpec spec;
  spec.in_channels = c;
  spec.out_channels = c;
  spec.h = x.h;
  spec.w = x.w;
  spec.kh = spec.kw = 1;
  int64_t oh = 0, ow = 0;
  std::vector<float> projected =
      Conv2d(attended, spec, weights.Get(prefix + ".proj_out.weight"),
             &weights.Get(prefix + ".proj_out.bias"), &oh, &ow);
  AudioMap out = x;
  for (size_t i = 0; i < out.data.size(); ++i) out.data[i] += projected[i];
  return out;
}

// Upsample (upsample.py:25-55): nearest 2x on BOTH axes, a causal conv, then DROP
// THE FIRST element on the causal axis — the head, not the tail, because only the
// first two interpolated elements depend on a single input element.
AudioMap Upsample(const AudioMap& x, const Ltx2AudioDecoderConfig& config,
                  const Ltx2VaeWeights& weights, const std::string& prefix) {
  AudioMap up;
  up.channels = x.channels;
  up.h = x.h * 2;
  up.w = x.w * 2;
  up.data.resize(static_cast<size_t>(up.channels * up.h * up.w));
  for (int64_t c = 0; c < up.channels; ++c) {
    for (int64_t y = 0; y < up.h; ++y) {
      for (int64_t z = 0; z < up.w; ++z) {
        up.data[static_cast<size_t>((c * up.h + y) * up.w + z)] =
            x.data[static_cast<size_t>((c * x.h + y / 2) * x.w + z / 2)];
      }
    }
  }
  AudioMap out = CausalConv(up, config, up.channels, 3, weights, prefix + ".conv");

  const bool drop_height = config.causality_axis == Ltx2CausalityAxis::kHeight;
  const bool drop_width = config.causality_axis == Ltx2CausalityAxis::kWidth;
  if (!drop_height && !drop_width) return out;

  AudioMap trimmed;
  trimmed.channels = out.channels;
  trimmed.h = drop_height ? out.h - 1 : out.h;
  trimmed.w = drop_width ? out.w - 1 : out.w;
  trimmed.data.resize(static_cast<size_t>(trimmed.channels * trimmed.h * trimmed.w));
  for (int64_t c = 0; c < trimmed.channels; ++c) {
    for (int64_t y = 0; y < trimmed.h; ++y) {
      for (int64_t z = 0; z < trimmed.w; ++z) {
        trimmed.data[static_cast<size_t>((c * trimmed.h + y) * trimmed.w + z)] =
            out.data[static_cast<size_t>((c * out.h + y + (drop_height ? 1 : 0)) * out.w + z +
                                         (drop_width ? 1 : 0))];
      }
    }
  }
  return trimmed;
}

}  // namespace

// kaiser_sinc_filter1d (vocoder.py:52-70). Identical to the BigVGAN filter the
// MiniMax-H3 audio VAE already ports, so this DELEGATES rather than standing up a
// second copy; the golden proves the shared code matches LTX's upstream too.
std::vector<float> Ltx2KaiserSincFilter1d(double cutoff, double half_width, int64_t kernel_size) {
  return MiniMaxH3KaiserSincFilter1d(cutoff, half_width, kernel_size);
}

// UpSample1d's HANN-windowed sinc (vocoder.py:116-128) — the BWE resampler's
// filter, equivalent to torchaudio.functional.resample, and NOT normalized to
// sum 1 the way the kaiser filter is.
std::vector<float> Ltx2HannSincResampleFilter1d(int64_t ratio, int64_t* kernel_size, int64_t* pad,
                                                int64_t* pad_left, int64_t* pad_right) {
  VT_CHECK(ratio >= 1, "ltx2 resampler: ratio must be positive");
  const double rolloff = 0.99;
  const int64_t lowpass_filter_width = 6;
  const int64_t width = static_cast<int64_t>(
      std::ceil(static_cast<double>(lowpass_filter_width) / rolloff));
  const int64_t size = 2 * width * ratio + 1;
  if (kernel_size != nullptr) *kernel_size = size;
  if (pad != nullptr) *pad = width;
  if (pad_left != nullptr) *pad_left = 2 * width * ratio;
  if (pad_right != nullptr) *pad_right = size - ratio;

  std::vector<float> filter(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    const double t = (static_cast<double>(i) / static_cast<double>(ratio) -
                      static_cast<double>(width)) *
                     rolloff;
    const double clamped =
        std::max(-static_cast<double>(lowpass_filter_width),
                 std::min(static_cast<double>(lowpass_filter_width), t));
    const double window =
        std::pow(std::cos(clamped * M_PI / static_cast<double>(lowpass_filter_width) / 2.0), 2.0);
    const double sinc = t == 0.0 ? 1.0 : std::sin(M_PI * t) / (M_PI * t);
    filter[static_cast<size_t>(i)] =
        static_cast<float>(sinc * window * rolloff / static_cast<double>(ratio));
  }
  return filter;
}

// AudioDecoder.forward (audio_vae.py:385-400).
Ltx2AudioSpectrogram Ltx2AudioDecoderForward(const Ltx2AudioDecoderConfig& config,
                                             const Ltx2VaeWeights& weights,
                                             const std::vector<float>& latent,
                                             int64_t latent_channels, int64_t latent_frames,
                                             int64_t latent_mel_bins) {
  // LATENT_DOWNSAMPLE_FACTOR (audio_vae.py:19) is a module constant, not config.
  constexpr int64_t kLatentDownsampleFactor = 4;
  const int64_t levels = config.num_resolutions();
  VT_CHECK(levels > 0, "ltx2 audio vae: ch_mult must not be empty");
  VT_CHECK(static_cast<int64_t>(latent.size()) == latent_channels * latent_frames * latent_mel_bins,
           "ltx2 audio vae: latent size does not match [C, T, F]");

  // --- _denormalize_latents (audio_vae.py:402-425) ---
  // The patchifier flattens (c, f) into ONE axis, so the per-channel statistics
  // are indexed by c * mel_bins + f, not by c.
  const int64_t patched_width = latent_channels * latent_mel_bins;
  const std::vector<float>& std_of_means =
      weights.Get(config.prefix + "per_channel_statistics.std-of-means");
  const std::vector<float>& mean_of_means =
      weights.Get(config.prefix + "per_channel_statistics.mean-of-means");
  VT_CHECK(static_cast<int64_t>(std_of_means.size()) == patched_width &&
               static_cast<int64_t>(mean_of_means.size()) == patched_width,
           "ltx2 audio vae: per-channel statistics must span latent channels x mel bins");

  AudioMap h;
  h.channels = latent_channels;
  h.h = latent_frames;
  h.w = latent_mel_bins;
  h.data.resize(latent.size());
  for (int64_t c = 0; c < latent_channels; ++c) {
    for (int64_t t = 0; t < latent_frames; ++t) {
      for (int64_t f = 0; f < latent_mel_bins; ++f) {
        const size_t i = static_cast<size_t>((c * latent_frames + t) * latent_mel_bins + f);
        const size_t k = static_cast<size_t>(c * latent_mel_bins + f);
        h.data[i] = static_cast<float>(static_cast<double>(latent[i]) * std_of_means[k] +
                                       mean_of_means[k]);
      }
    }
  }

  int64_t target_frames = latent_frames * kLatentDownsampleFactor;
  if (config.causality_axis != Ltx2CausalityAxis::kNone) {
    target_frames = std::max<int64_t>(target_frames - (kLatentDownsampleFactor - 1), 1);
  }
  const int64_t target_mel_bins = config.mel_bins != 0 ? config.mel_bins : latent_mel_bins;

  // --- conv_in -> mid -> up path -> norm/SiLU/conv_out ---
  const std::string p = config.prefix;
  const int64_t base = config.ch * config.ch_mult[static_cast<size_t>(levels - 1)];
  h = CausalConv(h, config, base, 3, weights, p + "conv_in");

  h = ResnetBlock(h, config, base, weights, p + "mid.block_1");
  if (config.mid_block_add_attention) h = AttnBlock(h, config, weights, p + "mid.attn_1");
  h = ResnetBlock(h, config, base, weights, p + "mid.block_2");

  int64_t curr_res = config.resolution / (int64_t{1} << (levels - 1));
  for (int64_t level = levels - 1; level >= 0; --level) {
    const std::string sp = p + "up." + std::to_string(level);
    const int64_t block_out = config.ch * config.ch_mult[static_cast<size_t>(level)];
    const bool has_attn = std::find(config.attn_resolutions.begin(), config.attn_resolutions.end(),
                                    curr_res) != config.attn_resolutions.end();
    for (int64_t i = 0; i < config.num_res_blocks + 1; ++i) {
      h = ResnetBlock(h, config, block_out, weights, sp + ".block." + std::to_string(i));
      if (has_attn) h = AttnBlock(h, config, weights, sp + ".attn." + std::to_string(i));
    }
    if (level != 0) {
      h = Upsample(h, config, weights, sp + ".upsample");
      curr_res *= 2;
    }
  }

  ApplyNorm(config, h.data, h.channels, h.h * h.w, weights, p + "norm_out");
  Silu(h.data);
  h = CausalConv(h, config, config.out_ch, 3, weights, p + "conv_out");
  // give_pre_end and tanh_out are both hardcoded False upstream (audio_vae.py:338-339).

  // --- _adjust_output_shape (audio_vae.py:427-472): crop, then zero-pad on the
  // RIGHT of the frequency axis and the BOTTOM of the time axis, then crop again.
  Ltx2AudioSpectrogram out;
  out.channels = config.out_ch;
  out.frames = target_frames;
  out.mel_bins = target_mel_bins;
  out.data.assign(static_cast<size_t>(out.channels * out.frames * out.mel_bins), 0.0f);
  const int64_t copy_c = std::min<int64_t>(h.channels, out.channels);
  const int64_t copy_t = std::min<int64_t>(h.h, out.frames);
  const int64_t copy_f = std::min<int64_t>(h.w, out.mel_bins);
  for (int64_t c = 0; c < copy_c; ++c) {
    for (int64_t t = 0; t < copy_t; ++t) {
      for (int64_t f = 0; f < copy_f; ++f) {
        out.data[static_cast<size_t>((c * out.frames + t) * out.mel_bins + f)] =
            h.data[static_cast<size_t>((c * h.h + t) * h.w + f)];
      }
    }
  }
  return out;
}

namespace {

// AMPBlock1 (vocoder.py:283-290) and ResBlock1 (resnet.py:73-80): the two residual
// stacks a Vocoder can carry.
std::vector<float> VocoderResBlock(const Ltx2VocoderConfig& config, const Ltx2VaeWeights& weights,
                                   const std::string& prefix, const MiniMaxH3AliasFreeActivation1d& act,
                                   const std::vector<float>& x, int64_t channels, int64_t length,
                                   int64_t kernel, const std::vector<int64_t>& dilations) {
  std::vector<float> current = x;
  for (size_t d = 0; d < dilations.size(); ++d) {
    const int64_t dilation = dilations[d];
    const std::string c1 = prefix + ".convs1." + std::to_string(d);
    const std::string c2 = prefix + ".convs2." + std::to_string(d);
    std::vector<float> xt;
    int64_t len = 0;

    // The two arms pad DIFFERENTLY, and the difference only shows up for even
    // kernels: AMPBlock1 passes an integer `padding=get_padding(k, d)` to
    // nn.Conv1d, which pads that many on BOTH sides (vocoder.py:245-277), while
    // ResBlock1 uses `padding="same"`, which splits the total dilation*(k-1) as
    // (total // 2, total - total // 2) (resnet.py:17-24).
    auto conv = [&](const std::string& name, const std::vector<float>& input, int64_t in_len,
                    int64_t dil) {
      int64_t left = 0, right = 0;
      if (config.amp) {
        left = right = GetPadding(kernel, dil);
      } else {
        const int64_t total = dil * (kernel - 1);
        left = total / 2;
        right = total - left;
      }
      int64_t padded_len = 0;
      const std::vector<float> padded =
          MiniMaxH3Pad1d(input, channels, in_len, left, right, /*replicate=*/false, &padded_len);
      int64_t produced = 0;
      std::vector<float> result =
          MiniMaxH3Conv1d(padded, channels, padded_len, weights.Get(name + ".weight"),
                 &weights.Get(name + ".bias"), channels, kernel, 1, dil, 1, &produced);
      VT_CHECK(produced == in_len, "ltx2 vocoder: resblock conv changed the length");
      return result;
    };

    if (config.amp) {
      const std::string a1 = prefix + ".acts1." + std::to_string(d) + ".act";
      const std::string a2 = prefix + ".acts2." + std::to_string(d) + ".act";
      const std::vector<float>* beta1 =
          config.snakebeta ? &weights.Get(a1 + ".beta") : nullptr;
      const std::vector<float>* beta2 =
          config.snakebeta ? &weights.Get(a2 + ".beta") : nullptr;
      xt = act.Apply(current, channels, length, weights.Get(a1 + ".alpha"), beta1,
                     config.snake_logscale, &len);
      VT_CHECK(len == length, "ltx2 vocoder: anti-aliased activation changed the length");
      xt = conv(c1, xt, length, dilation);
      xt = act.Apply(xt, channels, length, weights.Get(a2 + ".alpha"), beta2,
                     config.snake_logscale, &len);
      VT_CHECK(len == length, "ltx2 vocoder: anti-aliased activation changed the length");
      xt = conv(c2, xt, length, 1);
    } else {
      // resnet.py:74-79 — the legacy arm's LRELU_SLOPE is 0.1, and `padding="same"`
      // splits the total padding the same way get_padding does for odd kernels.
      xt = current;
      LeakyRelu(xt, 0.1);
      xt = conv(c1, xt, length, dilation);
      LeakyRelu(xt, 0.1);
      xt = conv(c2, xt, length, 1);
    }
    for (size_t i = 0; i < current.size(); ++i) current[i] += xt[i];
  }
  return current;
}

std::vector<float> VocoderForwardFromRows(const Ltx2VocoderConfig& config,
                                          const Ltx2VaeWeights& weights,
                                          const std::vector<float>& rows, int64_t row_count,
                                          int64_t frames, int64_t* out_samples) {
  const int64_t num_upsamples = static_cast<int64_t>(config.upsample_rates.size());
  const int64_t num_kernels = static_cast<int64_t>(config.resblock_kernel_sizes.size());
  VT_CHECK(num_upsamples > 0 && num_kernels > 0, "ltx2 vocoder: empty config");
  VT_CHECK(static_cast<int64_t>(config.upsample_kernel_sizes.size()) == num_upsamples,
           "ltx2 vocoder: upsample rates/kernels length mismatch");
  VT_CHECK(static_cast<int64_t>(config.resblock_dilation_sizes.size()) == num_kernels,
           "ltx2 vocoder: resblock kernels/dilations length mismatch");
  VT_CHECK(static_cast<int64_t>(rows.size()) == row_count * frames,
           "ltx2 vocoder: input size does not match [rows, frames]");

  const std::string p = config.prefix;
  MiniMaxH3AliasFreeActivation1d act;
  if (config.amp) act.Build();

  // conv_pre: MiniMaxH3Conv1d(128 -> upsample_initial_channel, k=7, padding=3).
  int64_t channels = config.upsample_initial_channel;
  int64_t length = 0;
  std::vector<float> x;
  {
    int64_t padded_len = 0;
    const std::vector<float> padded =
        MiniMaxH3Pad1d(rows, row_count, frames, 3, 3, /*replicate=*/false, &padded_len);
    x = MiniMaxH3Conv1d(padded, row_count, padded_len, weights.Get(p + "conv_pre.weight"),
               &weights.Get(p + "conv_pre.bias"), channels, 7, 1, 1, 1, &length);
  }

  for (int64_t i = 0; i < num_upsamples; ++i) {
    if (!config.amp) LeakyRelu(x, 0.1);  // LRELU_SLOPE (resnet.py:9)
    const int64_t stride = config.upsample_rates[static_cast<size_t>(i)];
    const int64_t kernel = config.upsample_kernel_sizes[static_cast<size_t>(i)];
    const int64_t out_channels = config.upsample_initial_channel / (int64_t{1} << (i + 1));
    const std::string up = p + "ups." + std::to_string(i);
    int64_t up_len = 0;
    x = MiniMaxH3ConvTranspose1d(x, channels, length, weights.Get(up + ".weight"),
                        &weights.Get(up + ".bias"), out_channels, kernel, stride,
                        /*padding=*/(kernel - stride) / 2, /*groups=*/1, &up_len);
    channels = out_channels;
    length = up_len;

    // Every resblock sees the SAME input, and their outputs are AVERAGED
    // (vocoder.py:426-430) — not chained.
    std::vector<double> accumulated(x.size(), 0.0);
    for (int64_t j = 0; j < num_kernels; ++j) {
      const std::string block = p + "resblocks." + std::to_string(i * num_kernels + j);
      const std::vector<float> produced = VocoderResBlock(
          config, weights, block, act, x, channels, length,
          config.resblock_kernel_sizes[static_cast<size_t>(j)],
          config.resblock_dilation_sizes[static_cast<size_t>(j)]);
      for (size_t n = 0; n < accumulated.size(); ++n) accumulated[n] += produced[n];
    }
    for (size_t n = 0; n < x.size(); ++n) {
      x[n] = static_cast<float>(accumulated[n] / static_cast<double>(num_kernels));
    }
  }

  if (config.amp) {
    const std::string a = p + "act_post.act";
    // act_post is UNCONDITIONALLY SnakeBeta, independently of `activation`:
    // `self.act_post = Activation1d(SnakeBeta(final_channels))` (vocoder.py:388),
    // inside `if self.is_amp` and taking no `activation=` argument — unlike the
    // resblocks, which do (vocoder.py:376). So the beta tensor is read here
    // whenever `amp` is set, even on the `activation="snake"` arm where every
    // resblock uses plain Snake. Gating it on `config.snakebeta` instead made this
    // one activation silently reuse ALPHA as its reciprocal scale.
    const std::vector<float>* beta = &weights.Get(a + ".beta");
    int64_t produced = 0;
    x = act.Apply(x, channels, length, weights.Get(a + ".alpha"), beta, config.snake_logscale,
                  &produced);
    VT_CHECK(produced == length, "ltx2 vocoder: act_post changed the length");
  } else {
    // nn.LeakyReLU() — torch's DEFAULT slope of 0.01, not LRELU_SLOPE
    // (vocoder.py:386).
    LeakyRelu(x, 0.01);
  }

  int64_t padded_len = 0;
  const std::vector<float> padded =
      MiniMaxH3Pad1d(x, channels, length, 3, 3, /*replicate=*/false, &padded_len);
  const std::vector<float>* post_bias =
      config.use_bias_at_final ? &weights.Get(p + "conv_post.bias") : nullptr;
  int64_t final_len = 0;
  std::vector<float> out = MiniMaxH3Conv1d(padded, channels, padded_len, weights.Get(p + "conv_post.weight"),
                                  post_bias, 2, 7, 1, 1, 1, &final_len);

  if (config.apply_final_activation) {
    for (float& value : out) {
      value = config.use_tanh_at_final ? static_cast<float>(std::tanh(value))
                                       : std::max(-1.0f, std::min(1.0f, value));
    }
  }
  if (out_samples != nullptr) *out_samples = final_len;
  return out;
}

// x.transpose(2, 3) then `b s c t -> b (s c) t` (vocoder.py:408-412). The row for
// stereo channel s and mel bin m is s * mel_bins + m; interleaving the two
// instead time-smears the audio while still producing something that plays.
std::vector<float> InterleaveStereoMel(const std::vector<float>& mel, int64_t channels,
                                       int64_t frames, int64_t mel_bins) {
  VT_CHECK(channels == 2, "ltx2 vocoder: upstream's conv_pre is stereo-only (vocoder.py:350-358)");
  std::vector<float> rows(static_cast<size_t>(channels * mel_bins * frames));
  for (int64_t s = 0; s < channels; ++s) {
    for (int64_t m = 0; m < mel_bins; ++m) {
      for (int64_t t = 0; t < frames; ++t) {
        rows[static_cast<size_t>((s * mel_bins + m) * frames + t)] =
            mel[static_cast<size_t>((s * frames + t) * mel_bins + m)];
      }
    }
  }
  return rows;
}

}  // namespace

std::vector<float> Ltx2VocoderForward(const Ltx2VocoderConfig& config,
                                      const Ltx2VaeWeights& weights, const std::vector<float>& mel,
                                      int64_t channels, int64_t frames, int64_t mel_bins,
                                      int64_t* out_samples) {
  VT_CHECK(static_cast<int64_t>(mel.size()) == channels * frames * mel_bins,
           "ltx2 vocoder: mel size does not match [C, T, mel_bins]");
  const std::vector<float> rows = InterleaveStereoMel(mel, channels, frames, mel_bins);
  return VocoderForwardFromRows(config, weights, rows, channels * mel_bins, frames, out_samples);
}

std::vector<float> Ltx2VocoderWithBweForward(const Ltx2VocoderBweConfig& config,
                                             const Ltx2VaeWeights& weights,
                                             const std::vector<float>& mel, int64_t channels,
                                             int64_t frames, int64_t mel_bins,
                                             int64_t* out_samples) {
  VT_CHECK(config.output_sampling_rate % config.input_sampling_rate == 0,
           "ltx2 bwe: the output sample rate must be a whole multiple of the input rate");
  const int64_t ratio = config.output_sampling_rate / config.input_sampling_rate;

  int64_t low_len = 0;
  std::vector<float> x =
      Ltx2VocoderForward(config.vocoder, weights, mel, channels, frames, mel_bins, &low_len);
  const int64_t out_channels = 2;
  const int64_t output_length = low_len * config.output_sampling_rate / config.input_sampling_rate;

  // Pad to a multiple of hop_length so the mel frame count is exact
  // (vocoder.py:617-619).
  int64_t padded_len = low_len;
  const int64_t remainder = low_len % config.hop_length;
  if (remainder != 0) {
    std::vector<float> grown;
    grown = MiniMaxH3Pad1d(x, out_channels, low_len, 0, config.hop_length - remainder, /*replicate=*/false,
                  &padded_len);
    x.swap(grown);
  }

  // --- MelSTFT.mel_spectrogram (vocoder.py:502-516) per waveform channel ---
  const int64_t n_freqs = config.filter_length / 2 + 1;
  const std::vector<float>& forward_basis =
      weights.Get(config.prefix + "mel_stft.stft_fn.forward_basis");
  const std::vector<float>& mel_basis = weights.Get(config.prefix + "mel_stft.mel_basis");
  VT_CHECK(static_cast<int64_t>(forward_basis.size()) == n_freqs * 2 * config.filter_length,
           "ltx2 bwe: forward_basis does not match [n_freqs*2, 1, filter_length]");
  VT_CHECK(static_cast<int64_t>(mel_basis.size()) == config.n_mel_channels * n_freqs,
           "ltx2 bwe: mel_basis does not match [n_mels, n_freqs]");

  const int64_t left_pad = std::max<int64_t>(0, config.win_length - config.hop_length);
  int64_t stft_len = 0;
  int64_t mel_frames = 0;
  std::vector<float> bwe_mel;
  for (int64_t c = 0; c < out_channels; ++c) {
    std::vector<float> channel(x.begin() + static_cast<ptrdiff_t>(c * padded_len),
                               x.begin() + static_cast<ptrdiff_t>((c + 1) * padded_len));
    int64_t padded_wave = 0;
    // CAUSAL: left-only padding, so a frame never depends on future samples
    // (vocoder.py:469-470).
    const std::vector<float> padded =
        MiniMaxH3Pad1d(channel, 1, padded_len, left_pad, 0, /*replicate=*/false, &padded_wave);
    const std::vector<float> spec =
        MiniMaxH3Conv1d(padded, 1, padded_wave, forward_basis, nullptr, n_freqs * 2, config.filter_length,
               config.hop_length, 1, 1, &stft_len);
    if (c == 0) {
      mel_frames = stft_len;
      bwe_mel.assign(static_cast<size_t>(out_channels * mel_frames * config.n_mel_channels), 0.0f);
    }
    VT_CHECK(stft_len == mel_frames, "ltx2 bwe: per-channel STFT frame counts disagree");
    for (int64_t t = 0; t < mel_frames; ++t) {
      for (int64_t m = 0; m < config.n_mel_channels; ++m) {
        double acc = 0.0;
        for (int64_t f = 0; f < n_freqs; ++f) {
          const double real = spec[static_cast<size_t>(f * stft_len + t)];
          const double imag = spec[static_cast<size_t>((n_freqs + f) * stft_len + t)];
          acc += static_cast<double>(mel_basis[static_cast<size_t>(m * n_freqs + f)]) *
                 std::sqrt(real * real + imag * imag);
        }
        // The bwe generator consumes (channel, frame, mel) — the same layout
        // Ltx2VocoderForward takes (vocoder.py:625).
        bwe_mel[static_cast<size_t>((c * mel_frames + t) * config.n_mel_channels + m)] =
            static_cast<float>(std::log(std::max(acc, kLtx2BweMelLogClamp)));
      }
    }
  }

  int64_t residual_len = 0;
  const std::vector<float> residual =
      Ltx2VocoderForward(config.bwe_generator, weights, bwe_mel, out_channels, mel_frames,
                         config.n_mel_channels, &residual_len);

  // --- the hann-sinc skip connection (vocoder.py:627) ---
  int64_t kernel_size = 0, pad = 0, pad_left = 0, pad_right = 0;
  const std::vector<float> filter =
      Ltx2HannSincResampleFilter1d(ratio, &kernel_size, &pad, &pad_left, &pad_right);
  std::vector<float> depthwise(static_cast<size_t>(out_channels * kernel_size));
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t k = 0; k < kernel_size; ++k) {
      depthwise[static_cast<size_t>(c * kernel_size + k)] = filter[static_cast<size_t>(k)];
    }
  }
  int64_t skip_padded = 0;
  const std::vector<float> skip_in =
      MiniMaxH3Pad1d(x, out_channels, padded_len, pad, pad, /*replicate=*/true, &skip_padded);
  int64_t skip_full = 0;
  std::vector<float> skip =
      MiniMaxH3ConvTranspose1d(skip_in, out_channels, skip_padded, depthwise, nullptr, out_channels,
                      kernel_size, ratio, /*padding=*/0, /*groups=*/out_channels, &skip_full);
  for (float& value : skip) value *= static_cast<float>(ratio);
  const int64_t skip_len = skip_full - pad_left - pad_right;
  VT_CHECK(skip_len == residual_len,
           "ltx2 bwe: the resampled skip and the generated residual must have equal length");

  std::vector<float> out(static_cast<size_t>(out_channels * output_length));
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t t = 0; t < output_length; ++t) {
      const double value =
          static_cast<double>(residual[static_cast<size_t>(c * residual_len + t)]) +
          static_cast<double>(skip[static_cast<size_t>(c * skip_full + pad_left + t)]);
      out[static_cast<size_t>(c * output_length + t)] =
          static_cast<float>(std::max(-1.0, std::min(1.0, value)));
    }
  }
  if (out_samples != nullptr) *out_samples = output_length;
  return out;
}

}  // namespace vllm
