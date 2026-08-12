// LTX-2.5 CONV VIDEO VAE — ConvVideoDecoder (conv_video_decoder.py) ported 1:1
// from upstream `ltx_core` and gated against it by
// scripts/gen-ltx2-vae-goldens.py, which EXECUTES the upstream module at reduced
// dimensions on CPU.
//
// ─── SCOPE, so nothing is discovered later ───────────────────────────────────
//  * The DIFFUSION decoder (`NADiffusionDecoder` / `DiffusionVideoDecoder`) is
//    NOT ported. Ltx2VideoDecode refuses it BY NAME and never falls back — see
//    the header, and .agents/specs/ltx-2-5.md section 0 item 2.
//  * `attn_res_x` is refused too, for a different reason: at this upstream
//    revision the block cannot be CONSTRUCTED, because `_make_decoder_block`
//    passes `attention_head_dim` to `UNetMidBlock3D`, whose __init__ does not
//    accept it (conv_video_decoder.py:85-96 vs resnet.py:210-222). Upstream
//    raises TypeError; this raises with the same reason named.
//  * `dims == 2` / `dims == (2, 1)` (Conv2d and DualConv3d, convolution.py:27-71)
//    are not ported: the decoder is built with `convolution_dimensions=3`.
//  * The ENCODER half and the TILED decode path (`tiled_decode`,
//    conv_video_decoder.py:383-484) are out of this phase and owed.
#include "vllm/model_executor/models/ltx2_video_vae.h"

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

namespace {

// A [C, T, H, W] volume at batch 1.
struct Volume {
  std::vector<float> data;
  int64_t channels = 0, t = 0, h = 0, w = 0;

  int64_t spatial() const { return t * h * w; }
  size_t At(int64_t c, int64_t ti, int64_t hi, int64_t wi) const {
    return static_cast<size_t>(((c * t + ti) * h + hi) * w + wi);
  }
};

int64_t ReflectIndex(int64_t index, int64_t size) {
  // torch's "reflect" excludes the edge sample: [a b c] -> b a b c b.
  if (size == 1) return 0;
  while (index < 0 || index >= size) {
    if (index < 0) index = -index;
    if (index >= size) index = 2 * (size - 1) - index;
  }
  return index;
}

int64_t SpatialIndex(int64_t index, int64_t size, Ltx2PaddingMode mode, bool* zero) {
  *zero = false;
  if (index >= 0 && index < size) return index;
  switch (mode) {
    case Ltx2PaddingMode::kZeros:
      *zero = true;
      return 0;
    case Ltx2PaddingMode::kReflect:
      return ReflectIndex(index, size);
    case Ltx2PaddingMode::kReplicate:
      return std::max<int64_t>(0, std::min<int64_t>(size - 1, index));
  }
  *zero = true;
  return 0;
}

// CausalConv3d (convolution.py:266-317). Two things that are NOT interchangeable
// with MiniMax-H3's causal Conv3d:
//   * the temporal pad REPLICATES FRAME 0 `k_t - 1` times (H3 pads with zeros);
//   * the non-causal branch replicates the FIRST and LAST frame `(k_t - 1) / 2`
//     times each, so the output frame count is the same either way.
// Spatial padding is `k // 2` on each side in `spatial_padding_mode`.
Volume CausalConv3d(const Volume& in, int64_t out_channels, int64_t kernel, bool causal,
                    Ltx2PaddingMode mode, const std::vector<float>& weight,
                    const std::vector<float>* bias) {
  const int64_t ci = in.channels;
  VT_CHECK(static_cast<int64_t>(in.data.size()) == ci * in.spatial(),
           "ltx2 conv3d: input size does not match [C, T, H, W]");
  VT_CHECK(static_cast<int64_t>(weight.size()) == out_channels * ci * kernel * kernel * kernel,
           "ltx2 conv3d: weight size does not match the kernel");

  const int64_t pad_front = causal ? kernel - 1 : (kernel - 1) / 2;
  const int64_t pad_back = causal ? 0 : (kernel - 1) / 2;
  const int64_t pad_spatial = kernel / 2;
  const int64_t pt = in.t + pad_front + pad_back;
  const int64_t ph = in.h + 2 * pad_spatial;
  const int64_t pw = in.w + 2 * pad_spatial;

  std::vector<float> padded(static_cast<size_t>(ci * pt * ph * pw), 0.0f);
  for (int64_t c = 0; c < ci; ++c) {
    for (int64_t ti = 0; ti < pt; ++ti) {
      // Temporal padding REPLICATES the edge frame, never zeros.
      const int64_t st = std::max<int64_t>(0, std::min<int64_t>(in.t - 1, ti - pad_front));
      for (int64_t hi = 0; hi < ph; ++hi) {
        bool zero_h = false;
        const int64_t sh = SpatialIndex(hi - pad_spatial, in.h, mode, &zero_h);
        for (int64_t wi = 0; wi < pw; ++wi) {
          bool zero_w = false;
          const int64_t sw = SpatialIndex(wi - pad_spatial, in.w, mode, &zero_w);
          if (zero_h || zero_w) continue;
          padded[static_cast<size_t>(((c * pt + ti) * ph + hi) * pw + wi)] =
              in.data[in.At(c, st, sh, sw)];
        }
      }
    }
  }

  Volume out;
  out.channels = out_channels;
  out.t = pt - kernel + 1;
  out.h = ph - kernel + 1;
  out.w = pw - kernel + 1;
  VT_CHECK(out.t > 0 && out.h > 0 && out.w > 0, "ltx2 conv3d: empty output");
  out.data.resize(static_cast<size_t>(out_channels * out.spatial()));
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t ti = 0; ti < out.t; ++ti) {
      for (int64_t hi = 0; hi < out.h; ++hi) {
        for (int64_t wi = 0; wi < out.w; ++wi) {
          double acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0;
          for (int64_t ic = 0; ic < ci; ++ic) {
            for (int64_t a = 0; a < kernel; ++a) {
              for (int64_t b = 0; b < kernel; ++b) {
                for (int64_t d = 0; d < kernel; ++d) {
                  acc += static_cast<double>(
                             padded[static_cast<size_t>(
                                 ((ic * pt + ti + a) * ph + hi + b) * pw + wi + d)]) *
                         static_cast<double>(weight[static_cast<size_t>(
                             (((oc * ci + ic) * kernel + a) * kernel + b) * kernel + d)]);
                }
              }
            }
          }
          out.data[out.At(oc, ti, hi, wi)] = static_cast<float>(acc);
        }
      }
    }
  }
  return out;
}

// make_linear_nd for dims == 3 (convolution.py:84-85): a 1x1x1 Conv3d.
Volume Linear3d(const Volume& in, int64_t out_channels, const std::vector<float>& weight,
                const std::vector<float>& bias) {
  Volume out;
  out.channels = out_channels;
  out.t = in.t;
  out.h = in.h;
  out.w = in.w;
  out.data.resize(static_cast<size_t>(out_channels * in.spatial()));
  const int64_t n = in.spatial();
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t i = 0; i < n; ++i) {
      double acc = bias[static_cast<size_t>(oc)];
      for (int64_t ic = 0; ic < in.channels; ++ic) {
        acc += static_cast<double>(in.data[static_cast<size_t>(ic * n + i)]) *
               static_cast<double>(weight[static_cast<size_t>(oc * in.channels + ic)]);
      }
      out.data[static_cast<size_t>(oc * n + i)] = static_cast<float>(acc);
    }
  }
  return out;
}

void Silu(std::vector<float>& x) {
  for (float& v : x) v = static_cast<float>(v / (1.0 + std::exp(-static_cast<double>(v))));
}

// PixelNorm() with its DEFAULT eps of 1e-8 (normalization.py:22, reached bare
// from video_vae/resnet.py:46 and conv_video_decoder.py:243) — NOT the 1e-6 the
// audio VAE gets through build_normalization_layer.
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

void ApplyNorm(const Ltx2ConvVideoDecoderConfig& config, std::vector<float>& x, int64_t channels,
               int64_t spatial, const Ltx2VaeWeights& weights, const std::string& prefix) {
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(x, channels, spatial, config.pixel_norm_eps);
    return;
  }
  MiniMaxH3GroupNorm3d(x, channels, spatial, config.norm_num_groups,
                       weights.Get(prefix + ".weight"), weights.Get(prefix + ".bias"),
                       config.norm_eps);
}

// ---------------------------------------------------------------------------
// PixArtAlphaCombinedTimestepSizeEmbeddings (timestep_embedding.py:118-141) at
// batch 1: Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0) followed
// by TimestepEmbedding(256 -> embedding_dim) with a SiLU between its two linears.
// ---------------------------------------------------------------------------
std::vector<float> TimestepEmbedding(double timestep, int64_t embedding_dim,
                                     const Ltx2VaeWeights& weights, const std::string& prefix) {
  constexpr int64_t kProjChannels = 256;
  constexpr double kMaxPeriod = 10000.0;
  const int64_t half = kProjChannels / 2;
  std::vector<double> proj(static_cast<size_t>(kProjChannels));
  for (int64_t i = 0; i < half; ++i) {
    // downscale_freq_shift = 0, so the divisor is exactly half_dim.
    const double exponent = -std::log(kMaxPeriod) * static_cast<double>(i) / static_cast<double>(half);
    const double angle = timestep * std::exp(exponent);
    // flip_sin_to_cos=True puts COS first (timestep_embedding.py:87-89).
    proj[static_cast<size_t>(i)] = std::cos(angle);
    proj[static_cast<size_t>(half + i)] = std::sin(angle);
  }

  const std::vector<float>& w1 = weights.Get(prefix + ".timestep_embedder.linear_1.weight");
  const std::vector<float>& b1 = weights.Get(prefix + ".timestep_embedder.linear_1.bias");
  const std::vector<float>& w2 = weights.Get(prefix + ".timestep_embedder.linear_2.weight");
  const std::vector<float>& b2 = weights.Get(prefix + ".timestep_embedder.linear_2.bias");
  VT_CHECK(static_cast<int64_t>(w1.size()) == embedding_dim * kProjChannels,
           "ltx2 timestep embedding: linear_1 shape does not match the embedding dim");

  std::vector<double> hidden(static_cast<size_t>(embedding_dim));
  for (int64_t o = 0; o < embedding_dim; ++o) {
    double acc = b1[static_cast<size_t>(o)];
    for (int64_t i = 0; i < kProjChannels; ++i) {
      acc += proj[static_cast<size_t>(i)] *
             static_cast<double>(w1[static_cast<size_t>(o * kProjChannels + i)]);
    }
    hidden[static_cast<size_t>(o)] = acc / (1.0 + std::exp(-acc));  // SiLU
  }
  std::vector<float> out(static_cast<size_t>(embedding_dim));
  for (int64_t o = 0; o < embedding_dim; ++o) {
    double acc = b2[static_cast<size_t>(o)];
    for (int64_t i = 0; i < embedding_dim; ++i) {
      acc += hidden[static_cast<size_t>(i)] *
             static_cast<double>(w2[static_cast<size_t>(o * embedding_dim + i)]);
    }
    out[static_cast<size_t>(o)] = static_cast<float>(acc);
  }
  return out;
}

// _feed_spatial_noise (resnet.py:104-119): ONE [H, W] draw, broadcast over batch,
// channels and TIME, scaled per channel. Drawing a full [C, T, H, W] block
// instead still yields a finite, plausible clip.
void FeedSpatialNoise(Volume& x, const std::vector<float>& per_channel_scale,
                      Ltx2NoiseStream* noise) {
  VT_CHECK(noise != nullptr,
           "ltx2 video vae: a block sets inject_noise but no noise stream was supplied");
  const std::vector<float> plane = noise->Draw(x.h * x.w);
  VT_CHECK(static_cast<int64_t>(plane.size()) == x.h * x.w,
           "ltx2 video vae: the noise stream returned the wrong element count");
  for (int64_t c = 0; c < x.channels; ++c) {
    const double scale = per_channel_scale[static_cast<size_t>(c)];
    for (int64_t ti = 0; ti < x.t; ++ti) {
      for (int64_t hi = 0; hi < x.h; ++hi) {
        for (int64_t wi = 0; wi < x.w; ++wi) {
          x.data[x.At(c, ti, hi, wi)] += static_cast<float>(
              static_cast<double>(plane[static_cast<size_t>(hi * x.w + wi)]) * scale);
        }
      }
    }
  }
}

// One ada-LN group applied in place: x * (1 + scale) + shift, with the pair taken
// from `table[row]` plus `embed[row]` (resnet.py:135-147).
void ApplyAdaLn(Volume& x, const std::vector<float>& table, const std::vector<float>& embed,
                int64_t rows, int64_t shift_row, int64_t scale_row) {
  const int64_t c = x.channels;
  VT_CHECK(static_cast<int64_t>(table.size()) == rows * c,
           "ltx2 video vae: scale_shift_table does not match the channel count");
  VT_CHECK(static_cast<int64_t>(embed.size()) == rows * c,
           "ltx2 video vae: timestep embedding does not match rows x channels");
  const int64_t n = x.spatial();
  for (int64_t ch = 0; ch < c; ++ch) {
    const double shift = static_cast<double>(table[static_cast<size_t>(shift_row * c + ch)]) +
                         static_cast<double>(embed[static_cast<size_t>(shift_row * c + ch)]);
    const double scale = static_cast<double>(table[static_cast<size_t>(scale_row * c + ch)]) +
                         static_cast<double>(embed[static_cast<size_t>(scale_row * c + ch)]);
    for (int64_t i = 0; i < n; ++i) {
      x.data[static_cast<size_t>(ch * n + i)] = static_cast<float>(
          static_cast<double>(x.data[static_cast<size_t>(ch * n + i)]) * (1.0 + scale) + shift);
    }
  }
}

// ResnetBlock3D.forward (resnet.py:121-186).
Volume ResnetBlock3d(const Ltx2ConvVideoDecoderConfig& config, const Ltx2VaeWeights& weights,
                     const std::string& prefix, const Volume& input, int64_t out_channels,
                     bool inject_noise, bool timestep_conditioning,
                     const std::vector<float>* timestep_embed, Ltx2NoiseStream* noise) {
  Volume hidden = input;
  ApplyNorm(config, hidden.data, hidden.channels, hidden.spatial(), weights, prefix + ".norm1");
  if (timestep_conditioning) {
    VT_CHECK(timestep_embed != nullptr,
             "ltx2 video vae: a timestep-conditioned block needs a timestep embedding");
    // ada_values rows are (shift1, scale1, shift2, scale2).
    ApplyAdaLn(hidden, weights.Get(prefix + ".scale_shift_table"), *timestep_embed, 4, 0, 1);
  }
  Silu(hidden.data);
  hidden = CausalConv3d(hidden, out_channels, 3, config.causal, config.spatial_padding_mode,
                        weights.Get(prefix + ".conv1.conv.weight"),
                        &weights.Get(prefix + ".conv1.conv.bias"));
  if (inject_noise) {
    FeedSpatialNoise(hidden, weights.Get(prefix + ".per_channel_scale1"), noise);
  }

  ApplyNorm(config, hidden.data, hidden.channels, hidden.spatial(), weights, prefix + ".norm2");
  if (timestep_conditioning) {
    ApplyAdaLn(hidden, weights.Get(prefix + ".scale_shift_table"), *timestep_embed, 4, 2, 3);
  }
  Silu(hidden.data);
  hidden = CausalConv3d(hidden, out_channels, 3, config.causal, config.spatial_padding_mode,
                        weights.Get(prefix + ".conv2.conv.weight"),
                        &weights.Get(prefix + ".conv2.conv.bias"));
  if (inject_noise) {
    FeedSpatialNoise(hidden, weights.Get(prefix + ".per_channel_scale2"), noise);
  }

  Volume residual = input;
  if (input.channels != out_channels) {
    // norm3 is GroupNorm with ONE group — a LayerNorm over (C, T, H, W) that
    // works in the (B, C, ...) layout without a rearrange (resnet.py:91-97).
    MiniMaxH3GroupNorm3d(residual.data, residual.channels, residual.spatial(), 1,
                         weights.Get(prefix + ".norm3.weight"), weights.Get(prefix + ".norm3.bias"),
                         config.norm_eps);
    residual = Linear3d(residual, out_channels, weights.Get(prefix + ".conv_shortcut.weight"),
                        weights.Get(prefix + ".conv_shortcut.bias"));
  }
  VT_CHECK(residual.data.size() == hidden.data.size(),
           "ltx2 video vae: resnet residual and main-branch shapes must match");
  for (size_t i = 0; i < hidden.data.size(); ++i) hidden.data[i] += residual.data[i];
  return hidden;
}

// DepthToSpaceUpsample.forward (sampling.py:93-123). The channel unpack is
// `(c p1 p2 p3)` with p1 temporal and p2/p3 spatial, and a temporal stride of 2
// DROPS THE FIRST FRAME afterwards.
Volume DepthToSpaceUpsample(const Ltx2ConvVideoDecoderConfig& config, const Ltx2VaeWeights& weights,
                            const std::string& prefix, const Volume& x, int64_t st, int64_t sh,
                            int64_t sw, int64_t reduction, bool residual) {
  const int64_t stride_product = st * sh * sw;
  const int64_t conv_out_channels = stride_product * x.channels / reduction;

  auto expand = [&](const Volume& packed) {
    Volume out;
    out.channels = packed.channels / stride_product;
    out.t = packed.t * st;
    out.h = packed.h * sh;
    out.w = packed.w * sw;
    out.data.resize(static_cast<size_t>(out.channels * out.spatial()));
    for (int64_t c = 0; c < out.channels; ++c) {
      for (int64_t p1 = 0; p1 < st; ++p1) {
        for (int64_t p2 = 0; p2 < sh; ++p2) {
          for (int64_t p3 = 0; p3 < sw; ++p3) {
            const int64_t src_c = ((c * st + p1) * sh + p2) * sw + p3;
            for (int64_t ti = 0; ti < packed.t; ++ti) {
              for (int64_t hi = 0; hi < packed.h; ++hi) {
                for (int64_t wi = 0; wi < packed.w; ++wi) {
                  out.data[out.At(c, ti * st + p1, hi * sh + p2, wi * sw + p3)] =
                      packed.data[packed.At(src_c, ti, hi, wi)];
                }
              }
            }
          }
        }
      }
    }
    return out;
  };
  auto drop_first_frame = [&](const Volume& v) {
    Volume out;
    out.channels = v.channels;
    out.t = v.t - 1;
    out.h = v.h;
    out.w = v.w;
    out.data.resize(static_cast<size_t>(out.channels * out.spatial()));
    for (int64_t c = 0; c < out.channels; ++c) {
      for (int64_t ti = 0; ti < out.t; ++ti) {
        for (int64_t hi = 0; hi < out.h; ++hi) {
          for (int64_t wi = 0; wi < out.w; ++wi) {
            out.data[out.At(c, ti, hi, wi)] = v.data[v.At(c, ti + 1, hi, wi)];
          }
        }
      }
    }
    return out;
  };

  Volume skip;
  if (residual) {
    // The residual expands the INPUT itself and then repeats it up to the output
    // width (sampling.py:98-110).
    Volume expanded = expand(x);
    const int64_t repeat = stride_product / reduction;
    Volume repeated;
    repeated.channels = expanded.channels * repeat;
    repeated.t = expanded.t;
    repeated.h = expanded.h;
    repeated.w = expanded.w;
    repeated.data.resize(static_cast<size_t>(repeated.channels * repeated.spatial()));
    for (int64_t r = 0; r < repeat; ++r) {
      std::copy(expanded.data.begin(), expanded.data.end(),
                repeated.data.begin() +
                    static_cast<ptrdiff_t>(r * expanded.channels * expanded.spatial()));
    }
    skip = st == 2 ? drop_first_frame(repeated) : repeated;
  }

  Volume packed = CausalConv3d(x, conv_out_channels, 3, config.causal, config.spatial_padding_mode,
                               weights.Get(prefix + ".conv.conv.weight"),
                               &weights.Get(prefix + ".conv.conv.bias"));
  Volume out = expand(packed);
  if (st == 2) out = drop_first_frame(out);
  if (residual) {
    VT_CHECK(skip.data.size() == out.data.size(),
             "ltx2 video vae: depth-to-space residual and main-branch shapes must match");
    for (size_t i = 0; i < out.data.size(); ++i) out.data[i] += skip.data[i];
  }
  return out;
}

// AttnBlock3D.forward (attention.py:58-69): SINGLE-HEAD spatial self-attention
// PER FRAME, with frames folded into the batch — there is deliberately no
// cross-frame interaction, so this block does not break temporal causality.
Volume AttnBlock3d(const Ltx2VaeWeights& weights, const std::string& prefix, const Volume& x) {
  const int64_t c = x.channels;
  const int64_t n = x.h * x.w;
  const std::vector<float>& gamma = weights.Get(prefix + ".norm.gamma");
  const std::vector<float>& qkv_w = weights.Get(prefix + ".to_qkv.weight");
  const std::vector<float>& qkv_b = weights.Get(prefix + ".to_qkv.bias");
  const std::vector<float>& proj_w = weights.Get(prefix + ".proj.weight");
  const std::vector<float>& proj_b = weights.Get(prefix + ".proj.bias");
  const double norm_scale = std::sqrt(static_cast<double>(c));
  const double attn_scale = 1.0 / std::sqrt(static_cast<double>(c));

  Volume out = x;
  std::vector<double> normed(static_cast<size_t>(c * n));
  std::vector<double> q(static_cast<size_t>(c * n)), k(static_cast<size_t>(c * n)),
      v(static_cast<size_t>(c * n));
  std::vector<double> scores(static_cast<size_t>(n));
  std::vector<double> attended(static_cast<size_t>(c * n));

  for (int64_t frame = 0; frame < x.t; ++frame) {
    // _RMSNorm2D: F.normalize(x, dim=1) * (sqrt(C) * gamma) — an L2 normalize with
    // torch's 1e-12 floor, not a mean-square RMS.
    for (int64_t i = 0; i < n; ++i) {
      double sum_sq = 0.0;
      for (int64_t ch = 0; ch < c; ++ch) {
        const double value = x.data[x.At(ch, frame, i / x.w, i % x.w)];
        sum_sq += value * value;
      }
      const double inv = 1.0 / std::max(std::sqrt(sum_sq), 1e-12);
      for (int64_t ch = 0; ch < c; ++ch) {
        normed[static_cast<size_t>(ch * n + i)] =
            x.data[x.At(ch, frame, i / x.w, i % x.w)] * inv * norm_scale *
            static_cast<double>(gamma[static_cast<size_t>(ch)]);
      }
    }
    // to_qkv is a 1x1 Conv2d emitting [q | k | v] along the channel axis, and the
    // rearrange to tokens keeps that split on the LAST axis (attention.py:63-64).
    for (int64_t oc = 0; oc < 3 * c; ++oc) {
      std::vector<double>& dst = oc < c ? q : (oc < 2 * c ? k : v);
      const int64_t row = oc % c;
      for (int64_t i = 0; i < n; ++i) {
        double acc = qkv_b[static_cast<size_t>(oc)];
        for (int64_t ic = 0; ic < c; ++ic) {
          acc += normed[static_cast<size_t>(ic * n + i)] *
                 static_cast<double>(qkv_w[static_cast<size_t>(oc * c + ic)]);
        }
        dst[static_cast<size_t>(row * n + i)] = acc;
      }
    }
    for (int64_t i = 0; i < n; ++i) {
      double max_score = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j < n; ++j) {
        double dot = 0.0;
        for (int64_t ch = 0; ch < c; ++ch) {
          dot += q[static_cast<size_t>(ch * n + i)] * k[static_cast<size_t>(ch * n + j)];
        }
        scores[static_cast<size_t>(j)] = dot * attn_scale;
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
          acc += scores[static_cast<size_t>(j)] * v[static_cast<size_t>(ch * n + j)];
        }
        attended[static_cast<size_t>(ch * n + i)] = acc / sum;
      }
    }
    for (int64_t oc = 0; oc < c; ++oc) {
      for (int64_t i = 0; i < n; ++i) {
        double acc = proj_b[static_cast<size_t>(oc)];
        for (int64_t ic = 0; ic < c; ++ic) {
          acc += attended[static_cast<size_t>(ic * n + i)] *
                 static_cast<double>(proj_w[static_cast<size_t>(oc * c + ic)]);
        }
        out.data[out.At(oc, frame, i / x.w, i % x.w)] += static_cast<float>(acc);
      }
    }
  }
  return out;
}

}  // namespace

Ltx2VideoDecoderKind Ltx2ParseVideoDecoderKind(const std::string& vae_class_name) {
  // model_configurator.py:18-34: the conv decoder is the DEFAULT when the field is
  // absent, and is otherwise selected by the exact class name.
  if (vae_class_name.empty() || vae_class_name == "CausalVideoAutoencoder") {
    return Ltx2VideoDecoderKind::kConv;
  }
  return Ltx2VideoDecoderKind::kDiffusion;
}

Ltx2VideoFrames Ltx2ConvVideoDecode(const Ltx2ConvVideoDecoderConfig& config,
                                    const Ltx2VaeWeights& weights,
                                    const std::vector<float>& latent, int64_t latent_channels,
                                    int64_t latent_t, int64_t latent_h, int64_t latent_w,
                                    Ltx2NoiseStream* noise, const double* timestep) {
  VT_CHECK(latent_channels == config.in_channels,
           "ltx2 video vae: latent channel count does not match in_channels");
  VT_CHECK(static_cast<int64_t>(latent.size()) == latent_channels * latent_t * latent_h * latent_w,
           "ltx2 video vae: latent size does not match [C, T, H, W]");
  const std::string p = config.prefix;

  Volume x;
  x.channels = latent_channels;
  x.t = latent_t;
  x.h = latent_h;
  x.w = latent_w;
  x.data = latent;

  // --- noise + denormalize (conv_video_decoder.py:286-301) ---
  if (config.timestep_conditioning) {
    VT_CHECK(noise != nullptr,
             "ltx2 video vae: timestep conditioning injects noise but no noise stream was supplied");
    const std::vector<float> drawn = noise->Draw(static_cast<int64_t>(x.data.size()));
    VT_CHECK(drawn.size() == x.data.size(),
             "ltx2 video vae: the noise stream returned the wrong element count");
    for (size_t i = 0; i < x.data.size(); ++i) {
      x.data[i] = static_cast<float>(
          static_cast<double>(drawn[i]) * config.decode_noise_scale +
          (1.0 - config.decode_noise_scale) * static_cast<double>(x.data[i]));
    }
  }
  {
    const std::vector<float>& std_of_means =
        weights.Get(p + "per_channel_statistics.std-of-means");
    const std::vector<float>& mean_of_means =
        weights.Get(p + "per_channel_statistics.mean-of-means");
    VT_CHECK(static_cast<int64_t>(std_of_means.size()) == latent_channels &&
                 static_cast<int64_t>(mean_of_means.size()) == latent_channels,
             "ltx2 video vae: per-channel statistics must have one value per latent channel");
    const int64_t n = x.spatial();
    for (int64_t c = 0; c < latent_channels; ++c) {
      for (int64_t i = 0; i < n; ++i) {
        x.data[static_cast<size_t>(c * n + i)] = static_cast<float>(
            static_cast<double>(x.data[static_cast<size_t>(c * n + i)]) *
                static_cast<double>(std_of_means[static_cast<size_t>(c)]) +
            static_cast<double>(mean_of_means[static_cast<size_t>(c)]));
      }
    }
  }

  const double scaled_timestep =
      config.timestep_conditioning
          ? (timestep != nullptr ? *timestep : config.decode_timestep) *
                static_cast<double>(weights.Get(p + "timestep_scale_multiplier")[0])
          : 0.0;

  // --- conv_in widens the latents to the bottleneck ---
  int64_t multiplier = 1;
  for (const Ltx2VideoDecoderBlock& block : config.decoder_blocks) {
    if (block.name == "compress_time" || block.name == "compress_space" ||
        block.name == "compress_all") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 1;
    } else if (block.name == "res_x_y") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 2;
    }
  }
  // conv_in is ALWAYS causal upstream (conv_video_decoder.py:216), independently
  // of `config.causal`, which only reaches the per-call `causal=` argument.
  x = CausalConv3d(x, config.base_channels * multiplier, 3, config.causal,
                   config.spatial_padding_mode, weights.Get(p + "conv_in.conv.weight"),
                   &weights.Get(p + "conv_in.conv.bias"));

  // --- the reversed block walk (conv_video_decoder.py:222-238, 315-326) ---
  int64_t index = 0;
  for (auto it = config.decoder_blocks.rbegin(); it != config.decoder_blocks.rend(); ++it, ++index) {
    const Ltx2VideoDecoderBlock& block = *it;
    const std::string bp = p + "up_blocks." + std::to_string(index);
    if (block.name == "res_x") {
      std::vector<float> embed;
      const std::vector<float>* embed_ptr = nullptr;
      if (config.timestep_conditioning) {
        embed = TimestepEmbedding(scaled_timestep, x.channels * 4, weights, bp + ".time_embedder");
        embed_ptr = &embed;
      }
      for (int64_t i = 0; i < block.num_layers; ++i) {
        x = ResnetBlock3d(config, weights, bp + ".res_blocks." + std::to_string(i), x, x.channels,
                          block.inject_noise, config.timestep_conditioning, embed_ptr, noise);
      }
    } else if (block.name == "res_x_y") {
      const int64_t out_channels = x.channels / (block.multiplier != 0 ? block.multiplier : 2);
      // _make_decoder_block forces timestep_conditioning=False for res_x_y
      // (conv_video_decoder.py:107).
      x = ResnetBlock3d(config, weights, bp, x, out_channels, block.inject_noise,
                        /*timestep_conditioning=*/false, nullptr, noise);
    } else if (block.name == "attn") {
      x = AttnBlock3d(weights, bp, x);
    } else if (block.name == "compress_time" || block.name == "compress_space" ||
               block.name == "compress_all") {
      const int64_t st = block.name == "compress_space" ? 1 : 2;
      const int64_t ss = block.name == "compress_time" ? 1 : 2;
      x = DepthToSpaceUpsample(config, weights, bp, x, st, ss, ss,
                               block.multiplier != 0 ? block.multiplier : 1,
                               block.name == "compress_all" && block.residual);
    } else if (block.name == "attn_res_x") {
      VT_CHECK(false,
               "ltx2 video vae: the `attn_res_x` decoder block cannot be built — upstream passes "
               "`attention_head_dim` to UNetMidBlock3D, which does not accept it "
               "(conv_video_decoder.py:85-96 vs video_vae/resnet.py:210-222)");
    } else {
      VT_CHECK(false, "ltx2 video vae: unknown decoder block `" + block.name + "`");
    }
  }

  // --- conv_norm_out -> ada-LN -> SiLU -> conv_out ---
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(x.data, x.channels, x.spatial(), config.pixel_norm_eps);
  } else {
    MiniMaxH3GroupNorm3d(x.data, x.channels, x.spatial(), config.norm_num_groups,
                         weights.Get(p + "conv_norm_out.weight"),
                         weights.Get(p + "conv_norm_out.bias"), config.norm_eps);
  }
  if (config.timestep_conditioning) {
    const std::vector<float> embed =
        TimestepEmbedding(scaled_timestep, x.channels * 2, weights, p + "last_time_embedder");
    // ada_values rows are (shift, scale) — two, not the resnet's four.
    ApplyAdaLn(x, weights.Get(p + "last_scale_shift_table"), embed, 2, 0, 1);
  }
  Silu(x.data);
  x = CausalConv3d(x, config.out_channels * config.patch_size * config.patch_size, 3, config.causal,
                   config.spatial_padding_mode, weights.Get(p + "conv_out.conv.weight"),
                   &weights.Get(p + "conv_out.conv.bias"));

  // --- unpatchify (ops.py:35-60): `b (c p r q) f h w -> b c (f p) (h q) (w r)`
  // with p = patch_size_t = 1. NOTE h takes q and w takes r; swapping them
  // transposes every patch.
  const int64_t q = config.patch_size;
  const int64_t r = config.patch_size;
  Ltx2VideoFrames out;
  out.channels = config.out_channels;
  out.frames = x.t;
  out.height = x.h * q;
  out.width = x.w * r;
  out.data.resize(
      static_cast<size_t>(out.channels * out.frames * out.height * out.width));
  for (int64_t c = 0; c < out.channels; ++c) {
    for (int64_t ri = 0; ri < r; ++ri) {
      for (int64_t qi = 0; qi < q; ++qi) {
        const int64_t src_c = (c * r + ri) * q + qi;
        for (int64_t f = 0; f < x.t; ++f) {
          for (int64_t hi = 0; hi < x.h; ++hi) {
            for (int64_t wi = 0; wi < x.w; ++wi) {
              out.data[static_cast<size_t>(
                  ((c * out.frames + f) * out.height + hi * q + qi) * out.width + wi * r + ri)] =
                  x.data[x.At(src_c, f, hi, wi)];
            }
          }
        }
      }
    }
  }
  return out;
}

Ltx2VideoFrames Ltx2VideoDecode(Ltx2VideoDecoderKind kind,
                                const Ltx2ConvVideoDecoderConfig& config,
                                const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                                int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                                int64_t latent_w, Ltx2NoiseStream* noise, const double* timestep) {
  // REFUSE, never downgrade: falling back to the conv decoder would return a
  // lower-quality render as if it were the requested one, and no gate this
  // project owns could detect that (.agents/specs/ltx-2-5.md section 0 item 2).
  VT_CHECK(kind != Ltx2VideoDecoderKind::kDiffusion,
           "ltx2 video vae: this checkpoint asks for the DIFFUSION video decoder "
           "(NADiffusionDecoder / DiffusionVideoDecoder), which is NOT implemented — it needs a "
           "neighborhood-attention kernel and has its own row. It is refused rather than "
           "downgraded to the Conv video VAE, which would silently return a worse render");
  return Ltx2ConvVideoDecode(config, weights, latent, latent_channels, latent_t, latent_h, latent_w,
                             noise, timestep);
}

}  // namespace vllm
