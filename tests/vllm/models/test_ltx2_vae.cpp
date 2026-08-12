// LTX-2.5 VAE parity gate — the audio decoder, its vocoder (both resblock arms
// plus the BWE chain), and the Conv video decoder, each compared against the
// UPSTREAM `ltx_core` module executed at reduced dimensions on CPU by
// scripts/gen-ltx2-vae-goldens.py.
//
// Both sides rebuild every weight and input from ONE deterministic stream, so no
// weight byte is checked in. Beyond the tensor comparison each brick also asserts
// its PARAMETER MANIFEST — name and element count, in state_dict order — against
// the generator's, so a parameter one side builds and the other does not is a
// failure rather than a silent no-op.
//
// Tolerances use `.scale(0.0)` wherever doctest::Approx appears: Approx's default
// scale of 1.0 puts a 1.19e-5 ABSOLUTE floor under any epsilon, which would make
// a tight relative tolerance silently accept anything.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "doctest/doctest.h"
#include "support/max_abs_diff.h"
#include "vllm/model_executor/models/ltx2_audio_vae.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
// kMiniMaxH3SnakeEps: the Snake/SnakeBeta stabilizer is SHARED with MiniMax-H3's
// BigVGAN, so the constant this suite pins lives in that header.
#include "vllm/model_executor/models/minimax_h3.h"

#include "ltx2_vae_goldens.inc"

namespace {

// ---------------------------------------------------------------------------
// Ltx2Rand — the exact mirror of the generator's stream
// (scripts/gen-ltx2-vae-goldens.py :: ltx_rand): a per-tensor FNV-1a seed plus a
// splitmix64 counter, so both sides build identical tensors from a NAME alone and
// cannot drift by reordering their parameter construction.
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& name) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char byte : name) {
    h ^= static_cast<uint64_t>(byte);
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::vector<double> Ltx2Rand(const std::string& name, int64_t count) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<double> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    out[static_cast<size_t>(i)] = (static_cast<double>(u >> 11) * 0x1p-53) * 2.0 - 1.0;
  }
  return out;
}

// A raw input tensor: ltx_rand * scale (the generator's `make_input`).
std::vector<float> Ltx2Input(const std::string& name, int64_t count, double scale) {
  const std::vector<double> raw = Ltx2Rand(name, count);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(raw[static_cast<size_t>(i)] * scale);
  }
  return out;
}

bool EndsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The generator's `param_values` rule, mirrored EXACTLY. `rank` is the upstream
// tensor's rank, which is why the bag builder passes shapes and not counts.
std::vector<float> Ltx2Param(const std::string& name, const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;
  const size_t rank = shape.size();

  double scale = 0.1;
  double offset = 0.0;
  double attenuate = 1.0;
  bool absolute = false;
  if (name == "timestep_scale_multiplier") {
    return std::vector<float>(static_cast<size_t>(count), 1000.0f);
  } else if (EndsWith(name, "mel_basis")) {
    scale = 0.2;
    offset = 0.05;
    absolute = true;  // a mel filterbank is non-negative
    // ...except on the arm that exists to SATURATE the mel log clamp. Attenuating
    // the basis by 1e-4 puts every bin under `kLtx2BweMelLogClamp`, which is the
    // only way the deterministic stream can reach a constant that otherwise only
    // real silence binds. Mirrors the generator's `param_values` exactly.
    if (name.find(".bwequiet.") != std::string::npos) attenuate = 1e-4;
  } else if (EndsWith(name, ".gamma")) {
    offset = 1.0;
  } else if (EndsWith(name, ".alpha") || EndsWith(name, ".beta")) {
    scale = 0.2;
  } else if (EndsWith(name, "std-of-means")) {
    offset = 1.0;
  } else if (EndsWith(name, "mean-of-means") || EndsWith(name, "scale_shift_table") ||
             EndsWith(name, "per_channel_scale1") || EndsWith(name, "per_channel_scale2")) {
    // scale 0.1, offset 0
  } else if (EndsWith(name, ".bias")) {
    scale = 0.05;
  } else if (rank == 1 && EndsWith(name, ".weight")) {
    offset = 1.0;  // a 1-D `.weight` is an affine norm gain, initialized to ones
  }

  const std::vector<double> raw = Ltx2Rand(name, count);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    double value = raw[static_cast<size_t>(i)];
    if (absolute) value = std::abs(value);
    out[static_cast<size_t>(i)] = static_cast<float>((value * scale + offset) * attenuate);
  }
  return out;
}

// A weight bag that also records the (name, count) manifest in build order, so a
// test can prove its parameter set matches the generator's state_dict exactly.
struct ParamBag {
  vllm::Ltx2VaeWeights weights;
  std::vector<std::string> names;
  std::vector<int64_t> counts;

  void Put(const std::string& name, const std::vector<int64_t>& shape) {
    std::vector<float> values = Ltx2Param(name, shape);
    counts.push_back(static_cast<int64_t>(values.size()));
    names.push_back(name);
    weights.tensors[name] = std::move(values);
  }
};

void CheckManifest(const ParamBag& bag, const char* const* want_names, const int64_t* want_counts,
                   size_t want_size) {
  REQUIRE(bag.names.size() == want_size);
  REQUIRE(bag.counts.size() == want_size);
  for (size_t i = 0; i < want_size; ++i) {
    CHECK(bag.names[i] == std::string(want_names[i]));
    CHECK(bag.counts[i] == want_counts[i]);
  }
}

// ---------------------------------------------------------------------------
// Tolerances, derived from the measurement rather than picked.
//
// The tolerances started 6-14x above what the port actually produces, which makes
// them decoration: a band that can never bind reports nothing. These are set from
// the WORST arm in this file plus a stated margin, so a real drift moves them.
//
//   worst tensor arm   1.81794e-06  (the non-causal Conv video decoder)
//   worst filter arm   2.98023e-08  (the kaiser-sinc window)
//
// The margin is for libm, not for us: our side accumulates every reduction in
// double against a FIXED golden constant, so reduction-order variation across
// platforms cannot reach it — but `sin`, `exp`, `tanh` and `sqrt` differ by ~1 ulp
// between libm implementations, and the vocoder composes them through a deep
// sequential chain. ~2.7x on the tensors and ~3.4x on the filters covers that
// without leaving room for a structural porting error, which would move these by
// orders of magnitude rather than ulps.
//
// If a platform ever exceeds these, that is a finding to investigate and record —
// not a number to raise. AGENTS.md forbids widening a band to go green.
constexpr double kLtx2GoldenTol = 5e-6;
constexpr double kLtx2FilterTol = 1e-7;

// The shared, NaN-hardened reduction. The local copy this replaces used
// `std::max(worst, ...)`, which is `a < b ? b : a`; `a < NaN` is false, so an
// all-NaN result against a correct golden reduced to 0.0 (issue #449).
using vllm_test::MaxAbsDiff;

// The reduced audio decoder the generator built (AUDIO_DEC).
vllm::Ltx2AudioDecoderConfig ReducedAudioDecoderConfig(int64_t mel_bins) {
  vllm::Ltx2AudioDecoderConfig cfg;
  cfg.ch = 8;
  cfg.out_ch = 2;
  cfg.ch_mult = {1, 2, 4};
  cfg.num_res_blocks = 1;
  cfg.attn_resolutions = {8};
  cfg.resolution = 32;
  cfg.z_channels = 4;
  cfg.norm_type = vllm::Ltx2NormType::kPixel;
  cfg.causality_axis = vllm::Ltx2CausalityAxis::kHeight;
  cfg.mid_block_add_attention = true;
  cfg.mel_bins = mel_bins;
  cfg.prefix = "ltx2.audiodec.";
  return cfg;
}

// Build the audio decoder's parameters in upstream state_dict ORDER:
// per_channel_statistics, conv_in, mid, up (block / attn / upsample per level),
// conv_out. PixelNorm carries no parameters, which is why no norm tensor appears.
ParamBag BuildAudioDecoderParams(const vllm::Ltx2AudioDecoderConfig& cfg) {
  ParamBag bag;
  const std::string p = cfg.prefix;
  const int64_t levels = static_cast<int64_t>(cfg.ch_mult.size());
  const int64_t base = cfg.ch * cfg.ch_mult[static_cast<size_t>(levels - 1)];

  bag.Put(p + "per_channel_statistics.std-of-means", {cfg.ch});
  bag.Put(p + "per_channel_statistics.mean-of-means", {cfg.ch});
  bag.Put(p + "conv_in.conv.weight", {base, cfg.z_channels, 3, 3});
  bag.Put(p + "conv_in.conv.bias", {base});

  auto put_resnet = [&](const std::string& prefix, int64_t in_ch, int64_t out_ch) {
    bag.Put(prefix + ".conv1.conv.weight", {out_ch, in_ch, 3, 3});
    bag.Put(prefix + ".conv1.conv.bias", {out_ch});
    bag.Put(prefix + ".conv2.conv.weight", {out_ch, out_ch, 3, 3});
    bag.Put(prefix + ".conv2.conv.bias", {out_ch});
    if (in_ch != out_ch) {
      bag.Put(prefix + ".nin_shortcut.conv.weight", {out_ch, in_ch, 1, 1});
      bag.Put(prefix + ".nin_shortcut.conv.bias", {out_ch});
    }
  };
  auto put_attn = [&](const std::string& prefix, int64_t channels) {
    for (const char* leaf : {"q", "k", "v", "proj_out"}) {
      bag.Put(prefix + "." + leaf + ".weight", {channels, channels, 1, 1});
      bag.Put(prefix + "." + leaf + ".bias", {channels});
    }
  };

  put_resnet(p + "mid.block_1", base, base);
  if (cfg.mid_block_add_attention) put_attn(p + "mid.attn_1", base);
  put_resnet(p + "mid.block_2", base, base);

  // build_upsampling_path (upsample.py:58-106) walks levels in REVERSE and
  // inserts each stage at the front, so `up.<level>` is indexed by level.
  int64_t curr_res = cfg.resolution / (int64_t{1} << (levels - 1));
  int64_t block_in = base;
  struct Stage {
    std::vector<std::pair<int64_t, int64_t>> blocks;  // (in, out)
    std::vector<int64_t> attn;                        // channels
    int64_t upsample = 0;                             // channels, 0 = none
  };
  std::vector<Stage> stages(static_cast<size_t>(levels));
  for (int64_t level = levels - 1; level >= 0; --level) {
    Stage& stage = stages[static_cast<size_t>(level)];
    const int64_t block_out = cfg.ch * cfg.ch_mult[static_cast<size_t>(level)];
    for (int64_t i = 0; i < cfg.num_res_blocks + 1; ++i) {
      stage.blocks.emplace_back(block_in, block_out);
      block_in = block_out;
      if (std::find(cfg.attn_resolutions.begin(), cfg.attn_resolutions.end(), curr_res) !=
          cfg.attn_resolutions.end()) {
        stage.attn.push_back(block_in);
      }
    }
    if (level != 0) {
      stage.upsample = block_in;
      curr_res *= 2;
    }
  }
  for (int64_t level = 0; level < levels; ++level) {
    const Stage& stage = stages[static_cast<size_t>(level)];
    const std::string sp = p + "up." + std::to_string(level);
    for (size_t i = 0; i < stage.blocks.size(); ++i) {
      put_resnet(sp + ".block." + std::to_string(i), stage.blocks[i].first,
                 stage.blocks[i].second);
    }
    for (size_t i = 0; i < stage.attn.size(); ++i) {
      put_attn(sp + ".attn." + std::to_string(i), stage.attn[i]);
    }
    if (stage.upsample != 0) {
      bag.Put(sp + ".upsample.conv.conv.weight", {stage.upsample, stage.upsample, 3, 3});
      bag.Put(sp + ".upsample.conv.conv.bias", {stage.upsample});
    }
  }

  bag.Put(p + "conv_out.conv.weight", {cfg.out_ch, block_in, 3, 3});
  bag.Put(p + "conv_out.conv.bias", {cfg.out_ch});
  return bag;
}

// The reduced BigVGAN v2 vocoder the generator built (VOC).
vllm::Ltx2VocoderConfig ReducedVocoderConfig() {
  vllm::Ltx2VocoderConfig cfg;
  cfg.resblock_kernel_sizes = {3, 7};
  cfg.upsample_rates = {2, 2};
  cfg.upsample_kernel_sizes = {4, 4};
  cfg.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}};
  cfg.upsample_initial_channel = 16;
  cfg.amp = true;
  cfg.snakebeta = true;
  cfg.use_tanh_at_final = true;
  cfg.apply_final_activation = true;
  cfg.use_bias_at_final = true;
  cfg.output_sampling_rate = 16000;
  cfg.prefix = "ltx2.voc.";
  return cfg;
}

// Vocoder.__init__ registration order: conv_pre, ups, resblocks, act_post,
// conv_post. `*.filter` buffers are absent on purpose — the kaiser-sinc windows
// are COMPUTED at construction, never loaded.
void PutVocoderParams(ParamBag& bag, const vllm::Ltx2VocoderConfig& cfg) {
  const std::string p = cfg.prefix;
  const int64_t initial = cfg.upsample_initial_channel;
  const int64_t num_kernels = static_cast<int64_t>(cfg.resblock_kernel_sizes.size());

  bag.Put(p + "conv_pre.weight", {initial, 128, 7});
  bag.Put(p + "conv_pre.bias", {initial});
  for (size_t i = 0; i < cfg.upsample_rates.size(); ++i) {
    const int64_t in_ch = initial / (int64_t{1} << i);
    const int64_t out_ch = initial / (int64_t{1} << (i + 1));
    // ConvTranspose1d weight is [in, out, k].
    bag.Put(p + "ups." + std::to_string(i) + ".weight",
            {in_ch, out_ch, cfg.upsample_kernel_sizes[i]});
    bag.Put(p + "ups." + std::to_string(i) + ".bias", {out_ch});
  }
  for (size_t i = 0; i < cfg.upsample_rates.size(); ++i) {
    const int64_t ch = initial / (int64_t{1} << (i + 1));
    for (int64_t j = 0; j < num_kernels; ++j) {
      const std::string block =
          p + "resblocks." + std::to_string(static_cast<int64_t>(i) * num_kernels + j);
      const int64_t kernel = cfg.resblock_kernel_sizes[static_cast<size_t>(j)];
      for (int64_t d = 0; d < 3; ++d) {
        bag.Put(block + ".convs1." + std::to_string(d) + ".weight", {ch, ch, kernel});
        bag.Put(block + ".convs1." + std::to_string(d) + ".bias", {ch});
      }
      for (int64_t d = 0; d < 3; ++d) {
        bag.Put(block + ".convs2." + std::to_string(d) + ".weight", {ch, ch, kernel});
        bag.Put(block + ".convs2." + std::to_string(d) + ".bias", {ch});
      }
      if (cfg.amp) {
        for (const char* group : {"acts1", "acts2"}) {
          for (int64_t d = 0; d < 3; ++d) {
            const std::string act = block + "." + group + "." + std::to_string(d) + ".act.";
            bag.Put(act + "alpha", {ch});
            if (cfg.snakebeta) bag.Put(act + "beta", {ch});
          }
        }
      }
    }
  }
  const int64_t final_channels = initial / (int64_t{1} << cfg.upsample_rates.size());
  if (cfg.amp) {
    bag.Put(p + "act_post.act.alpha", {final_channels});
    // act_post is UNCONDITIONALLY SnakeBeta, so its `.beta` is present whenever
    // `amp` is — even on the `activation="snake"` arm, where every resblock
    // activation above has ONLY `.alpha`. Upstream builds it as
    // `Activation1d(SnakeBeta(final_channels))` with no `activation=` argument
    // (vocoder.py:388), unlike the resblocks one line earlier (vocoder.py:376).
    // The generator's own state_dict manifest for the snake arm proves it, and
    // gating this on `cfg.snakebeta` is what made the port read the wrong scale.
    bag.Put(p + "act_post.act.beta", {final_channels});
  }
  bag.Put(p + "conv_post.weight", {2, final_channels, 7});
  if (cfg.use_bias_at_final) bag.Put(p + "conv_post.bias", {2});
}

}  // namespace

TEST_CASE("ltx2 vae: the audio decoder matches upstream ltx_core") {
  const vllm::Ltx2AudioDecoderConfig cfg =
      ReducedAudioDecoderConfig(vllm_test::kLtx2AudioDecOutMelBins);
  ParamBag bag = BuildAudioDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioDecParamNames, vllm_test::kLtx2AudioDecParamCounts,
                std::size(vllm_test::kLtx2AudioDecParamNames));

  const int64_t latent_t = vllm_test::kLtx2AudioDecLatentT;
  const int64_t latent_f = vllm_test::kLtx2AudioDecLatentF;
  const std::vector<float> latent =
      Ltx2Input("ltx2.audiodec.input", cfg.z_channels * latent_t * latent_f, 1.0);

  const vllm::Ltx2AudioSpectrogram out =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, latent, cfg.z_channels, latent_t, latent_f);
  CHECK(out.channels == cfg.out_ch);
  CHECK(out.frames == vllm_test::kLtx2AudioDecOutFrames);
  CHECK(out.mel_bins == vllm_test::kLtx2AudioDecOutMelBins);
  const double err =
      MaxAbsDiff(out.data, vllm_test::kLtx2AudioDecGolden, std::size(vllm_test::kLtx2AudioDecGolden));
  INFO("audio decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // The SHIPPED configuration is NOT end-to-end causal, and this asserts that
  // rather than papering over it: the AttnBlocks attend over the whole (time,
  // mel) map (attention.py:31-55), so a change anywhere reaches every output
  // frame. Upstream agrees — measured in the generator, section 1b. The
  // causality claim is about the CONVOLUTIONS, and it is gated below.
  std::vector<float> bumped = latent;
  for (int64_t f = 0; f < latent_f; ++f) {
    for (int64_t c = 0; c < cfg.z_channels; ++c) {
      bumped[static_cast<size_t>((c * latent_t + (latent_t - 1)) * latent_f + f)] += 3.0f;
    }
  }
  const vllm::Ltx2AudioSpectrogram perturbed =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, bumped, cfg.z_channels, latent_t, latent_f);
  REQUIRE(perturbed.data.size() == out.data.size());
  for (int64_t t = 0; t < perturbed.frames; ++t) {
    bool moved = false;
    for (int64_t c = 0; c < perturbed.channels && !moved; ++c) {
      for (int64_t m = 0; m < perturbed.mel_bins; ++m) {
        const size_t i = static_cast<size_t>((c * perturbed.frames + t) * perturbed.mel_bins + m);
        if (perturbed.data[i] != out.data[i]) {
          moved = true;
          break;
        }
      }
    }
    INFO("output frame " << t << " under global attention");
    CHECK(moved);
  }
}

TEST_CASE("ltx2 vae: the audio decoder's CONVOLUTIONS are one-sided in time") {
  // The trap: padding the time axis symmetrically instead of on the LEFT still
  // produces a plausible spectrogram that merely peeks into the future. Isolate
  // the convolutions by turning attention off — the reach that leaves is a
  // property of the padding alone, and section 1b records what upstream's own
  // reach is, so this is not a claim the port makes about itself.
  vllm::Ltx2AudioDecoderConfig cfg =
      ReducedAudioDecoderConfig(vllm_test::kLtx2AudioDecOutMelBins);
  cfg.attn_resolutions.clear();
  cfg.mid_block_add_attention = false;
  cfg.prefix = "ltx2.audiodeccausal.";
  ParamBag bag = BuildAudioDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioDecCausalParamNames,
                vllm_test::kLtx2AudioDecCausalParamCounts,
                std::size(vllm_test::kLtx2AudioDecCausalParamNames));

  const int64_t latent_t = vllm_test::kLtx2AudioDecLatentT;
  const int64_t latent_f = vllm_test::kLtx2AudioDecLatentF;
  const std::vector<float> latent =
      Ltx2Input("ltx2.audiodec.input", cfg.z_channels * latent_t * latent_f, 1.0);
  std::vector<float> bumped = latent;
  for (int64_t f = 0; f < latent_f; ++f) {
    for (int64_t c = 0; c < cfg.z_channels; ++c) {
      bumped[static_cast<size_t>((c * latent_t + (latent_t - 1)) * latent_f + f)] += 3.0f;
    }
  }
  const vllm::Ltx2AudioSpectrogram base =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, latent, cfg.z_channels, latent_t, latent_f);
  const vllm::Ltx2AudioSpectrogram moved =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, bumped, cfg.z_channels, latent_t, latent_f);
  REQUIRE(moved.data.size() == base.data.size());

  int64_t first_moved = base.frames;
  int64_t last_moved = -1;
  for (int64_t t = 0; t < base.frames; ++t) {
    bool differs = false;
    for (int64_t c = 0; c < base.channels && !differs; ++c) {
      for (int64_t m = 0; m < base.mel_bins; ++m) {
        const size_t i = static_cast<size_t>((c * base.frames + t) * base.mel_bins + m);
        if (base.data[i] != moved.data[i]) {
          differs = true;
          break;
        }
      }
    }
    if (differs) {
      if (first_moved == base.frames) first_moved = t;
      last_moved = t;
    }
  }
  INFO("convolution-only reach of a last-latent-frame bump: [" << first_moved << ", " << last_moved
                                                               << "]");
  CHECK(first_moved == vllm_test::kLtx2AudioDecCausalFirstMoved);
  CHECK(last_moved == vllm_test::kLtx2AudioDecCausalLastMoved);
  CHECK(first_moved > 0);  // the past cannot see the future
}

TEST_CASE("ltx2 vae: the other three causality axes match upstream ltx_core") {
  // Every arm elsewhere in this file runs `causality_axis=height`, the shipped
  // default (audio_vae/model_configurator.py:134). The remaining three branches of
  // CausalConv2d's padding switch (causality_axis.py:4-10) were never executed, so
  // the port's pad split for them was an untested claim rather than a gated one.
  struct Arm {
    vllm::Ltx2CausalityAxis axis;
    const char* name;
    const float* golden;
    size_t golden_size;
    int64_t frames;
    int64_t mel_bins;
  };
  const Arm arms[] = {
      {vllm::Ltx2CausalityAxis::kNone, "NONE", vllm_test::kLtx2AudioDecNoneGolden,
       std::size(vllm_test::kLtx2AudioDecNoneGolden), vllm_test::kLtx2AudioDecNoneOutFrames,
       vllm_test::kLtx2AudioDecNoneOutMelBins},
      {vllm::Ltx2CausalityAxis::kWidth, "WIDTH", vllm_test::kLtx2AudioDecWidthGolden,
       std::size(vllm_test::kLtx2AudioDecWidthGolden), vllm_test::kLtx2AudioDecWidthOutFrames,
       vllm_test::kLtx2AudioDecWidthOutMelBins},
      {vllm::Ltx2CausalityAxis::kWidthCompatibility, "WIDTH_COMPATIBILITY",
       vllm_test::kLtx2AudioDecWidthCompatGolden,
       std::size(vllm_test::kLtx2AudioDecWidthCompatGolden),
       vllm_test::kLtx2AudioDecWidthCompatOutFrames,
       vllm_test::kLtx2AudioDecWidthCompatOutMelBins},
  };

  std::vector<std::vector<float>> outputs;
  for (const Arm& arm : arms) {
    vllm::Ltx2AudioDecoderConfig cfg =
        ReducedAudioDecoderConfig(vllm_test::kLtx2AudioDecOutMelBins);
    cfg.causality_axis = arm.axis;
    ParamBag bag = BuildAudioDecoderParams(cfg);
    const std::vector<float> latent = Ltx2Input(
        "ltx2.audiodec.input",
        cfg.z_channels * vllm_test::kLtx2AudioDecLatentT * vllm_test::kLtx2AudioDecLatentF, 1.0);
    const vllm::Ltx2AudioSpectrogram out = vllm::Ltx2AudioDecoderForward(
        cfg, bag.weights, latent, cfg.z_channels, vllm_test::kLtx2AudioDecLatentT,
        vllm_test::kLtx2AudioDecLatentF);
    INFO("causality_axis = " << arm.name);
    CHECK(out.frames == arm.frames);
    CHECK(out.mel_bins == arm.mel_bins);
    const double err = MaxAbsDiff(out.data, arm.golden, arm.golden_size);
    INFO("max|diff| = " << err);
    CHECK(err <= kLtx2GoldenTol);
    outputs.push_back(out.data);
  }

  // WIDTH and WIDTH_COMPATIBILITY pad the width axis IDENTICALLY in the
  // convolutions, so a port could plausibly collapse them into one branch. It must
  // not: the UPSAMPLER treats them differently — upsample.py:44-48 does NOT drop
  // the first interpolated element for WIDTH_COMPATIBILITY, while every other axis
  // does. That is the whole difference, it is invisible in the padding code, and
  // this asserts the two arms actually diverge.
  REQUIRE(outputs[1].size() == outputs[2].size());
  CHECK(outputs[1] != outputs[2]);
}

TEST_CASE("ltx2 vae: the audio decoder pads the frequency axis to the target mel bins") {
  // audio_vae.py:458-467 zero-pads on the RIGHT of the frequency axis when the
  // configured mel_bins exceeds what the network produced. A port that returns
  // the unpadded tensor passes every other assertion in this file.
  const vllm::Ltx2AudioDecoderConfig cfg =
      ReducedAudioDecoderConfig(vllm_test::kLtx2AudioDecPadOutMelBins);
  ParamBag bag = BuildAudioDecoderParams(cfg);
  const std::vector<float> latent =
      Ltx2Input("ltx2.audiodec.input",
                cfg.z_channels * vllm_test::kLtx2AudioDecLatentT * vllm_test::kLtx2AudioDecLatentF,
                1.0);
  const vllm::Ltx2AudioSpectrogram out = vllm::Ltx2AudioDecoderForward(
      cfg, bag.weights, latent, cfg.z_channels, vllm_test::kLtx2AudioDecLatentT,
      vllm_test::kLtx2AudioDecLatentF);
  CHECK(out.mel_bins == vllm_test::kLtx2AudioDecPadOutMelBins);
  const double err = MaxAbsDiff(out.data, vllm_test::kLtx2AudioDecPadGolden,
                                std::size(vllm_test::kLtx2AudioDecPadGolden));
  INFO("padded audio decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the BigVGAN v2 vocoder matches upstream ltx_core") {
  // The anti-aliasing window is COMPUTED, never loaded. Gate it FIRST: a wrong
  // filter makes every SnakeBeta wrong and the decoder mismatch unlocalizable.
  const std::vector<float> filter = vllm::Ltx2KaiserSincFilter1d(0.5 / 2, 0.6 / 2, 12);
  REQUIRE(filter.size() == std::size(vllm_test::kLtx2VocUpFilterGolden));
  double filter_err = 0.0;
  double filter_sum = 0.0;
  for (size_t i = 0; i < filter.size(); ++i) {
    filter_err = std::max(filter_err, std::abs(static_cast<double>(filter[i]) -
                                               vllm_test::kLtx2VocUpFilterGolden[i]));
    filter_sum += filter[i];
  }
  INFO("kaiser-sinc filter max|diff| = " << filter_err);
  CHECK(filter_err <= kLtx2FilterTol);
  CHECK(filter_sum == doctest::Approx(1.0).epsilon(1e-6).scale(0.0));

  const vllm::Ltx2VocoderConfig cfg = ReducedVocoderConfig();
  ParamBag bag;
  PutVocoderParams(bag, cfg);
  CheckManifest(bag, vllm_test::kLtx2VocParamNames, vllm_test::kLtx2VocParamCounts,
                std::size(vllm_test::kLtx2VocParamNames));

  const int64_t frames = vllm_test::kLtx2VocFrames;
  const int64_t mel_bins = vllm_test::kLtx2VocMelBins;
  const std::vector<float> mel = Ltx2Input("ltx2.voc.input", 2 * frames * mel_bins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderForward(cfg, bag.weights, mel, 2, frames, mel_bins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2VocOutSamples);
  const double err =
      MaxAbsDiff(wave, vllm_test::kLtx2VocGolden, std::size(vllm_test::kLtx2VocGolden));
  INFO("vocoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
  // The golden is deliberately UNSATURATED: a tanh-saturated golden would hide
  // errors instead of catching them.
  for (float value : wave) {
    CHECK(value > -0.999f);
    CHECK(value < 0.999f);
  }
}

TEST_CASE("ltx2 vae: act_post stays SnakeBeta on the plain-snake vocoder arm") {
  // `resblock="AMP1"` with `activation="snake"`. Upstream builds act_post as
  // `Activation1d(SnakeBeta(final_channels))` inside `if self.is_amp`, taking NO
  // `activation=` argument (vocoder.py:388) — unlike the resblocks one line
  // earlier (vocoder.py:376). So on this arm every resblock activation is plain
  // Snake, which reuses ALPHA as its reciprocal scale (vocoder.py:198), while
  // act_post alone still reads `.beta`.
  //
  // Keying act_post off `activation` therefore reads the wrong scale for one
  // activation and still produces a plausible waveform. No other arm can catch
  // it: on the shipped snakebeta arm the two spellings agree, and on the legacy
  // arm there is no act_post at all. The manifest below is the second half of the
  // proof — upstream's own state_dict has `act_post.act.beta` present while every
  // `acts1/acts2` entry has only `.alpha`.
  vllm::Ltx2VocoderConfig cfg = ReducedVocoderConfig();
  cfg.snakebeta = false;  // activation == "snake"
  cfg.prefix = "ltx2.vocsnake.";

  ParamBag bag;
  PutVocoderParams(bag, cfg);
  CheckManifest(bag, vllm_test::kLtx2VocSnakeParamNames, vllm_test::kLtx2VocSnakeParamCounts,
                std::size(vllm_test::kLtx2VocSnakeParamNames));

  const std::vector<float> mel =
      Ltx2Input("ltx2.voc.input", 2 * vllm_test::kLtx2VocFrames * vllm_test::kLtx2VocMelBins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderForward(cfg, bag.weights, mel, 2, vllm_test::kLtx2VocFrames,
                               vllm_test::kLtx2VocMelBins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2VocSnakeOutSamples);
  const double err =
      MaxAbsDiff(wave, vllm_test::kLtx2VocSnakeGolden, std::size(vllm_test::kLtx2VocSnakeGolden));
  INFO("snake-arm vocoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the legacy resblock-1 vocoder arm matches upstream ltx_core") {
  vllm::Ltx2VocoderConfig cfg;
  cfg.resblock_kernel_sizes = {3};
  cfg.upsample_rates = {2};
  cfg.upsample_kernel_sizes = {4};
  cfg.resblock_dilation_sizes = {{1, 3, 5}};
  cfg.upsample_initial_channel = 8;
  cfg.amp = false;  // resblock "1": ResBlock1 + plain leaky ReLU, no anti-aliasing
  cfg.output_sampling_rate = 16000;
  cfg.prefix = "ltx2.vocleg.";

  ParamBag bag;
  PutVocoderParams(bag, cfg);
  CheckManifest(bag, vllm_test::kLtx2VocLegacyParamNames, vllm_test::kLtx2VocLegacyParamCounts,
                std::size(vllm_test::kLtx2VocLegacyParamNames));

  const std::vector<float> mel =
      Ltx2Input("ltx2.voc.input", 2 * vllm_test::kLtx2VocFrames * vllm_test::kLtx2VocMelBins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderForward(cfg, bag.weights, mel, 2, vllm_test::kLtx2VocFrames,
                               vllm_test::kLtx2VocMelBins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2VocLegacyOutSamples);
  const double err = MaxAbsDiff(wave, vllm_test::kLtx2VocLegacyGolden,
                                std::size(vllm_test::kLtx2VocLegacyGolden));
  INFO("legacy vocoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the BWE vocoder chain matches upstream ltx_core") {
  // The hann-sinc resampler window is a DIFFERENT filter from the kaiser one the
  // activations use, and is likewise computed rather than loaded (persistent=False).
  int64_t kernel_size = 0, pad = 0, pad_left = 0, pad_right = 0;
  const std::vector<float> resample_filter =
      vllm::Ltx2HannSincResampleFilter1d(2, &kernel_size, &pad, &pad_left, &pad_right);
  CHECK(kernel_size == vllm_test::kLtx2BweResamplerKernel);
  CHECK(pad == 7);
  CHECK(pad_left == 28);
  CHECK(pad_right == 27);
  const double filter_err = MaxAbsDiff(resample_filter, vllm_test::kLtx2BweResamplerFilterGolden,
                                       std::size(vllm_test::kLtx2BweResamplerFilterGolden));
  INFO("hann-sinc resampler filter max|diff| = " << filter_err);
  CHECK(filter_err <= kLtx2FilterTol);

  vllm::Ltx2VocoderBweConfig cfg;
  cfg.vocoder = ReducedVocoderConfig();
  cfg.vocoder.prefix = "ltx2.bwe.vocoder.";
  cfg.bwe_generator = ReducedVocoderConfig();
  cfg.bwe_generator.prefix = "ltx2.bwe.bwe_generator.";
  cfg.bwe_generator.resblock_kernel_sizes = {3};
  cfg.bwe_generator.resblock_dilation_sizes = {{1, 3, 5}};
  cfg.bwe_generator.upsample_rates = {4, 4};
  cfg.bwe_generator.upsample_kernel_sizes = {8, 8};
  cfg.bwe_generator.apply_final_activation = false;
  cfg.bwe_generator.output_sampling_rate = 32000;
  cfg.filter_length = 16;
  cfg.hop_length = 8;
  cfg.win_length = 16;
  cfg.n_mel_channels = 64;
  cfg.input_sampling_rate = 16000;
  cfg.output_sampling_rate = 32000;
  cfg.prefix = "ltx2.bwe.";

  ParamBag bag;
  PutVocoderParams(bag, cfg.vocoder);
  PutVocoderParams(bag, cfg.bwe_generator);
  bag.Put(cfg.prefix + "mel_stft.mel_basis", {cfg.n_mel_channels, cfg.filter_length / 2 + 1});
  bag.Put(cfg.prefix + "mel_stft.stft_fn.forward_basis",
          {(cfg.filter_length / 2 + 1) * 2, 1, cfg.filter_length});
  bag.Put(cfg.prefix + "mel_stft.stft_fn.inverse_basis",
          {(cfg.filter_length / 2 + 1) * 2, 1, cfg.filter_length});
  CheckManifest(bag, vllm_test::kLtx2BweParamNames, vllm_test::kLtx2BweParamCounts,
                std::size(vllm_test::kLtx2BweParamNames));

  const std::vector<float> mel =
      Ltx2Input("ltx2.voc.input", 2 * vllm_test::kLtx2VocFrames * vllm_test::kLtx2VocMelBins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderWithBweForward(cfg, bag.weights, mel, 2, vllm_test::kLtx2VocFrames,
                                      vllm_test::kLtx2VocMelBins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2BweOutSamples);
  const double err =
      MaxAbsDiff(wave, vllm_test::kLtx2BweGolden, std::size(vllm_test::kLtx2BweGolden));
  INFO("BWE vocoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

namespace {

// The reduced Conv video decoder the generator built (VIDEO_BLOCKS / VIDEO_DEC).
vllm::Ltx2ConvVideoDecoderConfig ReducedVideoDecoderConfig() {
  vllm::Ltx2ConvVideoDecoderConfig cfg;
  cfg.in_channels = 6;
  cfg.out_channels = 3;
  cfg.patch_size = 2;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  cfg.causal = true;
  cfg.timestep_conditioning = true;
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kReflect;
  cfg.base_channels = 8;
  cfg.prefix = "ltx2.videodec.";
  cfg.decoder_blocks = {
      {"res_x", 1, 0, /*inject_noise=*/true, false},
      {"compress_all", 1, 2, false, /*residual=*/true},
      {"res_x_y", 1, 2, false, false},
      {"compress_space", 1, 1, false, false},
      {"attn", 1, 0, false, false},
      {"compress_time", 1, 1, false, false},
      {"res_x", 2, 0, false, false},
  };
  return cfg;
}

// ConvVideoDecoder state_dict order: the module's own PARAMETERS first
// (timestep_scale_multiplier, last_scale_shift_table), then submodules in
// registration order (per_channel_statistics, conv_in, up_blocks, conv_out,
// last_time_embedder). conv_norm_out is a PixelNorm and carries nothing.
ParamBag BuildVideoDecoderParams(const vllm::Ltx2ConvVideoDecoderConfig& cfg) {
  ParamBag bag;
  const std::string p = cfg.prefix;

  int64_t multiplier = 1;
  for (const vllm::Ltx2VideoDecoderBlock& block : cfg.decoder_blocks) {
    if (block.name == "compress_time" || block.name == "compress_space" ||
        block.name == "compress_all") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 1;
    } else if (block.name == "res_x_y") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 2;
    }
  }
  int64_t channels = cfg.base_channels * multiplier;

  // The bottleneck width is what conv_in widens to; the final width is what the
  // reversed block walk ends at, which is where last_scale_shift_table lives.
  int64_t final_channels = channels;
  for (auto it = cfg.decoder_blocks.rbegin(); it != cfg.decoder_blocks.rend(); ++it) {
    if (it->name == "res_x_y") {
      final_channels /= (it->multiplier != 0 ? it->multiplier : 2);
    } else if (it->name == "compress_time" || it->name == "compress_space" ||
               it->name == "compress_all") {
      final_channels /= (it->multiplier != 0 ? it->multiplier : 1);
    }
  }

  // Both module-level parameters exist only under timestep conditioning
  // (conv_video_decoder.py:256-261), and torch emits _parameters before _modules.
  if (cfg.timestep_conditioning) {
    bag.Put(p + "timestep_scale_multiplier", {});
    bag.Put(p + "last_scale_shift_table", {2, final_channels});
  }
  bag.Put(p + "per_channel_statistics.std-of-means", {cfg.in_channels});
  bag.Put(p + "per_channel_statistics.mean-of-means", {cfg.in_channels});
  bag.Put(p + "conv_in.conv.weight", {channels, cfg.in_channels, 3, 3, 3});
  bag.Put(p + "conv_in.conv.bias", {channels});

  auto put_resnet3d = [&](const std::string& prefix, int64_t in_ch, int64_t out_ch,
                          bool inject_noise, bool timestep) {
    if (inject_noise) {
      bag.Put(prefix + ".per_channel_scale1", {in_ch, 1, 1});
      bag.Put(prefix + ".per_channel_scale2", {in_ch, 1, 1});
    }
    if (timestep) bag.Put(prefix + ".scale_shift_table", {4, in_ch});
    bag.Put(prefix + ".conv1.conv.weight", {out_ch, in_ch, 3, 3, 3});
    bag.Put(prefix + ".conv1.conv.bias", {out_ch});
    bag.Put(prefix + ".conv2.conv.weight", {out_ch, out_ch, 3, 3, 3});
    bag.Put(prefix + ".conv2.conv.bias", {out_ch});
    if (in_ch != out_ch) {
      bag.Put(prefix + ".conv_shortcut.weight", {out_ch, in_ch, 1, 1, 1});
      bag.Put(prefix + ".conv_shortcut.bias", {out_ch});
      bag.Put(prefix + ".norm3.weight", {in_ch});
      bag.Put(prefix + ".norm3.bias", {in_ch});
    }
  };

  int64_t index = 0;
  for (auto it = cfg.decoder_blocks.rbegin(); it != cfg.decoder_blocks.rend(); ++it, ++index) {
    const vllm::Ltx2VideoDecoderBlock& block = *it;
    const std::string bp = p + "up_blocks." + std::to_string(index);
    if (block.name == "res_x") {
      if (cfg.timestep_conditioning) {
        bag.Put(bp + ".time_embedder.timestep_embedder.linear_1.weight", {channels * 4, 256});
        bag.Put(bp + ".time_embedder.timestep_embedder.linear_1.bias", {channels * 4});
        bag.Put(bp + ".time_embedder.timestep_embedder.linear_2.weight",
                {channels * 4, channels * 4});
        bag.Put(bp + ".time_embedder.timestep_embedder.linear_2.bias", {channels * 4});
      }
      for (int64_t i = 0; i < block.num_layers; ++i) {
        put_resnet3d(bp + ".res_blocks." + std::to_string(i), channels, channels,
                     block.inject_noise, cfg.timestep_conditioning);
      }
    } else if (block.name == "res_x_y") {
      const int64_t out_ch = channels / (block.multiplier != 0 ? block.multiplier : 2);
      put_resnet3d(bp, channels, out_ch, block.inject_noise, /*timestep=*/false);
      channels = out_ch;
    } else if (block.name == "attn") {
      bag.Put(bp + ".norm.gamma", {channels, 1, 1});
      bag.Put(bp + ".to_qkv.weight", {channels * 3, channels, 1, 1});
      bag.Put(bp + ".to_qkv.bias", {channels * 3});
      bag.Put(bp + ".proj.weight", {channels, channels, 1, 1});
      bag.Put(bp + ".proj.bias", {channels});
    } else {
      int64_t stride_product = 2;
      if (block.name == "compress_space") stride_product = 4;
      if (block.name == "compress_all") stride_product = 8;
      const int64_t reduction = block.multiplier != 0 ? block.multiplier : 1;
      const int64_t conv_out = stride_product * channels / reduction;
      bag.Put(bp + ".conv.conv.weight", {conv_out, channels, 3, 3, 3});
      bag.Put(bp + ".conv.conv.bias", {conv_out});
      channels /= reduction;
    }
  }

  bag.Put(p + "conv_out.conv.weight",
          {cfg.out_channels * cfg.patch_size * cfg.patch_size, channels, 3, 3, 3});
  bag.Put(p + "conv_out.conv.bias", {cfg.out_channels * cfg.patch_size * cfg.patch_size});
  if (cfg.timestep_conditioning) {
    bag.Put(p + "last_time_embedder.timestep_embedder.linear_1.weight", {channels * 2, 256});
    bag.Put(p + "last_time_embedder.timestep_embedder.linear_1.bias", {channels * 2});
    bag.Put(p + "last_time_embedder.timestep_embedder.linear_2.weight",
            {channels * 2, channels * 2});
    bag.Put(p + "last_time_embedder.timestep_embedder.linear_2.bias", {channels * 2});
  }
  return bag;
}

// The generator's patched torch.randn: one deterministic draw per call, keyed by
// CALL INDEX. `counts` records the sequence so the test can prove the port
// consumes noise in the same order and the same sizes upstream did.
class GoldenNoise : public vllm::Ltx2NoiseStream {
 public:
  // `prefix` selects the arm's noise stream; the generator keys its patched
  // torch.randn by the same name and CALL INDEX.
  explicit GoldenNoise(std::string prefix = "ltx2.videodec.") : prefix_(std::move(prefix)) {}

  std::vector<float> Draw(int64_t count) override {
    std::vector<float> values =
        Ltx2Input(prefix_ + "noise." + std::to_string(counts_.size()), count, 1.0);
    counts_.push_back(count);
    return values;
  }
  const std::vector<int64_t>& counts() const { return counts_; }

 private:
  std::string prefix_;
  std::vector<int64_t> counts_;
};

}  // namespace

TEST_CASE("ltx2 vae: the Conv video decoder matches upstream ltx_core") {
  const vllm::Ltx2ConvVideoDecoderConfig cfg = ReducedVideoDecoderConfig();
  ParamBag bag = BuildVideoDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoDecParamNames, vllm_test::kLtx2VideoDecParamCounts,
                std::size(vllm_test::kLtx2VideoDecParamNames));

  const int64_t lc = vllm_test::kLtx2VideoDecLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecLatentW;
  const std::vector<float> latent = Ltx2Input("ltx2.videodec.input", lc * lt * lh * lw, 1.0);

  GoldenNoise noise;
  const vllm::Ltx2VideoFrames frames =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, latent, lc, lt, lh, lw, &noise);
  CHECK(frames.channels == vllm_test::kLtx2VideoDecOutC);
  CHECK(frames.frames == vllm_test::kLtx2VideoDecOutT);
  CHECK(frames.height == vllm_test::kLtx2VideoDecOutH);
  CHECK(frames.width == vllm_test::kLtx2VideoDecOutW);

  // Noise is consumed in upstream's own call order and sizes. A port that drew a
  // full [C,T,H,W] block where upstream draws only [H,W] would still produce a
  // finite, plausible clip.
  REQUIRE(static_cast<int64_t>(noise.counts().size()) == vllm_test::kLtx2VideoDecNoiseDraws);
  for (size_t i = 0; i < noise.counts().size(); ++i) {
    CHECK(noise.counts()[i] == vllm_test::kLtx2VideoDecNoiseCounts[i]);
  }

  const double err = MaxAbsDiff(frames.data, vllm_test::kLtx2VideoDecGolden,
                                std::size(vllm_test::kLtx2VideoDecGolden));
  INFO("Conv video decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the NON-causal Conv video decoder matches upstream ltx_core") {
  // `causal=False` is UPSTREAM'S OWN DEFAULT (`causal: bool = False`,
  // conv_video_decoder.py:184, and the class docstring calls it the standard
  // decoder), yet every other video arm in this file runs causal=True — so the
  // default configuration was the one nothing executed.
  //
  // It is a DIFFERENT padding rule, not a disabled one: CausalConv3d replicates
  // the FIRST and LAST frame (kernel-1)/2 times each instead of putting kernel-1
  // copies of frame 0 on the left (convolution.py:266-317). The frame count comes
  // out identical either way, which is exactly why getting it wrong shifts the
  // whole clip while every shape assertion still passes.
  // Shares the causal arm's WEIGHTS, INPUT and NOISE stream deliberately, so the
  // padding rule is the ONLY difference between the two goldens.
  vllm::Ltx2ConvVideoDecoderConfig cfg = ReducedVideoDecoderConfig();
  cfg.causal = false;
  ParamBag bag = BuildVideoDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoDecNcParamNames, vllm_test::kLtx2VideoDecNcParamCounts,
                std::size(vllm_test::kLtx2VideoDecNcParamNames));

  const int64_t lc = vllm_test::kLtx2VideoDecLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecLatentW;
  // The INPUT is the shared one — only the decoder's causality differs, so any
  // divergence is attributable to the padding rule and nothing else.
  const std::vector<float> latent = Ltx2Input("ltx2.videodec.input", lc * lt * lh * lw, 1.0);

  GoldenNoise noise;
  const vllm::Ltx2VideoFrames frames =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, latent, lc, lt, lh, lw, &noise);
  CHECK(frames.channels == vllm_test::kLtx2VideoDecOutC);
  CHECK(frames.frames == vllm_test::kLtx2VideoDecNcOutT);
  CHECK(frames.height == vllm_test::kLtx2VideoDecNcOutH);
  CHECK(frames.width == vllm_test::kLtx2VideoDecNcOutW);
  // Same frame count as the causal arm — the shapes cannot tell them apart.
  CHECK(vllm_test::kLtx2VideoDecNcOutT == vllm_test::kLtx2VideoDecOutT);

  REQUIRE(static_cast<int64_t>(noise.counts().size()) == vllm_test::kLtx2VideoDecNcNoiseDraws);
  for (size_t i = 0; i < noise.counts().size(); ++i) {
    CHECK(noise.counts()[i] == vllm_test::kLtx2VideoDecNcNoiseCounts[i]);
  }

  const double err = MaxAbsDiff(frames.data, vllm_test::kLtx2VideoDecNcGolden,
                                std::size(vllm_test::kLtx2VideoDecNcGolden));
  INFO("non-causal Conv video decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // Same weights, same latent, same noise — so if the two arms agreed, `causal`
  // would be doing nothing and this whole arm would gate nothing. Upstream asserts
  // the same thing in the generator.
  GoldenNoise causal_noise;
  vllm::Ltx2ConvVideoDecoderConfig causal_cfg = ReducedVideoDecoderConfig();
  ParamBag causal_bag = BuildVideoDecoderParams(causal_cfg);
  const vllm::Ltx2VideoFrames causal_frames = vllm::Ltx2ConvVideoDecode(
      causal_cfg, causal_bag.weights, latent, lc, lt, lh, lw, &causal_noise);
  REQUIRE(causal_frames.data.size() == frames.data.size());
  CHECK(causal_frames.data != frames.data);
}

TEST_CASE("ltx2 vae: video temporal causality is one-sided, proven by perturbation") {
  // The trap this catches: putting temporal padding on BOTH sides of a causal
  // Conv3d — or zero-padding it the way MiniMax-H3's Conv3d does instead of
  // replicating frame 0 — still yields a finite, plausible clip.
  //
  // The probe runs on a block list WITHOUT `res_x_y` and without timestep
  // conditioning, because `res_x_y`'s shortcut norm is a one-group GroupNorm over
  // (C, T, H, W) whose statistics span TIME (resnet.py:93-97): with it in the
  // stack nothing is causal however right the padding is. Section 5b records
  // upstream's OWN reach for this stripped config, so the expected window is the
  // oracle's, not a claim the port makes about itself.
  vllm::Ltx2ConvVideoDecoderConfig cfg = ReducedVideoDecoderConfig();
  cfg.timestep_conditioning = false;
  cfg.prefix = "ltx2.videodeccausal.";
  cfg.decoder_blocks = {
      {"res_x", 1, 0, false, false},
      {"compress_all", 1, 1, false, /*residual=*/false},
      {"res_x", 1, 0, false, false},
  };
  ParamBag bag = BuildVideoDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoDecCausalParamNames,
                vllm_test::kLtx2VideoDecCausalParamCounts,
                std::size(vllm_test::kLtx2VideoDecCausalParamNames));
  const int64_t lc = vllm_test::kLtx2VideoDecLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecLatentW;
  const std::vector<float> latent = Ltx2Input("ltx2.videodec.input", lc * lt * lh * lw, 1.0);

  GoldenNoise noise_a;
  const vllm::Ltx2VideoFrames base =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, latent, lc, lt, lh, lw, &noise_a);

  std::vector<float> bumped = latent;
  for (int64_t c = 0; c < lc; ++c) {
    for (int64_t h = 0; h < lh; ++h) {
      for (int64_t w = 0; w < lw; ++w) {
        bumped[static_cast<size_t>(((c * lt + (lt - 1)) * lh + h) * lw + w)] += 5.0f;
      }
    }
  }
  GoldenNoise noise_b;
  const vllm::Ltx2VideoFrames moved =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, bumped, lc, lt, lh, lw, &noise_b);
  REQUIRE(moved.data.size() == base.data.size());
  CHECK(base.frames == vllm_test::kLtx2VideoDecCausalOutT);
  // With timestep conditioning off and no inject_noise block, nothing may draw.
  CHECK(noise_a.counts().empty());
  CHECK(noise_b.counts().empty());

  const int64_t plane = base.height * base.width;
  int64_t first_moved = base.frames;
  int64_t last_moved = -1;
  for (int64_t t = 0; t < base.frames; ++t) {
    bool differs = false;
    for (int64_t c = 0; c < base.channels && !differs; ++c) {
      for (int64_t i = 0; i < plane; ++i) {
        const size_t idx = static_cast<size_t>((c * base.frames + t) * plane + i);
        if (base.data[idx] != moved.data[idx]) {
          differs = true;
          break;
        }
      }
    }
    if (differs) {
      if (first_moved == base.frames) first_moved = t;
      last_moved = t;
    }
  }
  INFO("frames moved by a last-latent-frame bump: [" << first_moved << ", " << last_moved << "]");
  CHECK(first_moved == vllm_test::kLtx2VideoDecCausalFirstMoved);
  CHECK(last_moved == vllm_test::kLtx2VideoDecCausalLastMoved);
  CHECK(first_moved > 0);   // the past cannot see the future
  CHECK(last_moved >= 0);   // and the bump must reach SOMETHING, or the probe is vacuous
}

TEST_CASE("ltx2 vae: the goldens carry the upstream revision they came from") {
  // AGENTS.md §"vLLM is the reference" requires a ported test to preserve the
  // upstream REVISION ANCHOR. Without one, a Lightricks change to (say)
  // video_vae/resnet.py plus a regeneration moves every number here with nothing
  // to say whether the PORT drifted or UPSTREAM did, and nothing to bisect from.
  //
  // The generator resolves `git -C <--ltx2> rev-parse HEAD` and emits it; this
  // asserts it equals the SHA the suite PINS, so regenerating against a different
  // checkout fails the gate instead of silently replacing the oracle. Advancing
  // the pin is a deliberate edit here and in the generator docstring — never a
  // side effect of regenerating.
  constexpr const char* kLtx2VaeUpstreamRevisionPin =
      "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca";

  const std::string revision = vllm_test::kLtx2VaeUpstreamRevision;
  INFO("goldens were generated from Lightricks/LTX-2 @ " << revision);
  // "unknown" is what a tarball checkout with no git metadata yields, and it is
  // NOT an acceptable anchor: provenance you cannot bisect is not provenance.
  CHECK(revision.size() == 40);
  for (char c : revision) {
    CHECK(((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')));
  }
  CHECK(revision == kLtx2VaeUpstreamRevisionPin);
}

TEST_CASE("ltx2 vae: every stabilizing epsilon is pinned to its upstream line") {
  // THE INVISIBLE-CONSTANT CLASS. An epsilon that exists to stabilize a division
  // is by construction invisible to a reduced-dimension parity gate: the
  // deterministic stream produces O(1) activations, the guarded term never binds,
  // and the tensor comparison accepts ANY value — including 0.0 and including one
  // 100x off. Each of these was mutated with every golden staying green, so each
  // is held HERE, cited to the upstream line that sets it. Adding a new constant
  // without adding it to this list reopens the hole.

  // ResnetBlock3D's `eps: float = 1e-6` (video_vae/resnet.py:31), handed to every
  // nn.GroupNorm it builds (resnet.py:44, 65, 94) and carried by UNetMidBlock3D as
  // `resnet_eps` (resnet.py:216). This is the norm `res_x_y`'s shortcut uses.
  // Mutation: 1e-6 -> 1e-4, a 100x change, left every golden green.
  CHECK(vllm::Ltx2ConvVideoDecoderConfig{}.norm_eps ==
        doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));

  // `torch.clamp(mel, min=1e-5)` before the log (audio_vae/vocoder.py:515). The
  // member of the class that BINDS IN PRODUCTION, because real silence reaches it.
  // Mutation: 1e-5 -> 1e-8 left every golden green — until the saturating arm
  // below, which is the one place the reduced stream can be pushed into reaching
  // it.
  CHECK(vllm::kLtx2BweMelLogClamp == doctest::Approx(1e-5).epsilon(1e-12).scale(0.0));

  // Snake/SnakeBeta's `self.eps = 1e-9` (audio_vae/vocoder.py:198 and :221), the
  // stabilizer in `1 / (beta + eps)`. Shared with MiniMax-H3's BigVGAN, which is
  // why it lives in minimax_h3.h. Mutation: 1e-9 -> 0.0 left every golden green,
  // because beta is O(1) here and never approaches zero.
  CHECK(vllm::kMiniMaxH3SnakeEps == doctest::Approx(1e-9).epsilon(1e-12).scale(0.0));

  // `_RMSNorm2D` is `F.normalize(x, dim=1) * (sqrt(C) * gamma)`
  // (video_vae/attention.py:11-30), so the floor is torch's `F.normalize` DEFAULT
  // eps of 1e-12 — an L2 normalize, not a mean-square RMS. Mutation: 1e-12 -> 0.0
  // left every golden green; it decides whether an all-zero channel vector
  // divides or produces NaN.
  CHECK(vllm::kLtx2RmsNorm2dEps == doctest::Approx(1e-12).epsilon(1e-12).scale(0.0));
}

TEST_CASE("ltx2 vae: the BWE mel log clamp is gated where it actually binds") {
  // The ordinary BWE arm leaves `torch.clamp(mel, min=1e-5)` (vocoder.py:515)
  // inert — its raw mel minimum is ~4.4e-3, and it stays there even for a zero
  // input, because the vocoder's conv biases keep the waveform off silence. So
  // that arm can never move under a mutation of the constant.
  //
  // This arm attenuates mel_basis by 1e-4 (see the generator's `param_values`) so
  // EVERY bin lands under the clamp and the constant alone decides what the
  // bwe_generator consumes — which is what real silence does in production. The
  // saturation count comes from the generator, so the probe is proven saturated
  // rather than assumed to be.
  REQUIRE(vllm_test::kLtx2BweQuietSaturatedBins > 0);

  vllm::Ltx2VocoderBweConfig cfg;
  cfg.vocoder = ReducedVocoderConfig();
  cfg.vocoder.prefix = "ltx2.bwequiet.vocoder.";
  cfg.bwe_generator = ReducedVocoderConfig();
  cfg.bwe_generator.prefix = "ltx2.bwequiet.bwe_generator.";
  cfg.bwe_generator.resblock_kernel_sizes = {3};
  cfg.bwe_generator.resblock_dilation_sizes = {{1, 3, 5}};
  cfg.bwe_generator.upsample_rates = {4, 4};
  cfg.bwe_generator.upsample_kernel_sizes = {8, 8};
  cfg.bwe_generator.apply_final_activation = false;
  cfg.bwe_generator.output_sampling_rate = 32000;
  cfg.filter_length = 16;
  cfg.hop_length = 8;
  cfg.win_length = 16;
  cfg.n_mel_channels = 64;
  cfg.input_sampling_rate = 16000;
  cfg.output_sampling_rate = 32000;
  cfg.prefix = "ltx2.bwequiet.";

  ParamBag bag;
  PutVocoderParams(bag, cfg.vocoder);
  PutVocoderParams(bag, cfg.bwe_generator);
  bag.Put(cfg.prefix + "mel_stft.mel_basis", {cfg.n_mel_channels, cfg.filter_length / 2 + 1});
  bag.Put(cfg.prefix + "mel_stft.stft_fn.forward_basis",
          {(cfg.filter_length / 2 + 1) * 2, 1, cfg.filter_length});
  bag.Put(cfg.prefix + "mel_stft.stft_fn.inverse_basis",
          {(cfg.filter_length / 2 + 1) * 2, 1, cfg.filter_length});
  CheckManifest(bag, vllm_test::kLtx2BweQuietParamNames, vllm_test::kLtx2BweQuietParamCounts,
                std::size(vllm_test::kLtx2BweQuietParamNames));

  const std::vector<float> mel =
      Ltx2Input("ltx2.voc.input", 2 * vllm_test::kLtx2VocFrames * vllm_test::kLtx2VocMelBins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderWithBweForward(cfg, bag.weights, mel, 2, vllm_test::kLtx2VocFrames,
                                      vllm_test::kLtx2VocMelBins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2BweQuietOutSamples);
  const double err =
      MaxAbsDiff(wave, vllm_test::kLtx2BweQuietGolden, std::size(vllm_test::kLtx2BweQuietGolden));
  INFO("saturated-clamp BWE max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the two PixelNorm epsilons stay different") {
  // This is a SOURCE-ANCHORED CONSTANT guard, not a numerical gate, and it exists
  // because mutation proved the numerical gate cannot do the job: flipping the
  // video decoder's eps from 1e-8 to 1e-6 leaves every golden green, since the
  // normalized activations are O(1) and the difference is ~1e-7 relative.
  //
  // The values are not interchangeable upstream. The audio VAE reaches PixelNorm
  // through build_normalization_layer, which passes eps=1e-6
  // (common/normalization.py:58); the video VAE constructs `PixelNorm()` bare and
  // gets its 1e-8 default (common/normalization.py:22, from video_vae/resnet.py:46
  // and conv_video_decoder.py:243). "Unifying" them is the mistake this catches.
  CHECK(vllm::Ltx2AudioDecoderConfig{}.pixel_norm_eps ==
        doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));
  CHECK(vllm::Ltx2ConvVideoDecoderConfig{}.pixel_norm_eps ==
        doctest::Approx(1e-8).epsilon(1e-12).scale(0.0));
  CHECK(vllm::Ltx2AudioDecoderConfig{}.pixel_norm_eps !=
        vllm::Ltx2ConvVideoDecoderConfig{}.pixel_norm_eps);
}

TEST_CASE("ltx2 vae: the diffusion video decoder is refused by name, never downgraded") {
  // .agents/specs/ltx-2-5.md section 0 item 2. A silent fall-back to the Conv
  // decoder would return a lower-quality render as if it were the requested one,
  // and no gate this project owns could tell.
  CHECK(vllm::Ltx2ParseVideoDecoderKind("CausalVideoAutoencoder") ==
        vllm::Ltx2VideoDecoderKind::kConv);
  CHECK(vllm::Ltx2ParseVideoDecoderKind("") == vllm::Ltx2VideoDecoderKind::kConv);
  CHECK(vllm::Ltx2ParseVideoDecoderKind("CausalDiffusionVAE") ==
        vllm::Ltx2VideoDecoderKind::kDiffusion);

  const vllm::Ltx2ConvVideoDecoderConfig cfg = ReducedVideoDecoderConfig();
  ParamBag bag = BuildVideoDecoderParams(cfg);
  const int64_t lc = vllm_test::kLtx2VideoDecLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecLatentW;
  const std::vector<float> latent = Ltx2Input("ltx2.videodec.input", lc * lt * lh * lw, 1.0);
  GoldenNoise noise;

  bool threw = false;
  std::string message;
  try {
    vllm::Ltx2VideoDecode(vllm::Ltx2VideoDecoderKind::kDiffusion, cfg, bag.weights, latent, lc, lt,
                          lh, lw, &noise);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }
  REQUIRE(threw);
  INFO("refusal message: " << message);
  // The message must NAME the missing piece, not just say "unsupported".
  CHECK(message.find("NADiffusionDecoder") != std::string::npos);
  CHECK(message.find("neighborhood") != std::string::npos);
  // And nothing may have been decoded on the way to the refusal.
  CHECK(noise.counts().empty());
}
