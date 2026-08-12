// LTX-2.5 TEXT CONDITIONING — see include/vllm/model_executor/models/ltx2_text_encoder.h
// for the port map, the four silent-failure traps and the DTYPE note.
//
// Ported from Lightricks/LTX-2,
// packages/ltx-core/src/ltx_core/text_encoders/gemma/{feature_extractor,
// embeddings_processor,gemma_assets}.py and encoders/encoder_configurator.py.
//
// Reduction accumulators are `double`. That is an accumulator width, not a memory
// format: every buffer this file produces, stores or returns is f32, so it does
// not widen the stream the way AGENTS.md's dtype-polarity rule is about. Upstream
// reduces in the tensor dtype; at the shipped 3840 x 49 = 188160-wide projection a
// naive f32 accumulation would be materially worse than torch's blocked GEMM, and
// a double accumulator lands strictly closer to it than a naive f32 one does.
#include "vllm/model_executor/models/ltx2_text_encoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("ltx2 text encoder: " + message);
}

void RequireF32(vt::DType dtype) {
  if (dtype != vt::DType::kF32) {
    Fail(
        "compute_dtype must be f32. The bf16 / FP8 / NVFP4 arms of the text tower "
        "are phase L6 of .agents/specs/ltx-2-5.md and are NOT implemented; "
        "computing them in f32 would silently return a wider-than-checkpoint "
        "result rather than the requested arm.");
  }
}

// `torch.nn.functional.linear`: out[b, o] = sum_i x[b, i] * W[o, i] + bias[o].
// `weight` is row-major [out_features, in_features], torch's own layout.
std::vector<float> Linear(const std::vector<float>& x, int64_t rows,
                          const Ltx2TextAggregateEmbed& w) {
  if (static_cast<int64_t>(x.size()) != rows * w.in_features)
    Fail("linear: input size does not match in_features");
  if (static_cast<int64_t>(w.weight.size()) != w.out_features * w.in_features)
    Fail("linear: weight size does not match [out_features, in_features]");
  const bool has_bias = !w.bias.empty();
  if (has_bias && static_cast<int64_t>(w.bias.size()) != w.out_features)
    Fail("linear: bias size does not match out_features");

  std::vector<float> out(static_cast<size_t>(rows * w.out_features));
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x.data() + static_cast<size_t>(r * w.in_features);
    for (int64_t o = 0; o < w.out_features; ++o) {
      const float* wr = w.weight.data() + static_cast<size_t>(o * w.in_features);
      double acc = has_bias ? static_cast<double>(w.bias[static_cast<size_t>(o)]) : 0.0;
      for (int64_t i = 0; i < w.in_features; ++i)
        acc += static_cast<double>(xr[i]) * static_cast<double>(wr[i]);
      out[static_cast<size_t>(r * w.out_features + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

// encoder_configurator.py:163-168 — the EXACT V2 marker set, values included.
struct V2Marker {
  const char* key;
  bool expected;
};
constexpr V2Marker kV2Markers[] = {
    {"caption_proj_before_connector", true},
    {"caption_projection_first_linear", false},
    {"caption_proj_input_norm", false},
    {"caption_projection_second_linear", false},
};

// The DECLARED contract, checked against the SUPPLIED weights.
//
// `config.aggregate_bias` and `config.*_out_features` are what the SELECTOR read
// out of the checkpoint's config (Ltx2SelectTextFeatureVariant, :130 and :166).
// `w.bias.empty()`, `w.out_features` and `w.in_features` are what the LOADER
// actually bound. Upstream builds both `nn.Linear`s from the one config object
// (encoder_configurator.py:187, 206-208) and therefore cannot disagree with
// itself; a port that resolves the config and the tensors on separate paths can,
// and nothing compared them.
//
// The concrete failure this refuses, which is the LTX-2.5 loader's most likely
// mistake: `video_aggregate_embed.weight` is U8/NVFP4 and `.bias` is BF16 — a
// different dtype on a different unpack path — so a loader can bind the weight
// and miss the bias while the config still says bias=True. Every conditioning row
// is then shifted by the missing bias, and every padded row projects to 0 instead
// of to the bias (feature_extractor.py:44-45 zeroes the NORM, not the
// projection). Finite, correctly shaped, wrong prompt.
void RequireDeclaredProjection(const char* which, const Ltx2TextAggregateEmbed& w,
                               int64_t declared_out, bool declared_bias, int64_t flat) {
  const bool has_bias = !w.bias.empty();
  if (has_bias != declared_bias)
    Fail(std::string(which) + " projection: the config declares aggregate_bias=" +
         (declared_bias ? "true" : "false") + " but the supplied weights carry " +
         (has_bias ? "a bias" : "NO bias") +
         ". A missing bias shifts every conditioning row and projects every PADDED "
         "row to 0 instead of to the bias — finite, correctly shaped, wrong prompt. "
         "The .weight is U8/NVFP4 and the .bias is BF16, so a loader can bind one "
         "and miss the other (encoder_configurator.py:187, 206-208).");
  if (w.out_features != declared_out)
    Fail(std::string(which) + " projection: the config declares out_features=" +
         std::to_string(declared_out) + " but the supplied weights are " +
         std::to_string(w.out_features) +
         " wide. The width that RUNS comes from the weights; the config's copy only "
         "feeds _rescale_norm (feature_extractor.py:121-129), so a mismatch rescales "
         "by one width and emits another.");
  if (w.in_features != flat)
    Fail(std::string(which) + " projection: the config declares in_features=" +
         std::to_string(flat) + " = embedding_dim x (num_hidden_layers + 1) but the "
         "supplied weights take " + std::to_string(w.in_features) +
         ". LTX conditions on EVERY Gemma hidden state plus the embedding output "
         "(encoder_configurator.py:182).");
}

int64_t RequireInt(const nlohmann::json& config, const char* key) {
  if (!config.contains(key) || !config.at(key).is_number_integer())
    Fail(std::string("transformer config is missing integer key '") + key + "'");
  return config.at(key).get<int64_t>();
}

}  // namespace

// ───────────────────────────── asset accessors ───────────────────────────────

const std::vector<uint8_t>& Ltx2GemmaAssets::SidecarBytes(const std::string& name) const {
  const auto it = sidecars.find(name);
  if (it == sidecars.end()) Fail("gemma assets are missing sidecar '" + name + "'");
  return it->second;
}

nlohmann::json Ltx2GemmaAssets::SidecarJson(const std::string& name) const {
  const std::vector<uint8_t>& bytes = SidecarBytes(name);
  return nlohmann::json::parse(std::string(bytes.begin(), bytes.end()));
}

// ─────────────────────────── the variant selection ───────────────────────────

Ltx2TextFeatureConfig Ltx2SelectTextFeatureVariant(const nlohmann::json& transformer_config,
                                                   int64_t gemma_hidden_size,
                                                   int64_t gemma_num_hidden_layers) {
  if (gemma_hidden_size <= 0 || gemma_num_hidden_layers <= 0)
    Fail("gemma hidden_size and num_hidden_layers must be positive");

  Ltx2TextFeatureConfig cfg;
  cfg.embedding_dim = gemma_hidden_size;
  // encoder_configurator.py:182 — "+1 for the embedding layer".
  cfg.num_layers = Ltx2GemmaHiddenStateContract::Count(gemma_num_hidden_layers);

  std::vector<std::string> present;
  std::vector<std::string> missing;
  for (const V2Marker& marker : kV2Markers) {
    if (transformer_config.contains(marker.key))
      present.emplace_back(marker.key);
    else
      missing.emplace_back(marker.key);
  }

  // encoder_configurator.py:185-188 — no marker at all means a pre-2.5 checkpoint
  // whose single projection lives in the DiT, reached through the V1 extractor.
  if (present.empty()) {
    cfg.variant = Ltx2TextNormVariant::kPaddedBatchV1;
    cfg.video_out_features = gemma_hidden_size;  // Linear(flat_dim, embedding_dim)
    cfg.audio_out_features = 0;
    cfg.aggregate_bias = false;  // encoder_configurator.py:187 — bias=False
    cfg.is_av = true;            // encoder_configurator.py:188
    return cfg;
  }

  // encoder_configurator.py:190-192 — a PARTIAL marker set is config drift.
  if (!missing.empty()) {
    std::sort(missing.begin(), missing.end());  // upstream sorts too
    std::string names;
    for (size_t i = 0; i < missing.size(); ++i) {
      if (i != 0) names += ", ";
      names += missing[i];
    }
    Fail("Partial V2 config — missing keys: " + names);
  }

  // encoder_configurator.py:194-201 — a marker present with the WRONG value.
  std::string drift;
  for (const V2Marker& marker : kV2Markers) {
    const nlohmann::json& value = transformer_config.at(marker.key);
    if (!value.is_boolean() || value.get<bool>() != marker.expected) {
      if (!drift.empty()) drift += ", ";
      drift += std::string(marker.key) + "=" + value.dump() + " (expected " +
               (marker.expected ? "True" : "False") + ")";
    }
  }
  if (!drift.empty()) Fail("Unknown config: " + drift);

  // encoder_configurator.py:203-209.
  cfg.variant = Ltx2TextNormVariant::kPerTokenRmsV2;
  cfg.video_out_features =
      RequireInt(transformer_config, "num_attention_heads") *
      RequireInt(transformer_config, "attention_head_dim");
  cfg.audio_out_features =
      RequireInt(transformer_config, "audio_num_attention_heads") *
      RequireInt(transformer_config, "audio_attention_head_dim");
  cfg.aggregate_bias = true;  // both Linears are bias=True
  cfg.is_av = false;
  return cfg;
}

// ───────────────────────────── feature aggregation ───────────────────────────

std::vector<float> Ltx2StackHiddenStates(const Ltx2TextHiddenStates& states) {
  const int64_t B = states.batch, T = states.seq, D = states.hidden;
  const int64_t L = static_cast<int64_t>(states.layers.size());
  if (B <= 0 || T <= 0 || D <= 0 || L <= 0) Fail("hidden states have a zero extent");
  // feature_extractor.py:120 — stack on a NEW LAST axis, so layer is the fastest
  // moving index and the later reshape interleaves as `d * L + l`.
  std::vector<float> out(static_cast<size_t>(B * T * D * L));
  for (int64_t l = 0; l < L; ++l) {
    const float* src = states.layers[static_cast<size_t>(l)];
    if (src == nullptr) Fail("hidden state layer pointer is null");
    for (int64_t bt = 0; bt < B * T; ++bt)
      for (int64_t d = 0; d < D; ++d)
        out[static_cast<size_t>((bt * D + d) * L + l)] =
            src[static_cast<size_t>(bt * D + d)];
  }
  return out;
}

std::vector<float> Ltx2NormAndConcatPaddedBatch(const float* stacked, const int32_t* mask,
                                                int64_t batch, int64_t seq,
                                                int64_t hidden, int64_t layers) {
  if (stacked == nullptr || mask == nullptr) Fail("null input to the V1 normalization");
  const int64_t B = batch, T = seq, D = hidden, L = layers;
  // feature_extractor.py:28 — the ONE `eps`, named in the header and pinned there
  // against the value measured out of upstream, because neither of its two uses is
  // reachable from a random fixture.
  constexpr double kEps = kLtx2TextNormV1Eps;
  const size_t count = static_cast<size_t>(B * T * D * L);
  std::vector<float> out(count);

  for (int64_t b = 0; b < B; ++b) {
    // feature_extractor.py:30 — sequence_lengths = attention_mask.sum(-1).
    int64_t seq_len = 0;
    for (int64_t t = 0; t < T; ++t)
      if (mask[static_cast<size_t>(b * T + t)] != 0) ++seq_len;
    // :34 — denom = (sequence_lengths * d), i.e. the number of VALID (t, d) pairs.
    // INT64, exactly as upstream: `attention_mask.sum(-1)` is an int64 tensor and
    // `* d` keeps it there. The width it is added to `eps` in is decided at :35.
    const int64_t denom = seq_len * D;

    for (int64_t l = 0; l < L; ++l) {
      // :33-38 — mean over the masked entries and min/max over them, both reduced
      // over BOTH the token and the hidden axis, per (batch, layer).
      //
      // f64 SUM ACCUMULATOR, and it is the deliberate escape rather than an
      // oversight: upstream's `masked.sum` is a float32 reduction, but a BLOCKED
      // one whose order no straight loop reproduces. Accumulating exactly and
      // rounding once is the closest single-rounding approximation to any order.
      // Measured against upstream on this fixture: f64 accumulate reads 2.38e-07
      // (left) / 4.77e-07 (right), a naive f32 accumulate reads 4.77e-07 / 2.38e-07
      // — it wins on one mask and loses on the other, which is noise, not a mirror.
      // The min/max stay f32 because they ARE f32 values and `hi - lo` is then the
      // same single rounding upstream's f32 subtraction does; taking the
      // difference in f64 and rounding after would double-round.
      double sum = 0.0;
      float lo = std::numeric_limits<float>::infinity();
      float hi = -std::numeric_limits<float>::infinity();
      for (int64_t t = 0; t < T; ++t) {
        if (mask[static_cast<size_t>(b * T + t)] == 0) continue;
        for (int64_t d = 0; d < D; ++d) {
          const float v = stacked[static_cast<size_t>((((b * T) + t) * D + d) * L + l)];
          sum += static_cast<double>(v);
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
      }
      // :35 — `denom + eps` where `denom` is an int64 TENSOR and `eps` a python
      // float. Torch promotes int64-tensor + python-float to the DEFAULT dtype, so
      // that add happens in FLOAT32, not float64: upstream's denominator for
      // seq_len*D == 18 is 18.000001907348633, where an f64 add gives 18.000001.
      // Adding in f64 is numerically finer and therefore WRONG here — it is a
      // dtype that is too wide, which no token gate and no random-input golden can
      // see. It costs one f32 ulp in `mean`, which `range_ + eps` below multiplies
      // by 8/eps = 8e6 as `range_` collapses: on a constant slice it moves the
      // output from 0.476837158 to 0.238418579. Gated on that input in
      // tests/vllm/models/test_ltx2_text_encoder.cpp.
      const float mean = static_cast<float>(sum) /
                         (static_cast<float>(denom) + static_cast<float>(kEps));
      const float range = hi - lo;

      // :41 — 8 * (x - mean) / (range + eps), applied to the UNMASKED tensor.
      for (int64_t t = 0; t < T; ++t) {
        const bool valid = mask[static_cast<size_t>(b * T + t)] != 0;
        for (int64_t d = 0; d < D; ++d) {
          const size_t src = static_cast<size_t>((((b * T) + t) * D + d) * L + l);
          // :42 — reshape [B, T, D, L] -> [B, T, D * L], so `d * L + l`.
          const size_t dst = static_cast<size_t>(((b * T) + t) * D * L + d * L + l);
          // :44-45 — the padded positions are zeroed AFTER the normalization.
          out[dst] = valid ? 8.0f * (stacked[src] - mean) /
                                 (range + static_cast<float>(kEps))
                           : 0.0f;
        }
      }
    }
  }
  return out;
}

std::vector<float> Ltx2NormAndConcatPerTokenRms(const float* stacked, const int32_t* mask,
                                                int64_t batch, int64_t seq,
                                                int64_t hidden, int64_t layers) {
  if (stacked == nullptr || mask == nullptr) Fail("null input to the V2 normalization");
  const int64_t B = batch, T = seq, D = hidden, L = layers;
  constexpr float kEps = kLtx2TextNormV2Eps;  // feature_extractor.py:61
  std::vector<float> out(static_cast<size_t>(B * T * D * L));

  for (int64_t bt = 0; bt < B * T; ++bt) {
    const bool valid = mask[static_cast<size_t>(bt)] != 0;
    for (int64_t l = 0; l < L; ++l) {
      // :60 — the variance reduces over dim=2, the HIDDEN axis, per (b, t, layer).
      // The mask does NOT participate: a padded token's own values set its own
      // scale, and the result is discarded at :64.
      double sum_sq = 0.0;
      for (int64_t d = 0; d < D; ++d) {
        const double v =
            static_cast<double>(stacked[static_cast<size_t>((bt * D + d) * L + l)]);
        sum_sq += v * v;
      }
      const float variance = static_cast<float>(sum_sq / static_cast<double>(D));
      const float inv = 1.0f / std::sqrt(variance + kEps);
      for (int64_t d = 0; d < D; ++d) {
        const size_t src = static_cast<size_t>((bt * D + d) * L + l);
        const size_t dst = static_cast<size_t>(bt * D * L + d * L + l);
        out[dst] = valid ? stacked[src] * inv : 0.0f;
      }
    }
  }
  return out;
}

double Ltx2RescaleNorm(int64_t target_dim, int64_t source_dim) {
  if (source_dim <= 0) Fail("rescale: source_dim must be positive");
  // feature_extractor.py:69 — `x * math.sqrt(target_dim / source_dim)`. torch
  // converts a Python float scalar to the TENSOR's dtype before multiplying, so
  // the factor that actually multiplies an f32 activation is the f32-rounded one.
  // Returning the double here instead would diverge from upstream in the last
  // ulps of every conditioning value.
  const double exact = std::sqrt(static_cast<double>(target_dim) /
                                 static_cast<double>(source_dim));
  return static_cast<double>(static_cast<float>(exact));
}

Ltx2TextFeatures Ltx2TextFeatureExtractorForward(const Ltx2TextHiddenStates& states,
                                                 const int32_t* mask,
                                                 const Ltx2TextEncoderWeights& weights,
                                                 const Ltx2TextFeatureConfig& config,
                                                 vt::DType compute_dtype) {
  RequireF32(compute_dtype);
  if (mask == nullptr) Fail("attention mask is null");
  if (static_cast<int64_t>(states.layers.size()) != config.num_layers)
    Fail("hidden state count " + std::to_string(states.layers.size()) +
         " != num_hidden_layers + 1 = " + std::to_string(config.num_layers) +
         ". LTX conditions on EVERY Gemma hidden state plus the embedding output "
         "(base_encoder.py:68-71), not on the last one.");
  if (states.hidden != config.embedding_dim)
    Fail("hidden width does not match the configured gemma hidden_size");

  const int64_t B = states.batch, T = states.seq;
  const int64_t flat = config.FlatDim();

  // The declared contract, checked BEFORE any work — see RequireDeclaredProjection.
  // Only the projections that actually run are checked: V1 uses `video` alone and
  // returns it twice under `is_av` (feature_extractor.py:95-96), and V2's audio arm
  // exists only when the config gave it a width.
  RequireDeclaredProjection("video", weights.video, config.video_out_features,
                            config.aggregate_bias, flat);
  if (config.variant != Ltx2TextNormVariant::kPaddedBatchV1 &&
      config.audio_out_features > 0)
    RequireDeclaredProjection("audio", weights.audio, config.audio_out_features,
                              config.aggregate_bias, flat);

  const std::vector<float> stacked = Ltx2StackHiddenStates(states);

  std::vector<float> normed =
      config.variant == Ltx2TextNormVariant::kPaddedBatchV1
          ? Ltx2NormAndConcatPaddedBatch(stacked.data(), mask, B, T, states.hidden,
                                         config.num_layers)
          : Ltx2NormAndConcatPerTokenRms(stacked.data(), mask, B, T, states.hidden,
                                         config.num_layers);

  Ltx2TextFeatures features;
  if (config.variant == Ltx2TextNormVariant::kPaddedBatchV1) {
    // feature_extractor.py:93-97 — ONE projection, no rescale, and `is_av` returns
    // the same tensor twice rather than running a second projection.
    features.video = Linear(normed, B * T, weights.video);
    if (config.is_av) features.audio = features.video;
    return features;
  }

  // feature_extractor.py:121-129 — the rescale is applied SEPARATELY per
  // projection, each with that projection's OWN out_features over the GEMMA
  // hidden size (not over the flat width).
  auto project = [&](const Ltx2TextAggregateEmbed& w, int64_t out_features) {
    const float scale = static_cast<float>(Ltx2RescaleNorm(out_features, config.embedding_dim));
    std::vector<float> scaled(normed.size());
    for (size_t i = 0; i < normed.size(); ++i) scaled[i] = normed[i] * scale;
    return Linear(scaled, B * T, w);
  };
  features.video = project(weights.video, config.video_out_features);
  if (config.audio_out_features > 0)
    features.audio = project(weights.audio, config.audio_out_features);
  return features;
}

// ──────────────────── the encoder -> conditioning hand-off ───────────────────

std::vector<float> Ltx2ConvertToAdditiveMask(const int32_t* mask, int64_t batch,
                                             int64_t seq) {
  if (mask == nullptr) Fail("attention mask is null");
  // embeddings_processor.py:18-20 — (mask - 1) * finfo(f32).max: 0.0 for a kept
  // position, -FLT_MAX for a pad.
  const float big = std::numeric_limits<float>::max();
  std::vector<float> out(static_cast<size_t>(batch * seq));
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<float>(mask[i] != 0 ? 0 : -1) * big;
  return out;
}

void Ltx2ComputeRightPadOrder(const float* additive_mask, int64_t batch, int64_t seq,
                              std::vector<int32_t>& sort_index,
                              std::vector<float>& reordered_additive_mask) {
  if (additive_mask == nullptr) Fail("additive mask is null");
  const float big = std::numeric_limits<float>::max();
  sort_index.assign(static_cast<size_t>(batch * seq), 0);
  reordered_additive_mask.assign(static_cast<size_t>(batch * seq), 0.0f);

  for (int64_t b = 0; b < batch; ++b) {
    // embeddings_processor.py:34-36 — binary = (additive >= 0), then a STABLE
    // descending argsort, i.e. valid positions first in their original relative
    // order, pads after in theirs. Written as a stable partition, which is the
    // same permutation for a 0/1 key and avoids a comparator that could reorder
    // equal elements.
    int64_t w = 0;
    for (int64_t t = 0; t < seq; ++t)
      if (additive_mask[static_cast<size_t>(b * seq + t)] >= 0.0f)
        sort_index[static_cast<size_t>(b * seq + w++)] = static_cast<int32_t>(t);
    const int64_t valid = w;
    for (int64_t t = 0; t < seq; ++t)
      if (additive_mask[static_cast<size_t>(b * seq + t)] < 0.0f)
        sort_index[static_cast<size_t>(b * seq + w++)] = static_cast<int32_t>(t);
    // :37 — the reordered mask is rebuilt from the reordered BINARY values, so it
    // carries the canonical -FLT_MAX rather than whatever the input held.
    for (int64_t t = 0; t < seq; ++t)
      reordered_additive_mask[static_cast<size_t>(b * seq + t)] =
          t < valid ? 0.0f : -big;
  }
}

std::vector<float> Ltx2ApplyRightPadOrder(const float* features, const int32_t* sort_index,
                                          int64_t batch, int64_t seq, int64_t dim) {
  if (features == nullptr || sort_index == nullptr) Fail("null input to the right-pad gather");
  std::vector<float> out(static_cast<size_t>(batch * seq * dim));
  for (int64_t b = 0; b < batch; ++b)
    for (int64_t t = 0; t < seq; ++t) {
      const int64_t src = sort_index[static_cast<size_t>(b * seq + t)];
      if (src < 0 || src >= seq) Fail("right-pad sort index out of range");
      std::copy_n(features + static_cast<size_t>((b * seq + src) * dim),
                  static_cast<size_t>(dim),
                  out.begin() + static_cast<ptrdiff_t>((b * seq + t) * dim));
    }
  return out;
}

std::vector<int32_t> Ltx2ToBinaryMask(const float* encoded_mask, int64_t batch,
                                      int64_t seq) {
  if (encoded_mask == nullptr) Fail("encoded mask is null");
  // embeddings_processor.py:48 — (encoded_mask < 0.000001). See the header: BOTH
  // masks upstream can pass in satisfy this everywhere.
  std::vector<int32_t> out(static_cast<size_t>(batch * seq));
  for (size_t i = 0; i < out.size(); ++i) out[i] = encoded_mask[i] < 1e-6f ? 1 : 0;
  return out;
}

Ltx2TextConditioning Ltx2TextEncoderConditioning(const Ltx2TextHiddenStates& states,
                                                 const int32_t* mask,
                                                 const Ltx2TextEncoderWeights& weights,
                                                 const Ltx2TextFeatureConfig& config,
                                                 vt::DType compute_dtype) {
  RequireF32(compute_dtype);
  // embeddings_processor.py:114-116 — extract, convert the mask, then normalize
  // the padding layout to right-padded before the connector sees anything.
  const Ltx2TextFeatures features =
      Ltx2TextFeatureExtractorForward(states, mask, weights, config, compute_dtype);
  const std::vector<float> additive =
      Ltx2ConvertToAdditiveMask(mask, states.batch, states.seq);

  Ltx2TextConditioning out;
  // :84 — the sort index depends only on the mask, so it is computed ONCE and
  // reused for the audio arm.
  Ltx2ComputeRightPadOrder(additive.data(), states.batch, states.seq, out.sort_index,
                           out.additive_mask);
  out.video = Ltx2ApplyRightPadOrder(features.video.data(), out.sort_index.data(),
                                     states.batch, states.seq, config.video_out_features);
  if (!features.audio.empty()) {
    const int64_t audio_dim =
        config.audio_out_features > 0 ? config.audio_out_features : config.video_out_features;
    out.audio = Ltx2ApplyRightPadOrder(features.audio.data(), out.sort_index.data(),
                                       states.batch, states.seq, audio_dim);
  }
  return out;
}

// ───────────────────── the embedded tokenizer / asset pack ───────────────────

namespace {

// gemma_assets.py:302-307 — the pack stores asset bytes as a U8 tensor; Comfy may
// emit I8 for the same bytes, which reinterprets identically.
std::vector<uint8_t> AssetBytes(const StTensor& tensor, const std::string& name) {
  if (tensor.dtype != "U8" && tensor.dtype != "I8")
    Fail("asset tensor '" + name + "' has dtype " + tensor.dtype + ", expected U8 or I8");
  return std::vector<uint8_t>(tensor.data, tensor.data + tensor.nbytes);
}

// gemma_assets.py:38-41 — the two sidecars a pack MUST carry.
constexpr const char* kRequiredSidecars[] = {"tokenizer_config.json",
                                             "processor_config.json"};
// gemma_assets.py:43-47 — older / Comfy packs may store these as metadata strings.
constexpr const char* kMetadataFallbacks[] = {"tokenizer_config.json",
                                              "processor_config.json",
                                              "chat_template.jinja",
                                              "generation_config.json"};

}  // namespace

Ltx2GemmaAssets Ltx2LoadGemmaAssets(const SafetensorsFile& file, bool require_config) {
  Ltx2GemmaAssets assets;
  const std::map<std::string, std::string>& metadata = file.Metadata();

  // gemma_assets.py:108-115 — the HF config, JSON-encoded inside __metadata__.
  const auto config_it = metadata.find("gemma_config");
  if (config_it == metadata.end()) {
    if (require_config)
      Fail(
          "safetensors text encoder is missing metadata key 'gemma_config' "
          "(JSON-encoded HuggingFace config). MEASURED: the shipped "
          "vonkaiser/LTX-2.5-FP8-NVFP4 text encoder carries NO __metadata__ block "
          "at all, and upstream's GemmaAssets.from_single_file refuses it too "
          "(gemma_assets.py:110-114). Supply the Gemma config out of band and pass "
          "require_config=false.");
  } else {
    assets.config = nlohmann::json::parse(config_it->second);
    assets.has_config = true;
  }

  // :117-120 — the tokenizer, as one ~32 MB U8 tensor.
  const std::vector<std::string>& names = file.Names();
  if (std::find(names.begin(), names.end(), std::string(kLtx2GemmaTokenizerTensor)) ==
      names.end())
    Fail(std::string("safetensors text encoder is missing tensor '") +
         kLtx2GemmaTokenizerTensor +
         "'. The LTX-2.5 pack embeds the tokenizer AS A TENSOR; a loader that "
         "expects a sibling tokenizer.json file cannot read this checkpoint.");
  assets.tokenizer_json =
      AssetBytes(file.Get(kLtx2GemmaTokenizerTensor), kLtx2GemmaTokenizerTensor);

  // :122-127 — every hf_asset__<name> tensor is a sidecar file.
  const std::string prefix(kLtx2GemmaAssetPrefix);
  for (const std::string& name : names) {
    if (name.compare(0, prefix.size(), prefix) != 0) continue;
    assets.sidecars[name.substr(prefix.size())] = AssetBytes(file.Get(name), name);
  }

  // :129-132 — metadata fallback for the small JSON sidecars.
  for (const char* name : kMetadataFallbacks) {
    if (assets.sidecars.count(name) != 0) continue;
    const auto it = metadata.find(name);
    if (it == metadata.end()) continue;
    assets.sidecars[name] = std::vector<uint8_t>(it->second.begin(), it->second.end());
  }

  // :141 / :153-159 — the required sidecars, named in the failure.
  std::string missing;
  for (const char* name : kRequiredSidecars) {
    if (assets.sidecars.count(name) != 0) continue;
    if (!missing.empty()) missing += ", ";
    missing += name;
  }
  if (!missing.empty())
    Fail("safetensors text encoder is missing required sidecar(s): " + missing +
         " (embed as " + prefix + "<name>, or as metadata for small JSON)");

  return assets;
}

}  // namespace vllm
