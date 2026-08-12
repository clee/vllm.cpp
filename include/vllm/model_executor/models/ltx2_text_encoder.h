// LTX-2.5 TEXT CONDITIONING — the Gemma-4 multi-layer feature aggregation, the
// two caption projections, and the embedded tokenizer/asset pack.
//
// ─── THE ONE THING THAT MAKES THIS DIFFERENT FROM EVERY OTHER TEXT ENCODER ────
//
// LTX-2.5 does NOT condition on the encoder's last hidden state. It takes EVERY
// Gemma-4 hidden state — the embedding output plus all 48 decoder outputs, 49 in
// total — stacks them on a new LAST axis to [batch, seq, hidden, layers],
// normalizes, concatenates ACROSS THE LAYER AXIS, and projects the flattened
// result twice (feature_extractor.py:114-129).
//
// Measured on the shipped `vonkaiser/LTX-2.5-FP8-NVFP4` text encoder
// (`gemma4-12b-with-proj-nvfp4-torchao.safetensors`, 1688 tensors):
//
//   text_embedding_projection.video_aggregate_embed.weight  U8 [4096, 94080]
//   text_embedding_projection.audio_aggregate_embed.weight  U8 [2048, 94080]
//   model.embed_tokens.weight                               U8 [262144, 1920]
//   model.norm.weight                                     BF16 [3840]
//   model.layers.{0..47}.*                                       48 layers
//
// NVFP4 packs TWO values per byte, so those U8 widths are HALF the real feature
// counts: the projections take 188160 = 3840 x 49 inputs, not 94080, and the
// Gemma hidden size is 3840, not 1920. `model.norm.weight [3840]` is the
// independent confirmation — it is stored BF16 and therefore unpacked.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/
//   OURS                              <-  UPSTREAM
//   Ltx2StackHiddenStates             <-  text_encoders/gemma/feature_extractor.py:120
//   Ltx2NormAndConcatPaddedBatch      <-  text_encoders/gemma/feature_extractor.py:12-45
//   Ltx2NormAndConcatPerTokenRms      <-  text_encoders/gemma/feature_extractor.py:48-64
//   Ltx2RescaleNorm                   <-  text_encoders/gemma/feature_extractor.py:67-69
//   Ltx2TextFeatureExtractorForward   <-  text_encoders/gemma/feature_extractor.py:85-129
//   Ltx2SelectTextFeatureVariant      <-  text_encoders/gemma/encoders/encoder_configurator.py:163-209
//   Ltx2ConvertToAdditiveMask         <-  text_encoders/gemma/embeddings_processor.py:16-20
//   Ltx2ComputeRightPadOrder          <-  text_encoders/gemma/embeddings_processor.py:23-38
//   Ltx2ApplyRightPadOrder            <-  text_encoders/gemma/embeddings_processor.py:41-43
//   Ltx2ToBinaryMask                  <-  text_encoders/gemma/embeddings_processor.py:46-48
//   Ltx2TextEncoderConditioning       <-  text_encoders/gemma/embeddings_processor.py:70-117
//   Ltx2LoadGemmaAssets               <-  text_encoders/gemma/gemma_assets.py:104-159
//   Ltx2GemmaHiddenStateContract      <-  text_encoders/gemma/encoders/base_encoder.py:49-71
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * THE CONCATENATION IS HIDDEN-MAJOR, LAYER-MINOR. `stack(..., dim=-1)` then
//    `.reshape(B, T, D*L)` puts layer `l` of channel `d` at flat index
//    `d * L + l`. A port that concatenates layer-major (`l * D + d`) produces a
//    correctly shaped, finite, PERMUTED conditioning vector.
//  * THERE ARE TWO NORMALIZATION VARIANTS AND THEY ARE NOT INTERCHANGEABLE.
//    V1 is a per-batch, per-layer masked mean/range with an `8 *` scale and
//    eps 1e-6; V2 is a per-token RMS over the HIDDEN axis with eps 1e-6 and no
//    scale. The choice comes from config (`Ltx2SelectTextFeatureVariant`), never
//    from a guess.
//  * THE `+1` IS THE EMBEDDING LAYER. `num_layers = num_hidden_layers + 1`
//    (encoder_configurator.py:182). Dropping it makes the projection 3840 inputs
//    too narrow, which a shape check catches; taking the LAST 48 of the 49
//    instead of the first 48 plus the embedding does NOT change any shape.
//  * V1'S PROJECTION HAS NO BIAS AND V2'S HAVE ONE (encoder_configurator.py:187,
//    206-208). Because the norm zeroes padded positions, a padded position's
//    PROJECTED value is exactly the bias — not zero. A port that force-zeroes the
//    projected pads silently diverges from upstream on every padded row. The
//    DECLARED contract below (`aggregate_bias`, `*_out_features`) is therefore
//    checked against the weights the loader actually supplied — see
//    `Ltx2TextFeatureExtractorForward`.
//
// ─── AND THE EPSILONS, WHICH ARE A CLASS AND NOT ONE INSTANCE ────────────────
// A constant that only changes the answer on a DEGENERATE input is invisible to
// any golden built from random values. Both normalizations have one, they are
// named below, and each is held two ways: its VALUE against upstream measured by
// probe, and the degenerate input on which it is the only thing between the port
// and a division by zero. When a fourth epsilon arrives, it owes the same pair —
// not a comment saying it matches upstream.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// Everything here is f32, exactly as phase L2 (ltx2.h) records for the DiT: that
// is NOT a widening of a bf16 path, it is the PARITY dtype of this gate, which
// compares the ALGORITHM against upstream run in torch float32. Upstream resolves
// ONE model dtype and every layer inherits it — `LTXGemmaTextEncoder` takes a
// single `dtype` (base_encoder.py:41) and `FeatureExtractorV2.forward` casts the
// normalized tensor straight back to `encoded.dtype` (feature_extractor.py:122),
// so the production bf16 / FP8 / NVFP4 arms are a single stream-dtype choice —
// phase L6 — and are OWED, not shipped. Every entry point that takes a
// `compute_dtype` REFUSES anything but `vt::DType::kF32` with a message naming
// the missing phase rather than silently computing in f32.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/dtype.h"

namespace vllm {

class SafetensorsFile;

// ─────────────────────────── the hidden-state contract ───────────────────────

// Which hidden states LTX consumes, and in which order.
//
// `LTXGemmaTextEncoder.encode` (base_encoder.py:68-71) calls the inner Gemma
// model with `output_hidden_states=True` and passes `outputs.hidden_states`
// straight through. In transformers that tuple has `num_hidden_layers + 1`
// entries in this exact order:
//
//   [0]              the token embeddings, AFTER the sqrt(hidden) embed scale
//   [1 .. L-1]       the output of decoder layers 0 .. L-2
//   [L]              the output of decoder layer L-1 AFTER `model.norm`
//
// The last entry is FINAL-NORMED and the raw output of the last decoder layer
// never appears. A port that appends L raw layer outputs plus the embeddings, or
// that appends the raw last layer and then the normed one, has 49 finite tensors
// of the right shape and the wrong content.
struct Ltx2GemmaHiddenStateContract {
  static constexpr const char* kOrder =
      "hidden_states[0] = embeddings * sqrt(hidden); hidden_states[i] = output of "
      "decoder layer i-1; hidden_states[num_hidden_layers] = model.norm(output of "
      "the LAST decoder layer). Count = num_hidden_layers + 1.";
  // The count the caption projections' in_features must agree with.
  static int64_t Count(int64_t num_hidden_layers) { return num_hidden_layers + 1; }
};

// A batch of Gemma hidden states, one pointer per layer, each [batch, seq, hidden]
// in row-major order. `layers.size()` must equal
// `Ltx2GemmaHiddenStateContract::Count(num_hidden_layers)`.
struct Ltx2TextHiddenStates {
  std::vector<const float*> layers;
  int64_t batch = 0;
  int64_t seq = 0;
  int64_t hidden = 0;
};

// ───────────────────────────── feature aggregation ───────────────────────────

// feature_extractor.py:28 — ONE `eps = 1e-6` bound at the top of
// `_norm_and_concat_padded_batch` and used TWICE: in the mean's denominator
// (`denom + eps`, :34) and in the range's (`range_ + eps`, :41). Named here so
// there is a single thing to pin, and pinned against the value MEASURED out of
// upstream in tests/vllm/models/test_ltx2_text_encoder.cpp.
//
// Reachability, stated because it is what makes the pin necessary: `range_ + eps`
// is reachable only when a whole (batch, layer) slice is CONSTANT over its valid
// positions, and `denom + eps` is reachable only when a batch row has NO valid
// token — and on that row every position is a pad that :44-45 zeroes, so the
// second use is UNOBSERVABLE at the output for every possible input. It is
// mirrored because upstream has it, and held by the constant, not by behaviour.
inline constexpr double kLtx2TextNormV1Eps = 1e-6;

// feature_extractor.py:61 — `torch.rsqrt(variance + 1e-6)`. Reachable only when a
// token's whole hidden slice is zero.
inline constexpr float kLtx2TextNormV2Eps = 1e-6f;

// feature_extractor.py:12-64. Which of the two normalizations runs.
enum class Ltx2TextNormVariant {
  // `_norm_and_concat_padded_batch` (:12-45) — per-batch, per-layer masked mean
  // and range, `8 * (x - mean) / (range + 1e-6)`. The 19B / V1 checkpoints.
  kPaddedBatchV1,
  // `norm_and_concat_per_token_rms` (:48-64) — per-token RMS over the HIDDEN
  // axis, `x * rsqrt(mean(x^2) + 1e-6)`. Upstream's docstring: "for V2 models".
  kPerTokenRmsV2,
};

// The resolved feature-extractor shape. Produced by `Ltx2SelectTextFeatureVariant`
// from the checkpoint config; never hand-assembled on a model path.
struct Ltx2TextFeatureConfig {
  Ltx2TextNormVariant variant = Ltx2TextNormVariant::kPerTokenRmsV2;
  int64_t embedding_dim = 0;       // gemma_text_config.hidden_size (3840)
  int64_t num_layers = 0;          // num_hidden_layers + 1        (49)
  int64_t video_out_features = 0;  // video_aggregate_embed.out_features (4096)
  int64_t audio_out_features = 0;  // audio_aggregate_embed.out_features (2048); 0 = absent
  bool aggregate_bias = false;     // V1 false (:187), V2 true (:206-208)
  bool is_av = false;              // V1 only (:188): the audio arm IS the video tensor
  int64_t FlatDim() const { return embedding_dim * num_layers; }
};

// encoder_configurator.py:163-209 — the selection, mirrored including both of its
// refusals. `transformer_config` is the diffusion checkpoint's `config.transformer`
// object.
//
//   none of the four V2 marker keys present -> V1 (projection lives in the DiT)
//   all four present with their exact expected values -> V2
//   a partial set, or a drifted value -> throws std::runtime_error naming the keys
//
// Never infers the variant from tensor shapes: 3840 x 49 is the flat width under
// BOTH variants, so shapes cannot distinguish them.
Ltx2TextFeatureConfig Ltx2SelectTextFeatureVariant(
    const nlohmann::json& transformer_config, int64_t gemma_hidden_size,
    int64_t gemma_num_hidden_layers);

// feature_extractor.py:120 — `torch.stack(hidden_states, dim=-1)`.
// Output is [batch, seq, hidden, layers], layer being the LAST (fastest) axis.
std::vector<float> Ltx2StackHiddenStates(const Ltx2TextHiddenStates& states);

// feature_extractor.py:12-45. `stacked` is [B, T, D, L]; `mask` is [B, T] in
// {0, 1}. Returns [B, T, D * L] with PADDED POSITIONS ZEROED. Padding-side
// agnostic — the binary mask alone decides which positions are valid.
std::vector<float> Ltx2NormAndConcatPaddedBatch(const float* stacked,
                                                const int32_t* mask, int64_t batch,
                                                int64_t seq, int64_t hidden,
                                                int64_t layers);

// feature_extractor.py:48-64. Same shapes; per-token RMS over the hidden axis.
// Padded positions ZEROED.
std::vector<float> Ltx2NormAndConcatPerTokenRms(const float* stacked,
                                                const int32_t* mask, int64_t batch,
                                                int64_t seq, int64_t hidden,
                                                int64_t layers);

// feature_extractor.py:67-69 — `x * sqrt(target_dim / source_dim)`, computed with
// the same `math.sqrt` of a double ratio upstream uses. V2 only.
double Ltx2RescaleNorm(int64_t target_dim, int64_t source_dim);

// One caption projection: `torch.nn.Linear(flat_dim, out_features, bias=...)`.
// `weight` is row-major [out_features, in_features] — torch's own layout, so the
// checkpoint tensor is used as stored.
struct Ltx2TextAggregateEmbed {
  std::vector<float> weight;
  std::vector<float> bias;  // empty when the Linear has bias=False
  int64_t out_features = 0;
  int64_t in_features = 0;
};

// The text encoder's projection weights. V1 populates `video` only and reports
// the same tensor for audio (`is_av`); V2 populates both.
struct Ltx2TextEncoderWeights {
  Ltx2TextAggregateEmbed video;  // text_embedding_projection.video_aggregate_embed
                                 // (V1: .aggregate_embed)
  Ltx2TextAggregateEmbed audio;  // text_embedding_projection.audio_aggregate_embed
};

// The extractor output. `audio` is empty when the config has no audio projection;
// under V1's `is_av` it is a COPY of `video`, matching upstream returning the same
// tensor twice (feature_extractor.py:95-96).
struct Ltx2TextFeatures {
  std::vector<float> video;  // [batch, seq, video_out_features]
  std::vector<float> audio;  // [batch, seq, audio_out_features] or empty
};

// feature_extractor.py:85-129 — the whole extractor: stack, normalize by the
// selected variant, (V2) rescale per projection, project. `compute_dtype` must be
// vt::DType::kF32 — see the DTYPE note at the top of this header.
//
// REFUSES, by name, any disagreement between what `config` DECLARES and what
// `weights` actually carries: `aggregate_bias` vs `w.bias.empty()`,
// `*_out_features` vs `w.out_features`, and `FlatDim()` vs `w.in_features`.
// Upstream builds both Linears from the one config object
// (encoder_configurator.py:187, 206-208) and so cannot disagree with itself; a
// port that loads the config and the tensors separately can. The concrete case is
// a loader that reads `video_aggregate_embed.weight` (U8/NVFP4) and misses
// `.bias` (BF16, a different unpack path) while the config still says bias=True:
// every conditioning row is then shifted by the missing bias and every padded row
// projects to 0 rather than to the bias — finite, correctly shaped, wrong prompt.
Ltx2TextFeatures Ltx2TextFeatureExtractorForward(
    const Ltx2TextHiddenStates& states, const int32_t* mask,
    const Ltx2TextEncoderWeights& weights, const Ltx2TextFeatureConfig& config,
    vt::DType compute_dtype = vt::DType::kF32);

// ──────────────────── the encoder -> conditioning hand-off ───────────────────

// embeddings_processor.py:16-20 — `(mask - 1) * finfo(f32).max`, i.e. 0.0 for a
// kept position and -FLT_MAX for a pad. Returns [batch, 1, 1, seq] flattened.
std::vector<float> Ltx2ConvertToAdditiveMask(const int32_t* mask, int64_t batch,
                                             int64_t seq);

// embeddings_processor.py:23-38 — the STABLE descending argsort of the binary
// mask that places valid positions before pads while preserving their relative
// order. Idempotent on an already right-padded input. Fills `sort_index`
// [batch, seq] and `reordered_additive_mask` [batch, 1, 1, seq].
void Ltx2ComputeRightPadOrder(const float* additive_mask, int64_t batch,
                              int64_t seq, std::vector<int32_t>& sort_index,
                              std::vector<float>& reordered_additive_mask);

// embeddings_processor.py:41-43 — gather `features` [batch, seq, dim] along seq
// by `sort_index`.
std::vector<float> Ltx2ApplyRightPadOrder(const float* features,
                                          const int32_t* sort_index, int64_t batch,
                                          int64_t seq, int64_t dim);

// embeddings_processor.py:46-48 — `(encoded_mask < 1e-6)` as {0, 1}, [batch, seq].
//
// MEASURED, and gated as measured: BOTH masks upstream can hand this function
// satisfy the predicate everywhere. With learnable registers on — which LTX-2.5
// has — the connector returns `zeros_like(additive_mask)` (embeddings_connector.py:152)
// and 0.0 < 1e-6; with them off it returns the additive mask and -FLT_MAX < 1e-6
// too. So the mask `EmbeddingsProcessor` hands the DiT is ALL ONES. That is
// upstream's behaviour, not ours to repair.
std::vector<int32_t> Ltx2ToBinaryMask(const float* encoded_mask, int64_t batch,
                                      int64_t seq);

// The conditioning `EmbeddingsProcessor.process_hidden_states` produces, up to
// the connector call. `video`/`audio` are RIGHT-PAD ORDERED features ready for
// `Embeddings1DConnector`; `additive_mask` is the matching reordered mask.
struct Ltx2TextConditioning {
  std::vector<float> video;          // [batch, seq, video_out_features]
  std::vector<float> audio;          // [batch, seq, audio_out_features] or empty
  std::vector<float> additive_mask;  // [batch, 1, 1, seq]
  std::vector<int32_t> sort_index;   // [batch, seq]
};

// embeddings_processor.py:70-117, minus the two connector calls.
//
// OWED, and recorded here rather than discovered later: `Embeddings1DConnector`
// (embeddings_connector.py:74-191) is NOT ported. It is built out of the DiT's
// own `Attention`, `FeedForward` and RoPE (embeddings_connector.py:4-11), which
// phase L2 owns, so it belongs to the change that can link against them. Until
// then this function stops at the connector's INPUT contract, which is what it
// gates.
Ltx2TextConditioning Ltx2TextEncoderConditioning(
    const Ltx2TextHiddenStates& states, const int32_t* mask,
    const Ltx2TextEncoderWeights& weights, const Ltx2TextFeatureConfig& config,
    vt::DType compute_dtype = vt::DType::kF32);

// ───────────────────── the embedded tokenizer / asset pack ───────────────────

// gemma_assets.py:58-159. The LTX-2.5 text encoder ships as ONE .safetensors file
// with its HuggingFace assets stored AS TENSORS, which is unusual enough that a
// loader assuming a sibling `tokenizer.json` fails on it:
//
//   tokenizer_json                    U8 [32169626]   ~32 MB, the whole tokenizer
//   hf_asset__tokenizer_config.json   U8 [3736]
//   hf_asset__processor_config.json   U8 [1382]
//   hf_asset__generation_config.json  U8 [255]
//   hf_asset__chat_template.jinja     U8 [18683]
//
// and the HF config itself in the file's `__metadata__` under `gemma_config`.
struct Ltx2GemmaAssets {
  std::vector<uint8_t> tokenizer_json;
  // Sidecar name (the part after `hf_asset__`) -> raw bytes.
  std::map<std::string, std::vector<uint8_t>> sidecars;
  // The parsed `__metadata__["gemma_config"]` JSON. Null when `require_config`
  // was false and the file carried no metadata.
  nlohmann::json config;
  bool has_config = false;

  // gemma_assets.py:144-151.
  const std::vector<uint8_t>& SidecarBytes(const std::string& name) const;
  nlohmann::json SidecarJson(const std::string& name) const;
};

// gemma_assets.py:34-36 — the two names the pack format is keyed on.
inline constexpr const char* kLtx2GemmaTokenizerTensor = "tokenizer_json";
inline constexpr const char* kLtx2GemmaAssetPrefix = "hf_asset__";

// gemma_assets.py:104-142 + `_require_sidecars` (:153-159). Throws
// std::runtime_error when `tokenizer_json` is missing or when either REQUIRED
// sidecar (`tokenizer_config.json`, `processor_config.json`, gemma_assets.py:38-41)
// is absent.
//
// MEASURED FINDING, reported rather than worked around: the shipped
// `vonkaiser/LTX-2.5-FP8-NVFP4` text encoder carries NO `__metadata__` block at
// all, so upstream's `GemmaAssets.from_single_file` raises on it before it reads a
// single tensor (gemma_assets.py:110-114). `require_config` mirrors that refusal
// by default; a caller that has the Gemma config from elsewhere passes false and
// gets the tensors, with `has_config` reporting which happened.
Ltx2GemmaAssets Ltx2LoadGemmaAssets(const SafetensorsFile& file,
                                    bool require_config = true);

}  // namespace vllm
