// LTX-2.5 TEXT CONDITIONING parity gate — the Gemma-4 multi-layer feature
// aggregation, both normalization variants, the two caption projections, the
// encoder -> conditioning hand-off and the embedded asset pack, each compared
// against the UPSTREAM `ltx_core` module executed at reduced dimensions on CPU by
// scripts/gen-ltx2-text-goldens.py.
//
// Both sides rebuild every weight and input from ONE deterministic stream, so no
// weight byte is checked in, and each extractor also asserts its PARAMETER
// MANIFEST — name and shape, in named_parameters() order — against upstream's, so
// a parameter one side builds and the other does not is a failure rather than a
// silent no-op.
//
// The traps this file exists to catch, each of which yields a finite,
// correctly-shaped, WRONG conditioning vector rather than an error:
//
//   * the layer axis concatenated layer-major instead of hidden-major;
//   * the wrong normalization variant (both are "a normalization" and both zero
//     the pads, so only the VALUES differ);
//   * a mask reduction over the wrong axes, or a padding side assumed;
//   * V1's bias-free projection given a bias, or V2's bias dropped.
//
// No doctest::Approx appears here: every tensor comparison is an explicit max
// absolute difference against a stated bound. Approx's default scale of 1.0 puts
// a 1.19e-5 ABSOLUTE floor under any epsilon, which would let a tight tolerance
// silently accept anything (including zero).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/ltx2_text_encoder.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

#include "ltx2_text_goldens.inc"

namespace fs = std::filesystem;

using vllm::Ltx2TextFeatureConfig;
using vllm::Ltx2TextHiddenStates;
using vllm::Ltx2TextNormVariant;

namespace {

// ---------------------------------------------------------------------------
// Ltx2Rand — the exact mirror of the generator's stream
// (scripts/gen-ltx2-text-goldens.py :: ltx2_rand), identical to the one L2's
// suite uses: a per-tensor FNV-1a seed plus a splitmix64 counter, so both sides
// build identical tensors from a NAME alone and cannot drift by reordering their
// parameter construction.
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

std::vector<float> Ltx2Make(const std::string& name, int64_t count, double scale,
                            double offset) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    const double unit = (static_cast<double>(u >> 11) * 0x1p-53) * 2.0 - 1.0;
    out[static_cast<size_t>(i)] = static_cast<float>(unit * scale + offset);
  }
  return out;
}

bool EndsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The generator's `param_spec` rule, mirrored EXACTLY (and it is L2's rule
// verbatim, so the three LTX-2.5 suites share one weight stream).
std::vector<float> Ltx2Param(const std::string& name,
                             const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;
  double scale = 0.05;
  double offset = 0.0;
  if (EndsWith(name, "q_norm.weight") || EndsWith(name, "k_norm.weight")) {
    scale = 0.1;
    offset = 1.0;
  } else if (EndsWith(name, ".bias")) {
    scale = 0.02;
  }
  return Ltx2Make(name, count, scale, offset);
}

// ---------------------------------------------------------------------------
// The reduced fixture the generator built.
// ---------------------------------------------------------------------------

constexpr int64_t kBatch = vllm_test::kLtxTeBatch;
constexpr int64_t kSeq = vllm_test::kLtxTeSeq;
constexpr int64_t kHidden = vllm_test::kLtxTeGemmaHidden;
constexpr int64_t kLayers = vllm_test::kLtxTeNumLayers;

std::vector<std::vector<float>> HiddenStateBuffers() {
  std::vector<std::vector<float>> states;
  for (int64_t l = 0; l < kLayers; ++l) {
    states.push_back(Ltx2Make("input.hidden." + std::to_string(l),
                              kBatch * kSeq * kHidden, 0.5, 0.0));
  }
  return states;
}

Ltx2TextHiddenStates MakeStates(const std::vector<std::vector<float>>& buffers) {
  Ltx2TextHiddenStates states;
  for (const std::vector<float>& b : buffers) states.layers.push_back(b.data());
  states.batch = kBatch;
  states.seq = kSeq;
  states.hidden = kHidden;
  return states;
}

// The masks the generator emits, as int32 (our mask dtype).
std::vector<int32_t> MaskFrom(const int64_t* golden) {
  std::vector<int32_t> mask(static_cast<size_t>(kBatch * kSeq));
  for (size_t i = 0; i < mask.size(); ++i) mask[i] = static_cast<int32_t>(golden[i]);
  return mask;
}

double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t count) {
  REQUIRE(got.size() == count);
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    worst = std::max(worst,
                     std::abs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
  }
  return worst;
}

// `additive_mask` holds -FLT_MAX, whose absolute difference saturates any bound;
// compare it EXACTLY instead, which is also what upstream produces (one multiply
// by finfo.max, no accumulation).
void CheckExact(const std::vector<float>& got, const float* want, size_t count) {
  REQUIRE(got.size() == count);
  for (size_t i = 0; i < count; ++i) CHECK(got[i] == want[i]);
}

void CheckExactI(const std::vector<int32_t>& got, const int64_t* want, size_t count) {
  REQUIRE(got.size() == count);
  for (size_t i = 0; i < count; ++i) CHECK(static_cast<int64_t>(got[i]) == want[i]);
}

// The tolerance every f32 brick is held to. The generator runs upstream in torch
// float32 and we accumulate in float32, so anything above this is an algorithm
// difference, not round-off.
constexpr double kTol = 1e-5;

Ltx2TextFeatureConfig V2Config() {
  Ltx2TextFeatureConfig cfg;
  cfg.variant = Ltx2TextNormVariant::kPerTokenRmsV2;
  cfg.embedding_dim = kHidden;
  cfg.num_layers = kLayers;
  cfg.video_out_features = vllm_test::kLtxTeVideoInner;
  cfg.audio_out_features = vllm_test::kLtxTeAudioInner;
  cfg.aggregate_bias = true;
  cfg.is_av = false;
  return cfg;
}

Ltx2TextFeatureConfig V1Config() {
  Ltx2TextFeatureConfig cfg;
  cfg.variant = Ltx2TextNormVariant::kPaddedBatchV1;
  cfg.embedding_dim = kHidden;
  cfg.num_layers = kLayers;
  cfg.video_out_features = kHidden;  // V1 projects back to the Gemma width
  cfg.audio_out_features = 0;
  cfg.aggregate_bias = false;
  cfg.is_av = true;
  return cfg;
}

vllm::Ltx2TextEncoderWeights V2Weights() {
  vllm::Ltx2TextEncoderWeights w;
  const int64_t flat = vllm_test::kLtxTeFlatDim;
  w.video.out_features = vllm_test::kLtxTeVideoInner;
  w.video.in_features = flat;
  w.video.weight = Ltx2Param("video_aggregate_embed.weight", {w.video.out_features, flat});
  w.video.bias = Ltx2Param("video_aggregate_embed.bias", {w.video.out_features});
  w.audio.out_features = vllm_test::kLtxTeAudioInner;
  w.audio.in_features = flat;
  w.audio.weight = Ltx2Param("audio_aggregate_embed.weight", {w.audio.out_features, flat});
  w.audio.bias = Ltx2Param("audio_aggregate_embed.bias", {w.audio.out_features});
  return w;
}

vllm::Ltx2TextEncoderWeights V1Weights() {
  vllm::Ltx2TextEncoderWeights w;
  const int64_t flat = vllm_test::kLtxTeFlatDim;
  w.video.out_features = kHidden;
  w.video.in_features = flat;
  w.video.weight = Ltx2Param("aggregate_embed.weight", {kHidden, flat});
  // bias stays EMPTY: encoder_configurator.py:187 builds V1's Linear with bias=False.
  return w;
}

nlohmann::json V2TransformerConfig() {
  return nlohmann::json{
      {"caption_proj_before_connector", true},
      {"caption_projection_first_linear", false},
      {"caption_proj_input_norm", false},
      {"caption_projection_second_linear", false},
      {"num_attention_heads", vllm_test::kLtxTeVideoHeads},
      {"attention_head_dim", vllm_test::kLtxTeVideoHeadDim},
      {"audio_num_attention_heads", vllm_test::kLtxTeAudioHeads},
      {"audio_attention_head_dim", vllm_test::kLtxTeAudioHeadDim},
  };
}

nlohmann::json V1TransformerConfig() {
  return nlohmann::json{
      {"num_attention_heads", vllm_test::kLtxTeVideoHeads},
      {"attention_head_dim", vllm_test::kLtxTeVideoHeadDim},
  };
}

// --- a synthetic single-file text-encoder pack, for the asset gate -----------

void AppendU64(std::string& out, uint64_t value) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

// Writes a minimal .safetensors carrying only U8 asset tensors, in the shape
// gemma_assets.py:335-386 packs them. `metadata` is written verbatim as the
// `__metadata__` object when non-empty.
std::string WritePack(const fs::path& path,
                      const std::vector<std::pair<std::string, std::string>>& tensors,
                      const std::string& metadata_json) {
  nlohmann::json header = nlohmann::json::object();
  if (!metadata_json.empty()) header["__metadata__"] = nlohmann::json::parse(metadata_json);
  uint64_t offset = 0;
  std::string payload;
  for (const auto& [name, bytes] : tensors) {
    header[name] = nlohmann::json{
        {"dtype", "U8"},
        {"shape", nlohmann::json::array({static_cast<int64_t>(bytes.size())})},
        {"data_offsets",
         nlohmann::json::array({offset, offset + static_cast<uint64_t>(bytes.size())})}};
    offset += bytes.size();
    payload += bytes;
  }
  const std::string head = header.dump();
  std::string out;
  AppendU64(out, head.size());
  out += head;
  out += payload;
  std::ofstream file(path, std::ios::binary);
  file.write(out.data(), static_cast<std::streamsize>(out.size()));
  file.close();
  return path.string();
}

std::string BytesToString(const std::vector<uint8_t>& bytes) {
  return std::string(bytes.begin(), bytes.end());
}

// --- the Gemma-4 per-layer hidden-state seam --------------------------------
//
// LTX-2.5 needs EVERY hidden state, so `Gemma4Model::ForwardHiddenStates` had to
// exist. A tiny CPU-synthetic Gemma-4 gates the ORDER of what it returns, which
// is the part that fails silently: 49 finite tensors of the right shape carrying
// the wrong states condition on the wrong thing and still render.
//
// The reduced config deliberately mirrors the SHIPPED LTX text tower's shape
// rather than unsloth's E4B: `model.layers.N.{input,post_attention,pre_feedforward,
// post_feedforward}_layernorm` + `layer_scalar`, uniform head_dim, GQA, and NO
// per-layer-embedding tensors at all — measured on
// gemma4-12b-with-proj-nvfp4-torchao.safetensors, whose 1688 tensors contain no
// `embed_tokens_per_layer` and no `per_layer_*`.

vt::Queue Qcpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed,
                           float scale = 0.08f) {
  vllm::OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  // The deterministic stream again, so the fixture is reproducible from a seed.
  for (int64_t i = 0; i < numel; ++i) {
    const uint64_t u = Splitmix64(static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ULL +
                                  static_cast<uint64_t>(i));
    const double unit = (static_cast<double>(u >> 11) * 0x1p-53) * 2.0 - 1.0;
    p[i] = vt::F32ToBF16(static_cast<float>(unit * scale));
  }
  return o;
}

vllm::HfConfig TinyGemma4Config() {
  vllm::HfConfig c;
  c.num_hidden_layers = 3;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.vocab_size = 96;
  c.sliding_window = 8;
  // No hidden_size_per_layer_input -> ple_dim 0, no global_head_dim -> one head
  // width, no final_logit_softcapping -> the lm_head product is a plain matmul,
  // which is what lets the last-state invariant below be checked directly.
  c.raw = nlohmann::json{{"tie_word_embeddings", true}};
  return c;
}

vllm::Gemma4Weights TinyGemma4Weights(const vllm::HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads;
  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  vllm::Gemma4Weights w;
  w.tie_word_embeddings = true;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.3f);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Gemma4LayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.pre_feedforward_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_feedforward_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.layer_scalar = MakeBf16({1}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    lw.attn.q_norm = MakeBf16({Dh}, false, seed++, 0.3f);
    lw.attn.k_norm = MakeBf16({Dh}, false, seed++, 0.3f);
    lw.mlp.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed++);
    lw.mlp.down_proj = MakeBf16({H, I}, /*nk=*/true, seed++);
    lw.head_dim = Dh;
    lw.num_kv_heads = Hkv;
    lw.is_full_attention = false;
    lw.is_kv_shared = false;
    lw.kv_target_layer = -1;
    w.layers.push_back(std::move(lw));
  }
  return w;
}

struct Gemma4CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<vllm::PagedKvCache> attn_kv;
  Gemma4CachePool(const vllm::HfConfig& c, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
    for (auto& b : buf) {
      vllm::PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = vt::DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }
};

vllm::v1::CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  vllm::v1::CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ltx2 text: the hidden-state stack is HIDDEN-major, layer-minor") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const std::vector<float> stacked = Ltx2StackHiddenStates(MakeStates(buffers));

  const size_t count = static_cast<size_t>(kBatch * kSeq * kHidden * kLayers);
  const double worst = MaxAbsDiff(stacked, vllm_test::kLtxTeStacked, count);
  MESSAGE("ltx2 text stack max|diff| = " << worst);
  CHECK(worst == 0.0);  // a pure permutation: it must be BIT-equal

  // The trap, made explicit: the layer-major alternative is a different tensor,
  // so a port that concatenates `l * D + d` cannot pass the check above.
  std::vector<float> layer_major(count);
  for (int64_t b = 0; b < kBatch; ++b)
    for (int64_t t = 0; t < kSeq; ++t)
      for (int64_t d = 0; d < kHidden; ++d)
        for (int64_t l = 0; l < kLayers; ++l) {
          const size_t src = static_cast<size_t>(((b * kSeq + t) * kHidden + d) * kLayers + l);
          const size_t dst = static_cast<size_t>(((b * kSeq + t) * kLayers + l) * kHidden + d);
          layer_major[dst] = stacked[src];
        }
  bool differs = false;
  for (size_t i = 0; i < count; ++i)
    if (layer_major[i] != stacked[i]) differs = true;
  CHECK(differs);
}

TEST_CASE("ltx2 text: the hidden-state contract is 48 layers + the embedding output") {
  // encoder_configurator.py:182 and base_encoder.py:68-71 — the shipped Gemma-4
  // has 48 decoder layers and LTX consumes 49 states, which is what makes the
  // caption projections 3840 x 49 = 188160 wide.
  CHECK(vllm::Ltx2GemmaHiddenStateContract::Count(48) == 49);
  CHECK(vllm::Ltx2GemmaHiddenStateContract::Count(48) * 3840 == 188160);
  CHECK(vllm::Ltx2GemmaHiddenStateContract::Count(vllm_test::kLtxTeGemmaHiddenLayers) ==
        kLayers);
}

TEST_CASE("ltx2 text: the normalization variant is SELECTED from config, never guessed") {
  const Ltx2TextFeatureConfig v2 = vllm::Ltx2SelectTextFeatureVariant(
      V2TransformerConfig(), vllm_test::kLtxTeGemmaHidden,
      vllm_test::kLtxTeGemmaHiddenLayers);
  CHECK(static_cast<int>(v2.variant == Ltx2TextNormVariant::kPerTokenRmsV2) ==
        static_cast<int>(vllm_test::kLtxTeSelectedV2IsV2));
  CHECK(v2.embedding_dim == vllm_test::kLtxTeV2EmbeddingDim);
  CHECK(v2.FlatDim() == vllm_test::kLtxTeV2VideoIn);
  CHECK(v2.video_out_features == vllm_test::kLtxTeV2VideoOut);
  CHECK(v2.FlatDim() == vllm_test::kLtxTeV2AudioIn);
  CHECK(v2.audio_out_features == vllm_test::kLtxTeV2AudioOut);
  CHECK(static_cast<int64_t>(v2.aggregate_bias) == vllm_test::kLtxTeV2VideoHasBias);
  CHECK(static_cast<int64_t>(v2.aggregate_bias) == vllm_test::kLtxTeV2AudioHasBias);
  CHECK_FALSE(v2.is_av);

  const Ltx2TextFeatureConfig v1 = vllm::Ltx2SelectTextFeatureVariant(
      V1TransformerConfig(), vllm_test::kLtxTeGemmaHidden,
      vllm_test::kLtxTeGemmaHiddenLayers);
  CHECK(static_cast<int>(v1.variant == Ltx2TextNormVariant::kPerTokenRmsV2) ==
        static_cast<int>(vllm_test::kLtxTeSelectedV1IsV2));
  CHECK(v1.variant == Ltx2TextNormVariant::kPaddedBatchV1);
  CHECK(v1.FlatDim() == vllm_test::kLtxTeV1AggregateIn);
  CHECK(v1.video_out_features == vllm_test::kLtxTeV1AggregateOut);
  CHECK(static_cast<int64_t>(v1.aggregate_bias) == vllm_test::kLtxTeV1AggregateHasBias);
  CHECK(static_cast<int64_t>(v1.is_av) == vllm_test::kLtxTeV1IsAv);
  CHECK(v1.audio_out_features == 0);

  // encoder_configurator.py:190-192 — a PARTIAL V2 marker set is
  // NotImplementedError, not a fall-back to V1.
  nlohmann::json partial = V2TransformerConfig();
  partial.erase("caption_proj_input_norm");
  CHECK_THROWS_AS(vllm::Ltx2SelectTextFeatureVariant(partial, kHidden,
                                                     vllm_test::kLtxTeGemmaHiddenLayers),
                  std::runtime_error);

  // encoder_configurator.py:194-201 — a marker key present with the WRONG value
  // is config drift and is refused too.
  nlohmann::json drifted = V2TransformerConfig();
  drifted["caption_projection_first_linear"] = true;
  CHECK_THROWS_AS(vllm::Ltx2SelectTextFeatureVariant(drifted, kHidden,
                                                     vllm_test::kLtxTeGemmaHiddenLayers),
                  std::runtime_error);
}

TEST_CASE("ltx2 text: the weight manifest matches upstream named_parameters()") {
  // V2 (LTX-2.5): two biased projections, in this order.
  const std::vector<std::string> v2_names = {
      "video_aggregate_embed.weight", "video_aggregate_embed.bias",
      "audio_aggregate_embed.weight", "audio_aggregate_embed.bias"};
  const std::vector<std::vector<int64_t>> v2_shapes = {
      {vllm_test::kLtxTeVideoInner, vllm_test::kLtxTeFlatDim},
      {vllm_test::kLtxTeVideoInner},
      {vllm_test::kLtxTeAudioInner, vllm_test::kLtxTeFlatDim},
      {vllm_test::kLtxTeAudioInner}};
  REQUIRE(static_cast<int64_t>(v2_names.size()) == vllm_test::kLtxTeV2ParamCount);
  size_t dim = 0;
  for (size_t i = 0; i < v2_names.size(); ++i) {
    CHECK(v2_names[i] == std::string(vllm_test::kLtxTeV2ParamNames[i]));
    CHECK(static_cast<int64_t>(v2_shapes[i].size()) == vllm_test::kLtxTeV2ParamRanks[i]);
    for (int64_t d : v2_shapes[i]) CHECK(d == vllm_test::kLtxTeV2ParamDims[dim++]);
  }

  // V1: ONE projection and NO bias.
  REQUIRE(vllm_test::kLtxTeV1ParamCount == 1);
  CHECK(std::string(vllm_test::kLtxTeV1ParamNames[0]) == "aggregate_embed.weight");
  CHECK(vllm_test::kLtxTeV1ParamRanks[0] == 2);
  CHECK(vllm_test::kLtxTeV1ParamDims[0] == vllm_test::kLtxTeV1AggregateOut);
  CHECK(vllm_test::kLtxTeV1ParamDims[1] == vllm_test::kLtxTeFlatDim);
}

TEST_CASE("ltx2 text: `_norm_and_concat_padded_batch`, both padding sides") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const std::vector<float> stacked = Ltx2StackHiddenStates(MakeStates(buffers));
  const size_t count = static_cast<size_t>(kBatch * kSeq * kHidden * kLayers);

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* want;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeNormV1Left},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeNormV1Right},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const std::vector<float> got = vllm::Ltx2NormAndConcatPaddedBatch(
        stacked.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
    const double worst = MaxAbsDiff(got, c.want, count);
    MESSAGE("ltx2 text norm V1 (" << std::string(c.tag) << ") max|diff| = " << worst);
    CHECK(worst < kTol);

    // feature_extractor.py:44-45 — the padded positions are ZEROED, exactly.
    for (int64_t b = 0; b < kBatch; ++b)
      for (int64_t t = 0; t < kSeq; ++t) {
        if (mask[static_cast<size_t>(b * kSeq + t)] != 0) continue;
        for (int64_t f = 0; f < kHidden * kLayers; ++f)
          CHECK(got[static_cast<size_t>((b * kSeq + t) * kHidden * kLayers + f)] == 0.0f);
      }
  }
}

TEST_CASE("ltx2 text: `norm_and_concat_per_token_rms`, both padding sides") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const std::vector<float> stacked = Ltx2StackHiddenStates(MakeStates(buffers));
  const size_t count = static_cast<size_t>(kBatch * kSeq * kHidden * kLayers);

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* want;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeNormV2Left},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeNormV2Right},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const std::vector<float> got = vllm::Ltx2NormAndConcatPerTokenRms(
        stacked.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
    const double worst = MaxAbsDiff(got, c.want, count);
    MESSAGE("ltx2 text norm V2 (" << std::string(c.tag) << ") max|diff| = " << worst);
    CHECK(worst < kTol);

    for (int64_t b = 0; b < kBatch; ++b)
      for (int64_t t = 0; t < kSeq; ++t) {
        if (mask[static_cast<size_t>(b * kSeq + t)] != 0) continue;
        for (int64_t f = 0; f < kHidden * kLayers; ++f)
          CHECK(got[static_cast<size_t>((b * kSeq + t) * kHidden * kLayers + f)] == 0.0f);
      }
  }

  // The two variants are NOT interchangeable — proven here rather than assumed,
  // so a config-selection bug cannot hide behind "it is still a normalization".
  const std::vector<int32_t> mask = MaskFrom(vllm_test::kLtxTeMaskLeft);
  const std::vector<float> v1 = vllm::Ltx2NormAndConcatPaddedBatch(
      stacked.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
  const std::vector<float> v2 = vllm::Ltx2NormAndConcatPerTokenRms(
      stacked.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
  double gap = 0.0;
  for (size_t i = 0; i < count; ++i)
    gap = std::max(gap, std::abs(static_cast<double>(v1[i]) - static_cast<double>(v2[i])));
  MESSAGE("ltx2 text: V1 vs V2 normalization max|diff| = " << gap);
  CHECK(gap > 1.0);
}

TEST_CASE("ltx2 text: `_rescale_norm` uses each projection's OWN width") {
  const double video = vllm::Ltx2RescaleNorm(vllm_test::kLtxTeVideoInner, kHidden);
  const double audio = vllm::Ltx2RescaleNorm(vllm_test::kLtxTeAudioInner, kHidden);
  MESSAGE("ltx2 text rescale video = " << video << " audio = " << audio);
  CHECK(std::abs(video - vllm_test::kLtxTeRescaleVideo) < 1e-12);
  CHECK(std::abs(audio - vllm_test::kLtxTeRescaleAudio) < 1e-12);
  // The shipped ratios sit on OPPOSITE sides of 1 (4096 > 3840 > 2048), so a
  // swapped numerator/denominator cannot pass both arms.
  CHECK(video > 1.0);
  CHECK(audio < 1.0);
  CHECK(vllm::Ltx2RescaleNorm(4096, 3840) > 1.0);
  CHECK(vllm::Ltx2RescaleNorm(2048, 3840) < 1.0);
}

TEST_CASE("ltx2 text: FeatureExtractorV2 — the two caption projections") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const Ltx2TextFeatureConfig cfg = V2Config();
  const vllm::Ltx2TextEncoderWeights weights = V2Weights();

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* video;
    const float* audio;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeV2VideoLeft,
       vllm_test::kLtxTeV2AudioLeft},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeV2VideoRight,
       vllm_test::kLtxTeV2AudioRight},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const vllm::Ltx2TextFeatures got =
        vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), weights, cfg);
    const double wv = MaxAbsDiff(got.video, c.video,
                                 static_cast<size_t>(kBatch * kSeq * cfg.video_out_features));
    const double wa = MaxAbsDiff(got.audio, c.audio,
                                 static_cast<size_t>(kBatch * kSeq * cfg.audio_out_features));
    MESSAGE("ltx2 text V2 extractor (" << std::string(c.tag) << ") max|diff| video = " << wv
                                       << " audio = " << wa);
    CHECK(wv < kTol);
    CHECK(wa < kTol);

    // The norm zeroes a padded position, so its PROJECTED value is exactly the
    // Linear's bias — NOT zero. A port that force-zeroes projected pads diverges
    // from upstream on every padded row while still looking "masked".
    for (int64_t b = 0; b < kBatch; ++b)
      for (int64_t t = 0; t < kSeq; ++t) {
        if (mask[static_cast<size_t>(b * kSeq + t)] != 0) continue;
        for (int64_t f = 0; f < cfg.video_out_features; ++f) {
          const size_t idx =
              static_cast<size_t>((b * kSeq + t) * cfg.video_out_features + f);
          CHECK(std::abs(static_cast<double>(got.video[idx]) -
                         static_cast<double>(weights.video.bias[static_cast<size_t>(f)])) <
                kTol);
        }
      }
  }
}

TEST_CASE("ltx2 text: FeatureExtractorV1 — one bias-free projection, is_av") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const Ltx2TextFeatureConfig cfg = V1Config();
  const vllm::Ltx2TextEncoderWeights weights = V1Weights();

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* video;
    const float* audio;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeV1VideoLeft,
       vllm_test::kLtxTeV1AudioLeft},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeV1VideoRight,
       vllm_test::kLtxTeV1AudioRight},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const vllm::Ltx2TextFeatures got =
        vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), weights, cfg);
    const size_t count = static_cast<size_t>(kBatch * kSeq * cfg.video_out_features);
    const double wv = MaxAbsDiff(got.video, c.video, count);
    const double wa = MaxAbsDiff(got.audio, c.audio, count);
    MESSAGE("ltx2 text V1 extractor (" << std::string(c.tag) << ") max|diff| video = " << wv
                                       << " audio = " << wa);
    CHECK(wv < kTol);
    CHECK(wa < kTol);
    // feature_extractor.py:95-96 — `is_av` returns the SAME tensor twice.
    CHECK(got.audio == got.video);

    // No bias: a padded position projects to exactly ZERO under V1.
    for (int64_t b = 0; b < kBatch; ++b)
      for (int64_t t = 0; t < kSeq; ++t) {
        if (mask[static_cast<size_t>(b * kSeq + t)] != 0) continue;
        for (int64_t f = 0; f < cfg.video_out_features; ++f)
          CHECK(got.video[static_cast<size_t>((b * kSeq + t) * cfg.video_out_features + f)] ==
                0.0f);
      }
  }
}

TEST_CASE("ltx2 text: additive mask, right-pad ordering and the binary mask") {
  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* additive;
    const int64_t* sort_idx;
    const float* reordered_mask;
    const int64_t* binary;
    const int64_t* binary_registers;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeAdditiveMaskLeft,
       vllm_test::kLtxTeSortIdxLeft, vllm_test::kLtxTeReorderedMaskLeft,
       vllm_test::kLtxTeBinaryMaskLeft, vllm_test::kLtxTeBinaryMaskFromRegistersLeft},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeAdditiveMaskRight,
       vllm_test::kLtxTeSortIdxRight, vllm_test::kLtxTeReorderedMaskRight,
       vllm_test::kLtxTeBinaryMaskRight, vllm_test::kLtxTeBinaryMaskFromRegistersRight},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const std::vector<float> additive =
        vllm::Ltx2ConvertToAdditiveMask(mask.data(), kBatch, kSeq);
    CheckExact(additive, c.additive, static_cast<size_t>(kBatch * kSeq));

    std::vector<int32_t> sort_index;
    std::vector<float> reordered;
    vllm::Ltx2ComputeRightPadOrder(additive.data(), kBatch, kSeq, sort_index, reordered);
    CheckExactI(sort_index, c.sort_idx, static_cast<size_t>(kBatch * kSeq));
    CheckExact(reordered, c.reordered_mask, static_cast<size_t>(kBatch * kSeq));

    const std::vector<int32_t> binary =
        vllm::Ltx2ToBinaryMask(reordered.data(), kBatch, kSeq);
    CheckExactI(binary, c.binary, static_cast<size_t>(kBatch * kSeq));
    const std::vector<float> zeros(static_cast<size_t>(kBatch * kSeq), 0.0f);
    const std::vector<int32_t> from_registers =
        vllm::Ltx2ToBinaryMask(zeros.data(), kBatch, kSeq);
    CheckExactI(from_registers, c.binary_registers, static_cast<size_t>(kBatch * kSeq));
  }

  // embeddings_processor.py:26-27 — idempotent on already right-padded input.
  const std::vector<int32_t> right = MaskFrom(vllm_test::kLtxTeMaskRight);
  const std::vector<float> additive =
      vllm::Ltx2ConvertToAdditiveMask(right.data(), kBatch, kSeq);
  std::vector<int32_t> sort_index;
  std::vector<float> reordered;
  vllm::Ltx2ComputeRightPadOrder(additive.data(), kBatch, kSeq, sort_index, reordered);
  for (int64_t b = 0; b < kBatch; ++b)
    for (int64_t t = 0; t < kSeq; ++t)
      CHECK(sort_index[static_cast<size_t>(b * kSeq + t)] == static_cast<int32_t>(t));
}

TEST_CASE("ltx2 text: the encoder -> conditioning hand-off") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const Ltx2TextFeatureConfig cfg = V2Config();
  const vllm::Ltx2TextEncoderWeights weights = V2Weights();

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* video;
    const float* audio;
    const float* reordered_mask;
    const int64_t* sort_idx;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeReorderedVideoLeft,
       vllm_test::kLtxTeReorderedAudioLeft, vllm_test::kLtxTeReorderedMaskLeft,
       vllm_test::kLtxTeSortIdxLeft},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeReorderedVideoRight,
       vllm_test::kLtxTeReorderedAudioRight, vllm_test::kLtxTeReorderedMaskRight,
       vllm_test::kLtxTeSortIdxRight},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const vllm::Ltx2TextConditioning got =
        vllm::Ltx2TextEncoderConditioning(states, mask.data(), weights, cfg);
    const double wv = MaxAbsDiff(got.video, c.video,
                                 static_cast<size_t>(kBatch * kSeq * cfg.video_out_features));
    const double wa = MaxAbsDiff(got.audio, c.audio,
                                 static_cast<size_t>(kBatch * kSeq * cfg.audio_out_features));
    MESSAGE("ltx2 text conditioning (" << std::string(c.tag) << ") max|diff| video = " << wv
                                       << " audio = " << wa);
    CHECK(wv < kTol);
    CHECK(wa < kTol);
    CheckExact(got.additive_mask, c.reordered_mask, static_cast<size_t>(kBatch * kSeq));
    CheckExactI(got.sort_index, c.sort_idx, static_cast<size_t>(kBatch * kSeq));
  }
}

TEST_CASE("ltx2 text: a non-f32 compute dtype is REFUSED, never silently widened") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const std::vector<int32_t> mask = MaskFrom(vllm_test::kLtxTeMaskLeft);
  CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), V2Weights(),
                                                        V2Config(), vt::DType::kBF16),
                  std::runtime_error);
  CHECK_THROWS_AS(vllm::Ltx2TextEncoderConditioning(states, mask.data(), V2Weights(),
                                                    V2Config(), vt::DType::kBF16),
                  std::runtime_error);
}

TEST_CASE("ltx2 text: the tokenizer and HF sidecars come out of TENSORS, not files") {
  const fs::path dir =
      fs::temp_directory_path() / ("ltx2_text_assets_" + std::to_string(::getpid()));
  fs::create_directories(dir);

  const std::string tokenizer = R"({"version":"1.0","model":{"type":"BPE"}})";
  const std::string tok_cfg = R"({"tokenizer_class":"PreTrainedTokenizerFast"})";
  const std::string proc_cfg = R"({"processor_class":"Gemma4Processor"})";
  const std::string chat = "{{ messages }}";

  SUBCASE("a complete pack round-trips, config included") {
    const std::string path = WritePack(
        dir / "full.safetensors",
        {{"tokenizer_json", tokenizer},
         {"hf_asset__tokenizer_config.json", tok_cfg},
         {"hf_asset__processor_config.json", proc_cfg},
         {"hf_asset__chat_template.jinja", chat}},
        R"({"format":"pt","gemma_config":"{\"model_type\":\"gemma4_unified\",\"hidden_size\":3840}"})");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    const vllm::Ltx2GemmaAssets assets = vllm::Ltx2LoadGemmaAssets(file);
    CHECK(BytesToString(assets.tokenizer_json) == tokenizer);
    CHECK(assets.sidecars.size() == 3);
    CHECK(BytesToString(assets.SidecarBytes("tokenizer_config.json")) == tok_cfg);
    CHECK(BytesToString(assets.SidecarBytes("chat_template.jinja")) == chat);
    CHECK(assets.SidecarJson("processor_config.json")["processor_class"] ==
          "Gemma4Processor");
    REQUIRE(assets.has_config);
    CHECK(assets.config["model_type"] == "gemma4_unified");
    CHECK(assets.config["hidden_size"] == 3840);
    // gemma_assets.py:148 — an absent sidecar throws, it does not return empty.
    CHECK_THROWS_AS(assets.SidecarBytes("nope.json"), std::runtime_error);
  }

  SUBCASE("a missing tokenizer tensor is refused by NAME") {
    const std::string path =
        WritePack(dir / "notok.safetensors",
                  {{"hf_asset__tokenizer_config.json", tok_cfg},
                   {"hf_asset__processor_config.json", proc_cfg}},
                  R"({"gemma_config":"{}"})");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaAssets(file), std::runtime_error);
  }

  SUBCASE("a missing REQUIRED sidecar is refused") {
    // gemma_assets.py:38-41 — tokenizer_config.json and processor_config.json are
    // required; chat_template and generation_config are not.
    const std::string path =
        WritePack(dir / "nosidecar.safetensors",
                  {{"tokenizer_json", tokenizer},
                   {"hf_asset__tokenizer_config.json", tok_cfg}},
                  R"({"gemma_config":"{}"})");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaAssets(file), std::runtime_error);
  }

  SUBCASE("the SHIPPED checkpoint's missing __metadata__ is refused, not invented") {
    // MEASURED: vonkaiser/LTX-2.5-FP8-NVFP4's
    // gemma4-12b-with-proj-nvfp4-torchao.safetensors has 1688 tensors and NO
    // __metadata__ block, so upstream's from_single_file raises on it
    // (gemma_assets.py:110-114). We refuse identically by default, and a caller
    // that sources the config elsewhere opts out explicitly.
    const std::string path = WritePack(dir / "nometa.safetensors",
                                       {{"tokenizer_json", tokenizer},
                                        {"hf_asset__tokenizer_config.json", tok_cfg},
                                        {"hf_asset__processor_config.json", proc_cfg}},
                                       "");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaAssets(file), std::runtime_error);
    const vllm::Ltx2GemmaAssets assets = vllm::Ltx2LoadGemmaAssets(file, false);
    CHECK_FALSE(assets.has_config);
    CHECK(BytesToString(assets.tokenizer_json) == tokenizer);
  }

  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ─────────────── the Gemma-4 seam LTX needs, and its ORDER ───────────────────

TEST_CASE("gemma4: ForwardHiddenStates returns L+1 states in transformers' order") {
  const vllm::HfConfig cfg = TinyGemma4Config();
  const vllm::Gemma4Weights weights = TinyGemma4Weights(cfg);
  const int64_t T = 5;
  const int64_t H = cfg.hidden_size;
  const int64_t L = cfg.num_hidden_layers;
  Gemma4CachePool pool(cfg, /*num_blocks=*/2, /*block_size=*/8);
  const vllm::v1::CommonAttentionMetadata meta = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Qcpu();

  const vllm::Gemma4HiddenStatesResult got = vllm::Gemma4Model::ForwardHiddenStates(
      tokens, positions, meta, pool.attn_kv, weights, cfg, q);

  // The count IS the contract: 48 + 1 on the shipped tower, L + 1 here.
  REQUIRE(static_cast<int64_t>(got.hidden_states.size()) == L + 1);
  for (const std::vector<float>& state : got.hidden_states) {
    REQUIRE(static_cast<int64_t>(state.size()) == T * H);
    for (float x : state) REQUIRE(std::isfinite(x));
  }

  // [0] is the EMBEDDING output, sqrt(hidden)-scaled — not the first layer's
  // output. Reconstructed here from the table, so a port that starts the tuple at
  // layer 0's output fails.
  const auto* table = reinterpret_cast<const uint16_t*>(weights.embed_tokens.bytes.data());
  const double normalizer =
      static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(std::sqrt(static_cast<float>(H)))));
  double worst_embed = 0.0;
  double scale_embed = 0.0;
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h) {
      const double want =
          static_cast<double>(vt::BF16ToF32(
              table[static_cast<size_t>(tokens[static_cast<size_t>(t)] * H + h)])) *
          normalizer;
      const double have =
          static_cast<double>(got.hidden_states[0][static_cast<size_t>(t * H + h)]);
      worst_embed = std::max(worst_embed, std::abs(have - want));
      scale_embed = std::max(scale_embed, std::abs(want));
    }
  MESSAGE("gemma4 hidden_states[0] vs embed*sqrt(H): max|diff| = "
          << worst_embed << " over max|value| = " << scale_embed);
  // The state is stored bf16, so equality is bf16 rounding of the product.
  CHECK(worst_embed < 0.01 * scale_embed);

  // [L] is the FINAL-NORMED state, not the raw output of the last decoder layer.
  // Proven by the invariant only the final-normed state satisfies: the logits are
  // exactly that state through the tied lm_head. A port that stored the raw last
  // layer output here would be off by the whole RMSNorm.
  const int64_t V = cfg.vocab_size;
  REQUIRE(static_cast<int64_t>(got.logits.size()) == T * V);
  double worst_logit = 0.0;
  double scale_logit = 0.0;
  for (int64_t t = 0; t < T; ++t)
    for (int64_t v = 0; v < V; ++v) {
      double acc = 0.0;
      for (int64_t h = 0; h < H; ++h)
        acc += static_cast<double>(
                   got.hidden_states[static_cast<size_t>(L)][static_cast<size_t>(t * H + h)]) *
               static_cast<double>(vt::BF16ToF32(table[static_cast<size_t>(v * H + h)]));
      const double have = static_cast<double>(got.logits[static_cast<size_t>(t * V + v)]);
      worst_logit = std::max(worst_logit, std::abs(have - acc));
      scale_logit = std::max(scale_logit, std::abs(acc));
    }
  MESSAGE("gemma4 hidden_states[L] @ lm_head vs logits: max|diff| = "
          << worst_logit << " over max|value| = " << scale_logit);
  CHECK(worst_logit < 0.01 * scale_logit);

  // Every state genuinely differs from its neighbour: a capture that pushed the
  // same buffer L+1 times would satisfy every check above except this one.
  for (int64_t i = 0; i + 1 <= L; ++i) {
    double gap = 0.0;
    for (size_t k = 0; k < got.hidden_states[static_cast<size_t>(i)].size(); ++k)
      gap = std::max(gap, std::abs(static_cast<double>(
                                       got.hidden_states[static_cast<size_t>(i)][k]) -
                                   static_cast<double>(
                                       got.hidden_states[static_cast<size_t>(i + 1)][k])));
    CHECK(gap > 0.0);
  }

  // The plain forward is unchanged by the capture: same logits, bit for bit.
  Gemma4CachePool pool2(cfg, 2, 8);
  const std::vector<float> plain = vllm::Gemma4Model::Forward(
      tokens, positions, meta, pool2.attn_kv, weights, cfg, q);
  REQUIRE(plain.size() == got.logits.size());
  for (size_t i = 0; i < plain.size(); ++i) CHECK(plain[i] == got.logits[i]);
}

TEST_CASE("gemma4 -> ltx2: the captured stack feeds the conditioning path") {
  // The seam end to end at reduced dims: every Gemma-4 hidden state goes straight
  // into the LTX feature extractor, which is the whole reason the capture exists.
  const vllm::HfConfig cfg = TinyGemma4Config();
  const vllm::Gemma4Weights weights = TinyGemma4Weights(cfg);
  const int64_t T = 5;
  Gemma4CachePool pool(cfg, 2, 8);
  const vllm::v1::CommonAttentionMetadata meta = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Qcpu();
  const vllm::Gemma4HiddenStatesResult run = vllm::Gemma4Model::ForwardHiddenStates(
      tokens, positions, meta, pool.attn_kv, weights, cfg, q);

  Ltx2TextFeatureConfig fcfg = vllm::Ltx2SelectTextFeatureVariant(
      V2TransformerConfig(), cfg.hidden_size, cfg.num_hidden_layers);
  REQUIRE(fcfg.num_layers == static_cast<int64_t>(run.hidden_states.size()));

  vllm::Ltx2TextEncoderWeights w;
  w.video.out_features = fcfg.video_out_features;
  w.video.in_features = fcfg.FlatDim();
  w.video.weight = Ltx2Param("video_aggregate_embed.weight",
                             {w.video.out_features, w.video.in_features});
  w.video.bias = Ltx2Param("video_aggregate_embed.bias", {w.video.out_features});
  w.audio.out_features = fcfg.audio_out_features;
  w.audio.in_features = fcfg.FlatDim();
  w.audio.weight = Ltx2Param("audio_aggregate_embed.weight",
                             {w.audio.out_features, w.audio.in_features});
  w.audio.bias = Ltx2Param("audio_aggregate_embed.bias", {w.audio.out_features});

  Ltx2TextHiddenStates states;
  for (const std::vector<float>& s : run.hidden_states) states.layers.push_back(s.data());
  states.batch = 1;
  states.seq = T;
  states.hidden = cfg.hidden_size;

  // One padded position, to keep the mask path live on the real stack too.
  const std::vector<int32_t> mask = {1, 1, 1, 1, 0};
  const vllm::Ltx2TextConditioning cond =
      vllm::Ltx2TextEncoderConditioning(states, mask.data(), w, fcfg);
  REQUIRE(cond.video.size() == static_cast<size_t>(T * fcfg.video_out_features));
  REQUIRE(cond.audio.size() == static_cast<size_t>(T * fcfg.audio_out_features));
  for (float x : cond.video) REQUIRE(std::isfinite(x));
  for (float x : cond.audio) REQUIRE(std::isfinite(x));

  // A stack missing the embedding state is REFUSED by count, not silently padded.
  Ltx2TextHiddenStates short_stack = states;
  short_stack.layers.pop_back();
  CHECK_THROWS_AS(
      vllm::Ltx2TextEncoderConditioning(short_stack, mask.data(), w, fcfg),
      std::runtime_error);
}
