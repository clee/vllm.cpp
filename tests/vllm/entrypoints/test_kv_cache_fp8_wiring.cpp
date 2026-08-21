// KV-FP8 W3 gate — the RUNNER integration: half-sized KV blocks, the
// `--kv-cache-dtype` thread from the flag to the block sizing, and the
// checkpoint `k_scale`/`v_scale` path.
//
// Upstream anchors, all verified in /home/mudler/_git/vllm at the parity pin
// `555967922`:
//   * `vllm/config/cache.py:19-36` CacheDType, `:76` cache_dtype default,
//     `:111` calculate_kv_scales (deprecated).
//   * `vllm/utils/torch_utils.py:32-52` STR_DTYPE_TO_TORCH_DTYPE (every fp8
//     CacheDType maps to `torch.uint8` — ONE byte), `:64-67`
//     MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP, `:75-80` is_quantized_kv_cache,
//     `:310-362` get_kv_cache_quant_algo_string, `:374-392`
//     resolve_kv_cache_dtype_string, `:394-401` kv_cache_dtype_str_to_dtype.
//   * `vllm/v1/worker/gpu_model_runner.py:484-486` — the runner resolves ONE
//     kv_cache_dtype and every attention spec is built with it.
//   * `vllm/v1/kv_cache_interface.py:204-218` AttentionSpec.real_page_size_bytes
//     — linear in `get_dtype_size(self.dtype)`, which is the whole halving.
//   * `vllm/model_executor/layers/quantization/kv_cache.py:18-30`
//     KVCacheScaleParameter (the -1.0 unloaded sentinel), `:100-102` the
//     is_quantized_kv_cache guard, `:104-127` the three loaded arms, `:150-156`
//     the uncalibrated warning.
//   * `vllm/engine/arg_utils.py:1915-1929` — the resolution happens ONCE, at
//     config construction, and CacheConfig receives the resolved string.
//
// THE CASES ARE ORDERED BY WHAT THEY WOULD LET THROUGH IF THEY WERE MISSING:
//   G1  the checkpoint declaration resolves, and an explicit flag outranks it
//   G2  a declared-but-absent scale is NOT the same state as no declaration
//   G3  the block arithmetic — an fp8 page is EXACTLY half a bf16 page
//   G4  the same halving through the LOADER: the same byte budget buys 2x blocks
//   G5  the fp8 KV path is REACHED from a production entry point (generation)
//   G6  storage dtype and fp8 interpretation cannot disagree
//   G7  an unrouted attention block is refused BY NAME, never silently
//   G8  the refusals: MLA, float16, e5m2, and a Mamba state left alone
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/cache.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/layers/quantization/kv_cache.h"
#include "vllm/model_executor/models/kv_cache_route.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using nlohmann::json;
using vllm::HfConfig;
using vllm::KvScaleOrigin;
using vllm::OwnedTensor;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vt::DType;

namespace {

// ─── The gate checkpoint's own declaration, transcribed ──────────────────────
//
// `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b8
// 7825680`, the subject of benchmark campaign #1574. Fetched from the public
// `hf_quant_config.json` on 2026-08-21 and trimmed to the three keys this
// resolver reads; the `quantized_layers` map (1900+ entries) is the WEIGHT half
// and is read elsewhere.
//
// The same revision's `model.safetensors.index.json` lists 2001 tensors and
// ZERO named `k_scale`, `v_scale` or `kv_scale` — measured, not assumed. That
// pair of facts is the whole reason G2 exists.
constexpr const char* kGateCheckpointQuantConfig = R"({
  "producer": {"name": "modelopt", "version": "0.46.0rc1"},
  "quantization": {
    "quant_algo": "MIXED_PRECISION",
    "kv_cache_quant_algo": "FP8",
    "quantized_layers": {
      "model.language_model.layers.0.mlp.gate_proj":
        {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    }
  }
})";

// A modelopt checkpoint that quantizes WEIGHTS and declares nothing about the
// KV cache — the case that must not reach a default scale.
constexpr const char* kNoKvDeclarationQuantConfig = R"({
  "producer": {"name": "modelopt", "version": "0.46.0rc1"},
  "quantization": {"quant_algo": "FP8", "quantized_layers": {}}
})";

// ─── Synthetic dense-hybrid model (the same shape as
// tests/vllm/entrypoints/test_loaded_engine_dense.cpp, which is the file whose
// LOADER path these cases enter through) ─────────────────────────────────────
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

constexpr int kVocab = 24;
constexpr int kMaxModelLen = 32;

HfConfig MakeDenseConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = kVocab;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = kMaxModelLen;
  c.raw = json::object();
  return c;
}

vllm::DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  vllm::DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

vllm::Qwen3_5DenseWeights MakeDenseWeights(const HfConfig& c) {
  vllm::Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

vllm::tok::Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_kvfp8_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},   {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},   {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11}, {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15}, {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  vocab[vllm::tok::MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[vllm::tok::MapBytesToUnicode("\x8C\x8D")] = 23;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

// The absolute KV budget both arms of G4 are given. Large enough that the fp8
// arm's doubled block count is well inside the pool the tiny model needs, and
// EXACTLY divisible by both per-block sizes so the 2x is an equality rather
// than a rounding coincidence.
constexpr int64_t kKvBudgetBytes = 1 << 20;

EngineParams ParamsWithCacheDType(const std::string& cache_dtype) {
  EngineParams p;
  p.kv_cache_memory_bytes = kKvBudgetBytes;
  p.kv_cache_dtype = cache_dtype;
  return p;
}

// The single full-attention group's spec out of a loaded engine's RESOLVED KV
// config. The synthetic model has exactly one (three GDN layers + one full
// attention layer), so "the first attention spec" is unambiguous.
const vllm::v1::AttentionSpec* SoleAttentionSpec(const LoadedEngine& eng) {
  for (const auto& group : eng.kv_cache_config().kv_cache_groups) {
    const auto* attn =
        dynamic_cast<const vllm::v1::AttentionSpec*>(group.kv_cache_spec.get());
    if (attn != nullptr) return attn;
  }
  return nullptr;
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.output_kind = vllm::RequestOutputKind::kCumulative;
  return sp;
}

class CerrRedirect {
 public:
  explicit CerrRedirect(std::streambuf* target)
      : previous_(std::cerr.rdbuf(target)) {}
  ~CerrRedirect() { std::cerr.rdbuf(previous_); }
  CerrRedirect(const CerrRedirect&) = delete;
  CerrRedirect& operator=(const CerrRedirect&) = delete;

 private:
  std::streambuf* previous_;
};

// A bare NHD KV cache pair for the op-level cases, on the CPU queue.
struct HostKvPair {
  std::vector<uint8_t> storage;
  vt::Tensor k;
  vt::Tensor v;
};

}  // namespace

// ─── G1. The checkpoint's declaration, and who outranks whom ─────────────────
TEST_CASE("kv-fp8 W3 G1: the gate checkpoint's kv_cache_quant_algo resolves") {
  // torch_utils.py:374-392 + :310-362 + :64-67. "FP8" (the modelopt spelling,
  // upper case) maps to vLLM's own `fp8_e4m3`, not to the bare "fp8" alias.
  const vllm::ResolvedCacheDTypeString r =
      vllm::ResolveKvCacheDTypeString("auto", kGateCheckpointQuantConfig);
  CHECK(r.cache_dtype == "fp8_e4m3");
  // The FACT that separates this from an operator who typed the flag.
  CHECK(r.declared_by_checkpoint);
}

TEST_CASE("kv-fp8 W3 G1: an explicit --kv-cache-dtype outranks the checkpoint") {
  // torch_utils.py:380-381 returns the explicit value UNCHANGED without ever
  // reading the config, and attention.py:279-290 re-applies the same precedence
  // with the comment "an explicit choice (e.g. bfloat16) must win".
  const vllm::ResolvedCacheDTypeString r =
      vllm::ResolveKvCacheDTypeString("bfloat16", kGateCheckpointQuantConfig);
  CHECK(r.cache_dtype == "bfloat16");
  CHECK_FALSE(r.declared_by_checkpoint);
}

TEST_CASE("kv-fp8 W3 G1: a checkpoint that declares no KV algo resolves auto") {
  const vllm::ResolvedCacheDTypeString none =
      vllm::ResolveKvCacheDTypeString("auto", kNoKvDeclarationQuantConfig);
  CHECK(none.cache_dtype == "auto");
  CHECK_FALSE(none.declared_by_checkpoint);

  // No quantization config at all — the ordinary bf16 checkpoint.
  const vllm::ResolvedCacheDTypeString empty =
      vllm::ResolveKvCacheDTypeString("auto", "");
  CHECK(empty.cache_dtype == "auto");
  CHECK_FALSE(empty.declared_by_checkpoint);
}

TEST_CASE("kv-fp8 W3 G1: an unrecognized kv_cache_quant_algo falls back to auto") {
  // torch_utils.py:351-361 — upstream's own safe fallback. It must NOT become
  // "declared", because a KV format we cannot serve is not a declaration we can
  // honour.
  const vllm::ResolvedCacheDTypeString r = vllm::ResolveKvCacheDTypeString(
      "auto",
      R"({"producer":{"name":"modelopt"},
          "quantization":{"quant_algo":"FP8","kv_cache_quant_algo":"INT3"}})");
  CHECK(r.cache_dtype == "auto");
  CHECK_FALSE(r.declared_by_checkpoint);
}

// ─── G2. Declared-but-absent is NOT the same state as never declared ─────────
TEST_CASE("kv-fp8 W3 G2: a DECLARED fp8 cache with no scale tensors takes 1.0") {
  // kv_cache.py:112-116 — this is the arm the gate checkpoint takes: it declares
  // `kv_cache_quant_algo: "FP8"` and ships zero k/v scale tensors, so both
  // sentinels survive and the documented default applies.
  const vllm::ResolvedKvCacheScales r = vllm::ResolveKvCacheScales(
      "fp8_e4m3", /*calculate_kv_scales=*/false, vllm::kKvScaleUnloaded,
      vllm::kKvScaleUnloaded);
  CHECK(r.origin == KvScaleOrigin::kDeclaredButAbsent);
  CHECK(r.k_scale == doctest::Approx(1.0F));
  CHECK(r.v_scale == doctest::Approx(1.0F));
  // kv_cache.py:150-156 — and it says so.
  CHECK(r.uncalibrated);

  // The consumer is happy to be handed this pair: it was DECLARED.
  float k = 0.0F;
  float v = 0.0F;
  vllm::ScalesForFp8Store(r, &k, &v);
  CHECK(k == doctest::Approx(1.0F));
  CHECK(v == doctest::Approx(1.0F));
}

TEST_CASE(
    "kv-fp8 W3 G2: NO declaration is a different state and yields NO scale") {
  // THE CASE THIS WHOLE FILE EXISTS FOR. Identical inputs to the one above
  // except the declaration, and identical NUMBERS out — 1.0/1.0 are the struct's
  // defaults — so a gate that only read k_scale/v_scale could not tell them
  // apart. `origin` can, and the consumer refuses on it.
  const vllm::ResolvedKvCacheScales r = vllm::ResolveKvCacheScales(
      "auto", /*calculate_kv_scales=*/false, vllm::kKvScaleUnloaded,
      vllm::kKvScaleUnloaded);
  CHECK(r.origin == KvScaleOrigin::kNotQuantized);
  // kv_cache.py:100-102: the scale block never ran, so the uncalibrated warning
  // is not owed either.
  CHECK_FALSE(r.uncalibrated);

  float k = 0.0F;
  float v = 0.0F;
  CHECK_THROWS_AS(vllm::ScalesForFp8Store(r, &k, &v), std::runtime_error);
  // And the refusal NAMES what is missing, so the next reader does not have to
  // rediscover the distinction.
  try {
    vllm::ScalesForFp8Store(r, &k, &v);
    FAIL("ScalesForFp8Store accepted a kNotQuantized pair");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    CHECK(msg.find("no fp8 KV cache was declared") != std::string::npos);
    CHECK(msg.find("kv_cache_quant_algo") != std::string::npos);
  }
  // Nothing was written into the outputs.
  CHECK(k == doctest::Approx(0.0F));
  CHECK(v == doctest::Approx(0.0F));
}

TEST_CASE("kv-fp8 W3 G2: the three LOADED arms mirror kv_cache.py:104-127") {
  // Both scales present (:104-111).
  const vllm::ResolvedKvCacheScales both = vllm::ResolveKvCacheScales(
      "fp8_e4m3", /*calculate_kv_scales=*/false, 0.5F, 0.25F);
  CHECK(both.origin == KvScaleOrigin::kCheckpoint);
  CHECK(both.k_scale == doctest::Approx(0.5F));
  CHECK(both.v_scale == doctest::Approx(0.25F));
  CHECK_FALSE(both.uncalibrated);  // not 1.0/1.0

  // A single `kv_scale`, remapped to k_scale at load and duplicated (:117-127).
  const vllm::ResolvedKvCacheScales dup = vllm::ResolveKvCacheScales(
      "fp8_e4m3", /*calculate_kv_scales=*/false, 0.5F, vllm::kKvScaleUnloaded);
  CHECK(dup.origin == KvScaleOrigin::kCheckpointKvScale);
  CHECK(dup.k_scale == doctest::Approx(0.5F));
  CHECK(dup.v_scale == doctest::Approx(0.5F));

  // e5m2 suppresses the uncalibrated warning (:153) — 1.0 is ordinary there.
  const vllm::ResolvedKvCacheScales e5m2 = vllm::ResolveKvCacheScales(
      "fp8_e5m2", /*calculate_kv_scales=*/false, vllm::kKvScaleUnloaded,
      vllm::kKvScaleUnloaded);
  CHECK(e5m2.origin == KvScaleOrigin::kDeclaredButAbsent);
  CHECK_FALSE(e5m2.uncalibrated);

  // The deprecated dynamic path is refused BY NAME rather than silently taking
  // the static arm (cache.py:111).
  CHECK_THROWS_AS(vllm::ResolveKvCacheScales("fp8_e4m3",
                                             /*calculate_kv_scales=*/true,
                                             vllm::kKvScaleUnloaded,
                                             vllm::kKvScaleUnloaded),
                  std::runtime_error);
}

// ─── G3. The block arithmetic ────────────────────────────────────────────────
TEST_CASE("kv-fp8 W3 G3: an fp8 KV page is EXACTLY half a bf16 page") {
  // kv_cache_interface.py:204-218 — `real_page_size_bytes` is
  // `2 * block_size * num_kv_heads * head_dim * get_dtype_size(dtype)`, and
  // torch_utils.py:38-40 makes every fp8 CacheDType one byte. Assert the CLOSED
  // FORM, not just the ratio: a ratio alone is satisfied by any pair of widths
  // in 2:1, including a pair that is wrong on both sides.
  constexpr int kBlock = 16;
  constexpr int kHkv = 4;
  constexpr int kDh = 64;
  constexpr int64_t kElems = 2LL * kBlock * kHkv * kDh;  // K + V

  vllm::v1::KVCacheConfig cfg;
  cfg.num_blocks = 8;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(kBlock, kHkv, kDh,
                                                    DType::kBF16));
  const auto* spec = dynamic_cast<const vllm::v1::AttentionSpec*>(
      cfg.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(spec != nullptr);
  const int64_t bf16_page = spec->page_size_bytes();
  CHECK(bf16_page == kElems * 2);

  vllm::v1::ApplyCacheDType(cfg, vllm::v1::ParseCacheDType("fp8", DType::kBF16),
                            1.0F, 1.0F);
  const int64_t fp8_page = spec->page_size_bytes();
  CHECK(fp8_page == kElems * 1);
  CHECK(fp8_page * 2 == bf16_page);
  // The storage dtype and the interpretation both landed, on the SAME spec.
  CHECK(spec->dtype == DType::kI8);
  CHECK(spec->fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3);
  // KVBytesPerBlock — the divisor the pool sizing actually uses — halves too.
  CHECK(vllm::v1::KVBytesPerBlock(cfg) == kElems);
}

TEST_CASE("kv-fp8 W3 G3: an auto cache_dtype leaves every spec untouched") {
  // The byte-identical default. `ApplyCacheDType` must not rewrite a spec the
  // model's factory already built at the model dtype.
  constexpr int kBlock = 16;
  vllm::v1::KVCacheConfig cfg;
  cfg.num_blocks = 8;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(kBlock, 4, 64,
                                                    DType::kBF16));
  const int64_t before = vllm::v1::KVBytesPerBlock(cfg);
  vllm::v1::ApplyCacheDType(cfg, vllm::v1::ParseCacheDType("auto", DType::kBF16),
                            1.0F, 1.0F);
  CHECK(vllm::v1::KVBytesPerBlock(cfg) == before);
  const auto* spec = dynamic_cast<const vllm::v1::AttentionSpec*>(
      cfg.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(spec != nullptr);
  CHECK(spec->dtype == DType::kBF16);
  CHECK(spec->fp8_kind == vt::Fp8KVCacheDataType::kAuto);
}

TEST_CASE("kv-fp8 W3 G3: the f32 A/B cache and an MLA spec still load on auto") {
  // The regression this early return exists for. `ApplyCacheDType` refuses
  // float16 and refuses MLA — correctly — so it must not REACH those refusals on
  // the default path. `VT_KV_CACHE_F32=1` builds an f32 KV spec and "auto"
  // resolves to f32; an MLA model on "auto" resolves to its own model dtype.
  // Both are asking for nothing to change, and both must survive.
  vllm::v1::KVCacheConfig f32;
  f32.num_blocks = 4;
  f32.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(16, 4, 64, DType::kF32));
  vllm::v1::ApplyCacheDType(
      f32, vllm::v1::ParseCacheDType("auto", DType::kF32), 1.0F, 1.0F);
  const auto* f32_spec = dynamic_cast<const vllm::v1::AttentionSpec*>(
      f32.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(f32_spec != nullptr);
  CHECK(f32_spec->dtype == DType::kF32);

  vllm::v1::KVCacheConfig mla;
  mla.num_blocks = 4;
  mla.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<vllm::v1::MLAAttentionSpec>(16, 576, DType::kBF16));
  // No throw: the MLA refusal is for an fp8 REQUEST, not for every load.
  vllm::v1::ApplyCacheDType(
      mla, vllm::v1::ParseCacheDType("auto", DType::kBF16), 1.0F, 1.0F);
  const auto* mla_spec = dynamic_cast<const vllm::v1::AttentionSpec*>(
      mla.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(mla_spec != nullptr);
  CHECK(mla_spec->dtype == DType::kBF16);
}

// ─── G4. The same halving, through the LOADER ────────────────────────────────
TEST_CASE(
    "kv-fp8 W3 G4: --kv-cache-dtype fp8 buys EXACTLY 2x the blocks at one "
    "--kv-cache-memory") {
  // This case enters through the production entry point — the LoadedEngine
  // constructor -> MakeKVCacheResolved -> ApplyResolvedCacheDType ->
  // ResolveNumBlocks — rather than calling the resolver, because what is under
  // test is that the storage dtype reaches the sizing BEFORE the sizing reads
  // the geometry. Applying it afterwards would leave this equality at 1x and is
  // the ordering mistake the comment in MakeKVCacheResolved names.
  const HfConfig c = MakeDenseConfig();

  LoadedEngine bf16(c, MakeDenseWeights(c), BuildFixture(),
                    ParamsWithCacheDType("auto"));
  LoadedEngine fp8(c, MakeDenseWeights(c), BuildFixture(),
                   ParamsWithCacheDType("fp8"));

  const int bf16_blocks = bf16.kv_cache_config().num_blocks;
  const int fp8_blocks = fp8.kv_cache_config().num_blocks;
  REQUIRE(bf16_blocks > 0);
  CHECK(fp8_blocks == 2 * bf16_blocks);

  // And the specs say why.
  const auto* bf16_spec = SoleAttentionSpec(bf16);
  const auto* fp8_spec = SoleAttentionSpec(fp8);
  REQUIRE(bf16_spec != nullptr);
  REQUIRE(fp8_spec != nullptr);
  CHECK(bf16_spec->dtype == DType::kBF16);
  CHECK(fp8_spec->dtype == DType::kI8);
  CHECK(fp8_spec->page_size_bytes() * 2 == bf16_spec->page_size_bytes());

  // The POOL is the same size in bytes — that is the point of the feature: the
  // same memory now holds twice the context.
  CHECK(static_cast<int64_t>(fp8_blocks) * vllm::v1::KVBytesPerBlock(
            fp8.kv_cache_config()) ==
        static_cast<int64_t>(bf16_blocks) *
            vllm::v1::KVBytesPerBlock(bf16.kv_cache_config()));
}

TEST_CASE("kv-fp8 W3 G4: the GDN/Mamba state is NOT retyped") {
  // config/cache.py:131-138 — recurrent state has its own mamba_cache_dtype
  // knob and `--kv-cache-dtype` never touches it. The synthetic model has three
  // linear-attention layers, so a rewrite that walked every group would corrupt
  // them.
  const HfConfig c = MakeDenseConfig();
  LoadedEngine fp8(c, MakeDenseWeights(c), BuildFixture(),
                   ParamsWithCacheDType("fp8"));
  bool saw_mamba = false;
  for (const auto& group : fp8.kv_cache_config().kv_cache_groups) {
    const auto* mamba =
        dynamic_cast<const vllm::v1::MambaSpec*>(group.kv_cache_spec.get());
    if (mamba == nullptr) continue;
    saw_mamba = true;
    for (const vt::DType dt : mamba->dtypes) {
      CHECK(dt != DType::kI8);
    }
  }
  CHECK(saw_mamba);  // the case would be vacuous without one
}

// ─── G5. Reachability: the fp8 KV path SERVES ────────────────────────────────
TEST_CASE("kv-fp8 W3 G5: an fp8 KV engine generates through the real forward") {
  // The reachability gate. Nothing here constructs a vt op or a PagedKvCache by
  // hand: the engine is built from the loader, the request goes through
  // LLMEngine, and the tokens come out of Qwen3_5DenseModel::Forward, whose
  // full-attention layer writes and reads THIS cache. If the store or the read
  // were not routed, `vt::ReshapeAndCache` refuses the kI8 page by name (G7)
  // and this case throws instead of counting tokens.
  const HfConfig c = MakeDenseConfig();
  EngineParams params = ParamsWithCacheDType("fp8");
  LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), params);

  // The cache really is one byte per element on the layer that serves.
  const auto* spec = SoleAttentionSpec(eng);
  REQUIRE(spec != nullptr);
  REQUIRE(spec->dtype == DType::kI8);
  REQUIRE(spec->fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3);

  constexpr int kMaxTokens = 4;
  const vllm::RequestOutput run1 =
      eng.engine().generate("hello world", Greedy(kMaxTokens), "req");
  REQUIRE(run1.finished);
  REQUIRE(run1.outputs.size() == 1);
  CHECK(static_cast<int>(run1.outputs[0].token_ids.size()) == kMaxTokens);

  // Deterministic across two fresh stacks on the fp8 cache — greedy decode over
  // an fp8 KV store is still a function of the inputs.
  LoadedEngine again(c, MakeDenseWeights(c), BuildFixture(), params);
  const vllm::RequestOutput run2 =
      again.engine().generate("hello world", Greedy(kMaxTokens), "req");
  REQUIRE(run2.outputs.size() == 1);
  CHECK(run2.outputs[0].token_ids == run1.outputs[0].token_ids);
}

TEST_CASE("kv-fp8 W3 G5: the uncalibrated-scale warning fires ONCE per load") {
  // kv_cache.py:150-156. The gate checkpoint's own case: an fp8 cache serving on
  // the default 1.0, said out loud.
  const HfConfig c = MakeDenseConfig();
  std::ostringstream captured;
  {
    CerrRedirect guard(captured.rdbuf());
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(),
                     ParamsWithCacheDType("fp8"));
    std::cerr.flush();
  }
  const std::string logged = captured.str();
  const std::string needle = "KV cache scaling factor 1.0";
  const size_t first = logged.find(needle);
  REQUIRE(first != std::string::npos);
  // Once, not once per ApplyResolvedCacheDType call (there are two per load).
  CHECK(logged.find(needle, first + needle.size()) == std::string::npos);
  // It names the checkpoint as the place to fix it.
  CHECK(logged.find("checkpoint") != std::string::npos);

  // A SECOND engine in the SAME process gets its own line. Upstream's
  // `logger.warning_once` is per-process; ours is per LOAD, because a
  // process-static latch would silence the second engine rather than the second
  // of the two ApplyResolvedCacheDType calls one load makes. Getting that wrong
  // reads as "the warning fired once" on both counts.
  std::ostringstream second;
  {
    CerrRedirect guard(second.rdbuf());
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(),
                     ParamsWithCacheDType("fp8"));
    std::cerr.flush();
  }
  const std::string logged2 = second.str();
  const size_t only = logged2.find(needle);
  REQUIRE(only != std::string::npos);
  CHECK(logged2.find(needle, only + needle.size()) == std::string::npos);

  // And an auto engine says nothing, so the line is a warning rather than noise.
  std::ostringstream quiet;
  {
    CerrRedirect guard(quiet.rdbuf());
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(),
                     ParamsWithCacheDType("auto"));
    std::cerr.flush();
  }
  CHECK(quiet.str().find(needle) == std::string::npos);
}

// ─── G6. Storage dtype and interpretation cannot disagree ────────────────────
TEST_CASE("kv-fp8 W3 G6: a half-sized page with no fp8 kind is refused") {
  // The silent-corruption shape, asserted at the routing seam. A `kI8` page is
  // sized at one byte per element; an fp8 kind of kAuto would send it to the
  // float store, which indexes at the source width. Neither half of the pair is
  // allowed to travel alone.
  vllm::PagedKvCache kv;
  kv.dtype = DType::kI8;
  kv.fp8_kind = vt::Fp8KVCacheDataType::kAuto;
  CHECK_THROWS_AS(vllm::dense_attn::IsFp8KvCache(kv), std::runtime_error);

  vllm::PagedKvCache other;
  other.dtype = DType::kBF16;
  other.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  CHECK_THROWS_AS(vllm::dense_attn::IsFp8KvCache(other), std::runtime_error);

  // The two consistent states answer without throwing.
  vllm::PagedKvCache floatkv;
  CHECK_FALSE(vllm::dense_attn::IsFp8KvCache(floatkv));
  vllm::PagedKvCache fp8kv;
  fp8kv.dtype = DType::kI8;
  fp8kv.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  CHECK(vllm::dense_attn::IsFp8KvCache(fp8kv));
}

TEST_CASE("kv-fp8 W3 G6: ApplyKvCacheQuant is inert on a float cache") {
  // Every existing caller must be byte-identical: the three additive
  // PagedAttentionArgs fields keep their defaults on a float cache, so the op
  // takes exactly the branch it took before W3.
  vllm::PagedKvCache floatkv;
  vt::PagedAttentionArgs args{0.125F, true};
  vllm::dense_attn::ApplyKvCacheQuant(args, floatkv);
  CHECK(args.kv_cache_dtype == vt::Fp8KVCacheDataType::kAuto);
  CHECK(args.k_scale == doctest::Approx(1.0F));
  CHECK(args.v_scale == doctest::Approx(1.0F));

  vllm::PagedKvCache fp8kv;
  fp8kv.dtype = DType::kI8;
  fp8kv.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  fp8kv.k_scale = 0.5F;
  fp8kv.v_scale = 0.25F;
  vt::PagedAttentionArgs fp8_args{0.125F, true};
  vllm::dense_attn::ApplyKvCacheQuant(fp8_args, fp8kv);
  CHECK(fp8_args.kv_cache_dtype == vt::Fp8KVCacheDataType::kFp8E4M3);
  CHECK(fp8_args.k_scale == doctest::Approx(0.5F));
  CHECK(fp8_args.v_scale == doctest::Approx(0.25F));
}

// ─── G7. An unrouted attention block is refused BY NAME ──────────────────────
TEST_CASE("kv-fp8 W3 G7: the float store refuses a 1-byte fp8 cache by name") {
  // W3 routes `dense_attn::AttnBlock` (the shared seam) and `qwen3_5.cpp`. Every
  // other architecture still calls `vt::ReshapeAndCache` directly, and this is
  // what happens when one of them is served `--kv-cache-dtype fp8`: a named
  // refusal at the first store, not a half-width write into a half-sized page.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  constexpr int64_t T = 2, Hkv = 2, Dh = 4, kBlocks = 2, kBlockSize = 4;
  std::vector<uint16_t> k_src(static_cast<size_t>(T * Hkv * Dh), 0);
  std::vector<uint16_t> v_src(k_src.size(), 0);
  // The NHD unbind-slice cache: ONE (num_blocks, 2, block_size, H, D) byte
  // allocation, k/v are the two dim-1 slices.
  std::vector<uint8_t> cache(
      static_cast<size_t>(kBlocks * 2 * kBlockSize * Hkv * Dh), 0);
  std::vector<int64_t> slots{0, 1};

  vt::Tensor k = vt::Tensor::Contiguous(k_src.data(), DType::kBF16, q.device,
                                        {T, Hkv, Dh});
  vt::Tensor v = vt::Tensor::Contiguous(v_src.data(), DType::kBF16, q.device,
                                        {T, Hkv, Dh});
  vt::Tensor slot_mapping = vt::Tensor::Contiguous(
      slots.data(), DType::kI64, q.device, {static_cast<int64_t>(slots.size())});

  const auto kv_slice = [&](int which) {
    vt::Tensor t;
    t.data = cache.data() + static_cast<size_t>(which) *
                                static_cast<size_t>(kBlockSize * Hkv * Dh);
    t.dtype = DType::kI8;
    t.device = q.device;
    t.rank = 4;
    t.shape[0] = kBlocks;
    t.shape[1] = kBlockSize;
    t.shape[2] = Hkv;
    t.shape[3] = Dh;
    t.stride[0] = 2 * kBlockSize * Hkv * Dh;
    t.stride[1] = Hkv * Dh;
    t.stride[2] = Dh;
    t.stride[3] = 1;
    return t;
  };
  vt::Tensor k_cache = kv_slice(0);
  vt::Tensor v_cache = kv_slice(1);

  try {
    vt::ReshapeAndCache(q, k, v, k_cache, v_cache, slot_mapping);
    FAIL("the float store accepted a 1-byte fp8 cache");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    // It names the OP that should have been called...
    CHECK(msg.find("vt::ReshapeAndCacheFp8") != std::string::npos);
    // ...and the missing part, so the reader knows this is an unwired
    // architecture rather than a corrupt tensor.
    CHECK(msg.find("not routed for fp8 KV") != std::string::npos);
  }
}

// ─── G8. The refusals that stop a mis-sized pool ─────────────────────────────
TEST_CASE("kv-fp8 W3 G8: an MLA cache refuses --kv-cache-dtype fp8 by name") {
  // kv_cache_interface.py:398-410 — upstream's MLA fp8 arm is `fp8_ds_mla` with
  // a different page formula (656 B/token on V3.2), not this one. Retyping an
  // MLA spec to kI8 would size the latent page at half and store it at full.
  vllm::v1::KVCacheConfig cfg;
  cfg.num_blocks = 4;
  cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<vllm::v1::MLAAttentionSpec>(16, 576, DType::kBF16));
  try {
    vllm::v1::ApplyCacheDType(
        cfg, vllm::v1::ParseCacheDType("fp8", DType::kBF16), 1.0F, 1.0F);
    FAIL("ApplyCacheDType retyped an MLA spec");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    CHECK(msg.find("MLA") != std::string::npos);
    CHECK(msg.find("fp8_ds_mla") != std::string::npos);
  }
}

TEST_CASE("kv-fp8 W3 G8: float16 and fp8_e5m2 are refused, not mis-stored") {
  const auto fresh = [] {
    vllm::v1::KVCacheConfig cfg;
    cfg.num_blocks = 4;
    cfg.kv_cache_groups.emplace_back(
        std::vector<std::string>{"fa"},
        std::make_shared<vllm::v1::FullAttentionSpec>(16, 4, 64, DType::kBF16));
    return cfg;
  };

  // float16 PARSES — the CacheDType surface is mirrored in full (cache.py:19-36)
  // — but no attention block casts K/V to f16 before the store, so applying it
  // would reach a dtype mismatch deep inside the op instead of a sentence.
  vllm::v1::KVCacheConfig f16 = fresh();
  try {
    vllm::v1::ApplyCacheDType(
        f16, vllm::v1::ParseCacheDType("float16", DType::kBF16), 1.0F, 1.0F);
    FAIL("ApplyCacheDType accepted float16");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("float16") != std::string::npos);
  }

  // fp8_e5m2 likewise: W1/W2 refuse the compute, so the SIZING must refuse too
  // rather than allocate a half-sized pool nothing can write.
  vllm::v1::KVCacheConfig e5m2 = fresh();
  try {
    vllm::v1::ApplyCacheDType(
        e5m2, vllm::v1::ParseCacheDType("fp8_e5m2", DType::kBF16), 1.0F, 1.0F);
    FAIL("ApplyCacheDType accepted fp8_e5m2");
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find("fp8_e5m2") != std::string::npos);
  }
}
