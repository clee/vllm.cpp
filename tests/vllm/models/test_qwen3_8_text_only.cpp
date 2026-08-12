// Qwen3.5-family TEXT-ONLY arms: `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`.
//
// AHEAD-OF-PIN FORWARD PORT of upstream PR #50210 (`ad5d29db7`). Our parity pin
// is `555967922`, whose registry.py carries only the `ForConditionalGeneration`
// strings, so this file is deliberately anchored on a POST-PIN upstream head and
// says so; it does not advance the pin.
//
//   upstream vllm/model_executor/models/registry.py:202-203 @ `ad5d29db7`
//     "Qwen3_5ForCausalLM":    ("qwen3_5", "Qwen3_5ForCausalLM"),
//     "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),
//   upstream vllm/model_executor/models/qwen3_5.py:439-449 @ `ad5d29db7`
//     `Qwen3_5ForCausalLM` IS `Qwen3_5ForCausalLMBase` unchanged; the MoE arm is
//     that same base plus `set_moe_parameters()` — one backbone, two arms.
//   upstream vllm/model_executor/models/qwen3_5.py:296-300 @ `ad5d29db7`
//     WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
//     — `model.` is CANONICAL and the VL-prefixed spelling is its accepted alias.
//
// WHAT THIS FILE DOES NOT CLAIM. `Qwen/Qwen3.8-2.4T-A95B` cannot be executed on
// this hardware (2.4T bf16 ≈ 4.8 TB, the FP8 variant ≈ 2.4 TB, GB10 has 128 GB
// unified and no smaller Qwen3.8 sibling exists), so there is NO token gate for
// that checkpoint and none is implied here. These cases pin architecture
// dispatch, flat-config resolution and weight-namespace resolution — nothing
// about generated tokens. See .agents/specs/qwen38-text-only.md §Gates.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"

using vllm::HfConfig;
using vllm::ModelRegistration;
using vllm::ModelRegistry;

namespace {

HfConfig ArchConfig(std::vector<std::string> architectures) {
  HfConfig config;
  config.architectures = std::move(architectures);
  return config;
}

// ---------------------------------------------------------------------------
// Synthetic checkpoint plumbing (same shape as tests/vllm/test_load_direct_upload
// .cpp: a real safetensors file on disk, opened through the production reader).
// ---------------------------------------------------------------------------

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes, const char* tag) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_qwen3_8_" + std::string(tag) + "_" +
              std::to_string(counter++) + ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

class TempJsonDir {
 public:
  explicit TempJsonDir(const std::string& config_body) {
    static int counter = 0;
    dir_ = (std::filesystem::temp_directory_path() /
            ("vllm_qwen3_8_cfg_" + std::to_string(counter++)))
               .string();
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ + "/config.json", std::ios::binary) << config_body;
  }
  ~TempJsonDir() { std::filesystem::remove_all(dir_); }
  TempJsonDir(const TempJsonDir&) = delete;
  TempJsonDir& operator=(const TempJsonDir&) = delete;
  std::string config_path() const { return dir_ + "/config.json"; }

 private:
  std::string dir_;
};

// One tiny synthetic tensor description: name + shape, BF16, filled with a
// deterministic per-tensor pattern so a wrong binding shows up in the VALUES,
// not only in a name.
struct Spec {
  std::string name;
  std::vector<int64_t> shape;
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return n;
}

// Builds a whole safetensors file from `specs`. The bytes of a tensor depend
// ONLY on its position in `specs`, so two files built from the same specs with
// different NAMES carry byte-identical payloads — which is what lets the
// namespace test compare two loads for byte equality.
std::string BuildSafetensors(const std::vector<Spec>& specs) {
  std::string header = "{";
  std::string body;
  uint64_t offset = 0;
  for (size_t i = 0; i < specs.size(); ++i) {
    const int64_t n = Numel(specs[i].shape);
    const auto nbytes = static_cast<uint64_t>(n) * 2;
    if (i != 0) header += ",";
    header += "\"" + specs[i].name + "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (size_t d = 0; d < specs[i].shape.size(); ++d) {
      if (d != 0) header += ",";
      header += std::to_string(specs[i].shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + nbytes) + "]}";
    offset += nbytes;

    std::vector<uint16_t> values(static_cast<size_t>(n));
    for (size_t e = 0; e < values.size(); ++e) {
      // A finite, distinct bf16 per (tensor index, element index).
      values[e] = static_cast<uint16_t>(0x3d00 + ((i * 37 + e * 7) & 0x1ff));
    }
    const size_t at = body.size();
    body.resize(at + static_cast<size_t>(nbytes));
    std::memcpy(body.data() + at, values.data(), static_cast<size_t>(nbytes));
  }
  header += "}";
  return U64Le(header.size()) + header + body;
}

// The full backbone tensor list of a ONE-layer full-attention Qwen3.5 dense
// checkpoint under prefix `p`, plus the top-level tied-head case (no lm_head).
// Names verified against the published `Qwen/Qwen3.8-2.4T-A95B` and
// `Qwen/Qwen3.6-35B-A3B` safetensors indices: identical modulo the prefix.
std::vector<Spec> DenseOneLayerSpecs(const std::string& p) {
  const std::string l = p + "layers.0.";
  const std::string sa = l + "self_attn.";
  const std::string mlp = l + "mlp.";
  constexpr int64_t kHidden = 8;
  constexpr int64_t kFfn = 16;
  constexpr int64_t kHeadDim = 4;
  constexpr int64_t kQ = 8;   // 2 heads x 4
  constexpr int64_t kKv = 4;  // 1 head  x 4
  return {
      {p + "embed_tokens.weight", {6, kHidden}},
      {p + "norm.weight", {kHidden}},
      {l + "input_layernorm.weight", {kHidden}},
      {l + "post_attention_layernorm.weight", {kHidden}},
      {sa + "q_proj.weight", {kQ, kHidden}},
      {sa + "k_proj.weight", {kKv, kHidden}},
      {sa + "v_proj.weight", {kKv, kHidden}},
      {sa + "o_proj.weight", {kHidden, kQ}},
      {sa + "q_norm.weight", {kHeadDim}},
      {sa + "k_norm.weight", {kHeadDim}},
      {mlp + "gate_proj.weight", {kFfn, kHidden}},
      {mlp + "up_proj.weight", {kFfn, kHidden}},
      {mlp + "down_proj.weight", {kHidden, kFfn}},
  };
}

// The tensor payload of a safetensors blob: everything after the 8-byte header
// length and the JSON header itself.
std::string Payload(const std::string& file) {
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) {
    header_len = (header_len << 8) | static_cast<uint8_t>(file[static_cast<size_t>(i)]);
  }
  return file.substr(8 + static_cast<size_t>(header_len));
}

std::vector<std::string> NamesOf(const std::vector<Spec>& specs) {
  std::vector<std::string> names;
  names.reserve(specs.size());
  for (const Spec& s : specs) names.push_back(s.name);
  return names;
}

HfConfig OneLayerDenseConfig() {
  HfConfig config;
  config.model_type = "qwen3_5_text";
  config.hidden_size = 8;
  config.num_hidden_layers = 1;
  config.layer_types = {"full_attention"};
  return config;
}

// Byte-for-byte comparison of a loaded tensor pair.
void CheckSameBytes(const vllm::OwnedTensor& a, const vllm::OwnedTensor& b,
                    const char* what) {
  CAPTURE(what);
  REQUIRE(a.rank == b.rank);
  for (int i = 0; i < a.rank; ++i) CHECK(a.shape[i] == b.shape[i]);
  REQUIRE(a.bytes.size() == b.bytes.size());
  REQUIRE(a.bytes.size() > 0);
  CHECK(std::memcmp(a.bytes.data(), b.bytes.data(), a.bytes.size()) == 0);
}

}  // namespace

// ===========================================================================
// 1. Architecture dispatch. Upstream registers both text-only arms against the
//    SAME `qwen3_5` module (registry.py:202-203 @ `ad5d29db7`), so ours must
//    resolve to the SAME factories the ForConditionalGeneration wrappers use —
//    a second factory would be a fork of a backbone we already gate.
// ===========================================================================
TEST_CASE("qwen3_8: both text-only architecture strings resolve to the Qwen3.5 factories") {
  const HfConfig moe_config = ArchConfig({"Qwen3_5MoeForCausalLM"});
  const ModelRegistration& moe = ModelRegistry::Resolve(moe_config);
  CHECK(moe.architecture == "Qwen3_5MoeForCausalLM");
  CHECK(moe.factory ==
        vllm::RegistrationFor("Qwen3_5MoeForConditionalGeneration").factory);
  CHECK_FALSE(moe.factory->is_dense_model);

  const HfConfig dense_config = ArchConfig({"Qwen3_5ForCausalLM"});
  const ModelRegistration& dense = ModelRegistry::Resolve(dense_config);
  CHECK(dense.architecture == "Qwen3_5ForCausalLM");
  CHECK(dense.factory ==
        vllm::RegistrationFor("Qwen3_5ForConditionalGeneration").factory);
  CHECK(dense.factory->is_dense_model);

  // Upstream's `Qwen3_5ForCausalLMBase` inherits IsHybrid + HasInnerState but
  // NOT SupportsMultiModal (qwen3_5.py:287-296 @ `ad5d29db7`): these are the
  // TEXT arms, and their multimodal wrappers are separate registrations. Same
  // convention as `KimiLinearForCausalLM` — hybrid yes, multimodal no.
  for (const ModelRegistration* registration : {&moe, &dense}) {
    CAPTURE(registration->architecture);
    CHECK(registration->info.is_text_generation_model);
    CHECK(registration->info.is_hybrid);
    CHECK_FALSE(registration->info.supports_multimodal);
    CHECK_FALSE(registration->info.is_pooling_model);
  }
}

// ===========================================================================
// 2. Config resolution on the REAL flat 3.8 shape. `Qwen/Qwen3.8-2.4T-A95B`
//    declares `model_type: qwen3_5_moe_text` at the TOP level with no
//    `text_config` wrapper, no `vision_config` and no `mrope_section` — the
//    composite-wrapper path our 27B/35B checkpoints take does not apply.
//    Values from .agents/specs/qwen38-text-only.md §"Why this is not a new port".
// ===========================================================================
TEST_CASE("qwen3_8: the flat 2.4T text config resolves through the shared Qwen3.5 path") {
  const TempJsonDir dir(R"({
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 8192,
    "num_hidden_layers": 92,
    "vocab_size": 248320,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 2048,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 2048,
    "shared_expert_intermediate_size": 2048,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 128,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "rope_theta": 10000000.0,
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16"
  })");
  const HfConfig config = vllm::LoadHfConfig(dir.config_path());

  // The architecture the registry will be asked for, straight off a flat doc.
  REQUIRE(config.architectures.size() == 1);
  CHECK(config.architectures[0] == "Qwen3_5MoeForCausalLM");
  CHECK(config.model_type == "qwen3_5_moe_text");
  CHECK_NOTHROW(ModelRegistry::Resolve(config));

  // Scale: every knob that differs from the 35B is CONFIG, never a constant.
  CHECK(config.hidden_size == 8192);
  CHECK(config.num_hidden_layers == 92);
  CHECK(config.num_attention_heads == 64);
  CHECK(config.num_key_value_heads == 4);
  CHECK(config.head_dim == 256);
  CHECK(config.vocab_size == 248320);
  CHECK(config.num_experts == 512);
  CHECK(config.num_experts_per_tok == 10);
  CHECK(config.moe_intermediate_size == 2048);
  CHECK(config.shared_expert_intermediate_size == 2048);
  CHECK(config.linear_num_key_heads == 16);
  CHECK(config.linear_num_value_heads == 128);
  CHECK(config.linear_key_head_dim == 128);
  CHECK(config.linear_value_head_dim == 128);
  CHECK(config.linear_conv_kernel_dim == 4);
  CHECK(config.rope_theta == doctest::Approx(1e7).scale(0.0));

  // `qwen3_5_moe_text` is already in IsQwen35Family, so a config with NO
  // rope_parameters block still gets the family's 0.25 partial-rotary default
  // (qwen3_next.py:240 / qwen3_5_moe.py:92) => rotary_dim = 0.25 * 256.
  CHECK(config.rope_parameters.partial_rotary_factor ==
        doctest::Approx(0.25).scale(0.0));
  CHECK(config.rotary_dim == 64);

  // A text-only checkpoint has no vision tower and no MRoPE sections. Both are
  // ABSENT rather than empty-but-present, and neither may be synthesized.
  CHECK(config.raw.find("vision_config") == config.raw.end());
  CHECK(config.raw.find("text_config") == config.raw.end());
  CHECK(config.rope_parameters.mrope_section.empty());
}

// ===========================================================================
// 3. Weight-namespace resolution. Upstream normalizes with ONE WeightsMapper
//    (qwen3_5.py:296-300 @ `ad5d29db7`); we mirror that with ONE resolution per
//    checkpoint rather than a per-lookup fallback, because a per-lookup fallback
//    would let a checkpoint bind half its tensors from each namespace and still
//    appear to load.
// ===========================================================================
TEST_CASE("qwen3_8: the backbone weight namespace is resolved ONCE per checkpoint") {
  SUBCASE("a clean `model.` index resolves to the canonical namespace") {
    CHECK(vllm::ResolveQwen3_5BackbonePrefix(
              NamesOf(DenseOneLayerSpecs("model."))) == "model.");
  }

  SUBCASE("a VL-prefixed index resolves to `model.language_model.`") {
    CHECK(vllm::ResolveQwen3_5BackbonePrefix(
              NamesOf(DenseOneLayerSpecs("model.language_model."))) ==
          "model.language_model.");
  }

  SUBCASE("a VISION-INCLUSIVE VL checkpoint is still the VL namespace") {
    // The 27B/35B vision-inclusive checkpoints carry `model.visual.*` NEXT TO
    // `model.language_model.*`. `model.visual.` is not a backbone spelling, so
    // it must never be mistaken for the canonical `model.` namespace and turn a
    // checkpoint we gate today into a refusal.
    std::vector<std::string> names = NamesOf(DenseOneLayerSpecs("model.language_model."));
    names.push_back("model.visual.patch_embed.proj.weight");
    names.push_back("model.visual.blocks.0.attn.qkv.weight");
    names.push_back("lm_head.weight");
    CHECK(vllm::ResolveQwen3_5BackbonePrefix(names) == "model.language_model.");
  }

  SUBCASE("the optional `mtp.*` draft head does not decide the namespace") {
    std::vector<std::string> names = NamesOf(DenseOneLayerSpecs("model."));
    names.push_back("mtp.fc.weight");
    names.push_back("mtp.layers.0.input_layernorm.weight");
    CHECK(vllm::ResolveQwen3_5BackbonePrefix(names) == "model.");
  }

  SUBCASE("a MIXED index is REFUSED, not silently half-loaded") {
    std::vector<std::string> names = NamesOf(DenseOneLayerSpecs("model."));
    names.push_back("model.language_model.layers.1.input_layernorm.weight");
    CHECK_THROWS_AS(vllm::ResolveQwen3_5BackbonePrefix(names),
                    std::runtime_error);
  }

  SUBCASE("an index with NEITHER namespace is refused") {
    const std::vector<std::string> names{"lm_head.weight", "mtp.fc.weight"};
    CHECK_THROWS_AS(vllm::ResolveQwen3_5BackbonePrefix(names),
                    std::runtime_error);
  }
}

// ===========================================================================
// 4. The loader must USE the resolved prefix. Two synthetic checkpoints with
//    byte-identical payloads and only the namespace differing must produce
//    byte-identical weights — which no amount of name-mapping unit testing can
//    show on its own.
// ===========================================================================
TEST_CASE("qwen3_8: the dense loader reads the SAME weights through either namespace") {
  const std::vector<Spec> vl = DenseOneLayerSpecs("model.language_model.");
  const std::vector<Spec> flat = DenseOneLayerSpecs("model.");
  REQUIRE(vl.size() == flat.size());

  const std::string vl_bytes = BuildSafetensors(vl);
  const std::string flat_bytes = BuildSafetensors(flat);
  // Same specs in the same order => IDENTICAL payloads, only the names differ.
  // Assert that here, so a later byte-equality of the two loads cannot be
  // satisfied by two identically-WRONG reads of two different payloads.
  REQUIRE(Payload(vl_bytes) == Payload(flat_bytes));
  REQUIRE(vl_bytes != flat_bytes);
  const TempFile vl_file(vl_bytes, "vl");
  const TempFile flat_file(flat_bytes, "flat");

  std::vector<vllm::SafetensorsFile> vl_shards;
  vl_shards.push_back(vllm::SafetensorsFile::Open(vl_file.path()));
  std::vector<vllm::SafetensorsFile> flat_shards;
  flat_shards.push_back(vllm::SafetensorsFile::Open(flat_file.path()));

  const HfConfig config = OneLayerDenseConfig();
  const vllm::Qwen3_5DenseWeights from_vl =
      vllm::LoadQwen3_5Dense(vl_shards, config);
  const vllm::Qwen3_5DenseWeights from_flat =
      vllm::LoadQwen3_5Dense(flat_shards, config);

  REQUIRE(from_vl.layers.size() == 1);
  REQUIRE(from_flat.layers.size() == 1);
  // No lm_head in either index => both tie the head to the embedding table.
  CHECK(from_vl.tied_lm_head);
  CHECK(from_flat.tied_lm_head);

  CheckSameBytes(from_vl.embed_tokens, from_flat.embed_tokens, "embed_tokens");
  CheckSameBytes(from_vl.final_norm, from_flat.final_norm, "final_norm");
  const vllm::Qwen3_5DenseLayerWeights& a = from_vl.layers[0];
  const vllm::Qwen3_5DenseLayerWeights& b = from_flat.layers[0];
  CHECK_FALSE(a.is_linear_attention);
  CHECK_FALSE(b.is_linear_attention);
  CheckSameBytes(a.input_layernorm, b.input_layernorm, "input_layernorm");
  CheckSameBytes(a.post_attention_layernorm, b.post_attention_layernorm,
                 "post_attention_layernorm");
  CheckSameBytes(a.attn.q_proj, b.attn.q_proj, "q_proj");
  CheckSameBytes(a.attn.k_proj, b.attn.k_proj, "k_proj");
  CheckSameBytes(a.attn.v_proj, b.attn.v_proj, "v_proj");
  CheckSameBytes(a.attn.o_proj, b.attn.o_proj, "o_proj");
  CheckSameBytes(a.attn.q_norm, b.attn.q_norm, "q_norm");
  CheckSameBytes(a.attn.k_norm, b.attn.k_norm, "k_norm");
  CheckSameBytes(a.mlp.gate_up_proj, b.mlp.gate_up_proj, "gate_up_proj");
  CheckSameBytes(a.mlp.down_proj, b.mlp.down_proj, "down_proj");
}

// ===========================================================================
// 5. INERTNESS of the gated rows. 27B / 35B / Coder are VL-prefixed
//    checkpoints; the per-layer public seams keep the VL prefix as their
//    DEFAULT, so every existing caller is unchanged by construction.
// ===========================================================================
TEST_CASE("qwen3_8: the VL prefix stays the default for the gated 27B/35B checkpoints") {
  // The two spellings are named constants, not literals scattered per lookup.
  CHECK(std::string(vllm::kQwen3_5VlBackbonePrefix) == "model.language_model.");
  CHECK(std::string(vllm::kQwen3_5TextBackbonePrefix) == "model.");

  // A 35B-shaped VL index (GDN + full-attn layers, 3D-stacked experts, shared
  // expert gate, top-level head) still resolves to the VL namespace.
  std::vector<std::string> names{
      "model.language_model.embed_tokens.weight",
      "model.language_model.norm.weight",
      "model.language_model.layers.0.input_layernorm.weight",
      "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
      "model.language_model.layers.0.mlp.experts.gate_up_proj",
      "model.language_model.layers.0.mlp.shared_expert_gate.weight",
      "model.language_model.layers.3.self_attn.q_proj.weight",
      "lm_head.weight",
  };
  CHECK(vllm::ResolveQwen3_5BackbonePrefix(names) == "model.language_model.");
}
