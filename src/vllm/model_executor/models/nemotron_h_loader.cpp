// Nemotron-H weight loader. See nemotron_h_loader.h for the scheme table, why
// the quantized forms are KEPT rather than dequantized at load, and what is
// deferred by name.
#include "vllm/model_executor/models/nemotron_h_loader.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& detail) {
  throw std::runtime_error("NemotronHForCausalLM weight load: " + detail);
}

// One tensor's on-disk view, keyed by the name the checkpoint ships.
using TensorIndex = std::map<std::string, const StTensor*>;

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

std::string ShapeStr(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

// The loader's accumulating state: the index it reads from, the names it has
// consumed (so the accounting is BOTH ways), and the report it fills.
struct Loader {
  const TensorIndex& index;
  NemotronHLoadReport& report;
  std::set<std::string> consumed;

  const StTensor& Need(const std::string& name) {
    const auto it = index.find(name);
    if (it == index.end()) {
      Refuse("the checkpoint does not ship '" + name +
             "', which this architecture's enumeration names");
    }
    if (!consumed.insert(name).second) {
      Refuse("'" + name + "' was claimed twice");
    }
    report.source_bytes += static_cast<int64_t>(it->second->nbytes);
    return *it->second;
  }

  void Expect(const StTensor& t, const std::string& name, const char* dtype,
              const std::vector<int64_t>& shape) {
    if (t.dtype != dtype) {
      Refuse("'" + name + "' ships dtype " + t.dtype + ", not the " + dtype +
             " its scheme declares");
    }
    if (t.shape != shape) {
      Refuse("'" + name + "' ships shape " + ShapeStr(t.shape) + ", not " +
             ShapeStr(shape));
    }
  }
};

// Release the source pages of a range the loader has finished with. The
// mappings stay open (the caller owns them), so without this the whole 20.1 GiB
// checkpoint stays resident alongside the owned mirror.
void Consumed(const StTensor& t) { MaybeReleaseSourcePages(t.data, t.nbytes); }

// ─── dense copies ───────────────────────────────────────────────────────────

// Copy a plain (unquantized) tensor into `want`. `disk_shape` is what the
// checkpoint ships; `logical_shape` is what the forward indexes it as — they
// differ only for `conv1d.weight`, which ships [Cd, 1, K] and is consumed as the
// squeezed [Cd, K] (mamba_mixer2.py's `self.conv1d.weight.view(conv_dim, K)`).
NemotronHOwned CopyDense(Loader& ld, const std::string& name, vt::DType want,
                         const std::vector<int64_t>& disk_shape,
                         std::vector<int64_t> logical_shape) {
  const StTensor& t = ld.Need(name);
  if (t.shape != disk_shape) {
    Refuse("'" + name + "' ships shape " + ShapeStr(t.shape) + ", not " +
           ShapeStr(disk_shape));
  }
  const int64_t n = Numel(disk_shape);
  NemotronHOwned w;
  w.dtype = want;
  w.shape = std::move(logical_shape);
  w.bytes.assign(static_cast<size_t>(n) * vt::SizeOf(want), 0);
  if (t.dtype == "BF16") {
    ld.report.bf16_tensors += 1;
    if (t.nbytes != static_cast<size_t>(n) * 2) {
      Refuse("'" + name + "' is BF16 but its byte count does not match its shape");
    }
    const auto* src = reinterpret_cast<const uint16_t*>(t.data);
    if (want == vt::DType::kBF16) {
      std::memcpy(w.bytes.data(), src, static_cast<size_t>(n) * 2);
    } else if (want == vt::DType::kF32) {
      // A WIDENING. Counted, because the only ones the released checkpoint asks
      // for are the three f32-by-contract SSM scalars; a widening anywhere else
      // is a defect that a token gate would absorb.
      ld.report.widened_tensors += 1;
      auto* dst = reinterpret_cast<float*>(w.bytes.data());
      for (int64_t i = 0; i < n; ++i) dst[i] = vt::BF16ToF32(src[i]);
    } else {
      Refuse("'" + name + "' cannot be materialized at the requested dtype");
    }
  } else if (t.dtype == "F32") {
    ld.report.f32_tensors += 1;
    if (t.nbytes != static_cast<size_t>(n) * 4) {
      Refuse("'" + name + "' is F32 but its byte count does not match its shape");
    }
    const auto* src = reinterpret_cast<const float*>(t.data);
    if (want == vt::DType::kF32) {
      std::memcpy(w.bytes.data(), src, static_cast<size_t>(n) * 4);
    } else if (want == vt::DType::kBF16) {
      // A NARROWING, which is what a bf16 model dtype asks for on a tensor the
      // producer happened to store wide. Never silent: it is the model dtype
      // every other layer inherits, and the router (the one f32 consumer) asks
      // for f32 explicitly.
      auto* dst = reinterpret_cast<uint16_t*>(w.bytes.data());
      for (int64_t i = 0; i < n; ++i) dst[i] = vt::F32ToBF16(src[i]);
    } else {
      Refuse("'" + name + "' cannot be materialized at the requested dtype");
    }
  } else {
    Refuse("'" + name + "' ships dtype " + t.dtype +
           ", which is not an unquantized dtype this loader reads");
  }
  Consumed(t);
  return w;
}

NemotronHOwned CopyDense(Loader& ld, const std::string& name, vt::DType want,
                         const std::vector<int64_t>& shape) {
  return CopyDense(ld, name, want, shape, shape);
}

// A per-tensor f32 scalar companion. ModelOpt writes `weight_scale_2` and
// `weight_scale` as rank-0 and `input_scale` / `k_scale` / `v_scale` as [1];
// both spellings are one number and both are accepted, nothing else is.
float ReadF32Scalar(Loader& ld, const std::string& name) {
  const StTensor& t = ld.Need(name);
  if (t.dtype != "F32") {
    Refuse("'" + name + "' ships dtype " + t.dtype + ", not the F32 a scale is");
  }
  if (!(t.shape.empty() || t.shape == std::vector<int64_t>{1})) {
    Refuse("'" + name + "' ships shape " + ShapeStr(t.shape) +
           ", not the scalar a per-tensor scale is");
  }
  if (t.nbytes != sizeof(float)) {
    Refuse("'" + name + "' is a scalar F32 but does not carry 4 bytes");
  }
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  Consumed(t);
  return v;
}

// ─── the two quantized schemes ──────────────────────────────────────────────

// `W4A16_NVFP4`, `group_size=16`. `rows`/`cols` are the LOGICAL [out, in].
NemotronHOwned LoadNvfp4(Loader& ld, const std::string& prefix, vt::DType logical,
                         int64_t rows, int64_t cols) {
  if (cols % kNvfp4GroupSize != 0) {
    Refuse("'" + prefix + "' has in_features " + std::to_string(cols) +
           ", which is not a multiple of the NVFP4 group size 16");
  }
  NemotronHOwned w;
  w.form = NemotronHWeightForm::kNvfp4W4A16G16;
  w.dtype = logical;
  w.shape = {rows, cols};

  const std::string wname = prefix + ".weight";
  const StTensor& packed = ld.Need(wname);
  ld.Expect(packed, wname, "U8", {rows, cols / 2});
  w.bytes.assign(packed.data, packed.data + packed.nbytes);
  Consumed(packed);

  const std::string sname = prefix + ".weight_scale";
  const StTensor& gs = ld.Need(sname);
  // The GROUP scale, not a per-tensor one. Binding a per-tensor scale here (or
  // this one to the wrong projection) produces a finite, correctly-shaped,
  // wrongly-scaled matrix — the x1.10-class error a token gate absorbs.
  ld.Expect(gs, sname, "F8_E4M3", {rows, cols / kNvfp4GroupSize});
  w.scale.assign(gs.data, gs.data + gs.nbytes);
  Consumed(gs);

  w.global_scale = ReadF32Scalar(ld, prefix + ".weight_scale_2");

  ld.report.nvfp4_weights += 1;
  ld.report.nvfp4_tensors += 3;
  return w;
}

// FP8 W8A8 static. Weight-only on this path: `input_scale` is carried, not
// applied (nemotron_h_forward.h records why).
NemotronHOwned LoadFp8(Loader& ld, const std::string& prefix, vt::DType logical,
                       int64_t rows, int64_t cols) {
  NemotronHOwned w;
  w.form = NemotronHWeightForm::kFp8W8A8Static;
  w.dtype = logical;
  w.shape = {rows, cols};

  const std::string wname = prefix + ".weight";
  const StTensor& q = ld.Need(wname);
  ld.Expect(q, wname, "F8_E4M3", {rows, cols});
  w.bytes.assign(q.data, q.data + q.nbytes);
  Consumed(q);

  w.global_scale = ReadF32Scalar(ld, prefix + ".weight_scale");
  w.input_scale = ReadF32Scalar(ld, prefix + ".input_scale");
  w.has_input_scale = true;

  ld.report.fp8_weights += 1;
  ld.report.fp8_tensors += 3;
  return w;
}

// ─── the blocks ─────────────────────────────────────────────────────────────

void LoadMamba(Loader& ld, const NemotronHParams& p, const std::string& mixer,
               vt::DType adt, NemotronHMambaWeights& out) {
  const int64_t H = p.hidden_size;
  const int64_t I = p.mamba_intermediate_size();
  const int64_t Cd = p.conv_dim();
  const int64_t K = p.conv_kernel;
  const int64_t Hh = p.mamba_num_heads;

  out.in_proj = LoadFp8(ld, mixer + ".in_proj", adt, p.in_proj_out_features(), H);
  out.out_proj = LoadFp8(ld, mixer + ".out_proj", adt, H, I);
  // The conv weight ships [Cd, 1, K] and is consumed squeezed.
  out.conv1d_weight =
      CopyDense(ld, mixer + ".conv1d.weight", adt, {Cd, 1, K}, {Cd, K});
  if (p.use_conv_bias) {
    out.conv1d_bias = CopyDense(ld, mixer + ".conv1d.bias", adt, {Cd});
  }
  // f32 BY CONTRACT, and bf16 on disk. Upstream keeps these f32 whatever the
  // model dtype (`self.A = -torch.exp(self.A_log.float())`, and D/dt_bias feed
  // the same f32 scan); `vt::Mamba2ChunkScan` validates all three as f32. This
  // is the annotated f32 escape AGENTS.md allows, it is upstream's own polarity,
  // and `report.widened_tensors` counts it so it stays exactly these three.
  out.A_log = CopyDense(ld, mixer + ".A_log", vt::DType::kF32, {Hh});
  out.D = CopyDense(ld, mixer + ".D", vt::DType::kF32, {Hh});
  out.dt_bias = CopyDense(ld, mixer + ".dt_bias", vt::DType::kF32, {Hh});
  // Mixer2RMSNormGated over the SSM intermediate width, NOT hidden_size.
  out.norm_weight = CopyDense(ld, mixer + ".norm.weight", adt, {I});
}

void LoadAttention(Loader& ld, const NemotronHParams& p, const std::string& mixer,
                   vt::DType adt, bool fp8_kv, NemotronHAttentionWeights& out) {
  const int64_t H = p.hidden_size;
  const int64_t qd = p.q_proj_out_features();
  const int64_t kvd = p.kv_proj_out_features();
  out.q_proj = CopyDense(ld, mixer + ".q_proj.weight", adt, {qd, H});
  out.k_proj = CopyDense(ld, mixer + ".k_proj.weight", adt, {kvd, H});
  out.v_proj = CopyDense(ld, mixer + ".v_proj.weight", adt, {kvd, H});
  out.o_proj = CopyDense(ld, mixer + ".o_proj.weight", adt, {H, qd});
  if (fp8_kv) {
    out.k_scale = ReadF32Scalar(ld, mixer + ".k_proj.k_scale");
    out.v_scale = ReadF32Scalar(ld, mixer + ".v_proj.v_scale");
    out.has_kv_scales = true;
    ld.report.fp8_kv_scale_tensors += 2;
  }
}

void LoadExpert(Loader& ld, const std::string& prefix, vt::DType adt, int64_t H,
                int64_t I, bool quantized, NemotronHExpertWeights& out) {
  // `ckpt_names=("up_proj","down_proj","")` (nemotron_h.py:220): there is no
  // gate_proj anywhere in this checkpoint.
  if (quantized) {
    out.up_proj = LoadNvfp4(ld, prefix + ".up_proj", adt, I, H);
    out.down_proj = LoadNvfp4(ld, prefix + ".down_proj", adt, H, I);
  } else {
    out.up_proj = CopyDense(ld, prefix + ".up_proj.weight", adt, {I, H});
    out.down_proj = CopyDense(ld, prefix + ".down_proj.weight", adt, {H, I});
  }
}

void LoadMoe(Loader& ld, const NemotronHParams& p, const std::string& mixer,
             vt::DType adt, bool quantized, NemotronHMoeWeights& out) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.n_routed_experts;
  // The ROUTER IS F32 AND THAT IS MIRRORED, NOT INHERITED:
  // `GateLinear(..., out_dtype=torch.float32, force_fp32_compute=True)`
  // (nemotron_h.py:150-156). The released backbone even ships it F32 on disk;
  // the MTP tower's twin ships BF16, which is exactly why the dtype is REQUESTED
  // here rather than taken from whatever the producer wrote.
  out.gate = CopyDense(ld, mixer + ".gate.weight", vt::DType::kF32, {E, H});
  out.e_score_correction_bias = CopyDense(
      ld, mixer + ".gate.e_score_correction_bias", vt::DType::kF32, {E});
  out.experts.resize(static_cast<size_t>(E));
  for (int64_t e = 0; e < E; ++e) {
    LoadExpert(ld, mixer + ".experts." + std::to_string(e), adt, H,
               p.moe_intermediate_size, quantized,
               out.experts[static_cast<size_t>(e)]);
  }
  if (p.n_shared_experts > 0) {
    LoadExpert(ld, mixer + ".shared_experts", adt, H,
               p.moe_shared_expert_intermediate_size * p.n_shared_experts,
               quantized, out.shared);
    out.has_shared = true;
  }
}

void LoadMlp(Loader& ld, const NemotronHParams& p, const std::string& mixer,
             vt::DType adt, bool quantized, NemotronHMlpWeights& out) {
  NemotronHExpertWeights e;
  LoadExpert(ld, mixer, adt, p.hidden_size, p.intermediate_size, quantized, e);
  out.up_proj = std::move(e.up_proj);
  out.down_proj = std::move(e.down_proj);
}

int64_t HostBytesOf(const NemotronHHostWeights& h) {
  int64_t n = h.embeddings.HostBytes() + h.norm_f.HostBytes() +
              h.lm_head.HostBytes();
  for (const NemotronHLayerWeights& l : h.layers) {
    n += l.norm.HostBytes();
    n += l.mamba.in_proj.HostBytes() + l.mamba.out_proj.HostBytes() +
         l.mamba.conv1d_weight.HostBytes() + l.mamba.conv1d_bias.HostBytes() +
         l.mamba.A_log.HostBytes() + l.mamba.D.HostBytes() +
         l.mamba.dt_bias.HostBytes() + l.mamba.norm_weight.HostBytes();
    n += l.attn.q_proj.HostBytes() + l.attn.k_proj.HostBytes() +
         l.attn.v_proj.HostBytes() + l.attn.o_proj.HostBytes();
    n += l.moe.gate.HostBytes() + l.moe.e_score_correction_bias.HostBytes();
    for (const NemotronHExpertWeights& e : l.moe.experts) {
      n += e.up_proj.HostBytes() + e.down_proj.HostBytes();
    }
    n += l.moe.shared.up_proj.HostBytes() + l.moe.shared.down_proj.HostBytes();
    n += l.mlp.up_proj.HostBytes() + l.mlp.down_proj.HostBytes();
  }
  return n;
}

}  // namespace

vt::DType ResolveNemotronHModelDType(const HfConfig& config) {
  // transformers serializes the model dtype as `dtype` (the released
  // NemotronH config.json ships `"dtype": "bfloat16"`); `torch_dtype` is the
  // legacy spelling and is the fallback, not the other way round. HfConfig
  // parses only the legacy key, so the modern one is read from the raw
  // document here.
  std::string name;
  if (config.raw.contains("dtype") && config.raw.at("dtype").is_string()) {
    name = config.raw.at("dtype").get<std::string>();
  }
  if (name.empty()) name = config.torch_dtype;
  if (name.empty() || name == "auto") name = "bfloat16";
  if (name == "bfloat16") return vt::DType::kBF16;
  if (name == "float32" || name == "float") return vt::DType::kF32;
  // f16 is deliberately refused rather than substituted: `vt::MoeRelu2` has no
  // f16 output arm (spec §6a), so a silent widen-to-bf16 would run a different
  // model than the checkpoint declares.
  throw std::runtime_error(
      "NemotronHForCausalLM weight load: model dtype '" + name +
      "' is not supported (this forward composes ops whose output dtypes are "
      "bf16 — the released checkpoint's — and f32)");
}

NemotronHHostWeights LoadNemotronHHostWeights(
    const std::vector<SafetensorsFile>& shards, const NemotronHParams& params,
    vt::DType act_dtype, NemotronHLoadReport* report) {
  NemotronHLoadReport local;
  NemotronHLoadReport& rep = report != nullptr ? *report : local;
  rep = NemotronHLoadReport{};

  TensorIndex index;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      if (!index.emplace(name, &shard.Get(name)).second) {
        Refuse("'" + name + "' appears in more than one shard");
      }
    }
  }
  rep.in_index = static_cast<int64_t>(index.size());

  const std::vector<NemotronHTensor> enumerated =
      EnumerateNemotronHTensors(params);
  rep.enumerated = static_cast<int64_t>(enumerated.size());

  NemotronHHostWeights host;
  host.act_dtype = act_dtype;
  const NemotronHParams& p = params;
  const bool quantized = p.quant.present;

  Loader ld{index, rep, {}};

  // --- root ---
  host.embeddings = CopyDense(ld, "backbone.embeddings.weight", act_dtype,
                              {p.vocab_size, p.hidden_size});
  host.norm_f = CopyDense(ld, "backbone.norm_f.weight", act_dtype, {p.hidden_size});
  if (!p.tie_word_embeddings) {
    host.lm_head = quantized
                       ? LoadNvfp4(ld, "lm_head", act_dtype, p.vocab_size,
                                   p.hidden_size)
                       : CopyDense(ld, "lm_head.weight", act_dtype,
                                   {p.vocab_size, p.hidden_size});
  }

  // --- the 52 backbone layers ---
  const int64_t L = p.num_hidden_layers();
  host.layers.resize(static_cast<size_t>(L));
  for (int64_t i = 0; i < L; ++i) {
    NemotronHLayerWeights& lw = host.layers[static_cast<size_t>(i)];
    const std::string layer = "backbone.layers." + std::to_string(i);
    const std::string mixer = layer + ".mixer";
    lw.block = p.layers_block_type[static_cast<size_t>(i)];
    lw.norm = CopyDense(ld, layer + ".norm.weight", act_dtype, {p.hidden_size});
    switch (lw.block) {
      case NemotronHBlock::kMamba:
        LoadMamba(ld, p, mixer, act_dtype, lw.mamba);
        break;
      case NemotronHBlock::kAttention:
        LoadAttention(ld, p, mixer, act_dtype,
                      quantized && p.quant.fp8_kv_cache, lw.attn);
        break;
      case NemotronHBlock::kMoe:
        LoadMoe(ld, p, mixer, act_dtype, quantized, lw.moe);
        break;
      case NemotronHBlock::kMlp:
        LoadMlp(ld, p, mixer, act_dtype, quantized, lw.mlp);
        break;
    }
  }

  rep.materialized = static_cast<int64_t>(ld.consumed.size());

  // --- the MTP tower: DEFERRED BY NAME (W5) ---------------------------------
  //
  // 270 unquantized bf16 tensors (`ignore` carries `mtp*`). W5 owns the head;
  // nothing here can consume it, and materializing it would cost 2.65 GB of
  // host memory nothing reads. It is counted and NAMED rather than skipped —
  // the whole point of the enumeration is that no tensor is in the "nobody
  // thought of it" state.
  std::set<std::string> deferred_tags;
  for (const NemotronHTensor& t : enumerated) {
    if (ld.consumed.count(t.name) != 0) continue;
    if (t.name.rfind("mtp.", 0) != 0) {
      Refuse("'" + t.name + "' (consumer '" + t.consumer +
             "') is enumerated but no host slot claimed it; every enumerated "
             "tensor must be materialized or deferred by name");
    }
    rep.deferred += 1;
    deferred_tags.insert(t.consumer);
  }
  for (const std::string& tag : deferred_tags) {
    rep.deferred_by_name.push_back(
        tag + " (MTP head, W5 of .agents/specs/nemotron-h-model.md)");
  }

  // The other direction: a tensor the checkpoint ships that nothing named.
  if (rep.materialized + rep.deferred != rep.enumerated) {
    Refuse("accounting mismatch: " + std::to_string(rep.materialized) +
           " materialized + " + std::to_string(rep.deferred) + " deferred != " +
           std::to_string(rep.enumerated) + " enumerated");
  }
  if (rep.enumerated != rep.in_index) {
    Refuse("the enumeration names " + std::to_string(rep.enumerated) +
           " tensors but the checkpoint ships " + std::to_string(rep.in_index));
  }

  rep.host_bytes = HostBytesOf(host);
  host.materialized = true;
  return host;
}

}  // namespace vllm
