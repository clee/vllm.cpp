// LTX-2.5 phase L6 — the quantized loaders. See
// include/vllm/model_executor/models/ltx2_loader.h for the port map, the one
// delta, the dtype polarity and the four unported families.
//
// Ported from / grounded in:
//   the swizzle inverted here      <- vllm/model_executor/layers/quantization/
//                                    qutlass_utils.py:165-180 (`to_blocked`) and
//                                    utils/nvfp4_utils.py:44-49
//                                    (`swizzle_blockscale`) — the same
//                                    permutation written twice. qutlass_utils.py
//                                    records its own provenance as a copy of
//                                    torchao/prototype/mx_formats, which is the
//                                    module that quantized this checkpoint.
//   the fp4 decode + group scale   <- REUSED VERBATIM: DequantNvfp4ToBf16
//                                    (nvfp4_dequant.h:59). No new quant scheme.
//   the per-tensor fp8 decode      <- REUSED VERBATIM: DequantFp8ToBf16 (:76).
//   the tensor-at-a-time staging   <- the shape MiniMaxH3 arrived at for the
//                                    same reason (minimax_h3.h:1598-1618,
//                                    minimax_h3_device.cpp:1259-1360).
//   the asset pack                 <- REUSED: Ltx2LoadGemmaAssets
//                                    (ltx2_text_encoder.h:372), phase L3.
#include "vllm/model_executor/models/ltx2_loader.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("ltx2 loader: " + message);
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

float ReadScalarF32(const std::string& name, const StTensor& t) {
  if (t.dtype != "F32") {
    Fail("'" + name + "' must be F32 (it is the per-tensor scale), not " + t.dtype);
  }
  if (t.nbytes < sizeof(float)) Fail("'" + name + "' is too small to hold an f32 scale");
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(v));
  return v;
}

float Bf16ToF32(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

int64_t RoundUp(int64_t v, int64_t m) { return ((v + m - 1) / m) * m; }

// A contiguous non-owning view of arbitrary rank. `vt::Tensor::Contiguous` takes
// an initializer_list, which a runtime-length shape cannot supply.
vt::Tensor MakeView(void* data, vt::DType dtype, vt::Device device,
                    const std::vector<int64_t>& shape) {
  vt::Tensor t;
  t.data = data;
  t.dtype = dtype;
  t.device = device;
  t.rank = static_cast<int>(shape.size());
  int64_t acc = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = acc;
    acc *= t.shape[i];
  }
  return t;
}

// The module prefix a tensor belongs to, for reporting an unported FAMILY once
// rather than its 129 tensors individually.
std::string FamilyOf(const std::string& name) {
  const size_t dot = name.find('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

}  // namespace

// ---------------------------------------------------------------------------
// torchao NVFP4: the marker, and the one delta
// ---------------------------------------------------------------------------

Ltx2TorchaoNvfp4Marker ParseLtx2TorchaoNvfp4Marker(const std::string& module,
                                                   const StTensor& marker) {
  if (marker.data == nullptr || marker.nbytes == 0) {
    Fail("'" + module + "': the torchao_nvfp4 marker is empty");
  }
  const std::string text(reinterpret_cast<const char*>(marker.data), marker.nbytes);
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(text);
  } catch (const std::exception& e) {
    Fail("'" + module + "': the torchao_nvfp4 marker is not JSON (" + e.what() +
         "). Its payload was: " + text.substr(0, 120));
  }
  if (!parsed.is_object()) Fail("'" + module + "': the torchao_nvfp4 marker is not an object");

  Ltx2TorchaoNvfp4Marker m;
  m.format = parsed.value("format", std::string());
  m.block_size = parsed.value("block_size", static_cast<int64_t>(0));
  m.scope = parsed.value("scope", std::string());
  m.config = parsed.value("config", std::string());
  m.is_swizzled_scales = parsed.value("is_swizzled_scales", false);
  m.use_triton_kernel = parsed.value("use_triton_kernel", false);
  m.use_dynamic_activation = parsed.value("use_dynamic_activation", false);
  m.use_dynamic_per_tensor_scale = parsed.value("use_dynamic_per_tensor_scale", false);

  if (m.format != "torchao_nvfp4") {
    Fail("'" + module + "': quantization format is '" + m.format +
         "', not 'torchao_nvfp4'. This port implements torchao NVFP4 only; the "
         "compressed-tensors layout stores its global scales as DIVISORS "
         "(nvfp4_emulation.h:18-23) and reading one as a multiplier is finite and wrong.");
  }
  if (m.block_size != kNvfp4GroupSize) {
    Fail("'" + module + "': torchao block_size is " + std::to_string(m.block_size) +
         ", and this port implements " + std::to_string(kNvfp4GroupSize) +
         " only (nvfp4_dequant.h:32). A different group size regroups every scale.");
  }
  if (!m.is_swizzled_scales) {
    Fail("'" + module +
         "': torchao marker says is_swizzled_scales=false, so its group scales are "
         "LINEAR. Applying the unswizzle to them would permute every scale within a "
         "128x4 tile — finite, correctly shaped and wrong. Read it linearly instead.");
  }
  return m;
}

void Ltx2UnswizzleNvfp4BlockScale(const uint8_t* swizzled, size_t swizzled_bytes,
                                  int64_t rows, int64_t cols, uint8_t* linear) {
  if (swizzled == nullptr || linear == nullptr) Fail("unswizzle: null buffer");
  if (rows <= 0 || cols <= 0) Fail("unswizzle: non-positive block-scale dims");
  const int64_t padded_rows = RoundUp(rows, 128);
  const int64_t padded_cols = RoundUp(cols, 4);
  const size_t want = static_cast<size_t>(padded_rows) * static_cast<size_t>(padded_cols);
  if (swizzled_bytes != want) {
    Fail("unswizzle: the block-scale buffer is " + std::to_string(swizzled_bytes) +
         " bytes but the layout for " + ShapeText({rows, cols}) + " stores " +
         std::to_string(want) + " (padded to " + ShapeText({padded_rows, padded_cols}) +
         "). A short buffer here means the stored shape was read as if it were linear.");
  }
  const int64_t col_tiles = padded_cols / 4;
  for (int64_t r = 0; r < rows; ++r) {
    const int64_t row_tile = r / 128;
    const int64_t within = r % 128;
    const int64_t quarter = within / 32;  // `a`: the 4-way split of the 128-row tile
    const int64_t lane = within % 32;     // `s`
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t col_tile = c / 4;
      const int64_t q = c % 4;
      const int64_t src =
          ((((row_tile * col_tiles) + col_tile) * 32 + lane) * 4 + quarter) * 4 + q;
      linear[static_cast<size_t>(r * cols + c)] = swizzled[static_cast<size_t>(src)];
    }
  }
}

void Ltx2DequantTorchaoNvfp4ToBf16(const std::string& module, const StTensor& packed,
                                   const StTensor& scale, const StTensor& scale_2,
                                   int64_t out_features, int64_t in_features,
                                   uint16_t* out_bf16) {
  if (packed.dtype != "U8") {
    Fail("'" + module + ".weight' must be U8 (two E2M1 values per byte), not " +
         packed.dtype);
  }
  if (packed.shape.size() != 2 || packed.shape[0] != out_features ||
      packed.shape[1] * 2 != in_features) {
    Fail("'" + module + ".weight' is " + ShapeText(packed.shape) + " but the module is [" +
         std::to_string(out_features) + ", " + std::to_string(in_features) +
         "]. NVFP4 packs TWO values per byte along the last dimension, so the stored "
         "width must be exactly half the logical one.");
  }
  if (in_features % kNvfp4GroupSize != 0) {
    Fail("'" + module + "': in_features " + std::to_string(in_features) +
         " is not a multiple of the group size " + std::to_string(kNvfp4GroupSize));
  }
  if (scale.dtype != "F8_E4M3") {
    Fail("'" + module + ".weight_scale' must be F8_E4M3, not " + scale.dtype);
  }
  const int64_t groups = in_features / kNvfp4GroupSize;
  const int64_t padded_rows = RoundUp(out_features, 128);
  const int64_t padded_cols = RoundUp(groups, 4);
  const std::vector<int64_t> want_shape = {padded_rows / 4, padded_cols * 4};
  if (scale.shape != want_shape) {
    Fail("'" + module + ".weight_scale' is " + ShapeText(scale.shape) +
         " but a SWIZZLED torchao scale for [" + std::to_string(out_features) + ", " +
         std::to_string(in_features) + "] is stored as " + ShapeText(want_shape) +
         ". The LINEAR shape " + ShapeText({out_features, groups}) +
         " has the same element count, so reading one as the other type-checks and "
         "permutes every scale within a 128x4 tile.");
  }
  const float global = ReadScalarF32(module + ".weight_scale_2", scale_2);

  std::vector<uint8_t> linear(static_cast<size_t>(out_features) *
                              static_cast<size_t>(groups));
  Ltx2UnswizzleNvfp4BlockScale(static_cast<const uint8_t*>(scale.data), scale.nbytes,
                               out_features, groups, linear.data());
  // From here the path is the EXISTING modelopt one, byte for byte.
  DequantNvfp4ToBf16(static_cast<const uint8_t*>(packed.data), linear.data(), global,
                     out_features, in_features, out_bf16);
}

// ---------------------------------------------------------------------------
// The DiT
// ---------------------------------------------------------------------------

namespace {

// Everything a DiT checkpoint's header tells us, before any payload is touched.
struct DitPlan {
  Ltx2DitQuant quant = Ltx2DitQuant::kFp8;
  std::string prefix;
  // contract name -> file name
  std::map<std::string, std::string> file_name;
  // contract name -> LOGICAL shape (U8 widths already doubled)
  std::map<std::string, std::vector<int64_t>> logical;
  std::vector<Ltx2TensorSpec> manifest;
  std::vector<std::string> unported;  // module families, header order, deduped
};

bool IsScaleSidecar(const std::string& bare) {
  return EndsWith(bare, ".weight_scale") || EndsWith(bare, ".weight_scale_2") ||
         EndsWith(bare, "_scale") || EndsWith(bare, kLtx2TorchaoNvfp4MarkerSuffix);
}

DitPlan PlanDit(const SafetensorsFile& file) {
  DitPlan plan;
  const std::string prefix = kLtx2DitCheckpointPrefix;
  const std::vector<std::string>& names = file.Names();
  if (names.empty()) Fail("the DiT checkpoint has no tensors");

  int64_t prefixed = 0;
  for (const std::string& n : names) {
    if (StartsWith(n, prefix)) ++prefixed;
  }
  if (prefixed != 0 && prefixed != static_cast<int64_t>(names.size())) {
    Fail("the DiT checkpoint mixes prefixed and unprefixed names (" +
         std::to_string(prefixed) + " of " + std::to_string(names.size()) +
         " carry '" + prefix +
         "'). Stripping one prefix from some names and not others would bind two "
         "different models' tensors into one contract.");
  }
  plan.prefix = prefixed != 0 ? prefix : std::string();

  bool saw_u8 = false, saw_f8 = false;
  for (const std::string& n : names) {
    const std::string bare = n.substr(plan.prefix.size());
    if (IsScaleSidecar(bare)) continue;
    const StTensor& t = file.Get(n);
    std::vector<int64_t> shape = t.shape;
    if (t.dtype == "U8") {
      saw_u8 = true;
      if (shape.size() != 2) {
        Fail("'" + bare + "' is U8 (NVFP4-packed) but rank " +
             std::to_string(shape.size()) + "; a packed weight is rank 2");
      }
      shape[1] *= 2;  // TWO values per byte, always
    } else if (t.dtype == "F8_E4M3") {
      saw_f8 = true;
    }
    plan.file_name[bare] = n;
    plan.logical[bare] = shape;
    plan.manifest.push_back({bare, shape});
  }
  if (saw_u8 && saw_f8) {
    Fail(
        "the DiT checkpoint carries BOTH U8-packed (NVFP4) and F8_E4M3 (FP8) weights. "
        "The two arms use different scale sidecars, so a mixed file would be loaded "
        "half one way and half the other.");
  }
  if (!saw_u8 && !saw_f8) {
    Fail(
        "the DiT checkpoint carries no quantized weights at all (no U8 and no "
        "F8_E4M3). A bf16 DiT is not what phase L6 loads; use the L2 path.");
  }
  plan.quant = saw_u8 ? Ltx2DitQuant::kNvfp4 : Ltx2DitQuant::kFp8;
  return plan;
}

// Materialize one contract tensor into `buffer`, returning its dtype.
vt::DType MaterializeDitTensor(const SafetensorsFile& file, const DitPlan& plan,
                               const Ltx2TensorSpec& spec, std::vector<uint8_t>& buffer) {
  const auto it = plan.file_name.find(spec.name);
  if (it == plan.file_name.end()) {
    Fail("the checkpoint is missing '" + spec.name + "' " + ShapeText(spec.shape) +
         ", which the LTX-2.5 DiT contract requires. Refusing rather than binding a "
         "zero-filled tensor.");
  }
  const std::string& fname = it->second;
  const std::vector<int64_t>& got = plan.logical.at(spec.name);
  if (got != spec.shape) {
    Fail("'" + spec.name + "' is " + ShapeText(got) + " in the checkpoint but the "
         "contract requires " + ShapeText(spec.shape));
  }
  const StTensor& t = file.Get(fname);
  int64_t numel = 1;
  for (int64_t d : spec.shape) numel *= d;

  if (t.dtype == "F32") {
    // Kept F32 because the CHECKPOINT stores it F32 (the scale_shift tables).
    // Narrowing a tensor the file itself widened would be the dtype rule
    // applied backwards.
    buffer.resize(static_cast<size_t>(numel) * sizeof(float));
    if (t.nbytes != buffer.size()) {
      Fail("'" + spec.name + "' declares " + std::to_string(t.nbytes) +
           " F32 bytes but its shape needs " + std::to_string(buffer.size()));
    }
    std::memcpy(buffer.data(), t.data, buffer.size());
    return vt::DType::kF32;
  }
  if (t.dtype == "BF16") {
    buffer.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
    if (t.nbytes != buffer.size()) {
      Fail("'" + spec.name + "' declares " + std::to_string(t.nbytes) +
           " BF16 bytes but its shape needs " + std::to_string(buffer.size()));
    }
    std::memcpy(buffer.data(), t.data, buffer.size());
    return vt::DType::kBF16;
  }
  if (t.dtype == "F8_E4M3") {
    const std::string sname = fname + "_scale";
    const float scale = ReadScalarF32(spec.name + "_scale", file.Get(sname));
    buffer.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
    DequantFp8ToBf16(static_cast<const uint8_t*>(t.data), scale, numel,
                     reinterpret_cast<uint16_t*>(buffer.data()));
    return vt::DType::kBF16;
  }
  if (t.dtype == "U8") {
    if (spec.shape.size() != 2) {
      Fail("'" + spec.name + "' is NVFP4-packed but rank " +
           std::to_string(spec.shape.size()));
    }
    const StTensor& scale = file.Get(fname + "_scale");
    const StTensor& global = file.Get(fname + "_scale_2");
    buffer.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
    Ltx2DequantTorchaoNvfp4ToBf16(spec.name, t, scale, global, spec.shape[0], spec.shape[1],
                                  reinterpret_cast<uint16_t*>(buffer.data()));
    return vt::DType::kBF16;
  }
  Fail("'" + spec.name + "' has dtype " + t.dtype + ", which this loader does not read");
}

// The families the file carries and the L2 contract does not.
std::vector<std::string> UnportedFamilies(const DitPlan& plan,
                                          const std::vector<Ltx2TensorSpec>& contract) {
  std::set<std::string> known;
  for (const Ltx2TensorSpec& spec : contract) known.insert(spec.name);
  std::vector<std::string> families;
  std::set<std::string> seen;
  for (const Ltx2TensorSpec& spec : plan.manifest) {
    if (known.count(spec.name) != 0) continue;
    const std::string family = FamilyOf(spec.name);
    if (seen.insert(family).second) families.push_back(family);
  }
  return families;
}

std::vector<Ltx2TensorSpec> ContractOf(const Ltx2DitParams& params) {
  return EnumerateLtx2DitTensors(params);
}

[[noreturn]] void RefuseUnported(const std::vector<std::string>& families) {
  std::string list;
  for (size_t i = 0; i < families.size(); ++i) {
    if (i != 0) list += ", ";
    list += families[i];
  }
  Fail(
      "the checkpoint carries modules phase L2 does NOT port: " + list +
      ". They are not dropped silently: prompt_adaln_single / "
      "audio_prompt_adaln_single mean use_prompt_adaln_single is TRUE, which "
      "contradicts .agents/specs/ltx-2-5.md section 1.2 and voids the prompt-K/V "
      "cache's premise; keyframes_abs_pos_embedding contradicts ltx2.h:47-49; the "
      "two *_embeddings_connector families are the Embeddings1DConnector "
      "ltx2_text_encoder.h:319-324 already records as owed. Pass "
      "Ltx2DitLoadOptions::allow_unported_modules to load the ported SUBSET, which "
      "still reports every one of them.");
}

}  // namespace

Ltx2DitParams Ltx2ParseDitParamsFromCheckpoint(const SafetensorsFile& file,
                                               Ltx2DitQuant* out_quant) {
  const DitPlan plan = PlanDit(file);
  if (out_quant != nullptr) *out_quant = plan.quant;
  return ParseLtx2DitParamsFromManifest(plan.manifest);
}

Ltx2DitCheckpoint Ltx2LoadDitFromSafetensors(const SafetensorsFile& file,
                                             const Ltx2DitLoadOptions& options) {
  const DitPlan plan = PlanDit(file);
  Ltx2DitCheckpoint out;
  out.quant = plan.quant;
  out.checkpoint_params = ParseLtx2DitParamsFromManifest(plan.manifest);
  out.params = out.checkpoint_params;
  // The one flag whose module this port does not carry. Cleared for the CONTRACT
  // only; `checkpoint_params` keeps what the file actually says.
  out.params.use_prompt_adaln_single = false;

  const std::vector<Ltx2TensorSpec> contract = ContractOf(out.params);
  out.unported = UnportedFamilies(plan, contract);
  if (!out.unported.empty() && !options.allow_unported_modules) {
    RefuseUnported(out.unported);
  }

  for (const Ltx2TensorSpec& spec : contract) {
    auto buffer = std::make_shared<Ltx2HostBuffer>();
    buffer->dtype = MaterializeDitTensor(file, plan, spec, buffer->bytes);
    out.views[spec.name] =
        MakeView(buffer->bytes.data(), buffer->dtype, vt::Device{}, spec.shape);
    out.storage.push_back(std::move(buffer));
  }

  if (options.widen_to_f32) Ltx2WidenDitToF32(out);
  out.weights = BindLtx2DitWeights(out.params, out.views);
  return out;
}

void Ltx2WidenDitToF32(Ltx2DitCheckpoint& checkpoint) {
  for (auto& kv : checkpoint.views) {
    vt::Tensor& view = kv.second;
    if (view.dtype != vt::DType::kBF16) continue;
    const int64_t numel = view.Numel();
    auto widened = std::make_shared<Ltx2HostBuffer>();
    widened->dtype = vt::DType::kF32;
    widened->bytes.resize(static_cast<size_t>(numel) * sizeof(float));
    const uint16_t* src = view.Ptr<uint16_t>();
    float* dst = reinterpret_cast<float*>(widened->bytes.data());
    for (int64_t i = 0; i < numel; ++i) dst[i] = Bf16ToF32(src[static_cast<size_t>(i)]);
    view.data = widened->bytes.data();
    view.dtype = vt::DType::kF32;
    checkpoint.storage.push_back(std::move(widened));
  }
  checkpoint.weights = BindLtx2DitWeights(checkpoint.params, checkpoint.views);
}

Ltx2DitCheckpoint Ltx2StreamDitToDevice(vt::Queue& queue, const SafetensorsFile& file,
                                        const Ltx2DitLoadOptions& options) {
  if (options.widen_to_f32) {
    Fail(
        "Ltx2StreamDitToDevice refuses widen_to_f32. Staging at load exists because "
        "GB10 runs host/ATS-retagged decode weights 20-30% slower, and widening while "
        "staging would move twice the bytes to save nothing. Widen a HOST load "
        "instead, for the f32 parity forward.");
  }
  const DitPlan plan = PlanDit(file);
  Ltx2DitCheckpoint out;
  out.quant = plan.quant;
  out.checkpoint_params = ParseLtx2DitParamsFromManifest(plan.manifest);
  out.params = out.checkpoint_params;
  out.params.use_prompt_adaln_single = false;

  const std::vector<Ltx2TensorSpec> contract = ContractOf(out.params);
  out.unported = UnportedFamilies(plan, contract);
  if (!out.unported.empty() && !options.allow_unported_modules) {
    RefuseUnported(out.unported);
  }

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  for (const Ltx2TensorSpec& spec : contract) {
    // ONE tensor's host buffer is live at a time: it is dequantized, uploaded,
    // and dropped before the next is read. That is what keeps peak residency at
    // the device copy plus one tensor rather than two whole models.
    std::vector<uint8_t> host;
    const vt::DType dtype = MaterializeDitTensor(file, plan, spec, host);
    void* device = backend.Alloc(host.size());
    backend.Copy(queue, device, host.data(), host.size());
    backend.Synchronize(queue);  // `host` dies at the end of this iteration
    out.views[spec.name] = MakeView(device, dtype, queue.device, spec.shape);
    // The device allocation's lifetime rides on the same storage vector the host
    // load uses, so a staged checkpoint frees exactly like a host one.
    out.device_storage.emplace_back(device, [&backend](void* p) { backend.Free(p); });
    load_stats::AddDeviceUpload(host.size());
  }
  out.weights = BindLtx2DitWeights(out.params, out.views);
  return out;
}

// ---------------------------------------------------------------------------
// The text encoder
// ---------------------------------------------------------------------------

namespace {

const StTensor* Find(const SafetensorsFile& file, const std::string& name) {
  const std::vector<std::string>& names = file.Names();
  if (std::find(names.begin(), names.end(), name) == names.end()) return nullptr;
  return &file.Get(name);
}

// Load one caption projection: the U8/NVFP4 weight AND the BF16 bias, which sit
// on different dtype paths — the split ltx2_text_encoder.h:264-269 names as the
// one a loader silently half-does.
Ltx2TextProjection LoadProjection(const SafetensorsFile& file, const std::string& module,
                                  int64_t in_features) {
  const StTensor* w = Find(file, module + ".weight");
  if (w == nullptr) Fail("the text encoder is missing '" + module + ".weight'");
  if (w->shape.size() != 2) {
    Fail("'" + module + ".weight' is rank " + std::to_string(w->shape.size()) +
         "; a caption projection is rank 2");
  }
  Ltx2TextProjection proj;
  proj.out_features = w->shape[0];
  proj.in_features = w->shape[1] * 2;  // NVFP4 packs TWO values per byte
  if (proj.in_features != in_features) {
    Fail("'" + module + ".weight' unpacks to in_features " +
         std::to_string(proj.in_features) + " but the Gemma geometry gives " +
         std::to_string(in_features) +
         " (hidden_size * (num_hidden_layers + 1), feature_extractor.py:120). Reading "
         "the STORED U8 width as logical is what halves it.");
  }
  const StTensor* s = Find(file, module + ".weight_scale");
  const StTensor* g = Find(file, module + ".weight_scale_2");
  if (s == nullptr) Fail("the text encoder is missing '" + module + ".weight_scale'");
  if (g == nullptr) Fail("the text encoder is missing '" + module + ".weight_scale_2'");
  proj.weight_bf16.resize(static_cast<size_t>(proj.out_features) *
                          static_cast<size_t>(proj.in_features));
  Ltx2DequantTorchaoNvfp4ToBf16(module, *w, *s, *g, proj.out_features, proj.in_features,
                                proj.weight_bf16.data());

  const StTensor* b = Find(file, module + ".bias");
  if (b != nullptr) {
    if (b->dtype != "BF16") {
      Fail("'" + module + ".bias' must be BF16 (it is not quantized), not " + b->dtype);
    }
    if (b->shape.size() != 1 || b->shape[0] != proj.out_features) {
      Fail("'" + module + ".bias' is " + ShapeText(b->shape) + " but the projection has " +
           std::to_string(proj.out_features) + " outputs");
    }
    proj.bias_bf16.resize(static_cast<size_t>(proj.out_features));
    std::memcpy(proj.bias_bf16.data(), b->data, b->nbytes);
  }
  return proj;
}

}  // namespace

Ltx2TextEncoderCheckpoint Ltx2LoadTextEncoderFromSafetensors(
    const SafetensorsFile& file, const Ltx2TextEncoderLoadOptions& options) {
  Ltx2TextEncoderCheckpoint out;

  // The width authority is the UNPACKED tensor. `model.norm.weight` is BF16, so
  // it is stored one value per element and cannot be misread by a factor of two.
  const StTensor* norm = Find(file, "model.norm.weight");
  if (norm == nullptr) {
    Fail(
        "the text encoder is missing 'model.norm.weight'. It is the only UNPACKED "
        "tensor that fixes the Gemma hidden size; every quantized width in this file "
        "is half its logical value, so without it the geometry is a guess.");
  }
  if (norm->dtype != "BF16" || norm->shape.size() != 1) {
    Fail("'model.norm.weight' must be BF16 rank 1, not " + norm->dtype + " " +
         ShapeText(norm->shape));
  }
  out.gemma_hidden_size = norm->shape[0];

  int64_t layers = 0;
  const std::string layer_prefix = "model.layers.";
  for (const std::string& name : file.Names()) {
    if (!StartsWith(name, layer_prefix)) continue;
    const size_t dot = name.find('.', layer_prefix.size());
    if (dot == std::string::npos) continue;
    const std::string index = name.substr(layer_prefix.size(), dot - layer_prefix.size());
    if (index.empty() || index.find_first_not_of("0123456789") != std::string::npos) continue;
    layers = std::max<int64_t>(layers, std::stoll(index) + 1);
  }
  if (layers <= 0) Fail("the text encoder has no 'model.layers.<i>.*' tensors");
  out.gemma_num_hidden_layers = layers;

  // Every quantized module in the file — the multimodal tower included — is
  // VALIDATED, so an unreadable one is a load-time refusal by name rather than a
  // phase-L7 surprise. Nothing but the two caption projections is materialized.
  const std::string marker_suffix = kLtx2TorchaoNvfp4MarkerSuffix;
  for (const std::string& name : file.Names()) {
    if (!EndsWith(name, marker_suffix)) continue;
    const std::string module = name.substr(0, name.size() - marker_suffix.size());
    out.quantized_modules.push_back(module);
    ParseLtx2TorchaoNvfp4Marker(module, file.Get(name));

    const StTensor* w = Find(file, module + ".weight");
    const StTensor* s = Find(file, module + ".weight_scale");
    const StTensor* g = Find(file, module + ".weight_scale_2");
    if (w == nullptr || s == nullptr || g == nullptr) {
      Fail("'" + module +
           "' carries a torchao_nvfp4 marker but is missing " +
           std::string(w == nullptr ? "weight " : "") +
           std::string(s == nullptr ? "weight_scale " : "") +
           std::string(g == nullptr ? "weight_scale_2 " : "") +
           "- an incomplete quantized module cannot be dequantized, and skipping it "
           "would read as zeros.");
    }
    if (w->dtype != "U8" || w->shape.size() != 2) {
      Fail("'" + module + ".weight' must be U8 rank 2, not " + w->dtype + " " +
           ShapeText(w->shape));
    }
    const int64_t out_features = w->shape[0];
    const int64_t in_features = w->shape[1] * 2;
    if (in_features % kNvfp4GroupSize != 0) {
      Fail("'" + module + "' unpacks to in_features " + std::to_string(in_features) +
           ", not a multiple of " + std::to_string(kNvfp4GroupSize));
    }
    const std::vector<int64_t> want = {RoundUp(out_features, 128) / 4,
                                       RoundUp(in_features / kNvfp4GroupSize, 4) * 4};
    if (s->dtype != "F8_E4M3" || s->shape != want) {
      Fail("'" + module + ".weight_scale' is " + s->dtype + " " + ShapeText(s->shape) +
           " but a swizzled torchao scale for [" + std::to_string(out_features) + ", " +
           std::to_string(in_features) + "] is F8_E4M3 " + ShapeText(want));
    }
    if (g->dtype != "F32") {
      Fail("'" + module + ".weight_scale_2' must be F32, not " + g->dtype);
    }
  }

  const int64_t in_features = out.gemma_hidden_size * (out.gemma_num_hidden_layers + 1);
  if (!options.skip_projections) {
    out.video = LoadProjection(file, "text_embedding_projection.video_aggregate_embed",
                               in_features);
    if (Find(file, "text_embedding_projection.audio_aggregate_embed.weight") != nullptr) {
      out.audio = LoadProjection(file, "text_embedding_projection.audio_aggregate_embed",
                                 in_features);
    }
  }

  // Phase L3's own asset reader, unchanged: the tokenizer ships AS A TENSOR.
  out.assets = Ltx2LoadGemmaAssets(file, options.require_config);
  return out;
}

Ltx2TextEncoderWeights Ltx2WidenTextProjectionsToF32(
    const Ltx2TextEncoderCheckpoint& checkpoint) {
  auto widen = [](const Ltx2TextProjection& src, Ltx2TextAggregateEmbed& dst) {
    dst.out_features = src.out_features;
    dst.in_features = src.in_features;
    dst.weight.resize(src.weight_bf16.size());
    for (size_t i = 0; i < src.weight_bf16.size(); ++i) {
      dst.weight[i] = Bf16ToF32(src.weight_bf16[i]);
    }
    dst.bias.resize(src.bias_bf16.size());
    for (size_t i = 0; i < src.bias_bf16.size(); ++i) {
      dst.bias[i] = Bf16ToF32(src.bias_bf16[i]);
    }
  };
  Ltx2TextEncoderWeights out;
  widen(checkpoint.video, out.video);
  widen(checkpoint.audio, out.audio);
  return out;
}

}  // namespace vllm
