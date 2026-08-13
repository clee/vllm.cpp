// LTX-2.5 behind the generalized video seam — implementation. See
// include/vllm/multimodal/ltx2_video.h for the port map and the three refusals.
//
// Row: MODEL-DIFFUSION-LTX25, .agents/specs/ltx-2-5.md phase L7. Issue #435.
#include "vllm/multimodal/ltx2_video.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/device_pool.h"  // ActivePool(b)/DevicePool::Drain
#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_audio_vae.h"
#include "vllm/model_executor/models/ltx2_connector.h"
#include "vllm/model_executor/models/ltx2_device.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm/model_executor/models/minimax_h3.h"

namespace vllm::multimodal {
namespace {

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("ltx-2.5 video: " + message);
}

std::string ReadFileBytes(const std::string& field, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) Fail(field + ": cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteFileBytes(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) Fail("cannot write " + path);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) Fail("short write " + path);
}

// A JSON object read from a FILE, for the one configuration a checkpoint may not
// carry itself (`dit_config_path`). Both failure modes are refusals by name: a
// path that will not open, and bytes that are not a JSON object. Returning an
// empty object on either would be indistinguishable from an empty config, which
// is exactly the silent default this extra exists to remove.
nlohmann::json ReadJsonFile(const std::string& field, const std::string& path) {
  const std::string bytes = ReadFileBytes(field, path);
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(bytes);
  } catch (const std::exception& e) {
    Fail(field + ": '" + path + "' is not JSON (" + e.what() + ")");
  }
  if (!parsed.is_object()) Fail(field + ": '" + path + "' is not a JSON object");
  return parsed;
}

std::vector<float> ReadF32File(const std::string& field, const std::string& path) {
  const std::string bytes = ReadFileBytes(field, path);
  if (bytes.size() % sizeof(float) != 0) {
    Fail(field + ": '" + path + "' is " + std::to_string(bytes.size()) +
         " bytes, not a whole number of little-endian f32 values");
  }
  std::vector<float> out(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

// ── the noise stream ────────────────────────────────────────────────────────
//
// Upstream draws every tensor from ONE seeded `torch.Generator`
// (distilled.py:214-215) and the ancestral loop draws from a second one offset
// by `ANCESTRAL_NOISE_SEED_OFFSET` (:70-73, :177-183). Reproducing torch's
// Mersenne/Philox stream bit-exactly would decide WHICH sample comes out, not
// whether the pipeline is right — the same call MiniMax-H3 made and recorded
// (minimax_h3.h:1895-1897). So this is a documented splitmix64 + Box-Muller
// stream: deterministic, seedable, drawn in the SAME ORDER upstream draws
// (video before audio, one draw per state per step), and NOT torch's.
//
// What that costs is stated rather than hidden: a clip rendered here is a
// different sample from the same distribution as upstream's, so it is not
// comparable to an upstream render frame by frame. Sample-level comparison
// against the binding oracle needs the noise supplied from outside, which is
// exactly why every brick below L7 takes its noise as an argument.
class SplitMixGaussian {
 public:
  explicit SplitMixGaussian(uint64_t seed) : state_(seed) {}

  void Fill(float* out, int64_t count) {
    for (int64_t i = 0; i < count; ++i) out[i] = Next();
  }
  std::vector<float> Draw(int64_t count) {
    std::vector<float> out(static_cast<size_t>(count));
    Fill(out.data(), count);
    return out;
  }

 private:
  uint64_t NextBits() {
    state_ += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double Uniform() {
    // (0, 1): Box-Muller takes a log of the first draw, so 0 must be excluded.
    return (static_cast<double>(NextBits() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
  float Next() {
    if (have_spare_) {
      have_spare_ = false;
      return spare_;
    }
    const double u1 = Uniform();
    const double u2 = Uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 6.283185307179586476925286766559 * u2;
    spare_ = static_cast<float>(r * std::sin(theta));
    have_spare_ = true;
    return static_cast<float>(r * std::cos(theta));
  }

  uint64_t state_ = 0;
  bool have_spare_ = false;
  float spare_ = 0.0F;
};

// The VAE decoder's `torch.randn` source (ltx2_video_vae.h:152-161), consumed in
// CALL ORDER — which is what makes the decode reproducible on both sides.
class EngineNoiseStream : public Ltx2NoiseStream {
 public:
  explicit EngineNoiseStream(uint64_t seed) : gen_(seed) {}
  std::vector<float> Draw(int64_t count) override { return gen_.Draw(count); }

 private:
  SplitMixGaussian gen_;
};

// ── one modality's state through the loop (ltx-core types.py LatentState) ───
// Every buffer is PATCHIFIED: [tokens, width]. `mask` and `clean` are what
// `post_process_latent` (utils/helpers.py:462-464) blends between.
struct StreamState {
  int64_t tokens = 0;
  int64_t width = 0;                 // channels per token
  std::vector<float> latent, clean;  // [tokens, width]
  std::vector<float> mask;           // [tokens], patchified denoise mask
  std::vector<double> positions;     // [n_pos_dims, tokens, 2]
};

// `post_process_latent` (utils/helpers.py:462-464):
//   denoised * mask + clean * (1 - mask)
// The mask is PER TOKEN and the latent is per token x channel, so the mask
// broadcasts along the channel axis exactly as torch's trailing-axis rule does.
std::vector<float> PostProcessLatent(const std::vector<float>& denoised, const StreamState& state) {
  std::vector<float> out(denoised.size());
  for (int64_t t = 0; t < state.tokens; ++t) {
    const float m = state.mask[static_cast<size_t>(t)];
    for (int64_t c = 0; c < state.width; ++c) {
      const size_t i = static_cast<size_t>(t * state.width + c);
      out[i] = denoised[i] * m + state.clean[i] * (1.0F - m);
    }
  }
  return out;
}

// `GaussianNoiser.__call__` (components/noisers.py:30-37), on the PATCHIFIED
// state. The second lerp is the one a port gets backwards; `Ltx2GaussianNoise`
// already implements both and is gated, so this only broadcasts the per-token
// mask onto the per-token x channel latent before calling it.
void ApplyGaussianNoise(StreamState& state, const std::vector<float>& noise, float noise_scale) {
  std::vector<float> broadcast_mask(state.latent.size());
  for (int64_t t = 0; t < state.tokens; ++t) {
    for (int64_t c = 0; c < state.width; ++c) {
      broadcast_mask[static_cast<size_t>(t * state.width + c)] = state.mask[static_cast<size_t>(t)];
    }
  }
  state.latent = Ltx2GaussianNoise(state.latent.data(), state.clean.data(), broadcast_mask.data(),
                                   noise.data(), static_cast<int64_t>(state.latent.size()),
                                   noise_scale);
}

// `timesteps_from_mask` (utils/helpers.py:494-503): the per-token timestep is
// `denoise_mask * sigma`, so a conditioned token is at timestep 0 while an
// unconditioned one is at the schedule's sigma.
std::vector<float> TimestepsFromMask(const StreamState& state, float sigma) {
  std::vector<float> out(static_cast<size_t>(state.tokens));
  for (int64_t t = 0; t < state.tokens; ++t) out[static_cast<size_t>(t)] = state.mask[static_cast<size_t>(t)] * sigma;
  return out;
}

// `X0Model.forward` (model/transformer/model.py:601-604) composed with
// `to_denoised` (utils.py:38-50): the DiT emits a VELOCITY and the loop consumes
// a denoised prediction, `x - sigma * v`, with the PER-TOKEN sigma. Getting the
// per-token part wrong (using the scalar schedule sigma for every token) is
// invisible until a request carries conditioned tokens, and then it re-noises
// them.
std::vector<float> ToDenoised(const std::vector<float>& sample, const std::vector<float>& velocity,
                              const std::vector<float>& timesteps, int64_t tokens, int64_t width) {
  std::vector<float> out(sample.size());
  for (int64_t t = 0; t < tokens; ++t) {
    const double sigma = timesteps[static_cast<size_t>(t)];
    for (int64_t c = 0; c < width; ++c) {
      const size_t i = static_cast<size_t>(t * width + c);
      out[i] = static_cast<float>(static_cast<double>(sample[i]) -
                                  sigma * static_cast<double>(velocity[i]));
    }
  }
  return out;
}

std::string JoinPath(const std::string& dir, const std::string& leaf) {
  if (dir.empty()) return leaf;
  return dir.back() == '/' ? dir + leaf : dir + "/" + leaf;
}

int64_t ExtraInt(const std::map<std::string, std::string>& extras, const std::string& key,
                 int64_t fallback) {
  const std::string raw = VideoExtra(extras, key);
  if (raw.empty()) return fallback;
  try {
    size_t consumed = 0;
    const long long value = std::stoll(raw, &consumed);
    if (consumed != raw.size()) throw std::invalid_argument("trailing");
    return static_cast<int64_t>(value);
  } catch (const std::exception&) {
    Fail("the load extra '" + key + "' is '" + raw + "', which is not an integer");
  }
}

// Every extra key this family DEFINES. An extra outside this set is refused
// rather than ignored, for the same reason H3 refuses one
// (minimax_h3_video.cpp): a mistyped knob that is silently dropped renders the
// DEFAULT and looks like the feature not working.
const char* const kKnownLoadExtras[] = {
    kLtx2AudioPromptEmbedsExtra, kLtx2PipelineKindExtra,   kLtx2ModelVersionExtra,
    kLtx2AllowUnportedExtra,     kLtx2MaxPhaseExtra,       kLtx2DitConfigPathExtra,
    kLtx2PromptValidRowsExtra,   "upsampler_path",         "duration_head_path",
};

void CheckKnownExtras(const std::map<std::string, std::string>& extras) {
  for (const auto& kv : extras) {
    bool known = false;
    for (const char* name : kKnownLoadExtras) {
      if (kv.first == name) known = true;
    }
    if (!known) {
      std::string listing;
      for (const char* name : kKnownLoadExtras) listing += std::string(listing.empty() ? "" : ", ") + name;
      Fail("unknown load extra '" + kv.first + "'. This family defines: " + listing);
    }
  }
}

// `detect_model_version` normalizes the separator before parsing
// (utils/constants.py:161), and the recipe table is keyed on the two-component
// spelling ("2.5"), not on the checkpoint's three-component "2.5.0". Reduce it
// with the SAME parser the table's own callers use, so "2.5.0", "2.5" and
// "2.5-rc1" all resolve to one row instead of two of them missing it.
std::string RecipeVersionKey(const std::string& declared) {
  const std::vector<int64_t> parsed = Ltx2ParseModelVersion(declared);
  if (parsed.empty()) {
    Fail("the DiT checkpoint declares model_version '" + declared +
         "', whose numeric prefix is empty (loader/helpers.py:62-81). A recipe is resolved "
         "from an EXACT (kind, version) pair and never defaulted.");
  }
  std::string key = std::to_string(parsed[0]);
  if (parsed.size() > 1) key += "." + std::to_string(parsed[1]);
  return key;
}

}  // namespace

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

// Does this checkpoint set hold an LTX-2.5 DiT? The discriminator is the DUAL
// PATCHIFY PROJECTION, `patchify_proj.weight` + `audio_patchify_proj.weight` —
// the two names phase L2's contract binds first (`EnumerateLtx2DitTensors`) and
// the two every shipped arm carries, ComfyUI prefix or not.
//
// It CANNOT collide with MiniMax-H3's, which keys on `video_patch_proj.weight` +
// `audio_patch_proj.weight` (minimax_h3_video.cpp:820-831). The names are
// different words, not a prefix of one another, so neither detector can see the
// other's checkpoint however either file was repackaged.
//
// Deliberately NOT keyed on the file extension, on the directory layout, or on
// the presence of a `model.diffusion_model.` prefix: which container a checkpoint
// was repackaged into, and whether the repackager kept ComfyUI's prefix, say
// nothing about which model it holds. Both shipped LTX-2.5 DiTs carry the
// prefix; a de-prefixed re-export must still be found.
bool DetectLtx2Video(const VideoModelParams& params) {
  std::vector<std::string> names;
  std::string why;
  if (!ReadVideoCheckpointTensorNames(params.dit_path, &names, &why)) return false;
  const std::string prefix = kLtx2DitCheckpointPrefix;
  bool video_patchify = false, audio_patchify = false;
  for (const std::string& n : names) {
    std::string bare = n;
    if (bare.size() > prefix.size() && bare.compare(0, prefix.size(), prefix) == 0) {
      bare = bare.substr(prefix.size());
    }
    if (bare == "patchify_proj.weight") video_patchify = true;
    if (bare == "audio_patchify_proj.weight") audio_patchify = true;
    if (video_patchify && audio_patchify) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

struct Ltx2VideoEngine::Impl {
  VideoModelParams params;
  // CPU, or the CUDA device `params.device - 1` names. Phase L8 made the second
  // real: the DiT is staged with `Ltx2StreamDitToDevice` and driven by
  // `Ltx2DitForwardDevice`, so a CUDA handle now denotes a CUDA forward.
  vt::Device device;
  // The stream dtype the DiT was staged at, and the one the forward computes in.
  // bf16 on an accelerator — upstream resolves ONE model dtype and every layer
  // inherits it, and bf16 is what `Ltx2StreamDitToDevice` puts on the device.
  // f32 on the CPU, which is the L2 parity forward's declared dtype.
  vt::DType compute_dtype = vt::DType::kF32;
  bool on_device = false;
  std::optional<vt::Queue> queue;

  Ltx2DitCheckpoint dit;
  std::string model_version, pipeline_kind;
  Ltx2PipelineRecipe recipe;
  int64_t max_phase = -1;  // -1 => every phase

  Ltx2VideoDecoderKind video_kind = Ltx2VideoDecoderKind::kConv;
  Ltx2ConvVideoDecoderConfig video_cfg;
  Ltx2VaeWeights video_weights;

  Ltx2AudioDecoderConfig audio_cfg;
  Ltx2VaeWeights audio_weights;
  Ltx2VocoderBweConfig vocoder_cfg;
  Ltx2VaeWeights vocoder_weights;

  bool has_upsampler = false;
  Ltx2UpsamplerConfig upsampler_cfg;
  Ltx2VaeWeights upsampler_weights;

  // Conditioning. `has_encoder` is false for every checkpoint today; see the
  // header's refusal 2.
  bool has_encoder = false;
  // What the DiT's cross-attention consumes. When the checkpoint carries the
  // embeddings connector these are the CONNECTOR's output; without one they are
  // the supplied prompt embeds verbatim, which is what every render before L9c
  // did on every checkpoint.
  std::vector<float> video_prompt_embeds, audio_prompt_embeds;
  int64_t prompt_tokens = 0;

  // The connector's CONFIGURATION is kept; its WEIGHTS are not. They are ~8 GB
  // of f32 at the shipped widths (ltx2_loader.h), the conditioning they process
  // is resolved once at load, and this box reboots rather than OOM-killing — so
  // they are loaded, used and dropped inside one scope below.
  bool has_connector = false;
  Ltx2ConnectorConfig video_connector_cfg, audio_connector_cfg;
  int64_t prompt_valid_rows = 0;

  std::mutex mutex;
};

Ltx2VideoEngine::Ltx2VideoEngine() = default;
Ltx2VideoEngine::Ltx2VideoEngine(Ltx2VideoEngine&&) noexcept = default;
Ltx2VideoEngine& Ltx2VideoEngine::operator=(Ltx2VideoEngine&&) noexcept = default;
Ltx2VideoEngine::~Ltx2VideoEngine() = default;

std::string Ltx2VideoEngine::family() const { return kLtx2VideoFamily; }
vt::Device Ltx2VideoEngine::device() const { return impl_->device; }
bool Ltx2VideoEngine::has_encoder() const { return impl_->has_encoder; }
bool Ltx2VideoEngine::has_prompt_embeds() const { return !impl_->video_prompt_embeds.empty(); }
const std::string& Ltx2VideoEngine::model_version() const { return impl_->model_version; }
const std::string& Ltx2VideoEngine::pipeline_kind() const { return impl_->pipeline_kind; }
const Ltx2DitParams& Ltx2VideoEngine::dit_params() const { return impl_->dit.params; }

std::unique_ptr<Ltx2VideoEngine> Ltx2VideoEngine::Load(const VideoModelParams& params) {
  if (params.dit_path.empty()) Fail("dit_path is required");
  CheckKnownExtras(params.extras);

  auto engine = std::unique_ptr<Ltx2VideoEngine>(new Ltx2VideoEngine());
  engine->impl_ = std::make_unique<Impl>();
  Impl& im = *engine->impl_;
  im.params = params;

  // ── where this engine runs (phase L8) ─────────────────────────────────────
  //
  // `device` is 0 for the CPU and 1 + <cuda index> for an accelerator, which is
  // the mapping the seam already documents. Phase L7 REFUSED anything but 0,
  // because the f32-only forward and the bf16-only staging did not meet; phase
  // L8 is the forward that closes that, so the refusal is gone and the handle
  // now means what it says.
  //
  // What is NOT gone is the refusal to fake it: if the CUDA backend is not
  // registered in this build, the load is refused BY NAME rather than served the
  // CPU forward behind a CUDA-looking handle. That substitution is exactly what
  // would make every later timing and every "it ran on the GPU" claim false.
  im.on_device = params.device != 0;
  if (im.on_device) {
    vt::Backend* backend = vt::TryGetBackend(vt::DeviceType::kCUDA);
    if (backend == nullptr) {
      Fail("device " + std::to_string(params.device) +
           " asks for CUDA, but no CUDA backend is registered in this build. The LTX-2.5 "
           "device-resident forward is present (Ltx2DitForwardDevice); what is missing is "
           "the backend. Refusing rather than running the CPU forward behind a CUDA handle.");
    }
    // `vt::CreateQueue(Device)`, NOT `Backend::CreateQueue()`. backend.h:212-217
    // records the method as a "temporary index-0 migration shim" and says new
    // adapter code must use the free function, and the difference is not
    // cosmetic: the shim creates the queue on device 0 and relabelling its
    // `device.index` afterwards MOVES THE LABEL, not the stream. On a multi-GPU
    // host `device = 2` would then run every kernel on GPU 0 while every
    // residency check in this file reported index 1 and agreed with itself. GB10
    // has one GPU so it cannot be reached there, which is exactly why it has to
    // be right before a second device exists.
    const int32_t index = static_cast<int32_t>(params.device - 1);
    im.device = vt::Device{vt::DeviceType::kCUDA, index};
    if (vt::TryGetBackend(im.device) == nullptr) {
      Fail("device " + std::to_string(params.device) + " names CUDA device index " +
           std::to_string(index) +
           ", and no backend is registered for it. Refusing rather than creating a queue "
           "on device 0 and labelling it with an index nothing runs on.");
    }
    im.queue = vt::CreateQueue(im.device);
    // bf16: upstream resolves ONE model dtype and every layer inherits it, and it
    // is what `Ltx2StreamDitToDevice` stages. f32 on an accelerator would move
    // twice the bytes to reach a gate dtype, which is the polarity this project
    // inverted deliberately.
    im.compute_dtype = vt::DType::kBF16;
  } else {
    im.device = vt::Device{};
    im.compute_dtype = vt::DType::kF32;
  }

  // ── the DiT ───────────────────────────────────────────────────────────────
  const SafetensorsFile dit_file = SafetensorsFile::Open(params.dit_path);
  Ltx2DitLoadOptions dit_options;
  dit_options.allow_unported_modules = VideoExtra(params.extras, kLtx2AllowUnportedExtra) == "1";
  // On the CPU, f32 is what `Ltx2DitForward` requires: it is the PARITY dtype,
  // not a widening of a bf16 path. On an accelerator nothing is widened at all —
  // `Ltx2StreamDitToDevice` dequantizes and uploads ONE TENSOR AT A TIME, so peak
  // residency is the device copy plus one tensor rather than two whole models.
  dit_options.widen_to_f32 = !im.on_device;
  im.dit = im.on_device ? Ltx2StreamDitToDevice(*im.queue, dit_file, dit_options)
                        : Ltx2LoadDitFromSafetensors(dit_file, dit_options);

  // ── the config the SHAPES cannot see ──────────────────────────────────────
  //
  // FOUND AT L7 AND REPAIRED HERE. `Ltx2LoadDitFromSafetensors` derives its
  // geometry with `ParseLtx2DitParamsFromManifest`, which reads SHAPES — the way
  // a ComfyUI checkpoint carrying no config has to be read. A config states
  // things no shape encodes:
  //
  //     "frequencies_precision": "float64"          -> double_precision_rope
  //     "av_ca_timestep_scale_multiplier": 1000.0   -> default is 1
  //     "positional_embedding_max_pos": [20,2048,2048], "..._theta": 10000.0
  //     "norm_eps", "timestep_scale_multiplier", "use_middle_indices_grid"
  //
  // Each moves every RoPE angle or every modulation while leaving the tensor
  // set byte-identical, so the manifest path resolves a DIFFERENT MODEL from the
  // same file and nothing downstream can tell. `ParseLtx2DitParams` already
  // mirrors `LTXModelConfigurator.from_metadata` (model_configurator.py:19-83)
  // for exactly this; it was simply never reached.
  //
  // AND ONLY ONE OF THE TWO SHIPPED LTX-2.5 DiTs CARRIES ONE. An earlier revision
  // of this comment said "both shipped LTX-2.5 DiTs DO carry one". That was FALSE
  // and it mattered, because the copy it was false about is the one phases L1-L6
  // gated against and the one L8 ran on the GPU. Read from the files on the NAS,
  // 2026-08-12:
  //
  //   Lightricks/LTX-2.5 ...-transformer-nvfp4.safetensors
  //       __metadata__ = ['config','gemma_source_checkpoint','license',
  //                       'model_version']                       -> declares one
  //   vonkaiser/LTX-2.5-FP8-NVFP4 ltx-2.5-22b-distilled-fp8.safetensors
  //       has __metadata__ key: FALSE                            -> declares NONE
  //
  // So the adoption below never executed for the FP8 arm, and its DiT silently
  // took `av_ca_timestep_scale_multiplier = 1` (ltx2.h:106) and
  // `double_precision_rope = false` (ltx2.h:112) against LTX-2.5's declared 1000
  // and float64. That is a different model, not a different precision, and
  // nothing said so.
  //
  // THE REPAIR IS THAT A DiT WITHOUT A CONFIG IS REFUSED, not defaulted. The
  // caller supplies one through `dit_config_path` and it is adopted through the
  // IDENTICAL weight-contract check the declared path uses. `model_version` was
  // already treated this way one block down — never defaulted, supplied by an
  // extra when the file carries none — and this is the same rule applied to the
  // configuration that decides the arithmetic rather than the schedule.
  //
  // A config is adopted only when it produces the IDENTICAL weight contract —
  // which is what proves it describes THIS file rather than another checkpoint's
  // config pasted into it. A disagreement is refused by name.
  const std::string config_path = VideoExtra(params.extras, kLtx2DitConfigPathExtra);
  const bool declares_config = dit_file.Metadata().count("config") != 0;
  if (declares_config && !config_path.empty()) {
    Fail("the DiT checkpoint declares its own __metadata__[\"config\"] and the '" +
         std::string(kLtx2DitConfigPathExtra) + "' extra names '" + config_path +
         "'. Refusing rather than preferring one: they decide values no shape can see "
         "(frequencies_precision, av_ca_timestep_scale_multiplier, the "
         "positional-embedding bounds), so the wrong choice renders with the wrong RoPE "
         "instead of failing. Drop the extra to use the checkpoint's own config.");
  }
  if (!declares_config && config_path.empty()) {
    Fail("the DiT checkpoint declares no __metadata__[\"config\"][\"transformer\"], and no '" +
         std::string(kLtx2DitConfigPathExtra) +
         "' extra was supplied. The tensor SHAPES resolve the geometry but not the values "
         "no shape encodes: double_precision_rope would default to false and "
         "av_ca_timestep_scale_multiplier to 1, where LTX-2.5 declares float64 and 1000. "
         "Both move every RoPE angle and every audio<->video modulation, so defaulting them "
         "renders a DIFFERENT MODEL confidently. This is the shape the shipped vonkaiser FP8 "
         "DiT is in: it carries no __metadata__ at all. Supply the config rather than "
         "inheriting a default that contradicts the model family.");
  }
  // Hoisted out of the block it used to live in because the EMBEDDINGS CONNECTOR
  // is configured from the SAME object: its geometry, its RoPE bounds and its
  // gating are `connector_*` keys of the DiT's transformer config, and reading
  // them from a second source would let the two disagree.
  const nlohmann::json dit_config =
      declares_config ? Ltx2ReadCheckpointConfig(dit_file)
                      : ReadJsonFile(kLtx2DitConfigPathExtra, config_path);
  {
    const std::string source =
        declares_config
            ? std::string("the DiT checkpoint's own __metadata__[\"config\"][\"transformer\"]")
            : "the '" + std::string(kLtx2DitConfigPathExtra) + "' file '" + config_path + "'";
    // `Ltx2AdoptDeclaredDitParams` (ltx2_loader.h) is the ONE place the adoption
    // rule lives, because the device gate drives `Ltx2StreamDitToDevice` without
    // this engine and owes the same check; two copies would be two rules.
    const Ltx2DitParams declared = Ltx2AdoptDeclaredDitParams(
        dit_config, im.dit.params, dit_options.allow_unported_modules, source);
    im.dit.params = declared;
  }

  // ── the embeddings connector (phase L9c) ──────────────────────────────────
  //
  // WHAT THIS CLOSES. Until L9c the conditioning this engine handed the DiT's
  // cross-attention was the prompt-embeds file VERBATIM. Upstream never does
  // that: `EmbeddingsProcessor` runs an 8-layer 1-D transformer over the caption
  // projections before the DiT sees them (embeddings_processor.py:70-95), and
  // that module SHIPS IN THE DiT FILE — 129 tensors per stream, which this
  // loader used to refuse as "unported" and then step over. The bricks existed
  // (`Ltx2ConnectorForward`, landed at L5, gated against upstream on five arms);
  // nothing called them outside a test. This is the call.
  //
  // PRESENCE DECIDES, not a config flag. The connector is applied exactly when
  // the checkpoint carries it, which is the same rule `DetectLtx2Video` uses on
  // the DiT itself: which modules a file HOLDS is a fact, and a flag saying
  // otherwise would be a second, disagreeable authority. A checkpoint carrying
  // one stream's connector and not the other is refused, because rendering the
  // video stream through eight transformer layers and the audio stream through
  // none conditions the two halves of one clip on two different things.
  const bool has_video_connector =
      Ltx2CheckpointHasConnector(dit_file, Ltx2ConnectorStream::kVideo);
  const bool has_audio_connector =
      Ltx2CheckpointHasConnector(dit_file, Ltx2ConnectorStream::kAudio);
  if (has_video_connector != has_audio_connector) {
    Fail(std::string("the DiT checkpoint carries the ") +
         (has_video_connector ? "video" : "audio") +
         " embeddings connector but not the other stream's. Both conditioning paths run "
         "through one `EmbeddingsProcessor` upstream (embeddings_processor.py:60-68 refuses "
         "an audio connector without audio features and vice versa), so half a connector "
         "would condition one modality on eight transformer layers and the other on none.");
  }
  im.has_connector = has_video_connector;
  if (im.has_connector) {
    im.video_connector_cfg = Ltx2ParseConnectorConfig(dit_config, Ltx2ConnectorStream::kVideo);
    im.audio_connector_cfg = Ltx2ParseConnectorConfig(dit_config, Ltx2ConnectorStream::kAudio);
    // The connector is dimension-PRESERVING: it consumes the caption projection's
    // output and hands the DiT's cross-attention the same width. Asserted rather
    // than assumed — a connector whose inner_dim disagrees with the stream it
    // feeds would still forward, and the mismatch would surface as a wrong-shaped
    // GEMM deep inside the DiT rather than as a load refusal.
    if (im.video_connector_cfg.inner_dim() != im.dit.params.cross_attention_dim) {
      Fail("the video embeddings connector is " +
           std::to_string(im.video_connector_cfg.inner_dim()) +
           " wide but the DiT's cross-attention takes " +
           std::to_string(im.dit.params.cross_attention_dim));
    }
    if (im.audio_connector_cfg.inner_dim() != im.dit.params.audio_cross_attention_dim) {
      Fail("the audio embeddings connector is " +
           std::to_string(im.audio_connector_cfg.inner_dim()) +
           " wide but the DiT's audio cross-attention takes " +
           std::to_string(im.dit.params.audio_cross_attention_dim));
    }
  }

  // ── the recipe ────────────────────────────────────────────────────────────
  const std::string declared_version = Ltx2ReadCheckpointModelVersion(dit_file);
  const std::string override_version = VideoExtra(params.extras, kLtx2ModelVersionExtra);
  if (!declared_version.empty() && !override_version.empty() &&
      RecipeVersionKey(declared_version) != RecipeVersionKey(override_version)) {
    Fail("the DiT checkpoint declares model_version '" + declared_version + "' but the '" +
         kLtx2ModelVersionExtra + "' extra says '" + override_version +
         "'. Refusing rather than preferring one: they resolve DIFFERENT sigma schedules, "
         "and the wrong one renders a video instead of failing.");
  }
  const std::string version = !declared_version.empty() ? declared_version : override_version;
  if (version.empty()) {
    Fail("the DiT checkpoint declares no __metadata__[\"model_version\"] and no '" +
         std::string(kLtx2ModelVersionExtra) +
         "' extra was supplied, so no recipe can be resolved. A recipe is never defaulted "
         "(ltx2_recipes.py:170-175).");
  }
  im.model_version = RecipeVersionKey(version);
  im.pipeline_kind = VideoExtra(params.extras, kLtx2PipelineKindExtra, "distilled_two_stage");
  im.recipe = ResolveLtx2PipelineRecipe(im.pipeline_kind, im.model_version);
  im.max_phase = ExtraInt(params.extras, kLtx2MaxPhaseExtra, -1);
  if (im.max_phase >= static_cast<int64_t>(im.recipe.phases.size())) {
    Fail("the '" + std::string(kLtx2MaxPhaseExtra) + "' extra is " +
         std::to_string(im.max_phase) + " but the '" + im.pipeline_kind + "'/'" +
         im.model_version + "' recipe has " + std::to_string(im.recipe.phases.size()) +
         " phases");
  }

  // ── the video VAE ─────────────────────────────────────────────────────────
  if (params.video_vae_path.empty()) Fail("video_vae_path is required");
  {
    const SafetensorsFile f = SafetensorsFile::Open(params.video_vae_path);
    im.video_cfg = Ltx2ParseConvVideoDecoderConfig(Ltx2ReadCheckpointConfig(f), &im.video_kind);
    im.video_weights = Ltx2LoadVaeWeights(f, Ltx2VideoVaeDecoderKeyRules());
  }
  if (im.video_cfg.in_channels != im.dit.params.out_channels) {
    Fail("the video VAE takes " + std::to_string(im.video_cfg.in_channels) +
         " latent channels but the DiT emits " + std::to_string(im.dit.params.out_channels));
  }

  // ── the audio VAE + its vocoder ───────────────────────────────────────────
  if (params.audio_vae_path.empty()) Fail("audio_vae_path is required");
  {
    const SafetensorsFile f = SafetensorsFile::Open(params.audio_vae_path);
    const nlohmann::json config = Ltx2ReadCheckpointConfig(f);
    im.audio_cfg = Ltx2ParseAudioDecoderConfig(config);
    im.audio_weights = Ltx2LoadVaeWeights(f, Ltx2AudioVaeDecoderKeyRules());
    im.vocoder_cfg = Ltx2ParseVocoderBweConfig(config);
    im.vocoder_weights = Ltx2LoadVaeWeights(f, Ltx2VocoderKeyRules());
  }

  // ── the optional latent spatial upsampler (the two-stage recipe's phase 2) ─
  const std::string upsampler_path = VideoExtra(params.extras, "upsampler_path");
  if (!upsampler_path.empty()) {
    const SafetensorsFile f = SafetensorsFile::Open(upsampler_path);
    const nlohmann::json config = Ltx2ReadCheckpointConfig(f);
    im.upsampler_cfg = Ltx2ParseUpsamplerConfig(config);
    im.upsampler_weights = Ltx2LoadVaeWeights(f);
    im.has_upsampler = true;
  }

  // ── conditioning ──────────────────────────────────────────────────────────
  // REFUSAL 2 (header): the Gemma-4 TOWER is owed, so `encoder_path` cannot be
  // turned into hidden states here. Saying so at LOAD, when the caller supplied
  // one, is better than accepting the path and refusing every request later.
  if (!params.encoder_path.empty()) {
    Fail(
        "encoder_path was supplied, but this family cannot encode a prompt yet: phase L6 "
        "loads the caption projections and the embedded tokenizer and records the Gemma-4 "
        "TOWER itself as owed (ltx2_loader.h:318-324), so there is nothing to produce the "
        "49 hidden states `Ltx2TextFeatureExtractorForward` consumes. Condition through "
        "prompt_embeds_path + the '" +
        std::string(kLtx2AudioPromptEmbedsExtra) +
        "' extra meanwhile; that is the seam's own documented fallback.");
  }
  const std::string audio_embeds_path = VideoExtra(params.extras, kLtx2AudioPromptEmbedsExtra);
  if (params.prompt_embeds_path.empty() != audio_embeds_path.empty()) {
    Fail(
        "LTX-2.5 conditions TWO streams at two widths, so prompt_embeds_path (the video "
        "stream, " +
        std::to_string(im.dit.params.cross_attention_dim) + " wide) and the '" +
        std::string(kLtx2AudioPromptEmbedsExtra) + "' extra (the audio stream, " +
        std::to_string(im.dit.params.audio_cross_attention_dim) +
        " wide) are supplied together or not at all. One of them alone would leave a stream "
        "unconditioned, which renders.");
  }
  if (!params.prompt_embeds_path.empty()) {
    im.video_prompt_embeds = ReadF32File("prompt_embeds_path", params.prompt_embeds_path);
    im.audio_prompt_embeds = ReadF32File(kLtx2AudioPromptEmbedsExtra, audio_embeds_path);
    const int64_t vw = im.dit.params.cross_attention_dim;
    const int64_t aw = im.dit.params.audio_cross_attention_dim;
    if (im.video_prompt_embeds.size() % static_cast<size_t>(vw) != 0) {
      Fail("prompt_embeds_path holds " + std::to_string(im.video_prompt_embeds.size()) +
           " floats, which is not a whole number of " + std::to_string(vw) + "-wide rows");
    }
    if (im.audio_prompt_embeds.size() % static_cast<size_t>(aw) != 0) {
      Fail(std::string(kLtx2AudioPromptEmbedsExtra) + " holds " +
           std::to_string(im.audio_prompt_embeds.size()) +
           " floats, which is not a whole number of " + std::to_string(aw) + "-wide rows");
    }
    const int64_t v_rows = static_cast<int64_t>(im.video_prompt_embeds.size()) / vw;
    const int64_t a_rows = static_cast<int64_t>(im.audio_prompt_embeds.size()) / aw;
    if (v_rows != a_rows) {
      // Upstream's two encodings come from ONE tokenization
      // (feature_extractor.py:114-129 projects the SAME hidden states twice), so
      // differing row counts mean two different prompts, and the render would be
      // a picture of one with a soundtrack of the other.
      Fail("prompt_embeds_path has " + std::to_string(v_rows) + " rows but " +
           std::string(kLtx2AudioPromptEmbedsExtra) + " has " + std::to_string(a_rows) +
           "; both projections are built from ONE tokenization and must agree");
    }
    if (v_rows == 0) Fail("the prompt embeds are empty");
    im.prompt_tokens = v_rows;

    // How much of the supplied conditioning is REAL. Everything by default; the
    // extra is what a caller with a tokenizer's mask supplies.
    im.prompt_valid_rows = ExtraInt(params.extras, kLtx2PromptValidRowsExtra, v_rows);
    if (im.prompt_valid_rows < 0 || im.prompt_valid_rows > v_rows) {
      Fail("the '" + std::string(kLtx2PromptValidRowsExtra) + "' extra is " +
           std::to_string(im.prompt_valid_rows) + " but the prompt embeds hold " +
           std::to_string(v_rows) + " rows");
    }
    if (im.has_connector) {
      const int64_t registers = im.video_connector_cfg.num_learnable_registers;
      if (registers > 0 && v_rows % registers != 0) {
        // Upstream asserts this (embeddings_connector.py:144) because the
        // register table is TILED across the sequence rather than indexed by
        // which positions were padded.
        Fail("the prompt embeds hold " + std::to_string(v_rows) +
             " rows, which is not a multiple of the connector's " +
             std::to_string(registers) +
             " learnable registers. The register table is tiled across the sequence "
             "(embeddings_connector.py:144), so upstream asserts the same thing.");
      }
      // The additive mask `_prepare_attention_mask` would produce: 0.0 for a real
      // token, -finfo(f32).max for padding.
      std::vector<float> additive(static_cast<size_t>(v_rows), 0.0f);
      for (int64_t s = im.prompt_valid_rows; s < v_rows; ++s) {
        additive[static_cast<size_t>(s)] = -std::numeric_limits<float>::max();
      }
      // The weights live and die inside this scope. See the Impl comment: 129
      // tensors per stream is ~8 GB of f32 at the shipped widths, and holding
      // them for the engine's lifetime would spend that on a module that runs
      // exactly once. The DiT file is still open if a per-request conditioning
      // path ever needs to re-read them.
      //
      // THE f32 IS AN ANNOTATED ESCAPE, not an inherited default. Upstream runs
      // this module at the model dtype, so f32 here is WIDER — the polarity
      // AGENTS.md says a value gate cannot catch. It is taken because
      // `Ltx2ConnectorForward` is L5's declared PARITY dtype and this is the
      // arm its goldens cover, and it costs nothing on the production path: the
      // connector runs ONCE per load over 128 rows, and its output is narrowed
      // to the stream dtype on the first upload like every other activation.
      // What it does cost is the transient above, which is why it is scoped.
      const Ltx2VaeWeights video_connector =
          Ltx2LoadConnectorWeights(dit_file, im.video_connector_cfg);
      const Ltx2VaeWeights audio_connector =
          Ltx2LoadConnectorWeights(dit_file, im.audio_connector_cfg);
      const Ltx2ConnectorEmbeddings encoded = Ltx2ConnectorCreateEmbeddings(
          im.video_connector_cfg, video_connector, im.video_prompt_embeds.data(),
          im.audio_connector_cfg, audio_connector, im.audio_prompt_embeds.data(),
          additive.data(), /*batch=*/1, v_rows);
      // The processor returns the mask the DiT's cross-attention is supposed to
      // honour (embeddings_processor.py:89). `Ltx2ModalityInput` carries no
      // context mask, so a mask with a masked position would be silently dropped
      // — the DiT would attend over register-free padding as if it were caption.
      //
      // THIS LOOP IS UNREACHABLE ON EVERY INPUT EITHER REFERENCE PRODUCES, and
      // saying which case reaches it is how the claim stays checkable. It is NOT
      // the `num_learnable_registers = 0` case, which an earlier version of this
      // comment named and which cannot get here: with registers disabled
      // `Ltx2ConnectorForward` passes the caller's ADDITIVE mask straight
      // through, and `_to_binary_mask`'s `encoded_mask < 0.000001`
      // (embeddings_processor.py:46-48) is satisfied by BOTH values an additive
      // mask holds — 0.0 and -finfo(f32).max — so the binary mask is one
      // everywhere there too. With registers enabled :152 returns
      // `torch.zeros_like(mask)` and the answer is one everywhere for the same
      // reason. What would reach it is a connector whose output mask carries a
      // value at or above +1e-6, which no path in `ltx_core` or `diffusers`
      // emits today. It is kept as a guard on that future, refused by name
      // rather than ignored, and it is deliberately not gated: a test would have
      // to fabricate a mask neither reference can produce.
      for (const float m : encoded.mask) {
        if (m == 1.0f) continue;
        Fail(
            "the embeddings connector returned a cross-attention mask with masked "
            "positions, and `Ltx2ModalityInput` carries no context mask to pass it through. "
            "No path in either reference emits such a mask — `_to_binary_mask` is one "
            "everywhere for both values an additive mask holds — so this is a connector "
            "whose output mask this port does not model. Refusing rather than dropping "
            "the mask, which would condition the DiT on unmasked padding.");
      }
      im.video_prompt_embeds = encoded.video;
      im.audio_prompt_embeds = encoded.audio;
    }
  }
  return engine;
}

VideoResult Ltx2VideoEngine::Generate(const VideoGenParams& gen) {
  Impl& im = *impl_;
  std::lock_guard<std::mutex> guard(im.mutex);

  if (gen.output_dir.empty()) Fail("output_dir is required");
  if (!gen.extras.empty()) {
    Fail("unknown per-generation extra '" + gen.extras.begin()->first +
         "' (this family defines none)");
  }
  if (!gen.prompt.empty()) {
    Fail(
        "a prompt was supplied but no text tower is loaded, so it cannot condition this "
        "render. Rendering the prompt-embeds conditioning INSTEAD would silently ignore the "
        "request's own prompt. See the encoder_path refusal at load.");
  }
  if (im.video_prompt_embeds.empty()) {
    Fail("no conditioning is loaded; supply prompt_embeds_path and the '" +
         std::string(kLtx2AudioPromptEmbedsExtra) + "' extra");
  }
  // Image / reference conditioning is `ImageConditioner` upstream
  // (ltx-pipelines/utils/blocks.py:936-993, called at distilled.py:212). The
  // ENCODER it needs is no longer what is missing — phase L11 ported it as
  // `Ltx2ConvVideoEncode` — so the refusal names what actually is: this engine
  // holds no encoder to call. Refused by name rather than dropped: a keyframe
  // that is silently ignored renders an unconditioned clip that looks like the
  // feature not working.
  if (!gen.first_frame_path.empty() || !gen.first_frame_ppm.empty() ||
      !gen.last_frame_path.empty() || !gen.ref_image_paths.empty() ||
      !gen.ref_video_dir.empty() || !gen.ref_audio_path.empty() || !gen.ref_audio_wav.empty()) {
    Fail(
        "keyframe / reference conditioning is not ported for this family. The video VAE "
        "ENCODER itself landed in phase L11 (Ltx2ConvVideoEncode), but nothing can reach it "
        "from here: this engine materializes the DECODER key filter only, so no "
        "VAE_ENCODER_COMFY_KEYS_FILTER / VideoEncoderConfigurator path "
        "(video_vae/model_configurator.py:72, 267) puts encoder weights in memory, and "
        "upstream resolves each image conditioning's CRF against the checkpoint's "
        "default_image_crf when the caller left it unset (ImageConditioner.resolve_crf, "
        "ltx-pipelines/utils/blocks.py:977-983) and then re-compresses through an H.264 "
        "round trip unless that CRF is 0 (media_io/decode.py:413-435, from "
        "load_image_and_preprocess :75), which this build does not do. Recorded as owed.");
  }

  // ── geometry ──────────────────────────────────────────────────────────────
  const Ltx2PipelineRecipe& recipe = im.recipe;
  const double fps = recipe.frame_rate;
  const int64_t height = gen.height > 0 ? gen.height : recipe.height;
  const int64_t width = gen.width > 0 ? gen.width : recipe.width;
  int64_t frames = gen.num_frames > 1 ? gen.num_frames : recipe.num_frames;
  if (gen.duration_seconds > 0.0) {
    // `resolve_num_frames` (utils/blocks.py) turns an AUTO duration into frames
    // through the DURATION HEAD, which needs the encoded prompt this engine
    // cannot produce. An explicit duration is exact arithmetic, so it is served;
    // the AUTO path is what is missing, and `num_frames` is how to avoid it.
    frames = static_cast<int64_t>(std::llround(gen.duration_seconds * fps));
  }
  if (frames < 1) Fail("num_frames resolved to " + std::to_string(frames));

  const Ltx2ScaleFactors factors;  // VIDEO_SCALE_FACTORS (types.py:70) — the conv
                                   // arm's fixed (8, 32, 32), not derived
                                   // (utils/helpers.py:66-72)
  // `AudioLatentShape.from_video_pixel_shape` (types.py:184-200) and
  // `VideoLatentShape.from_pixel_shape` (:108-123) defaults. Asserted against the
  // DiT rather than assumed: the audio latent's channels x mel_bins IS the audio
  // stream's input width, and a mismatch would reinterpret the spectrogram.
  constexpr int64_t kAudioLatentChannels = 8;
  constexpr int64_t kAudioLatentMelBins = 16;
  if (kAudioLatentChannels * kAudioLatentMelBins != im.dit.params.audio_in_channels) {
    Fail("the audio latent is " + std::to_string(kAudioLatentChannels) + " x " +
         std::to_string(kAudioLatentMelBins) + " = " +
         std::to_string(kAudioLatentChannels * kAudioLatentMelBins) +
         " wide (types.py:184-200) but the DiT's audio stream takes " +
         std::to_string(im.dit.params.audio_in_channels));
  }

  const int64_t last_phase =
      im.max_phase >= 0 ? im.max_phase : static_cast<int64_t>(recipe.phases.size()) - 1;

  // One generator for the state noise (distilled.py:214-215) — see
  // SplitMixGaussian for what this is and is not.
  const uint64_t seed = gen.has_seed ? gen.seed : static_cast<uint64_t>(recipe.num_frames);
  SplitMixGaussian state_noise(seed);

  std::vector<float> video_latent_volume;  // [C, F, H, W], unpatchified
  int64_t video_lc = 0, video_lf = 0, video_lh = 0, video_lw = 0;
  std::vector<float> audio_latent_volume;  // [C, F, M], unpatchified
  int64_t audio_lc = 0, audio_lf = 0, audio_lm = 0;

  for (int64_t phase_index = 0; phase_index <= last_phase; ++phase_index) {
    const Ltx2PhaseRecipe& phase = recipe.phases[static_cast<size_t>(phase_index)];
    const int64_t phase_h = height / phase.spatial_downscale;
    const int64_t phase_w = width / phase.spatial_downscale;

    Ltx2VideoLatentShape vshape;
    vshape.channels = im.dit.params.in_channels;
    vshape.frames = (frames - 1) / factors.time + 1;
    vshape.height = phase_h / factors.height;
    vshape.width = phase_w / factors.width;
    if (vshape.frames < 1 || vshape.height < 1 || vshape.width < 1) {
      Fail("phase '" + phase.name + "' resolves a latent of " + std::to_string(vshape.frames) +
           "x" + std::to_string(vshape.height) + "x" + std::to_string(vshape.width) +
           " from " + std::to_string(frames) + " frames at " + std::to_string(phase_w) + "x" +
           std::to_string(phase_h) + "; the VAE downscales by " +
           std::to_string(factors.time) + "x" + std::to_string(factors.height) + "x" +
           std::to_string(factors.width) + " so the request is below one latent cell");
    }

    Ltx2AudioLatentShape ashape;
    ashape.channels = kAudioLatentChannels;
    ashape.mel_bins = kAudioLatentMelBins;
    {
      // `AudioLatentShape.from_duration` (types.py:164-181).
      const Ltx2AudioPatchifierParams ap;
      const double latents_per_second = static_cast<double>(ap.sample_rate) /
                                        static_cast<double>(ap.hop_length) /
                                        static_cast<double>(ap.audio_latent_downsample_factor);
      ashape.frames = static_cast<int64_t>(
          std::llround(static_cast<double>(frames) / fps * latents_per_second));
    }
    if (ashape.frames < 1) Fail("the audio latent resolved to zero frames");

    // ── the input transform (ltx2_recipes.py:38) ────────────────────────────
    std::vector<float> video_initial;
    if (phase.input_transform == Ltx2PhaseInputTransform::kSpatialUpsample) {
      if (video_latent_volume.empty()) {
        Fail("phase '" + phase.name +
             "' asks for the spatial-upsample input transform but no earlier phase produced a "
             "latent");
      }
      if (!im.has_upsampler) {
        Fail("phase '" + phase.name +
             "' needs the latent spatial x2 upsampler, and no 'upsampler_path' load extra was "
             "supplied. Refusing rather than skipping the phase: its 3-step refinement is what "
             "makes the upscaled latent valid, and running the decode on the half-resolution "
             "latent instead would return a smaller clip that looks like a completed request. "
             "Supply 'upsampler_path', or stop before this phase with '" +
             std::string(kLtx2MaxPhaseExtra) + "'.");
      }
      Ltx2LatentVolume in;
      in.batch = 1;
      in.channels = video_lc;
      in.frames = video_lf;
      in.height = video_lh;
      in.width = video_lw;
      in.data = video_latent_volume;
      // `upsample_video` (upsampler/model.py:129-143): un-normalize by the video
      // ENCODER's per-channel statistics, upsample, re-normalize. Those live in
      // the VAE checkpoint, not in the upsampler's.
      const Ltx2LatentVolume up = Ltx2UpsampleVideoLatent(
          im.upsampler_cfg, im.upsampler_weights, in,
          im.video_weights.Get("per_channel_statistics.std-of-means"),
          im.video_weights.Get("per_channel_statistics.mean-of-means"));
      if (up.channels != vshape.channels || up.frames != vshape.frames ||
          up.height != vshape.height || up.width != vshape.width) {
        Fail("the upsampled latent is " + std::to_string(up.channels) + "x" +
             std::to_string(up.frames) + "x" + std::to_string(up.height) + "x" +
             std::to_string(up.width) + " but phase '" + phase.name + "' needs " +
             std::to_string(vshape.channels) + "x" + std::to_string(vshape.frames) + "x" +
             std::to_string(vshape.height) + "x" + std::to_string(vshape.width));
      }
      video_initial = up.data;
    }

    // ── build the two states (create_noised_state, helpers.py:428-447) ───────
    StreamState video;
    video.width = vshape.channels;  // patch_size 1 (VideoLatentPatchifier(1))
    video.tokens = Ltx2VideoTokenCount(vshape, 1);
    {
      std::vector<float> volume(static_cast<size_t>(vshape.channels) *
                                static_cast<size_t>(vshape.frames) *
                                static_cast<size_t>(vshape.height) *
                                static_cast<size_t>(vshape.width));
      if (!video_initial.empty()) {
        if (video_initial.size() != volume.size()) Fail("the initial video latent is the wrong size");
        volume = video_initial;
      }
      video.latent = Ltx2VideoPatchify(volume.data(), vshape, 1);
      video.clean = video.latent;
      video.mask.assign(static_cast<size_t>(video.tokens), 1.0F);
      const std::vector<int64_t> bounds = Ltx2VideoPatchBounds(vshape, 1);
      const std::vector<int64_t> pixels = Ltx2PixelCoords(bounds, 1, video.tokens, factors, true);
      // `positions = get_pixel_coords(...).float()` then
      // `positions[:, 0, ...] /= fps` (tools.py:169-174). The division is f32
      // upstream, so it is done in float here and only then widened to the
      // double the DiT's `positions` field takes.
      video.positions.resize(pixels.size());
      for (size_t i = 0; i < pixels.size(); ++i) {
        const float value = static_cast<float>(pixels[i]);
        const bool temporal = i < static_cast<size_t>(video.tokens) * 2;
        video.positions[i] = temporal ? static_cast<double>(value / static_cast<float>(fps))
                                      : static_cast<double>(value);
      }
    }

    StreamState audio;
    audio.width = ashape.channels * ashape.mel_bins;
    audio.tokens = ashape.frames;
    {
      std::vector<float> volume(static_cast<size_t>(ashape.channels) *
                                static_cast<size_t>(ashape.frames) *
                                static_cast<size_t>(ashape.mel_bins));
      if (phase_index > 0) {
        // Stage 2 re-noises stage 1's AUDIO latent rather than starting from
        // zeros (distilled.py:307-311). Dropping that carry-over regenerates the
        // soundtrack from scratch with a 3-step schedule, which produces audio.
        if (audio_latent_volume.size() != volume.size()) {
          Fail("the audio latent changed size between phases");
        }
        volume = audio_latent_volume;
      }
      audio.latent = Ltx2AudioPatchify(volume.data(), ashape);
      audio.clean = audio.latent;
      audio.mask.assign(static_cast<size_t>(audio.tokens), 1.0F);
      const std::vector<float> timings = Ltx2AudioPatchTimings(ashape, Ltx2AudioPatchifierParams{});
      audio.positions.assign(timings.begin(), timings.end());
    }

    // The noiser draws VIDEO first, AUDIO second, from one generator
    // (blocks.py:576-580 builds the video state before the audio one).
    const float noise_scale = static_cast<float>(phase.noise_scale);
    ApplyGaussianNoise(video, state_noise.Draw(static_cast<int64_t>(video.latent.size())),
                       noise_scale);
    ApplyGaussianNoise(audio, state_noise.Draw(static_cast<int64_t>(audio.latent.size())),
                       noise_scale);

    // ── the schedule ────────────────────────────────────────────────────────
    std::vector<float> sigmas = phase.sigmas;
    if (sigmas.empty()) {
      int64_t steps = gen.steps > 0 ? gen.steps : recipe.num_inference_steps;
      if (steps < 1) Fail("num_inference_steps resolved to " + std::to_string(steps));
      sigmas = Ltx2SigmaSchedule(steps, video.tokens);
    } else if (gen.steps > 0 && !recipe.allow_request_sigmas) {
      // `fixed_num_inference_steps` (ltx2_recipes.py:53-87): a distilled recipe's
      // schedule is trained INTO the model, so honouring a step override would
      // sample a trajectory the weights were never distilled for.
      Fail("this recipe fixes its own distilled schedule (" +
           std::to_string(static_cast<int64_t>(phase.sigmas.size()) - 1) + " steps for phase '" +
           phase.name + "'), so a `steps` override is refused rather than applied");
    }

    // ── the denoise loop (samplers.py:39-79 / :488-558) ─────────────────────
    // The ancestral arm's loop generator is seeded from the pipeline seed plus
    // the recipe's own offset (distilled.py:69-73, :177-183) — a separate stream
    // from the state noise, so its first draw is not the initial latent's.
    SplitMixGaussian loop_noise(seed + static_cast<uint64_t>(phase.noise_seed_offset));
    const int64_t sigma_count = static_cast<int64_t>(sigmas.size());
    for (int64_t step = 0; step + 1 < sigma_count; ++step) {
      const float sigma = sigmas[static_cast<size_t>(step)];
      const std::vector<float> v_timesteps = TimestepsFromMask(video, sigma);
      const std::vector<float> a_timesteps = TimestepsFromMask(audio, sigma);
      const float sigma_row = sigma;

      Ltx2ModalityInput vin;
      vin.batch = 1;
      vin.tokens = video.tokens;
      vin.context_tokens = im.prompt_tokens;
      vin.latent = video.latent.data();
      vin.timesteps = v_timesteps.data();
      vin.sigma = &sigma_row;
      vin.positions = video.positions.data();
      vin.context = im.video_prompt_embeds.data();

      Ltx2ModalityInput ain;
      ain.batch = 1;
      ain.tokens = audio.tokens;
      ain.context_tokens = im.prompt_tokens;
      ain.latent = audio.latent.data();
      ain.timesteps = a_timesteps.data();
      ain.sigma = &sigma_row;
      ain.positions = audio.positions.data();
      ain.context = im.audio_prompt_embeds.data();

      // One graph, two residencies. On the CPU this is the L2 parity forward in
      // its declared f32; on an accelerator it is the phase-L8 device-resident
      // forward over the bf16 the DiT was STAGED at, and the two agree on
      // everything but where the bytes live and how wide they are.
      const Ltx2DitOutputs velocity =
          im.on_device ? Ltx2DitForwardDevice(*im.queue, im.dit.params, im.dit.weights, &vin,
                                              &ain, im.compute_dtype)
                       : Ltx2DitForward(im.device, im.dit.params, im.dit.weights, &vin, &ain,
                                        im.compute_dtype);

      const std::vector<float> v_denoised = PostProcessLatent(
          ToDenoised(video.latent, velocity.video, v_timesteps, video.tokens, video.width), video);
      const std::vector<float> a_denoised = PostProcessLatent(
          ToDenoised(audio.latent, velocity.audio, a_timesteps, audio.tokens, audio.width), audio);

      const bool terminal = sigmas[static_cast<size_t>(step + 1)] == 0.0F;
      if (phase.stepper == Ltx2StepperKind::kEulerAncestral) {
        if (terminal) {
          // samplers.py:545-547 — the terminal step IS the denoised prediction;
          // taking an ancestral step there would re-noise the finished latent.
          video.latent = v_denoised;
          audio.latent = a_denoised;
          continue;
        }
        const std::vector<float> v_noise =
            loop_noise.Draw(static_cast<int64_t>(video.latent.size()));
        const std::vector<float> a_noise =
            loop_noise.Draw(static_cast<int64_t>(audio.latent.size()));
        video.latent = PostProcessLatent(
            Ltx2EulerAncestralStep(video.latent.data(), v_denoised.data(), sigmas.data(),
                                   sigma_count, step, static_cast<int64_t>(video.latent.size()),
                                   phase.stepper_eta, phase.stepper_s_noise, v_noise.data()),
            video);
        audio.latent = PostProcessLatent(
            Ltx2EulerAncestralStep(audio.latent.data(), a_denoised.data(), sigmas.data(),
                                   sigma_count, step, static_cast<int64_t>(audio.latent.size()),
                                   phase.stepper_eta, phase.stepper_s_noise, a_noise.data()),
            audio);
      } else {
        video.latent = Ltx2EulerStep(video.latent.data(), v_denoised.data(), sigmas.data(),
                                     sigma_count, step,
                                     static_cast<int64_t>(video.latent.size()));
        audio.latent = Ltx2EulerStep(audio.latent.data(), a_denoised.data(), sigmas.data(),
                                     sigma_count, step,
                                     static_cast<int64_t>(audio.latent.size()));
      }
    }

    // `clear_conditioning` + `unpatchify` (blocks.py:575-580). There are no
    // conditioning tokens on this path, so the clear is the identity; the
    // unpatchify is not.
    video_latent_volume = Ltx2VideoUnpatchify(video.latent.data(), vshape, 1);
    video_lc = vshape.channels;
    video_lf = vshape.frames;
    video_lh = vshape.height;
    video_lw = vshape.width;
    audio_latent_volume = Ltx2AudioUnpatchify(audio.latent.data(), ashape);
    audio_lc = ashape.channels;
    audio_lf = ashape.frames;
    audio_lm = ashape.mel_bins;

    // PHASE CHANGE. The denoise loop just left the shared scratch pool holding
    // every activation size class this phase touched, and on an UNCAPPED pool
    // (`device_pool_cap_bytes == 0`, which is GB10 and Thor today) those blocks
    // are never returned to the driver. What comes next allocates DIFFERENT
    // classes — the next phase runs at twice the resolution, and after the last
    // phase the VAE decode runs on the host — so none of them can be reused and
    // all of them are headroom the next stage does not get.
    //
    // On GB10 that headroom is HOST memory: the unified pool is one ~119 GiB
    // arena, and this class of box REBOOTS rather than OOM-killing when the
    // driver runs out (`NVRM ... Out of memory [NV_ERR_NO_MEMORY]`), which it did
    // twice during phase L9b's 320x192/25-frame attempt. MiniMax-H3 drains at
    // exactly this boundary and for exactly this reason
    // (minimax_h3_pipeline.cpp:546-563); the LTX path drained nowhere.
    //
    // Draining costs one free per retained block, once per phase.
    if (im.on_device) {
      // A MEASUREMENT LANE, in the same shape `VT_POOL_BYPASS` and
      // `VT_POOL_EXACT` already take (device_pool.h): it exists so the A/B that
      // establishes what the drain is worth runs on ONE binary, which is what
      // AGENTS.md asks of a measurement. It is never a configuration — the drain
      // is on by default and there is no supported reason to turn it off.
      const char* off = std::getenv("VLLM_LTX2_POOL_DRAIN");
      if (off == nullptr || off[0] != '0') {
        vt::Backend& backend = vt::GetBackend(im.device.type);
        // `ActivePool(backend)`, not the device-less `ActivePool()` this line was
        // written against (#516): "the pool" without a device is what handed one
        // device's block to another, and draining it through `backend` was the
        // same assumption twice — resolve a pool with no device, then free its
        // blocks through an allocator that may not have made them. There is no
        // device-less spelling any more, and `Drain` refuses a foreign backend.
        const size_t drained = ActivePool(backend).Drain(backend);
        if (std::getenv("VT_POOL_STATS") != nullptr) {
          std::fprintf(stderr, "[ltx2] phase '%s' drained %.2f GiB of denoise scratch\n",
                       phase.name.c_str(),
                       static_cast<double>(drained) / (1024.0 * 1024.0 * 1024.0));
        }
      }
    }
  }

  // ── decode (distilled.py:314-315) ─────────────────────────────────────────
  EngineNoiseStream decode_noise(seed ^ 0x1D7Cull);
  const Ltx2VideoFrames rendered =
      Ltx2VideoDecode(im.video_kind, im.video_cfg, im.video_weights, video_latent_volume, video_lc,
                      video_lf, video_lh, video_lw, &decode_noise);

  const Ltx2AudioSpectrogram mel = Ltx2AudioDecoderForward(
      im.audio_cfg, im.audio_weights, audio_latent_volume, audio_lc, audio_lf, audio_lm);
  int64_t audio_samples = 0;
  const std::vector<float> waveform =
      Ltx2VocoderWithBweForward(im.vocoder_cfg, im.vocoder_weights, mel.data, mel.channels,
                                mel.frames, mel.mel_bins, &audio_samples);

  // ── artifacts (the library WRITES these, spawns nothing) ──────────────────
  std::error_code ec;
  std::filesystem::create_directories(gen.output_dir, ec);
  if (ec) Fail("cannot create " + gen.output_dir + ": " + ec.message());

  VideoResult result;
  result.frame_dir = gen.output_dir;
  MiniMaxH3VideoFrameShape shape;
  shape.channels = rendered.channels;
  shape.t = rendered.frames;
  shape.h = rendered.height;
  shape.w = rendered.width;
  for (int64_t f = 0; f < rendered.frames; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "frame_%06lld.ppm", static_cast<long long>(f));
    // The PPM/WAV/mux writers are the SHARED serialization the spec's §5 reuse
    // list names, not H3 behaviour: they take a buffer and a shape and contain
    // no model. Reimplementing them here would be the parallel path §"Shared
    // seams" forbids.
    WriteFileBytes(JoinPath(gen.output_dir, name),
                   MiniMaxH3WritePpmFrame(rendered.data, shape, f));
  }
  result.audio_path = JoinPath(gen.output_dir, "audio.wav");
  WriteFileBytes(result.audio_path,
                 MiniMaxH3WriteWav(waveform, mel.channels, audio_samples,
                                   im.vocoder_cfg.output_sampling_rate));
  result.frame_count = rendered.frames;
  result.width = rendered.width;
  result.height = rendered.height;
  result.fps = static_cast<int64_t>(std::llround(fps));
  result.sample_rate = im.vocoder_cfg.output_sampling_rate;

  MiniMaxH3MuxRequest mux;
  mux.frame_pattern = JoinPath(gen.output_dir, "frame_%06d.ppm");
  mux.audio_path = result.audio_path;
  mux.output_path = JoinPath(gen.output_dir, "video.mp4");
  mux.fps = result.fps;
  result.mux_argv = MiniMaxH3BuildMp4MuxArgs(mux);
  result.mux_output_path = mux.output_path;
  return result;
}

namespace {

std::unique_ptr<VideoEngine> LoadLtx2VideoFamily(const VideoModelParams& params) {
  return Ltx2VideoEngine::Load(params);
}

}  // namespace

REGISTER_VLLM_VIDEO_FAMILY(ltx2, kLtx2VideoFamily, DetectLtx2Video, LoadLtx2VideoFamily)

}  // namespace vllm::multimodal
