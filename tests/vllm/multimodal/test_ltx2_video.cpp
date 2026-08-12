// LTX-2.5 phase L7 gate — the family behind `vllm::multimodal::VideoEngine`,
// and the driving loop that turns the L2-L6 bricks into artifacts.
//
// Spec: .agents/specs/ltx-2-5.md §6 (L7). Issue #435.
//
// WHAT THIS CAN AND CANNOT SHOW. It runs the REAL path over a reduced-dimension
// checkpoint set written in the SHIPPED FILE FORMAT: the ComfyUI-prefixed FP8
// DiT through phase L6's quantized loader, the two VAEs through their own
// embedded `__metadata__["config"]`, the recipe table, the denoise loop, both
// decoders, the vocoder, and the artifact writers. So it gates COMPOSITION.
//
// It is NOT a render-quality result and nothing here should be read as one. The
// weights are a deterministic stream, so the frames are what those weights
// produce and no more — MiniMax-H3's fp4-resident e2e RAN and emitted a valid
// mp4 of a non-scene patch grid, which is exactly the failure a structural gate
// cannot see. What is asserted is what a structural gate CAN see: the geometry
// the pipeline resolved, that every value is finite, that the artifacts exist at
// the sizes the result reports, and that every refusal fires by name.
#include "vllm/multimodal/ltx2_video.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "ltx2_video_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vllm/multimodal/video_engine.h"

namespace {

struct Workspace {
  std::string root, fixture;
  ltx2_fixture::Paths paths;
  Workspace() {
    static int counter = 0;
    root = "/tmp/vllm_ltx2_video_" + std::to_string(::getpid()) + "_" + std::to_string(counter++);
    ::mkdir(root.c_str(), 0755);
    fixture = root + "/fixture";
    paths = ltx2_fixture::WriteFixture(fixture);
  }
  // `VLLM_KEEP_TEST_ARTIFACTS=1` leaves the workspace on disk. The e2e evidence
  // this phase owes is a FRAME SET and a WAV somebody can open, and a test that
  // deletes them can only ever report numbers about files nobody saw.
  ~Workspace() {
    const char* keep = std::getenv("VLLM_KEEP_TEST_ARTIFACTS");
    if (keep != nullptr && keep[0] == '1') {
      std::printf("[ltx2] kept workspace: %s\n", root.c_str());
      return;
    }
    const int rc = std::system(("rm -rf '" + root + "'").c_str());
    (void)rc;
  }
};

vllm::multimodal::VideoModelParams FixtureParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp;
  mp.dit_path = paths.dit;
  mp.video_vae_path = paths.video_vae;
  mp.audio_vae_path = paths.audio_vae;
  mp.prompt_embeds_path = paths.video_embeds;
  mp.extras[vllm::multimodal::kLtx2AudioPromptEmbedsExtra] = paths.audio_embeds;
  mp.device = 0;
  return mp;
}

// The smallest request the fixture's own scale factors admit: (8, 32, 32) means
// 64x64 pixels is a 2x2 latent and 9 frames is 2 latent frames.
vllm::multimodal::VideoGenParams FixtureGen(const std::string& out_dir) {
  vllm::multimodal::VideoGenParams gen;
  gen.num_frames = 9;
  gen.height = 64;
  gen.width = 64;
  gen.has_seed = true;
  gen.seed = 7;
  gen.output_dir = out_dir;
  return gen;
}

std::string ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open ", path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool Registered(const std::string& family) {
  const std::vector<std::string> all = vllm::multimodal::RegisteredVideoFamilies();
  return std::find(all.begin(), all.end(), family) != all.end();
}

// A binary PPM's declared width/height, so the test reads the FILE rather than
// trusting the result struct's own numbers about it.
void ParsePpmHeader(const std::string& bytes, int* width, int* height, size_t* payload_at) {
  REQUIRE(bytes.size() > 2);
  REQUIRE(bytes.compare(0, 2, "P6") == 0);
  size_t at = 2;
  int values[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    while (at < bytes.size() && (std::isspace(static_cast<unsigned char>(bytes[at])) != 0)) ++at;
    int v = 0;
    while (at < bytes.size() && (std::isdigit(static_cast<unsigned char>(bytes[at])) != 0)) {
      v = v * 10 + (bytes[at] - '0');
      ++at;
    }
    values[i] = v;
  }
  ++at;  // the single whitespace byte before the payload
  *width = values[0];
  *height = values[1];
  *payload_at = at;
}

}  // namespace

// ─── registration and detection ─────────────────────────────────────────────

TEST_CASE("ltx2 video: the family self-registers under its stable name") {
  CHECK(Registered(vllm::multimodal::kLtx2VideoFamily));
  CHECK(Registered("minimax-h3"));
  const std::vector<std::string> all = vllm::multimodal::RegisteredVideoFamilies();
  CHECK(std::is_sorted(all.begin(), all.end()));
  CHECK(std::adjacent_find(all.begin(), all.end()) == all.end());
}

TEST_CASE("ltx2 video: detection resolves the checkpoint by what it HOLDS") {
  Workspace ws;
  const std::vector<std::string> got =
      vllm::multimodal::DetectVideoFamilies(FixtureParams(ws.paths));
  REQUIRE(got.size() == 1);
  CHECK(got[0] == vllm::multimodal::kLtx2VideoFamily);

  // The VAE is a perfectly good safetensors file carrying no DiT signature. If
  // it claimed a family, "there is only one family so it must be that one" would
  // be back by another door.
  vllm::multimodal::VideoModelParams not_a_dit = FixtureParams(ws.paths);
  not_a_dit.dit_path = ws.paths.video_vae;
  CHECK(vllm::multimodal::DetectVideoFamilies(not_a_dit).empty());

  // And it must be found WITHOUT the ComfyUI prefix, because which prefix a
  // re-export kept is the repackager's choice and says nothing about the model.
  SUBCASE("a de-prefixed re-export is still detected") {
    const std::string stripped = ws.root + "/stripped.safetensors";
    // Rewrite the header with the prefix removed, payload untouched.
    const std::string bytes = ReadAll(ws.paths.dit);
    uint64_t n = 0;
    std::memcpy(&n, bytes.data(), sizeof(n));
    std::string header = bytes.substr(sizeof(n), n);
    const std::string prefix = vllm::kLtx2DitCheckpointPrefix;
    for (size_t at = header.find(prefix); at != std::string::npos;
         at = header.find(prefix, at)) {
      header.erase(at, prefix.size());
    }
    header.append(prefix.size() * 0, ' ');
    std::string rebuilt;
    const uint64_t m = header.size();
    rebuilt.append(reinterpret_cast<const char*>(&m), sizeof(m));
    rebuilt += header;
    rebuilt += bytes.substr(sizeof(n) + n);
    std::ofstream out(stripped, std::ios::binary);
    out.write(rebuilt.data(), static_cast<std::streamsize>(rebuilt.size()));
    out.close();

    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.dit_path = stripped;
    const std::vector<std::string> claimed = vllm::multimodal::DetectVideoFamilies(mp);
    REQUIRE(claimed.size() == 1);
    CHECK(claimed[0] == vllm::multimodal::kLtx2VideoFamily);
  }
}

// ─── the composition gate ───────────────────────────────────────────────────

TEST_CASE("ltx2 video: an auto-detected load renders frames, a WAV and a mux argv") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  // Stop after phase 0: the recipe's second phase needs the latent spatial
  // upsampler, and running without one is a REFUSAL (gated in its own case
  // below), not a silently shorter render.
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  CHECK(engine->family() == vllm::multimodal::kLtx2VideoFamily);
  // The text tower is OWED, so this must read false rather than "we have an
  // encoder_path field".
  CHECK(!engine->has_encoder());
  CHECK(engine->has_prompt_embeds());

  const std::string out_dir = ws.root + "/out";
  const vllm::multimodal::VideoResult result = engine->Generate(FixtureGen(out_dir));

  // Phase 0 of the distilled two-stage recipe runs at HALF the requested size
  // (spatial_downscale 2), which is upstream's own stage-1 geometry — so a
  // 64x64 request decodes a 32x32 clip here. Asserting the halved size is the
  // point: a port that ignored `spatial_downscale` would return 64x64 and look
  // more correct while sampling the wrong stage.
  CHECK(result.frame_count == 9);
  CHECK(result.width == 32);
  CHECK(result.height == 32);
  CHECK(result.fps == 24);
  CHECK(result.sample_rate == 48000);
  CHECK(result.frame_dir == out_dir);
  CHECK(result.mux_output_path == out_dir + "/video.mp4");

  // Every frame exists, at the size the RESULT claims, read off the file.
  //
  // AND CARRIES MORE THAN ONE BYTE VALUE. That second assertion is the one that
  // earns its place: the PPM writer clamps, and `std::max(-1.0, NaN)` returns
  // -1.0, so an ALL-NaN decode serializes as a perfectly well-formed, uniformly
  // black frame of exactly the right size. Every size/geometry check above would
  // pass over it. This is a floor, not a quality claim — it separates "the
  // pipeline produced something" from "the pipeline produced NaN", and nothing
  // more.
  for (int64_t f = 0; f < result.frame_count; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    const std::string bytes = ReadAll(out_dir + name);
    int w = 0, h = 0;
    size_t at = 0;
    ParsePpmHeader(bytes, &w, &h, &at);
    INFO("frame ", f);
    CHECK(w == static_cast<int>(result.width));
    CHECK(h == static_cast<int>(result.height));
    CHECK(bytes.size() == at + static_cast<size_t>(w) * h * 3);
    size_t distinct = 0;
    bool seen[256] = {false};
    for (size_t i = at; i < bytes.size(); ++i) {
      const unsigned char v = static_cast<unsigned char>(bytes[i]);
      if (!seen[v]) {
        seen[v] = true;
        ++distinct;
      }
    }
    CHECK_MESSAGE(distinct > 1, "frame ", f, " is a single flat value (",
                  static_cast<int>(static_cast<unsigned char>(bytes[at])),
                  "), which is what an all-NaN decode serializes as");
  }
  // One more frame than the clip must NOT exist.
  {
    char extra[64];
    std::snprintf(extra, sizeof(extra), "/frame_%06lld.ppm",
                  static_cast<long long>(result.frame_count));
    std::ifstream beyond(out_dir + extra, std::ios::binary);
    CHECK_MESSAGE(!beyond.good(), "the render produced more frames than it reported");
  }

  // The WAV: RIFF, 16-bit PCM, stereo, at the vocoder's OUTPUT rate, and long
  // enough to be the clip's own duration rather than a stub. 9 frames at 24 fps
  // is 0.375 s; the causal audio decoder trims 3 mel frames, so the waveform is
  // shorter than the nominal duration by a known amount, and the assertion is a
  // BAND rather than an equality for exactly that reason.
  const std::string wav = ReadAll(result.audio_path);
  REQUIRE(wav.size() > 44);
  CHECK(wav.compare(0, 4, "RIFF") == 0);
  CHECK(wav.compare(8, 4, "WAVE") == 0);
  uint16_t channels = 0;
  uint32_t rate = 0;
  std::memcpy(&channels, wav.data() + 22, sizeof(channels));
  std::memcpy(&rate, wav.data() + 24, sizeof(rate));
  CHECK(channels == 2);
  CHECK(rate == 48000u);
  const double seconds =
      static_cast<double>(wav.size() - 44) / (2.0 * 2.0 * static_cast<double>(rate));
  INFO("wav seconds = " << seconds);
  CHECK(seconds > 0.2);
  CHECK(seconds < 0.4);
  // The same floor as the frames': a NaN waveform serializes as digital silence
  // through the WAV writer's own cast, and silence is a valid RIFF file of
  // exactly the right length.
  int64_t nonzero = 0;
  for (size_t i = 44; i + 1 < wav.size(); i += 2) {
    int16_t sample = 0;
    std::memcpy(&sample, wav.data() + i, sizeof(sample));
    if (sample != 0) ++nonzero;
  }
  CHECK_MESSAGE(nonzero > 0, "the waveform is digital silence, which is what a NaN decode writes");

  // The mux argv the CALLER execs. The library spawns nothing, so what is gated
  // is that the argv names the artifacts that were actually written.
  REQUIRE(!result.mux_argv.empty());
  CHECK(result.mux_argv[0] == "ffmpeg");
  std::string joined;
  for (const std::string& a : result.mux_argv) joined += a + " ";
  CHECK(joined.find(out_dir + "/frame_%06d.ppm") != std::string::npos);
  CHECK(joined.find(result.audio_path) != std::string::npos);
  CHECK(joined.find(result.mux_output_path) != std::string::npos);
}

TEST_CASE("ltx2 video: the second phase upsamples, and refuses when it cannot") {
  Workspace ws;
  SUBCASE("without an upsampler the phase is refused BY NAME, not skipped") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    try {
      (void)engine->Generate(FixtureGen(ws.root + "/no_ups"));
      FAIL("a missing spatial upsampler must be refused, not skipped");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("upsampler_path") != std::string::npos);
      CHECK(msg.find("refine") != std::string::npos);
    }
  }
  SUBCASE("with one, the render lands at the FULL requested size") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    const std::string out_dir = ws.root + "/two_stage";
    const vllm::multimodal::VideoResult result = engine->Generate(FixtureGen(out_dir));
    CHECK(result.width == 64);
    CHECK(result.height == 64);
    CHECK(result.frame_count == 9);
  }
}

// ─── the refusals, each of which would otherwise RENDER ─────────────────────

// PHASE L8 CHANGED WHAT THIS CASE ASSERTS, and the change is the phase.
//
// L7 had to refuse `device = 1` outright: the forward was f32-only by
// declaration and the staging was bf16 and refused to widen, so no combination
// put the DiT on an accelerator. L8 is the device-resident forward that closes
// that (`Ltx2DitForwardDevice`), so a CUDA handle now denotes a CUDA forward and
// the load must SUCCEED where a CUDA backend exists.
//
// What must never come back is the substitution: on a build with no CUDA backend
// the load is still refused, and the refusal must name the missing BACKEND. If it
// ever again names the f32/bf16 gap, the device forward has been un-wired; if it
// silently succeeds with a CPU device, the engine is lying about where it ran.
TEST_CASE("ltx2 video: device 1 runs on CUDA, and is refused by name without it") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.device = 1;
  vt::Backend* cuda = vt::TryGetBackend(vt::DeviceType::kCUDA);
  if (cuda == nullptr) {
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("a CUDA load must be refused when no CUDA backend is registered");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("no CUDA backend") != std::string::npos);
      // The L7 gap must NOT be what is named any more; naming it would mean the
      // device forward is no longer wired in.
      CHECK(msg.find("kF32") == std::string::npos);
    }
    return;
  }
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  // The handle means what it says: a CUDA device, not a CPU one behind it.
  CHECK(engine->device().type == vt::DeviceType::kCUDA);
  CHECK(engine->family() == std::string(vllm::multimodal::kLtx2VideoFamily));
}

TEST_CASE("ltx2 video: a prompt with no text tower is refused, never quietly ignored") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(FixtureParams(ws.paths));
  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/prompted");
  gen.prompt = "a cat riding a bicycle";
  try {
    (void)engine->Generate(gen);
    FAIL("a prompt that cannot be encoded must be refused, not replaced by the embeds");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("text tower") != std::string::npos);
  }

  // And supplying an encoder_path is refused at LOAD, where the caller can still
  // act on it, rather than at every request.
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.encoder_path = ws.paths.video_vae;  // any path: the refusal is unconditional
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("encoder_path must be refused while the Gemma-4 tower is owed");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("Gemma-4") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: one prompt-embeds file alone leaves a stream unconditioned") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras.erase(vllm::multimodal::kLtx2AudioPromptEmbedsExtra);
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("the video embeds alone must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("TWO streams") != std::string::npos);
  }

  // Two files whose ROW COUNTS disagree are two different prompts.
  SUBCASE("mismatched row counts are two prompts, and are refused") {
    const std::string short_audio = ws.root + "/short_audio.f32";
    ltx2_fixture::WritePromptEmbeds(short_audio, "ltx2.embeds.audio.short", 2,
                                    ltx2_fixture::ReducedDitParams().audio_cross_attention_dim);
    vllm::multimodal::VideoModelParams two = FixtureParams(ws.paths);
    two.extras[vllm::multimodal::kLtx2AudioPromptEmbedsExtra] = short_audio;
    try {
      (void)vllm::multimodal::LoadVideoEngine(two);
      FAIL("two prompt-embeds files of different lengths must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("ONE tokenization") != std::string::npos);
    }
  }
}

TEST_CASE("ltx2 video: an unknown extra is refused, not ignored") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras["partition"] = "fl2va";  // H3's knob, meaningless here
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("an extra this family does not define must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("partition") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: the recipe comes from the CHECKPOINT's own model_version") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> seam =
      vllm::multimodal::LoadVideoEngine(FixtureParams(ws.paths));
  const auto* engine = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(seam.get());
  REQUIRE(engine != nullptr);
  // "2.5.0" in the file, reduced to the table's two-component key.
  CHECK(engine->model_version() == "2.5");
  CHECK(engine->pipeline_kind() == "distilled_two_stage");

  SUBCASE("a checkpoint of another generation is refused, never defaulted onto 2.5") {
    const std::string other = ws.root + "/v2_3.safetensors";
    ltx2_fixture::WriteReducedDit(ltx2_fixture::ReducedDitParams(), other, "2.3.0");
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.dit_path = other;
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("distilled_two_stage/2.3 is not in the recipe table and must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("2.3") != std::string::npos);
    }
  }

  SUBCASE("a declared version and an overriding extra that disagree are refused") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2ModelVersionExtra] = "2.3";
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("two disagreeing model versions must be refused, not silently ordered");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("DIFFERENT sigma schedules") != std::string::npos);
    }
  }
}

TEST_CASE("ltx2 video: keyframe and reference conditioning is refused by name") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(FixtureParams(ws.paths));
  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/keyframed");
  gen.first_frame_path = ws.paths.video_embeds;  // any path: the refusal precedes the read
  try {
    (void)engine->Generate(gen);
    FAIL("keyframe conditioning must be refused while the VAE encoder is unported");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("ImageConditioner") != std::string::npos);
  }
}


// ─── the floor under everything above ───────────────────────────────────────
//
// THIS CASE EARNED ITS PLACE BY FIRING. The fixture's first FP8 encoder let a
// value below 2^-6 fall through its subnormal branch into the normal one with a
// mantissa under 1.0; the resulting negative 3-bit fraction OR'd into the byte,
// and 0xFF is one of E4M3's two NaNs. 460 of 40,452 dequantized weights came
// back NaN, the DiT emitted NaN, and every geometry assertion above still
// passed: the PPM writer clamps, and `std::max(-1.0, NaN)` is -1.0, so the
// render serialized as nine perfectly well-formed uniformly black frames.
//
// It is a gate on the FIXTURE as much as on the loader, and that is the point —
// a fixture that cannot tell a right implementation from a wrong one is the same
// defect wearing different clothes.
TEST_CASE("ltx2 video: the loaded DiT and its forward are finite, weight by weight") {
  Workspace ws;
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(ws.paths.dit);
  vllm::Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;
  const vllm::Ltx2DitCheckpoint checkpoint =
      vllm::Ltx2LoadDitFromSafetensors(file, options);

  int64_t total = 0, nans = 0;
  for (const auto& entry : checkpoint.views) {
    const vt::Tensor& view = entry.second;
    const float* values = view.Ptr<float>();
    for (int64_t i = 0; i < view.Numel(); ++i) {
      ++total;
      if (!(values[i] == values[i])) ++nans;
    }
  }
  INFO("dequantized weights: " << total);
  REQUIRE(total > 0);
  CHECK_MESSAGE(nans == 0, nans, " of ", total, " dequantized DiT weights are NaN");

  // And the forward over them, at the geometry the engine drives it with.
  const vllm::Ltx2DitParams& params = checkpoint.params;
  const int64_t video_tokens = 2, audio_tokens = 4, context_tokens = 3;
  std::vector<float> video_latent(video_tokens * params.in_channels, 0.1F);
  std::vector<float> audio_latent(audio_tokens * params.audio_in_channels, 0.1F);
  std::vector<float> video_timesteps(video_tokens, 1.0F);
  std::vector<float> audio_timesteps(audio_tokens, 1.0F);
  float sigma = 1.0F;
  std::vector<double> video_positions(3 * video_tokens * 2);
  std::vector<double> audio_positions(audio_tokens * 2);
  for (int64_t d = 0; d < 3; ++d) {
    for (int64_t t = 0; t < video_tokens; ++t) {
      video_positions[static_cast<size_t>((d * video_tokens + t) * 2)] = static_cast<double>(t);
      video_positions[static_cast<size_t>((d * video_tokens + t) * 2 + 1)] =
          static_cast<double>(t + 1);
    }
  }
  for (int64_t t = 0; t < audio_tokens; ++t) {
    audio_positions[static_cast<size_t>(t * 2)] = static_cast<double>(t);
    audio_positions[static_cast<size_t>(t * 2 + 1)] = static_cast<double>(t + 1);
  }
  std::vector<float> video_context(context_tokens * params.cross_attention_dim, 0.05F);
  std::vector<float> audio_context(context_tokens * params.audio_cross_attention_dim, 0.05F);

  vllm::Ltx2ModalityInput video;
  video.tokens = video_tokens;
  video.context_tokens = context_tokens;
  video.latent = video_latent.data();
  video.timesteps = video_timesteps.data();
  video.sigma = &sigma;
  video.positions = video_positions.data();
  video.context = video_context.data();
  vllm::Ltx2ModalityInput audio;
  audio.tokens = audio_tokens;
  audio.context_tokens = context_tokens;
  audio.latent = audio_latent.data();
  audio.timesteps = audio_timesteps.data();
  audio.sigma = &sigma;
  audio.positions = audio_positions.data();
  audio.context = audio_context.data();

  const vllm::Ltx2DitOutputs out = vllm::Ltx2DitForward(
      vt::Device{}, params, checkpoint.weights, &video, &audio, vt::DType::kF32);
  REQUIRE(out.video.size() == static_cast<size_t>(video_tokens * params.out_channels));
  REQUIRE(out.audio.size() == static_cast<size_t>(audio_tokens * params.audio_out_channels));
  int64_t video_nans = 0, audio_nans = 0;
  for (const float v : out.video) {
    if (!(v == v)) ++video_nans;
  }
  for (const float v : out.audio) {
    if (!(v == v)) ++audio_nans;
  }
  CHECK(video_nans == 0);
  CHECK(audio_nans == 0);
}

// ─── reachable through include/vllm.h, which is the actual "done" bar ───────
//
// AGENTS.md §"Shared seams": a capability that is not reachable through the
// shared surface is not done, and for this project the shared surface is the C
// ABI. L1 made the video slice family-generic at ABI v18 (`family`,
// `extra_keys` / `extra_values`, `vllm_video_engine_family`), so registering a
// family is supposed to be all it takes — this asserts that it IS, by driving a
// whole generation through the ABI and nothing else.
TEST_CASE("ltx2 video: an ABI client loads, detects and generates through vllm.h") {
  Workspace ws;
  const std::string audio_embeds = ws.paths.audio_embeds;
  const std::string max_phase = "0";
  const char* keys[] = {vllm::multimodal::kLtx2AudioPromptEmbedsExtra,
                        vllm::multimodal::kLtx2MaxPhaseExtra};
  const char* values[] = {audio_embeds.c_str(), max_phase.c_str()};

  vllm_video_model_params mp = vllm_video_model_params_default();
  mp.dit_path = ws.paths.dit.c_str();
  mp.video_vae_path = ws.paths.video_vae.c_str();
  mp.audio_vae_path = ws.paths.audio_vae.c_str();
  mp.prompt_embeds_path = ws.paths.video_embeds.c_str();
  mp.extra_keys = keys;
  mp.extra_values = values;
  mp.n_extras = 2;
  mp.device = 0;  // the family refuses 1 by name; the ABI carries the refusal

  vllm_video_engine* engine = nullptr;
  const vllm_status loaded = vllm_video_engine_load(&mp, &engine);
  const std::string load_error = vllm_last_error() == nullptr ? "" : vllm_last_error();
  INFO(load_error);
  REQUIRE(loaded == VLLM_OK);
  REQUIRE(engine != nullptr);
  // Detection ran through the ABI, with no `family` declared.
  REQUIRE(vllm_video_engine_family(engine) != nullptr);
  CHECK(std::string(vllm_video_engine_family(engine)) == vllm::multimodal::kLtx2VideoFamily);

  const std::string out_dir = ws.root + "/abi_out";
  vllm_video_params gen = vllm_video_params_default();
  gen.width = 64;
  gen.height = 64;
  gen.num_frames = 9;
  gen.seed = 7;
  gen.has_seed = 1;
  gen.output_dir = out_dir.c_str();

  vllm_video_result result;
  std::memset(&result, 0, sizeof(result));
  const vllm_status generated = vllm_video_generate(engine, &gen, &result);
  const std::string gen_error = vllm_last_error() == nullptr ? "" : vllm_last_error();
  INFO(gen_error);
  REQUIRE(generated == VLLM_OK);
  CHECK(result.frame_count == 9);
  CHECK(result.width == 32);
  CHECK(result.height == 32);
  CHECK(result.sample_rate == 48000);
  REQUIRE(result.mux_argc > 0);
  REQUIRE(result.mux_argv != nullptr);
  CHECK(std::string(result.mux_argv[0]) == "ffmpeg");
  // execvp-ready: the argv is NULL-terminated past mux_argc.
  CHECK(result.mux_argv[result.mux_argc] == nullptr);
  const std::string first_frame = out_dir + "/frame_000000.ppm";
  std::ifstream frame(first_frame, std::ios::binary);
  CHECK_MESSAGE(frame.good(), "the ABI render wrote no ", first_frame);

  // The artifact INVENTORY, printed rather than only asserted: a phase whose
  // deliverable is "frames + a WAV + the mux argv the caller execs" owes the
  // reader those three things, not a claim that they were checked.
  std::string joined;
  for (int32_t i = 0; i < result.mux_argc; ++i) {
    joined += (i == 0 ? "" : " ") + std::string(result.mux_argv[i]);
  }
  const std::string wav_bytes = ReadAll(result.audio_path);
  const double wav_seconds =
      static_cast<double>(wav_bytes.size() - 44) / (2.0 * 2.0 * static_cast<double>(result.sample_rate));
  MESSAGE("ABI artifacts: family=" << std::string(vllm_video_engine_family(engine)) << " frames="
          << result.frame_count << " " << result.width << "x" << result.height << " fps="
          << result.fps << " wav=" << wav_bytes.size() << "B " << wav_seconds << "s @"
          << result.sample_rate << "Hz dir=" << std::string(result.frame_dir));
  MESSAGE("ABI mux argv: " << joined);

  vllm_video_result_free(&result);
  vllm_video_engine_free(engine);
}

// ─── the SHIPPED checkpoints, when the box has them ─────────────────────────
//
// Everything above runs over a reduced fixture, which proves the composition and
// says nothing about the 23 GB of bytes Lightricks actually ships. This case
// reads those bytes when `LTX2_CHECKPOINT_ROOT` points at them, and is SKIPPED
// otherwise rather than silently passing: a gate that quietly does nothing when
// its input is absent is how "we tested the real checkpoint" becomes untrue.
//
// It deliberately stops short of the DiT FORWARD. At the shipped 21.00B geometry
// the f32 parity forward needs ~76 GB of weights and ~2.6e14 FLOPs per step, and
// the device path cannot feed it (see the `device = 1` refusal). What is checked
// is everything before that line: that the first-party NVFP4 DiT's header
// resolves onto the L2 contract, and that BOTH VAEs and the upsampler load and
// configure from their own embedded metadata.
//
// WHICH DiT: `Lightricks/LTX-2.5` `ltx-2.5-22b-distilled-transformer-nvfp4`.
// It is NOT interchangeable with the `vonkaiser` FP8 copy — they differ in a
// TRAINED `keyframes_abs_pos_embedding` (spec section 3.1) — so the file this
// case reads is named here and in every report of its result.
TEST_CASE("ltx2 video: the SHIPPED Lightricks checkpoints parse and load") {
  const char* root_env = std::getenv("LTX2_CHECKPOINT_ROOT");
  if (root_env == nullptr) {
    MESSAGE("SKIPPED: set LTX2_CHECKPOINT_ROOT to the Lightricks/LTX-2.5 tree to run this");
    return;
  }
  const std::string root = root_env;

  SUBCASE("the first-party NVFP4 DiT resolves onto the L2 contract") {
    const std::string path =
        root + "/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    vllm::Ltx2DitQuant quant = vllm::Ltx2DitQuant::kFp8;
    vllm::Ltx2DitParams from_shapes = vllm::Ltx2ParseDitParamsFromCheckpoint(file, &quant);
    // The manifest parser leaves `use_prompt_adaln_single` at its default; the
    // LOADER clears it for the contract (ltx2_loader.cpp), and so does the
    // engine. Mirror that here so the two contracts are compared like for like.
    from_shapes.use_prompt_adaln_single = false;
    CHECK(quant == vllm::Ltx2DitQuant::kNvfp4);
    CHECK(from_shapes.num_layers == 48);
    CHECK(from_shapes.inner_dim() == 4096);
    CHECK(from_shapes.audio_inner_dim() == 2048);
    CHECK(from_shapes.in_channels == 128);
    CHECK(from_shapes.audio_in_channels == 128);

    // The declared config, which is what the engine adopts and what the SHAPES
    // cannot see. The two must agree on the weight contract.
    CHECK(vllm::Ltx2ReadCheckpointModelVersion(file) == "2.5.0");
    nlohmann::json config = vllm::Ltx2ReadCheckpointConfig(file);
    // The shipped DiT DECLARES `use_keyframes_abs_pos_embedding: true`, which
    // `ParseLtx2DitParams` refuses by name because the module is unported. The
    // engine clears it in a copy under `allow_unported_modules`; this mirrors
    // that, and asserting the file declares it is the point.
    REQUIRE(config["transformer"]["use_keyframes_abs_pos_embedding"].get<bool>());
    config["transformer"]["use_keyframes_abs_pos_embedding"] = false;
    nlohmann::json wrapper;
    wrapper["config"] = config;
    vllm::Ltx2DitParams declared = vllm::ParseLtx2DitParams(wrapper);
    declared.use_prompt_adaln_single = false;
    CHECK(declared.double_precision_rope);              // frequencies_precision float64
    CHECK(declared.av_ca_timestep_scale_multiplier == 1000);
    CHECK(declared.apply_gated_attention);
    CHECK(declared.cross_attention_adaln);
    CHECK(!declared.ff_bias);
    const std::vector<vllm::Ltx2TensorSpec> a = vllm::EnumerateLtx2DitTensors(from_shapes);
    const std::vector<vllm::Ltx2TensorSpec> b = vllm::EnumerateLtx2DitTensors(declared);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      INFO("tensor " << i);
      CHECK(a[i].name == b[i].name);
      CHECK(a[i].shape == b[i].shape);
    }
    MESSAGE("shipped NVFP4 DiT: " << a.size() << " contract tensors, "
            << file.Names().size() << " in the file");
  }

  SUBCASE("the Conv video VAE loads and configures from its own metadata") {
    const std::string path = root + "/vae/ltx-2.5-video-vae-conv-bf16.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    vllm::Ltx2VideoDecoderKind kind = vllm::Ltx2VideoDecoderKind::kDiffusion;
    const vllm::Ltx2ConvVideoDecoderConfig cfg =
        vllm::Ltx2ParseConvVideoDecoderConfig(vllm::Ltx2ReadCheckpointConfig(file), &kind);
    CHECK(kind == vllm::Ltx2VideoDecoderKind::kConv);
    CHECK(cfg.in_channels == 128);
    CHECK(cfg.out_channels == 3);
    CHECK(cfg.patch_size == 4);
    CHECK(cfg.base_channels == 128);
    CHECK(!cfg.timestep_conditioning);
    CHECK(cfg.decoder_blocks.size() == 9);
    // The block list has to multiply out to VIDEO_SCALE_FACTORS, because that
    // constant is what the pipeline derives every latent shape from.
    // `multiplier` is the CHANNEL reduction, not the stride: the resample factor
    // is fixed per block KIND (ltx2_video_vae.cpp:662-668 — compress_space is a
    // 2x2 spatial stride, compress_time a 2x temporal one, compress_all both).
    // Reading `multiplier` as the stride gives 16 and 4 here, which is what this
    // check first did, and which would have made every latent shape wrong by 2x.
    int64_t spatial = cfg.patch_size, temporal = 1;
    for (const vllm::Ltx2VideoDecoderBlock& b : cfg.decoder_blocks) {
      if (b.name == "compress_space") spatial *= 2;
      if (b.name == "compress_time") temporal *= 2;
      if (b.name == "compress_all") {
        spatial *= 2;
        temporal *= 2;
      }
    }
    CHECK(spatial == 32);
    CHECK(temporal == 8);
    const vllm::Ltx2VaeWeights weights =
        vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2VideoVaeDecoderKeyRules());
    CHECK(weights.Has("conv_in.conv.weight"));
    CHECK(weights.Has("conv_out.conv.weight"));
    CHECK(weights.Has("per_channel_statistics.std-of-means"));
    // The encoder half of the file is DROPPED by the key rules, exactly as
    // upstream's SDOps drop it.
    CHECK(!weights.Has("encoder.conv_in.conv.weight"));
    MESSAGE("shipped conv video VAE: " << weights.tensors.size() << " decoder tensors");
  }

  SUBCASE("the audio VAE and its BWE vocoder load and configure") {
    const std::string path = root + "/vae/ltx-2.5-audio-vae-bf16.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    const nlohmann::json config = vllm::Ltx2ReadCheckpointConfig(file);
    const vllm::Ltx2AudioDecoderConfig audio = vllm::Ltx2ParseAudioDecoderConfig(config);
    CHECK(audio.z_channels == 8);   // == the audio latent's channel count
    CHECK(audio.mel_bins == 64);    // == the vocoder's hardcoded 128 / 2 channels
    CHECK(audio.out_ch == 2);
    CHECK(!audio.mid_block_add_attention);
    const vllm::Ltx2VocoderBweConfig voc = vllm::Ltx2ParseVocoderBweConfig(config);
    CHECK(voc.input_sampling_rate == 16000);
    CHECK(voc.output_sampling_rate == 48000);
    CHECK(voc.hop_length == 80);
    CHECK(voc.n_mel_channels == 64);
    CHECK(voc.vocoder.amp);
    CHECK(voc.vocoder.snakebeta);
    CHECK(voc.vocoder.apply_final_activation);
    CHECK(!voc.bwe_generator.apply_final_activation);
    const vllm::Ltx2VaeWeights decoder =
        vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2AudioVaeDecoderKeyRules());
    const vllm::Ltx2VaeWeights vocoder =
        vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2VocoderKeyRules());
    CHECK(decoder.Has("conv_in.conv.weight"));
    CHECK(decoder.Has("per_channel_statistics.std-of-means"));
    CHECK(vocoder.Has("vocoder.conv_pre.weight"));
    CHECK(vocoder.Has("bwe_generator.conv_pre.weight"));
    CHECK(vocoder.Has("mel_stft.mel_basis"));
    MESSAGE("shipped audio VAE: " << decoder.tensors.size() << " decoder + "
            << vocoder.tensors.size() << " vocoder tensors");
  }

  SUBCASE("the latent spatial x2 upsampler loads and configures") {
    const std::string path =
        root + "/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    const vllm::Ltx2UpsamplerConfig cfg =
        vllm::Ltx2ParseUpsamplerConfig(vllm::Ltx2ReadCheckpointConfig(file));
    CHECK(cfg.in_channels == 128);
    CHECK(cfg.mid_channels == 1024);
    CHECK(cfg.spatial_upsample);
    CHECK(!cfg.temporal_upsample);
    CHECK(cfg.spatial_scale == doctest::Approx(2.0).scale(0.0));
    const vllm::Ltx2VaeWeights weights = vllm::Ltx2LoadVaeWeights(file);
    // Every tensor the contract asks for is in the file, by NAME.
    int64_t missing = 0;
    std::string first_missing;
    for (const vllm::Ltx2UpsamplerTensorSpec& spec :
         vllm::EnumerateLtx2UpsamplerTensors(cfg)) {
      if (!weights.Has(spec.name)) {
        ++missing;
        if (first_missing.empty()) first_missing = spec.name;
      }
    }
    INFO("first missing: " << first_missing);
    CHECK(missing == 0);
    MESSAGE("shipped upsampler: " << weights.tensors.size() << " tensors");
  }
}
