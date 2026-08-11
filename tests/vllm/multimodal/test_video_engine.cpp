// LTX-2.5 phase L1 gate — the GENERALIZED video seam
// (vllm::multimodal::VideoEngine + its checkpoint-detected family registry),
// with MiniMax-H3 moved behind it and NOTHING about H3 changed.
//
// The byte-identity contract is what makes "moved behind a seam" a fact rather
// than a claim: a render driven through the ABSTRACT seam
// (LoadVideoEngine -> VideoEngine::Generate) must reproduce the SAME committed
// pre-fold goldens (fixtures/minimax_h3_video_fold/) that the concrete
// MiniMaxH3VideoEngine arm is held to in
// tests/vllm/models/test_minimax_h3_video_fold.cpp. Those two tests are
// deliberately separate binaries over ONE golden set: the concrete arm proves
// H3 did not move, this one proves the generic path lands on the same bytes.
//
// The registry's refusals are gated as hard as its successes. AGENTS.md's
// "an arm that is not implemented is refused with a message naming the missing
// piece" applies to family resolution too: an unknown family name and an
// unrecognizable checkpoint must both throw NAMING what was asked/seen and what
// is registered, never fall through to "the only family we have".
//
// Spec: .agents/specs/ltx-2-5.md §5 (design, the generalized seam) + §6 (L1).
// Issue: #435.
#include "vllm/multimodal/video_engine.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "minimax_h3_video_fold_fixture.h"
#include "vllm/multimodal/minimax_h3_video.h"

namespace {

std::string ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open ", path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string GoldenDir() { return MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR; }

// The pre-fold golden clip length (latent_t 2 x the ViT3D decoder's patch_size_t
// 4) — pinned so a render of a DIFFERENT length cannot pass the per-frame loop.
constexpr int kGoldenFrames = 8;

struct SeamWorkspace {
  std::string root, fixture;
  SeamWorkspace() {
    static int counter = 0;
    root = "/tmp/vllm_video_engine_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++);
    ::mkdir(root.c_str(), 0755);
    fixture = root + "/fixture";
    minimax_h3_fold::WriteFoldFixture(fixture);
  }
  ~SeamWorkspace() {
    const int rc = std::system(("rm -rf '" + root + "'").c_str());
    (void)rc;
  }
};

// The SAME checkpoint set the concrete-arm fold gate loads, expressed in the
// GENERIC params — the H3-specific `partition` riding in `extras`.
vllm::multimodal::VideoModelParams FixtureParams(const std::string& dir) {
  vllm::multimodal::VideoModelParams mp;
  mp.dit_path = dir + "/dit.gguf";
  mp.video_vae_path = dir + "/video_vae.safetensors";
  mp.video_vae_config_path = dir + "/video_vae_config.json";
  mp.audio_vae_path = dir + "/audio_vae.safetensors";
  mp.audio_vae_config_path = dir + "/audio_vae_config.json";
  mp.prompt_embeds_path = dir + "/prompt_embeds.f32";
  mp.extras["partition"] = "fl2va";
  mp.device = 0;
  mp.dequant_bf16 = 0;
  return mp;
}

vllm::multimodal::VideoGenParams FixtureGen(const std::string& out_dir) {
  vllm::multimodal::VideoGenParams gen;
  gen.num_frames = 5;
  gen.height = 32;
  gen.width = 32;
  gen.steps = 3;
  gen.output_dir = out_dir;
  return gen;
}

void CheckAgainstGoldens(const std::string& out_dir) {
  for (int f = 0; f < kGoldenFrames; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06d.ppm", f);
    const std::string got = ReadAll(out_dir + name);
    const std::string want = ReadAll(GoldenDir() + name);
    INFO("frame ", f);
    REQUIRE(got.size() == want.size());
    CHECK_MESSAGE(got == want, "frame ", f, " diverged from the pre-fold golden");
  }
  const std::string got_wav = ReadAll(out_dir + "/audio.wav");
  const std::string want_wav = ReadAll(GoldenDir() + "/audio.wav");
  REQUIRE(got_wav.size() == want_wav.size());
  CHECK_MESSAGE(got_wav == want_wav, "audio.wav diverged from the pre-fold golden");
  char extra[64];
  std::snprintf(extra, sizeof(extra), "/frame_%06d.ppm", kGoldenFrames);
  std::ifstream ninth(out_dir + extra, std::ios::binary);
  CHECK_MESSAGE(!ninth.good(), "the render produced more frames than the golden clip");
}

bool Registered(const std::string& family) {
  const std::vector<std::string> all = vllm::multimodal::RegisteredVideoFamilies();
  return std::find(all.begin(), all.end(), family) != all.end();
}

}  // namespace

TEST_CASE("video engine registry: minimax-h3 self-registers under a stable family name") {
  CHECK(Registered("minimax-h3"));
  // The listing is the message an unresolved load prints, so it must be sorted
  // and duplicate-free however static init happened to order the TUs.
  const std::vector<std::string> all = vllm::multimodal::RegisteredVideoFamilies();
  CHECK(std::is_sorted(all.begin(), all.end()));
  CHECK(std::adjacent_find(all.begin(), all.end()) == all.end());
}

TEST_CASE("video engine registry: detection resolves the H3 checkpoint by what it HOLDS") {
  SeamWorkspace ws;
  const std::vector<std::string> got =
      vllm::multimodal::DetectVideoFamilies(FixtureParams(ws.fixture));
  REQUIRE(got.size() == 1);
  CHECK(got[0] == "minimax-h3");

  // A checkpoint that is a perfectly good safetensors file but carries NO
  // family's DiT signature (this is the video VAE) resolves to nothing. This is
  // the case a "there is only one family, so it must be that one" fallback
  // would silently mis-load.
  vllm::multimodal::VideoModelParams not_a_dit = FixtureParams(ws.fixture);
  not_a_dit.dit_path = ws.fixture + "/video_vae.safetensors";
  CHECK(vllm::multimodal::DetectVideoFamilies(not_a_dit).empty());
}

// ─── the byte-identity gate: the ABSTRACT seam lands on the pre-fold bytes ───
TEST_CASE("video engine seam: an auto-detected H3 render reproduces the pre-fold goldens") {
  SeamWorkspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(FixtureParams(ws.fixture));
  REQUIRE(engine != nullptr);
  CHECK(engine->family() == "minimax-h3");
  CHECK(!engine->has_encoder());
  CHECK(engine->has_prompt_embeds());

  const std::string out_dir = ws.root + "/generic_out";
  const vllm::multimodal::VideoResult result = engine->Generate(FixtureGen(out_dir));
  CHECK(result.frame_dir == out_dir);
  CHECK(result.frame_count == kGoldenFrames);
  CHECK(result.width == 32);
  CHECK(result.height == 32);
  CHECK(result.fps == 24);
  CHECK(result.sample_rate == 32000);
  CHECK(result.mux_output_path == out_dir + "/video.mp4");
  CheckAgainstGoldens(out_dir);

  // The mux argv the generic seam hands back is the H3 seam's, unchanged.
  std::string joined;
  for (size_t i = 0; i < result.mux_argv.size(); ++i) {
    joined += (i == 0 ? "" : " ") + result.mux_argv[i];
  }
  std::string golden_argv = ReadAll(GoldenDir() + "/golden_mux_argv.txt");
  while (!golden_argv.empty() && (golden_argv.back() == '\n' || golden_argv.back() == '\r')) {
    golden_argv.pop_back();
  }
  size_t pos = 0;
  while ((pos = golden_argv.find("W/", pos)) != std::string::npos) {
    golden_argv.replace(pos, 1, out_dir);
    pos += out_dir.size() + 1;
  }
  CHECK(joined == golden_argv);
}

TEST_CASE("video engine seam: an EXPLICITLY declared family renders the same bytes") {
  SeamWorkspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.fixture);
  mp.family = "minimax-h3";  // skip detection entirely
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  CHECK(engine->family() == "minimax-h3");
  const std::string out_dir = ws.root + "/declared_out";
  (void)engine->Generate(FixtureGen(out_dir));
  CheckAgainstGoldens(out_dir);
}

// ─── the refusals: never guess a family ─────────────────────────────────────
TEST_CASE("video engine registry: an unknown family names what was asked and what exists") {
  SeamWorkspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.fixture);
  mp.family = "ltx-2.5";  // not registered until the LTX phases land
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("an unregistered family must be refused, not detected around");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("ltx-2.5") != std::string::npos);
    CHECK(msg.find("minimax-h3") != std::string::npos);
  }
}

TEST_CASE("video engine registry: an unrecognizable checkpoint refuses instead of guessing") {
  SeamWorkspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.fixture);
  mp.dit_path = ws.fixture + "/video_vae.safetensors";  // real file, wrong role
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("an undetectable checkpoint must be refused, not handed to the only family");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("video_vae.safetensors") != std::string::npos);
    CHECK(msg.find("minimax-h3") != std::string::npos);
  }

  // An empty dit_path is refused with the same never-guess rule.
  vllm::multimodal::VideoModelParams empty = FixtureParams(ws.fixture);
  empty.dit_path.clear();
  CHECK_THROWS(vllm::multimodal::LoadVideoEngine(empty));
}

// ─── extras carry the family-specific knob, in BOTH directions ──────────────
TEST_CASE("video engine seam: H3's partition rides in extras and still guards") {
  SeamWorkspace ws;

  SUBCASE("a ref2va partition declared through extras refuses a t2va render") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.fixture);
    mp.extras["partition"] = "ref2va";
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    try {
      (void)engine->Generate(FixtureGen(ws.root + "/out"));
      FAIL("the ref2va partition must refuse t2va through the generic seam too");
    } catch (const std::exception& e) {
      CHECK(std::string(e.what()).find("ref2va") != std::string::npos);
    }
  }

  SUBCASE("no partition extra is declared-but-unknown, and refuses every render") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.fixture);
    mp.extras.erase("partition");
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    try {
      (void)engine->Generate(FixtureGen(ws.root + "/out"));
      FAIL("an undeclared partition must refuse to render");
    } catch (const std::exception& e) {
      CHECK(std::string(e.what()).find("partition") != std::string::npos);
    }
  }

  SUBCASE("an unknown partition extra is refused at load, exactly as the H3 field is") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.fixture);
    mp.extras["partition"] = "nonsense";
    CHECK_THROWS(vllm::multimodal::LoadVideoEngine(mp));
  }
}

TEST_CASE("video engine seam: the H3 params round-trip through the generic superset") {
  vllm::multimodal::VideoModelParams generic;
  generic.dit_path = "/d";
  generic.encoder_path = "/e";
  generic.tokenizer_path = "/t";
  generic.video_vae_path = "/vv";
  generic.video_vae_config_path = "/vvc";
  generic.audio_vae_path = "/av";
  generic.audio_vae_config_path = "/avc";
  generic.prompt_embeds_path = "/pe";
  generic.device = 1;
  generic.dequant_bf16 = 1;
  generic.fp4_resident = 1;
  generic.encoder_max_layers = 3;
  generic.extras["partition"] = "ref2va";

  const vllm::multimodal::MiniMaxH3VideoModelParams h3 =
      vllm::multimodal::MiniMaxH3VideoModelParamsFromGeneric(generic);
  CHECK(h3.dit_path == "/d");
  CHECK(h3.encoder_path == "/e");
  CHECK(h3.tokenizer_path == "/t");
  CHECK(h3.video_vae_path == "/vv");
  CHECK(h3.video_vae_config_path == "/vvc");
  CHECK(h3.audio_vae_path == "/av");
  CHECK(h3.audio_vae_config_path == "/avc");
  CHECK(h3.prompt_embeds_path == "/pe");
  CHECK(h3.device == 1);
  CHECK(h3.dequant_bf16 == 1);
  CHECK(h3.fp4_resident == 1);
  CHECK(h3.encoder_max_layers == 3);
  CHECK(h3.partition == "ref2va");  // OUT of extras

  const vllm::multimodal::VideoModelParams back =
      vllm::multimodal::MiniMaxH3VideoModelParamsToGeneric(h3);
  CHECK(back.family == "minimax-h3");
  CHECK(back.extras.at("partition") == "ref2va");  // INTO extras
  CHECK(back.dit_path == generic.dit_path);
  CHECK(back.encoder_path == generic.encoder_path);
  CHECK(back.tokenizer_path == generic.tokenizer_path);
  CHECK(back.video_vae_path == generic.video_vae_path);
  CHECK(back.video_vae_config_path == generic.video_vae_config_path);
  CHECK(back.audio_vae_path == generic.audio_vae_path);
  CHECK(back.audio_vae_config_path == generic.audio_vae_config_path);
  CHECK(back.prompt_embeds_path == generic.prompt_embeds_path);
  CHECK(back.device == generic.device);
  CHECK(back.dequant_bf16 == generic.dequant_bf16);
  CHECK(back.fp4_resident == generic.fp4_resident);
  CHECK(back.encoder_max_layers == generic.encoder_max_layers);
}

TEST_CASE("video engine seam: every generic gen field reaches the H3 request") {
  vllm::multimodal::VideoGenParams gen;
  gen.prompt = "a cat";
  gen.task = "ref2va";
  gen.duration_seconds = 2.5;
  gen.num_frames = 29;
  gen.height = 320;
  gen.width = 640;
  gen.steps = 7;
  gen.flow_shift = 11.0;
  gen.audio_flow_shift = 2.5;
  gen.seed = 42;
  gen.has_seed = true;
  gen.first_frame_path = "/x/first.ppm";
  gen.last_frame_path = "/x/last.ppm";
  gen.first_frame_ppm = "P6";
  gen.noise_aug = 0.5;
  gen.ref_image_paths = {"/x/a.ppm", "/x/b.ppm"};
  gen.ref_video_dir = "/x/clip";
  gen.ref_audio_path = "/x/a.wav";
  gen.ref_audio_wav = "RIFF";
  gen.output_dir = "/x/job0";

  const vllm::multimodal::MiniMaxH3VideoGenParams h3 =
      vllm::multimodal::MiniMaxH3VideoGenParamsFromGeneric(gen);
  CHECK(h3.prompt == "a cat");
  CHECK(h3.task == "ref2va");
  CHECK(h3.duration_seconds == 2.5);
  CHECK(h3.num_frames == 29);
  CHECK(h3.height == 320);
  CHECK(h3.width == 640);
  CHECK(h3.steps == 7);
  CHECK(h3.flow_shift == 11.0);
  CHECK(h3.audio_flow_shift == 2.5);
  CHECK(h3.seed == 42);
  CHECK(h3.has_seed);
  CHECK(h3.first_frame_path == "/x/first.ppm");
  CHECK(h3.last_frame_path == "/x/last.ppm");
  CHECK(h3.first_frame_ppm == "P6");
  CHECK(h3.noise_aug == 0.5);
  CHECK(h3.ref_image_paths == std::vector<std::string>{"/x/a.ppm", "/x/b.ppm"});
  CHECK(h3.ref_video_dir == "/x/clip");
  CHECK(h3.ref_audio_path == "/x/a.wav");
  CHECK(h3.ref_audio_wav == "RIFF");
  CHECK(h3.output_dir == "/x/job0");
}

TEST_CASE("video engine seam: the concrete H3 handle IS a VideoEngine") {
  // The concrete entry point every pre-L1 consumer drives still exists, still
  // returns the H3-typed handle, and that handle satisfies the abstract seam —
  // which is what lets the C ABI and /v1/videos migrate without a second path.
  SeamWorkspace ws;
  vllm::multimodal::MiniMaxH3VideoModelParams mp;
  mp.dit_path = ws.fixture + "/dit.gguf";
  mp.video_vae_path = ws.fixture + "/video_vae.safetensors";
  mp.video_vae_config_path = ws.fixture + "/video_vae_config.json";
  mp.audio_vae_path = ws.fixture + "/audio_vae.safetensors";
  mp.audio_vae_config_path = ws.fixture + "/audio_vae_config.json";
  mp.prompt_embeds_path = ws.fixture + "/prompt_embeds.f32";
  mp.partition = "fl2va";

  const std::unique_ptr<vllm::multimodal::MiniMaxH3VideoEngine> concrete =
      vllm::multimodal::MiniMaxH3VideoEngine::Load(mp);
  vllm::multimodal::VideoEngine* seam = concrete.get();
  REQUIRE(seam != nullptr);
  CHECK(seam->family() == "minimax-h3");
  CHECK(seam->device() == concrete->device());
  CHECK(seam->has_prompt_embeds());

  // Both spellings of Generate must land on the same bytes: the H3-typed one
  // (what examples/server drive today) and the generic override.
  const std::string a = ws.root + "/concrete_out";
  vllm::multimodal::MiniMaxH3VideoGenParams h3_gen;
  h3_gen.num_frames = 5;
  h3_gen.height = 32;
  h3_gen.width = 32;
  h3_gen.steps = 3;
  h3_gen.output_dir = a;
  (void)concrete->Generate(h3_gen);
  CheckAgainstGoldens(a);

  const std::string b = ws.root + "/seam_out";
  (void)seam->Generate(FixtureGen(b));
  CheckAgainstGoldens(b);
}
