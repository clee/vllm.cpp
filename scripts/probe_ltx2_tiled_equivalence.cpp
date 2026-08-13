// Is a streamed render the SAME render? — measured on the shipped conv VAE.
//
// Loads the REAL ltx-2.5-video-vae-conv checkpoint, decodes one latent twice —
// once through `Ltx2ConvVideoDecode` and once through `Ltx2VideoDecodeStreaming`
// under upstream's own AUTO layout — and reports max|diff| plus the count of
// floats that are not bit-identical. Touches no product code.
//
// WHY IT EXISTS. `include/vllm/model_executor/models/ltx2_tiling.h`, the pipeline
// comment, the spec and docs/USAGE.md all claimed the AUTO layout was a no-op
// below "121 frames". It is a no-op below **81**: `latent_t = (frames - 1) / 8 + 1`
// reaches 11 at 81 frames and `split_temporal_causal` short-circuits only while
// `latent_t <= 10` (tiling.py:239-240). 81..120 frames is therefore a window in
// which the render silently stopped being the old render, and this probe is what
// turns that from an argument into a number.
//
// ─── BUILD AND RUN (there is no CMake target; this is the recorded recipe) ───
//   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
//   ninja -C build vllm
//   g++ -O2 -std=c++20 -Iinclude -Ithird_party \
//       scripts/probe_ltx2_tiled_equivalence.cpp build/libvllm.a -o /tmp/ltx2_equiv -pthread
//   /tmp/ltx2_equiv <vae.safetensors> <frames> <pixel_h> <pixel_w>
//
// The sibling `scripts/probe_ltx2_decode_memory.cpp` builds with the identical
// line; only the .cpp changes.
//
//   /tmp/ltx2_equiv .../ltx-2.5-video-vae-conv-bf16.safetensors 81 64 64
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_tiling.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"

namespace {

// The shipped conv VAE sets `timestep_conditioning: false` and no block sets
// `inject_noise`, so nothing draws — but the seam requires a stream, and a
// deterministic one keeps the two arms comparable if a checkpoint ever does draw.
class ZeroNoise : public vllm::Ltx2NoiseStream {
 public:
  std::vector<float> Draw(int64_t count) override {
    return std::vector<float>(static_cast<size_t>(count), 0.0f);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: ltx2_equiv <vae.safetensors> <frames> <pixel_h> <pixel_w>\n");
    return 2;
  }
  const std::string path = argv[1];
  const int64_t frames = std::strtoll(argv[2], nullptr, 10);
  const int64_t px_h = std::strtoll(argv[3], nullptr, 10);
  const int64_t px_w = std::strtoll(argv[4], nullptr, 10);

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  vllm::Ltx2VideoDecoderKind kind = vllm::Ltx2VideoDecoderKind::kConv;
  const vllm::Ltx2ConvVideoDecoderConfig cfg =
      vllm::Ltx2ParseConvVideoDecoderConfig(vllm::Ltx2ReadCheckpointConfig(file), &kind);
  const vllm::Ltx2VaeWeights weights =
      vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2VideoVaeDecoderKeyRules());

  const vllm::Ltx2ScaleFactors factors =
      vllm::Ltx2VideoScaleFactorsFromBlocks(cfg.decoder_blocks, cfg.patch_size);
  const int64_t lt = (frames - 1) / factors.time + 1;
  const int64_t lh = px_h / factors.height;
  const int64_t lw = px_w / factors.width;
  const int64_t lc = cfg.in_channels;
  if (lt < 1 || lh < 1 || lw < 1) {
    std::fprintf(stderr, "[equiv] the request does not fill one latent cell\n");
    return 2;
  }

  const vllm::Ltx2TileSizeConfig tiling =
      vllm::Ltx2AutoTileSizeConfig(lh * factors.height, lw * factors.width, factors);
  std::fprintf(stderr,
               "[equiv] %lldx%lld, latent %lld,%lld,%lld  AUTO h=%lld/%lld w=%lld/%lld "
               "f=%lld/%lld\n",
               (long long)px_w, (long long)px_h, (long long)lt, (long long)lh, (long long)lw,
               (long long)tiling.height.tile_size, (long long)tiling.height.overlap,
               (long long)tiling.width.tile_size, (long long)tiling.width.overlap,
               (long long)tiling.frames.tile_size, (long long)tiling.frames.overlap);

  const std::vector<vllm::Ltx2Tile> tiles = vllm::Ltx2CreateTiles(lt, lh, lw, tiling, factors);
  const size_t groups = vllm::Ltx2GroupTilesByTemporalSlice(tiles).size();

  std::vector<float> latent(static_cast<size_t>(lc * lt * lh * lw));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i < latent.size(); ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    latent[i] = static_cast<float>(static_cast<int32_t>(s >> 33)) / 2147483648.0f - 0.5f;
  }

  ZeroNoise noise_a;
  const vllm::Ltx2VideoFrames untiled =
      vllm::Ltx2ConvVideoDecode(cfg, weights, latent, lc, lt, lh, lw, &noise_a);

  ZeroNoise noise_b;
  std::vector<float> streamed;
  int64_t streamed_frames = 0;
  int64_t chunks = 0;
  vllm::Ltx2VideoDecodeStreaming(kind, cfg, weights, latent, lc, lt, lh, lw, &noise_b, tiling,
                                 [&](const vllm::Ltx2VideoChunk& chunk) {
                                   ++chunks;
                                   streamed_frames += chunk.frames.frames;
                                   streamed.insert(streamed.end(), chunk.frames.data.begin(),
                                                   chunk.frames.data.end());
                                 });

  std::fprintf(stderr,
               "[equiv] tiles=%zu groups=%zu  ->  untiled [%lld,%lld,%lld,%lld]  streamed "
               "[%lld,%lld,%lld,%lld] chunks=%lld\n",
               tiles.size(), groups, (long long)untiled.channels, (long long)untiled.frames,
               (long long)untiled.height, (long long)untiled.width, (long long)untiled.channels,
               (long long)streamed_frames, (long long)untiled.height, (long long)untiled.width,
               (long long)chunks);

  if (streamed.size() != untiled.data.size()) {
    std::fprintf(stderr, "[equiv] SIZE MISMATCH %zu vs %zu\n", streamed.size(),
                 untiled.data.size());
    return 1;
  }
  // Concatenating the chunks reproduces the clip only because each chunk is a
  // contiguous [C, t, H, W] block in frame order; the C-major layout means the
  // comparison below is elementwise regardless.
  double max_diff = 0.0;
  size_t differing = 0;
  for (size_t i = 0; i < streamed.size(); ++i) {
    if (std::memcmp(&streamed[i], &untiled.data[i], sizeof(float)) != 0) ++differing;
    const double d = std::fabs(static_cast<double>(streamed[i]) -
                               static_cast<double>(untiled.data[i]));
    if (d > max_diff) max_diff = d;
  }
  double span = 0.0;
  for (float v : untiled.data) span = std::max(span, std::fabs(static_cast<double>(v)));
  std::fprintf(stderr, "[equiv] max|diff| = %.17g   non-bit-identical floats = %zu / %zu\n",
               max_diff, differing, streamed.size());
  std::fprintf(stderr, "[equiv] untiled |out|max = %.17g\n", span);
  return 0;
}
