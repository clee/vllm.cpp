// LTX-2.5 phase L6 — the quantized loaders.
//
// Row MODEL-DIFFUSION-LTX25, .agents/specs/ltx-2-5.md, issue #435.
//
// THREE KINDS OF GATE, and they are not interchangeable:
//
//  * REAL MANIFEST. ltx2_fp8_dit_manifest.inc / ltx2_nvfp4_te_manifest.inc are
//    the SHIPPED checkpoints' own safetensors headers — 6124 and 1688 entries,
//    names/dtypes/shapes, not one weight byte. Every claim this port makes about
//    what those files contain is asserted against them, so a claim cannot drift
//    from the artifact. This mirrors how MiniMax-H3 gated its GGUF and NVFP4
//    arms (test_minimax_h3.cpp:2348).
//  * REAL BYTES. ltx2_quant_goldens.inc carries a few hundred bytes read at
//    their own offsets out of those same files, with the expected values decoded
//    by TORCH. The fp8 half of both dequant paths is therefore held against an
//    implementation that is not ours.
//  * SYNTHETIC FILES. Whole-model materialization, missing-tensor refusals and
//    device staging need a file small enough to build in a test, so those are
//    written here from a deterministic byte stream whose header MIRRORS the real
//    layout (prefix, dtypes, packed widths, swizzled scale shapes).
//
// The NVFP4 DiT arm is gated on a SYNTHETIC file only. Lightricks' first-party
// NVFP4 DiT sits behind an un-accepted HF gate (HTTP 403) and was NOT
// downloaded; that is recorded as owed in .agents/porting-inventory.md rather
// than papered over with a fabricated manifest.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "ltx2_fp8_dit_manifest.inc"
#include "ltx2_nvfp4_te_manifest.inc"
#include "ltx2_quant_goldens.inc"

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vt/backend.h"

using vllm::Ltx2DitLoadOptions;
using vllm::Ltx2DitParams;
using vllm::Ltx2DitQuant;
using vllm::Ltx2TensorSpec;
using vllm::SafetensorsFile;

namespace {

// ---------------------------------------------------------------------------
// The deterministic byte stream, mirrored bit-for-bit by
// scripts/gen-ltx2-quant-goldens.py (fnv1a64 + splitmix64).
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& s) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::vector<uint8_t> RandBytes(const std::string& name, size_t count) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<uint8_t> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = static_cast<uint8_t>((SplitMix64(seed + i) >> 24) & 0xFFU);
  }
  return out;
}

float Bf16ToF32(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t F32ToBf16(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t rounded = bits + 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(rounded >> 16);
}

// ---------------------------------------------------------------------------
// A synthetic .safetensors writer. Same shape as
// test_minimax_h3.cpp:509 WriteSafetensorsFromEntries.
// ---------------------------------------------------------------------------

struct StEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

void WriteSafetensors(const std::vector<StEntry>& entries, const std::string& path) {
  std::string header = "{";
  size_t offset = 0;
  for (size_t i = 0; i < entries.size(); ++i) {
    const StEntry& e = entries[i];
    if (i != 0) header += ",";
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t d = 0; d < e.shape.size(); ++d) {
      if (d != 0) header += ",";
      header += std::to_string(e.shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  header += "}";
  FILE* fh = std::fopen(path.c_str(), "wb");
  REQUIRE(fh != nullptr);
  const uint64_t len = header.size();
  uint8_t le[8];
  for (int i = 0; i < 8; ++i) le[i] = static_cast<uint8_t>((len >> (8 * i)) & 0xFFU);
  std::fwrite(le, 1, 8, fh);
  std::fwrite(header.data(), 1, header.size(), fh);
  for (const StEntry& e : entries) std::fwrite(e.bytes.data(), 1, e.bytes.size(), fh);
  std::fclose(fh);
}

std::string PackF32(const std::vector<float>& v) {
  std::string out(v.size() * sizeof(float), '\0');
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

std::string PackBf16(const std::vector<float>& v) {
  std::string out(v.size() * sizeof(uint16_t), '\0');
  for (size_t i = 0; i < v.size(); ++i) {
    const uint16_t b = F32ToBf16(v[i]);
    std::memcpy(out.data() + i * sizeof(uint16_t), &b, sizeof(b));
  }
  return out;
}

std::string PackBytes(const std::vector<uint8_t>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size());
}

// The forward swizzle, so the synthetic files carry a genuinely swizzled scale
// and the loader's inverse has something real to invert. Transcribed from the
// SAME source the header cites (vLLM qutlass_utils.py:177-179): source (r, c)
// with r = 128*rt + 32*a + s, c = 4*ct + q lands at
// ((((rt * ctiles) + ct) * 32 + s) * 4 + a) * 4 + q.
std::vector<uint8_t> SwizzleBlockScale(const std::vector<uint8_t>& linear, int64_t rows,
                                       int64_t cols) {
  REQUIRE(rows % 128 == 0);
  REQUIRE(cols % 4 == 0);
  const int64_t ctiles = cols / 4;
  std::vector<uint8_t> out(static_cast<size_t>(rows * cols), 0);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t rt = r / 128, rem = r % 128, a = rem / 32, s = rem % 32;
      const int64_t ct = c / 4, q = c % 4;
      out[static_cast<size_t>(((((rt * ctiles) + ct) * 32 + s) * 4 + a) * 4 + q)] =
          linear[static_cast<size_t>(r * cols + c)];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// A REDUCED-dimension LTX-2.5 DiT, written as a real checkpoint would be:
// `model.diffusion_model.` prefixed, FP8 or NVFP4 weights with their scale
// sidecars, BF16 biases and norms, F32 tables.
// ---------------------------------------------------------------------------

Ltx2DitParams TinyParams() {
  Ltx2DitParams p;
  p.num_layers = 2;
  p.num_attention_heads = 2;
  p.attention_head_dim = 64;   // dim = 128
  p.audio_num_attention_heads = 2;
  p.audio_attention_head_dim = 32;  // adim = 64
  p.in_channels = 16;
  p.out_channels = 16;
  p.audio_in_channels = 16;
  p.audio_out_channels = 16;
  p.cross_attention_dim = 128;
  p.audio_cross_attention_dim = 64;
  p.apply_gated_attention = true;
  p.cross_attention_adaln = true;
  p.use_prompt_adaln_single = false;
  p.ff_bias = false;
  p.audio_ff_bias = true;
  return p;
}

bool IsTable(const std::string& name) {
  return name.find("scale_shift_table") != std::string::npos;
}

// Deterministic f32 "true" value for element `i` of tensor `name`, so both the
// file and the expectation come from one rule.
float TrueValue(const std::string& name, size_t i) {
  const uint64_t u = SplitMix64(Fnv1a64(name) + i);
  return static_cast<float>(static_cast<double>(u >> 11) * 0x1p-53 * 2.0 - 1.0) * 0.05F;
}

struct SyntheticDit {
  std::vector<StEntry> entries;
  // name (contract, unprefixed) -> the exact bf16 the loader must produce.
  std::map<std::string, std::vector<uint16_t>> expected;
};

SyntheticDit BuildSyntheticDit(const Ltx2DitParams& p, Ltx2DitQuant quant,
                               const std::vector<std::string>& extra_modules) {
  SyntheticDit out;
  const std::string pre = vllm::kLtx2DitCheckpointPrefix;
  for (const Ltx2TensorSpec& spec : vllm::EnumerateLtx2DitTensors(p)) {
    int64_t numel = 1;
    for (int64_t d : spec.shape) numel *= d;
    const bool table = IsTable(spec.name);
    const bool quantized = spec.shape.size() == 2 && !table;

    if (table) {
      std::vector<float> v(static_cast<size_t>(numel));
      for (size_t i = 0; i < v.size(); ++i) v[i] = TrueValue(spec.name, i);
      out.entries.push_back({pre + spec.name, "F32", spec.shape, PackF32(v)});
      std::vector<uint16_t> keep(v.size());
      for (size_t i = 0; i < v.size(); ++i) std::memcpy(&keep[i], &v[i], 0);
      continue;  // f32 tables are checked separately
    }
    if (!quantized) {
      // Biases and q/k norms: BF16, stored as-is.
      std::vector<float> v(static_cast<size_t>(numel));
      std::vector<uint16_t> want(v.size());
      for (size_t i = 0; i < v.size(); ++i) {
        v[i] = TrueValue(spec.name, i);
        want[i] = F32ToBf16(v[i]);
      }
      out.entries.push_back({pre + spec.name, "BF16", spec.shape, PackBf16(v)});
      out.expected[spec.name] = want;
      continue;
    }

    const int64_t rows = spec.shape[0], cols = spec.shape[1];
    if (quant == Ltx2DitQuant::kFp8) {
      const std::vector<uint8_t> raw = RandBytes(spec.name, static_cast<size_t>(numel));
      const float scale = 0.00390625F;  // exact power of two: no rounding of its own
      std::vector<uint16_t> want(raw.size());
      for (size_t i = 0; i < raw.size(); ++i) {
        want[i] = F32ToBf16(vllm::F8E4M3ToF32(raw[i]) * scale);
      }
      out.entries.push_back({pre + spec.name, "F8_E4M3", spec.shape, PackBytes(raw)});
      out.entries.push_back({pre + spec.name + "_scale", "F32", {},
                             std::string(reinterpret_cast<const char*>(&scale), 4)});
      out.expected[spec.name] = want;
    } else {
      REQUIRE(cols % 16 == 0);
      const std::vector<uint8_t> packed =
          RandBytes(spec.name, static_cast<size_t>(rows * cols / 2));
      // The scale grid is [rows, cols/16]; the swizzle needs rows % 128 == 0 and
      // (cols/16) % 4 == 0, which a real checkpoint always satisfies. The tiny
      // config does not, so the synthetic NVFP4 arm pads the scale grid the way
      // torchao's own producer does.
      const int64_t groups = cols / 16;
      const int64_t prows = ((rows + 127) / 128) * 128;
      const int64_t pcols = ((groups + 3) / 4) * 4;
      std::vector<uint8_t> lin(static_cast<size_t>(prows * pcols), 0);
      const std::vector<uint8_t> live =
          RandBytes(spec.name + ".scale", static_cast<size_t>(rows * groups));
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g = 0; g < groups; ++g) {
          uint8_t b = live[static_cast<size_t>(r * groups + g)];
          if (b == 0x7F || b == 0xFF) b = 0x38;  // never emit the NaN encoding
          lin[static_cast<size_t>(r * pcols + g)] = b;
        }
      }
      const std::vector<uint8_t> sw = SwizzleBlockScale(lin, prows, pcols);
      const float scale2 = 0.0078125F;
      std::vector<uint8_t> lin_live(static_cast<size_t>(rows * groups));
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g = 0; g < groups; ++g) {
          lin_live[static_cast<size_t>(r * groups + g)] =
              lin[static_cast<size_t>(r * pcols + g)];
        }
      }
      std::vector<uint16_t> want(static_cast<size_t>(rows * cols));
      vllm::DequantNvfp4ToBf16(packed.data(), lin_live.data(), scale2, rows, cols,
                               want.data());
      out.entries.push_back({pre + spec.name, "U8", {rows, cols / 2}, PackBytes(packed)});
      out.entries.push_back({pre + spec.name + "_scale", "F8_E4M3",
                             {prows / 4, pcols * 4}, PackBytes(sw)});
      out.entries.push_back({pre + spec.name + "_scale_2", "F32", {},
                             std::string(reinterpret_cast<const char*>(&scale2), 4)});
      out.expected[spec.name] = want;
    }
  }
  // Whatever unported family the caller wants present.
  for (const std::string& m : extra_modules) {
    const std::vector<float> v(4, 0.5F);
    out.entries.push_back({pre + m, "F32", {2, 2}, PackF32(v)});
  }
  return out;
}

std::string TmpPath(const char* stem) {
  return std::string("/tmp/ltx2_loader_") + stem + ".safetensors";
}

// ---------------------------------------------------------------------------
// Real-manifest helpers
// ---------------------------------------------------------------------------

std::vector<int64_t> ShapeOf(const int64_t* s, int32_t rank) {
  return std::vector<int64_t>(s, s + rank);
}

const vllm_test::Ltx25Fp8DitTensor* FindDit(const std::string& name) {
  for (const auto& t : vllm_test::kLtx25Fp8DitTensors) {
    if (name == t.name) return &t;
  }
  return nullptr;
}

const vllm_test::Ltx25Nvfp4TeTensor* FindTe(const std::string& name) {
  for (const auto& t : vllm_test::kLtx25Nvfp4TeTensors) {
    if (name == t.name) return &t;
  }
  return nullptr;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

// ===========================================================================
// 1. The one delta: the scale swizzle
// ===========================================================================

TEST_CASE("ltx2 loader: the unswizzle inverts vLLM's own block-scale permutation") {
  for (int64_t c = 0; c < vllm_test::kLtx2BlockedCaseCount; ++c) {
    const auto& cs = vllm_test::kLtx2BlockedCases[c];
    const uint8_t* swizzled = nullptr;
    const uint8_t* linear = nullptr;
    int64_t count = 0;
    switch (c) {
      case 0:
        swizzled = vllm_test::kLtx2BlockedSwizzled0;
        linear = vllm_test::kLtx2BlockedLinear0;
        count = vllm_test::kLtx2BlockedLinear0Count;
        break;
      case 1:
        swizzled = vllm_test::kLtx2BlockedSwizzled1;
        linear = vllm_test::kLtx2BlockedLinear1;
        count = vllm_test::kLtx2BlockedLinear1Count;
        break;
      case 2:
        swizzled = vllm_test::kLtx2BlockedSwizzled2;
        linear = vllm_test::kLtx2BlockedLinear2;
        count = vllm_test::kLtx2BlockedLinear2Count;
        break;
      case 3:
        swizzled = vllm_test::kLtx2BlockedSwizzled3;
        linear = vllm_test::kLtx2BlockedLinear3;
        count = vllm_test::kLtx2BlockedLinear3Count;
        break;
      default:
        swizzled = vllm_test::kLtx2BlockedSwizzled4;
        linear = vllm_test::kLtx2BlockedLinear4;
        count = vllm_test::kLtx2BlockedLinear4Count;
        break;
    }
    REQUIRE(count == cs.rows * cs.cols);
    std::vector<uint8_t> got(static_cast<size_t>(count), 0);
    vllm::Ltx2UnswizzleNvfp4BlockScale(swizzled, static_cast<size_t>(count), cs.rows,
                                       cs.cols, got.data());
    int64_t mismatches = 0;
    int64_t first_bad = -1;
    for (int64_t i = 0; i < count; ++i) {
      if (got[static_cast<size_t>(i)] != linear[i]) {
        ++mismatches;
        if (first_bad < 0) first_bad = i;
      }
    }
    INFO("case " << c << " rows=" << cs.rows << " cols=" << cs.cols
                 << " mismatches=" << mismatches << " first_bad=" << first_bad);
    CHECK(mismatches == 0);
  }
}

TEST_CASE("ltx2 loader: the unswizzle refuses a buffer that is not the padded size") {
  std::vector<uint8_t> src(128 * 4, 0);
  std::vector<uint8_t> dst(128 * 4, 0);
  // rows=100 pads to 128, so 100*4 bytes is NOT what the layout stores.
  CHECK_THROWS(vllm::Ltx2UnswizzleNvfp4BlockScale(src.data(), 100 * 4, 100, 4, dst.data()));
}

// ===========================================================================
// 2. The torchao marker — read, not assumed
// ===========================================================================

TEST_CASE("ltx2 loader: the shipped torchao marker says swizzled fp4, and is parsed") {
  const std::string json = vllm_test::kLtx2RealTeMarkerJson;
  vllm::StTensor t;
  t.dtype = "U8";
  t.shape = {static_cast<int64_t>(json.size())};
  t.data = reinterpret_cast<const uint8_t*>(json.data());
  t.nbytes = json.size();
  const vllm::Ltx2TorchaoNvfp4Marker m =
      vllm::ParseLtx2TorchaoNvfp4Marker("video_aggregate_embed", t);
  CHECK(m.format == "torchao_nvfp4");
  CHECK(m.block_size == 16);
  CHECK(m.is_swizzled_scales);
  CHECK(m.config == "NVFP4DynamicActivationNVFP4WeightConfig");

  // The refusals. Each is a layout this port does NOT implement, and each would
  // otherwise dequantize to finite, wrongly-scaled weights.
  const char* rejects[] = {
      R"({"format": "compressed_tensors", "block_size": 16, "is_swizzled_scales": true})",
      R"({"format": "torchao_nvfp4", "block_size": 32, "is_swizzled_scales": true})",
      R"({"format": "torchao_nvfp4", "block_size": 16, "is_swizzled_scales": false})",
      "not json at all",
  };
  for (const char* bad : rejects) {
    vllm::StTensor b;
    b.dtype = "U8";
    b.shape = {static_cast<int64_t>(std::strlen(bad))};
    b.data = reinterpret_cast<const uint8_t*>(bad);
    b.nbytes = std::strlen(bad);
    const std::string payload_msg = std::string("payload ") + bad;
    INFO(payload_msg);
    CHECK_THROWS(vllm::ParseLtx2TorchaoNvfp4Marker("m", b));
  }
}

// ===========================================================================
// 3. The REAL checkpoints' own bytes
// ===========================================================================

TEST_CASE("ltx2 loader: the shipped text encoder's swizzled scale tile unswizzles") {
  REQUIRE(vllm_test::kLtx2RealTeScaleTileSwizzledCount == 512);
  REQUIRE(vllm_test::kLtx2RealTeScaleTileLinearCount == 512);
  std::vector<uint8_t> got(512, 0);
  vllm::Ltx2UnswizzleNvfp4BlockScale(vllm_test::kLtx2RealTeScaleTileSwizzled, 512, 128, 4,
                                     got.data());
  int64_t mismatches = 0;
  for (int i = 0; i < 512; ++i) {
    if (got[static_cast<size_t>(i)] != vllm_test::kLtx2RealTeScaleTileLinear[i]) ++mismatches;
  }
  INFO("mismatches=" << mismatches);
  CHECK(mismatches == 0);

  // And the bytes decode to the values TORCH read out of them.
  double max_abs = 0.0;
  for (int i = 0; i < 512; ++i) {
    const float ours = vllm::F8E4M3ToF32(got[static_cast<size_t>(i)]);
    const double d = std::abs(static_cast<double>(ours) -
                              vllm_test::kLtx2RealTeScaleTileLinearF32[i]);
    if (d > max_abs) max_abs = d;
  }
  INFO("max abs diff vs torch fp8-e4m3 = " << max_abs);
  CHECK(max_abs == 0.0);
}

TEST_CASE("ltx2 loader: the shipped FP8 DiT's own bytes dequantize to torch's values") {
  const int64_t n = vllm_test::kLtx2RealDitFp8HeadCount;
  REQUIRE(n == vllm_test::kLtx2RealDitFp8HeadF32Count);
  std::vector<uint16_t> bf16(static_cast<size_t>(n), 0);
  vllm::DequantFp8ToBf16(vllm_test::kLtx2RealDitFp8Head, vllm_test::kLtx2RealDitFp8Scale, n,
                         bf16.data());
  double max_abs = 0.0, max_rel = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double want = vllm_test::kLtx2RealDitFp8HeadF32[i];
    const double got = Bf16ToF32(bf16[static_cast<size_t>(i)]);
    const double d = std::abs(got - want);
    if (d > max_abs) max_abs = d;
    if (want != 0.0 && d / std::abs(want) > max_rel) max_rel = d / std::abs(want);
  }
  INFO("max abs = " << max_abs << " max rel = " << max_rel);
  // bf16 carries 8 significand bits, so RNE's worst relative error is half an
  // ulp = 2^-8. (An earlier revision of this line wrote 2^-9 by counting the 7
  // EXPLICIT mantissa bits and forgetting the implicit one — a miscounted
  // constant, not a tolerance anyone widened to pass.)
  CHECK(max_rel <= 0.00390625);
  // And tighter, because it can be: our path computes f32(byte) * scale and
  // rounds, which is torch's association exactly, so the bf16 bits must match
  // bit for bit. A rounding-mode or association change breaks this and not the
  // bound above.
  int64_t bit_mismatches = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (bf16[static_cast<size_t>(i)] != F32ToBf16(vllm_test::kLtx2RealDitFp8HeadF32[i])) {
      ++bit_mismatches;
    }
  }
  INFO("bf16 bit mismatches = " << bit_mismatches);
  CHECK(bit_mismatches == 0);
}

TEST_CASE("ltx2 loader: a torchao module built from the shipped bytes dequantizes") {
  // out=128, in=64 is exactly the geometry of the (0,0) scale tile: the scale
  // grid is [128, 4] and 4 groups of 16 is 64 inputs. Row 0's packed nibbles are
  // the shipped weight's first 32 bytes, so row 0 IS the golden.
  const int64_t out_features = 128, in_features = 64;
  std::vector<uint8_t> packed(static_cast<size_t>(out_features * in_features / 2), 0);
  std::memcpy(packed.data(), vllm_test::kLtx2RealTePackedHead,
              static_cast<size_t>(vllm_test::kLtx2RealTePackedHeadCount));

  vllm::StTensor w;
  w.dtype = "U8";
  w.shape = {out_features, in_features / 2};
  w.data = packed.data();
  w.nbytes = packed.size();
  vllm::StTensor s;
  s.dtype = "F8_E4M3";
  s.shape = {out_features / 4, (in_features / 16) * 4};
  s.data = vllm_test::kLtx2RealTeScaleTileSwizzled;
  s.nbytes = 512;
  const float scale2 = vllm_test::kLtx2RealTeScale2;
  vllm::StTensor g;
  g.dtype = "F32";
  g.shape = {};
  g.data = reinterpret_cast<const uint8_t*>(&scale2);
  g.nbytes = sizeof(float);

  std::vector<uint16_t> bf16(static_cast<size_t>(out_features * in_features), 0);
  vllm::Ltx2DequantTorchaoNvfp4ToBf16("video_aggregate_embed", w, s, g, out_features,
                                      in_features, bf16.data());
  double max_abs = 0.0, max_rel = 0.0;
  for (int64_t i = 0; i < vllm_test::kLtx2RealTeWeightHeadF32Count; ++i) {
    const double want = vllm_test::kLtx2RealTeWeightHeadF32[i];
    const double got = Bf16ToF32(bf16[static_cast<size_t>(i)]);
    const double d = std::abs(got - want);
    if (d > max_abs) max_abs = d;
    if (want != 0.0 && d / std::abs(want) > max_rel) max_rel = d / std::abs(want);
  }
  INFO("max abs = " << max_abs << " max rel = " << max_rel);
  // Half an ulp of bf16 (2^-8). Not bit-exact here, unlike the FP8 case: the
  // generator multiplies nibble * group_scale * global while
  // DequantNvfp4ToBf16 folds group_scale * global FIRST (nvfp4_dequant.h:17),
  // so the two f32 associations differ by the last ulp before the bf16 round.
  CHECK(max_rel <= 0.00390625);

  // Row 0 alone is NOT enough: for a [128, 4] scale grid the (0, 0..3) cells sit
  // at swizzled offsets 0..3, so row 0 reads the same whether the unswizzle runs
  // or not. Probe EVERY row of the real tile by dequantizing an all-1.0 fp4
  // pattern, which makes each output literally its own group scale.
  std::vector<uint8_t> ones(packed.size(), 0x22);  // both nibbles -> e2m1 1.0
  vllm::StTensor w1 = w;
  w1.data = ones.data();
  std::vector<uint16_t> probe(bf16.size(), 0);
  vllm::Ltx2DequantTorchaoNvfp4ToBf16("video_aggregate_embed", w1, s, g, out_features,
                                      in_features, probe.data());
  double probe_max_rel = 0.0;
  for (int64_t r = 0; r < out_features; ++r) {
    for (int64_t i = 0; i < in_features; ++i) {
      const double want =
          static_cast<double>(vllm_test::kLtx2RealTeScaleTileLinearF32[r * 4 + i / 16]) *
          static_cast<double>(scale2);
      const double got = Bf16ToF32(probe[static_cast<size_t>(r * in_features + i)]);
      if (want != 0.0) {
        const double rel = std::abs(got - want) / std::abs(want);
        if (rel > probe_max_rel) probe_max_rel = rel;
      }
    }
  }
  INFO("per-row group-scale probe max rel = " << probe_max_rel);
  CHECK(probe_max_rel <= 0.00390625);

  // And the SHAPE trap: reading the swizzled scale as if it were linear
  // [out, in/16] type-checks on element count. It must not be accepted.
  vllm::StTensor bad = s;
  bad.shape = {out_features, in_features / 16};
  CHECK_THROWS(vllm::Ltx2DequantTorchaoNvfp4ToBf16("video_aggregate_embed", w, bad, g,
                                                   out_features, in_features, bf16.data()));
}

// ===========================================================================
// 4. The REAL DiT manifest
// ===========================================================================

TEST_CASE("ltx2 loader: the shipped FP8 DiT manifest is fully accounted for") {
  REQUIRE(vllm_test::kLtx25Fp8DitTensorCount == 6124);
  const std::string pre = vllm::kLtx2DitCheckpointPrefix;

  int64_t scales = 0, f8 = 0, bf16 = 0, f32 = 0, unprefixed = 0;
  std::set<std::string> families;
  for (const auto& t : vllm_test::kLtx25Fp8DitTensors) {
    const std::string name = t.name;
    if (name.compare(0, pre.size(), pre) != 0) {
      ++unprefixed;
      continue;
    }
    const std::string bare = name.substr(pre.size());
    if (EndsWith(bare, "_scale")) {
      ++scales;
      continue;
    }
    if (std::string(t.dtype) == "F8_E4M3") ++f8;
    if (std::string(t.dtype) == "BF16") ++bf16;
    if (std::string(t.dtype) == "F32") ++f32;
    families.insert(bare.substr(0, bare.find('.')));
  }
  // EVERY tensor carries the ComfyUI prefix; the loader strips exactly one.
  CHECK(unprefixed == 0);
  CHECK(scales == 1775);
  CHECK(f8 == 1775);   // one scale per quantized weight, and no more
  CHECK(bf16 == 2284);
  CHECK(f32 == 290);   // the tables; the 1775 scalar scales were counted above

  // The four families outside the phase-L2 contract, named so their absence
  // from the port cannot be discovered later.
  CHECK(families.count("prompt_adaln_single") == 1);
  CHECK(families.count("audio_prompt_adaln_single") == 1);
  CHECK(families.count("keyframes_abs_pos_embedding") == 1);
  CHECK(families.count("video_embeddings_connector") == 1);
  CHECK(families.count("audio_embeddings_connector") == 1);

  // Spot-check the two shapes the ported forward is most sensitive to, straight
  // out of the shipped header. `audio_to_video_attn` is the asymmetric pair the
  // spec's test-trap list names: query/out width from the VIDEO stream, key/value
  // from the AUDIO one. A square assumption transposes these and still runs.
  const auto* a2v_q = FindDit(pre + "transformer_blocks.0.audio_to_video_attn.to_q.weight");
  const auto* a2v_o =
      FindDit(pre + "transformer_blocks.0.audio_to_video_attn.to_out.0.weight");
  REQUIRE(a2v_q != nullptr);
  REQUIRE(a2v_o != nullptr);
  CHECK(a2v_q->shape[0] == 2048);
  CHECK(a2v_q->shape[1] == 4096);
  CHECK(a2v_o->shape[0] == 4096);
  CHECK(a2v_o->shape[1] == 2048);
  // The FP8 arm's scale really is a per-tensor SCALAR, not a per-channel vector.
  const auto* scale = FindDit(pre + "transformer_blocks.0.attn1.to_q.weight_scale");
  REQUIRE(scale != nullptr);
  CHECK(std::string(scale->dtype) == "F32");
  CHECK(scale->rank == 0);
  // The video `ff` has NO bias and the audio one does — upstream's
  // ff_bias=false / audio_ff_bias=true asymmetry, in the shipped weights.
  CHECK(FindDit(pre + "transformer_blocks.0.ff.net.0.proj.bias") == nullptr);
  CHECK(FindDit(pre + "transformer_blocks.0.audio_ff.net.0.proj.bias") != nullptr);
}

TEST_CASE("ltx2 loader: the L2 contract's every name is present in the shipped DiT") {
  const std::string pre = vllm::kLtx2DitCheckpointPrefix;
  std::vector<Ltx2TensorSpec> manifest;
  std::set<std::string> present;
  for (const auto& t : vllm_test::kLtx25Fp8DitTensors) {
    const std::string bare = std::string(t.name).substr(pre.size());
    if (EndsWith(bare, "_scale")) continue;
    present.insert(bare);
    manifest.push_back({bare, ShapeOf(t.shape, t.rank)});
  }

  Ltx2DitParams p = vllm::ParseLtx2DitParamsFromManifest(manifest);
  CHECK(p.num_layers == 48);
  CHECK(p.num_attention_heads == 32);
  CHECK(p.audio_num_attention_heads == 32);
  CHECK(p.inner_dim() == 4096);
  CHECK(p.audio_inner_dim() == 2048);
  CHECK(p.attention_head_dim == 128);
  CHECK(p.audio_attention_head_dim == 64);
  CHECK(p.in_channels == 128);
  CHECK(p.out_channels == 128);
  CHECK(p.cross_attention_dim == 4096);
  CHECK(p.audio_cross_attention_dim == 2048);
  CHECK(p.apply_gated_attention);
  CHECK(p.cross_attention_adaln);
  CHECK_FALSE(p.ff_bias);
  CHECK(p.audio_ff_bias);
  // MEASURED, and it contradicts .agents/specs/ltx-2-5.md section 1.2 and
  // ltx2.h:115-117: the SHIPPED checkpoint carries prompt_adaln_single, which
  // upstream builds only when use_prompt_adaln_single is TRUE (model.py:222-226).
  // The prompt-K/V cache's premise does not hold for this checkpoint.
  CHECK(p.use_prompt_adaln_single);

  // Enumerate the contract for the subset this port DOES carry, and require
  // every one of its names in the file.
  Ltx2DitParams contract = p;
  contract.use_prompt_adaln_single = false;
  const std::vector<Ltx2TensorSpec> want = vllm::EnumerateLtx2DitTensors(contract);
  CHECK(want.size() == 4078);
  int64_t missing = 0;
  std::string first_missing;
  for (const Ltx2TensorSpec& spec : want) {
    if (present.count(spec.name) == 0) {
      ++missing;
      if (first_missing.empty()) first_missing = spec.name;
    }
  }
  const std::string missing_msg = "missing=" + std::to_string(missing) + " first=" + first_missing;
  INFO(missing_msg);
  CHECK(missing == 0);

  // ... and account for every name the file has that the contract does not, so
  // "the rest is fine" is a counted claim rather than a hope.
  //   258 = 2 connectors x (8 blocks x 16 + 1 learnable_registers)
  //    12 = prompt_adaln_single + audio_prompt_adaln_single, 6 tensors each
  //     1 = keyframes_abs_pos_embedding
  std::set<std::string> want_set;
  for (const Ltx2TensorSpec& spec : want) want_set.insert(spec.name);
  int64_t extra = 0;
  for (const std::string& name : present) {
    if (want_set.count(name) == 0) ++extra;
  }
  CHECK(extra == 258 + 12 + 1);
}

// ===========================================================================
// 5. The REAL text-encoder manifest
// ===========================================================================

TEST_CASE("ltx2 loader: the shipped torchao text encoder's widths are the LOGICAL ones") {
  REQUIRE(vllm_test::kLtx25Nvfp4TeTensorCount == 1688);

  // model.norm.weight is BF16 and therefore UNPACKED — the width authority.
  const auto* norm = FindTe("model.norm.weight");
  REQUIRE(norm != nullptr);
  CHECK(std::string(norm->dtype) == "BF16");
  CHECK(norm->rank == 1);
  CHECK(norm->shape[0] == 3840);

  // The two caption projections, the trap this campaign already fell into once.
  struct Proj {
    const char* module;
    int64_t out_features;
  };
  const Proj projections[] = {
      {"text_embedding_projection.video_aggregate_embed", 4096},
      {"text_embedding_projection.audio_aggregate_embed", 2048},
  };
  for (const Proj& pr : projections) {
    const auto* w = FindTe(std::string(pr.module) + ".weight");
    const auto* s = FindTe(std::string(pr.module) + ".weight_scale");
    const auto* g = FindTe(std::string(pr.module) + ".weight_scale_2");
    const auto* b = FindTe(std::string(pr.module) + ".bias");
    const auto* m = FindTe(std::string(pr.module) + ".torchao_nvfp4");
    REQUIRE(w != nullptr);
    REQUIRE(s != nullptr);
    REQUIRE(g != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(m != nullptr);
    CHECK(std::string(w->dtype) == "U8");
    CHECK(w->shape[0] == pr.out_features);
    // TWO values per byte: the stored width is HALF the logical one, and the
    // logical one is the Gemma hidden size times (num_hidden_layers + 1).
    const int64_t in_features = w->shape[1] * 2;
    CHECK(in_features == 3840 * 49);
    CHECK(std::string(s->dtype) == "F8_E4M3");
    CHECK(s->shape[0] == pr.out_features / 4);
    CHECK(s->shape[1] == (in_features / 16) * 4);
    CHECK(std::string(g->dtype) == "F32");
    CHECK(g->rank == 0);
    // The bias is BF16 — a DIFFERENT unpack path from the weight, which is
    // exactly the module ltx2_text_encoder.h:264-269 warns a loader drops.
    CHECK(std::string(b->dtype) == "BF16");
    CHECK(b->shape[0] == pr.out_features);
  }

  // Every quantized module carries all four tensors, tower and vision included.
  int64_t markers = 0, complete = 0;
  std::set<std::string> towers;
  for (const auto& t : vllm_test::kLtx25Nvfp4TeTensors) {
    const std::string name = t.name;
    if (!EndsWith(name, ".torchao_nvfp4")) continue;
    ++markers;
    const std::string module = name.substr(0, name.size() - std::strlen(".torchao_nvfp4"));
    towers.insert(module.substr(0, module.find('.')));
    if (FindTe(module + ".weight") != nullptr &&
        FindTe(module + ".weight_scale") != nullptr &&
        FindTe(module + ".weight_scale_2") != nullptr) {
      ++complete;
    }
  }
  CHECK(markers == 334);
  CHECK(complete == markers);
  // The file ships the FULL multimodal Gemma-4. Text conditioning is the scope,
  // but a loader that chokes on these cannot read this checkpoint at all.
  CHECK(towers.count("vision_model") == 1);
  CHECK(towers.count("multi_modal_projector") == 1);
  CHECK(towers.count("audio_projector") == 1);

  // The tokenizer ships AS A TENSOR, with its sidecars.
  const auto* tok = FindTe("tokenizer_json");
  REQUIRE(tok != nullptr);
  CHECK(std::string(tok->dtype) == "U8");
  CHECK(tok->shape[0] == 32169626);
  for (const char* asset : {"hf_asset__tokenizer_config.json",
                            "hf_asset__processor_config.json",
                            "hf_asset__generation_config.json",
                            "hf_asset__chat_template.jinja"}) {
    const std::string asset_msg = std::string("asset ") + asset;
    INFO(asset_msg);
    CHECK(FindTe(asset) != nullptr);
  }
}

// ===========================================================================
// 6. Whole-model materialization, on synthetic files
// ===========================================================================

TEST_CASE("ltx2 loader: the FP8 DiT materializes onto the L2 contract, exactly") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("fp8");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);
  CHECK(ck.quant == Ltx2DitQuant::kFp8);
  CHECK(ck.unported.empty());
  CHECK(ck.params.num_layers == p.num_layers);
  CHECK(ck.params.inner_dim() == p.inner_dim());
  CHECK(ck.params.audio_inner_dim() == p.audio_inner_dim());
  CHECK_FALSE(ck.params.ff_bias);
  CHECK(ck.params.audio_ff_bias);

  // Every quantized/bf16 tensor lands bit-exactly on the bf16 the checkpoint
  // encodes. bf16 is the DEFAULT: a wider default would still pass a value gate.
  int64_t checked = 0, bad = 0;
  std::string first_bad;
  for (const auto& kv : syn.expected) {
    auto it = ck.views.find(kv.first);
    REQUIRE(it != ck.views.end());
    const vt::Tensor& t = it->second;
    REQUIRE(t.dtype == vt::DType::kBF16);
    const uint16_t* got = t.Ptr<uint16_t>();
    for (size_t i = 0; i < kv.second.size(); ++i) {
      ++checked;
      if (got[i] != kv.second[i]) {
        ++bad;
        if (first_bad.empty()) first_bad = kv.first;
      }
    }
  }
  const std::string count_msg = "checked=" + std::to_string(checked) + " bad=" +
                                std::to_string(bad) + " first=" + first_bad;
  INFO(count_msg);
  CHECK(checked > 0);
  CHECK(bad == 0);

  // The tables stay F32 because the CHECKPOINT stores them F32.
  auto tbl = ck.views.find("scale_shift_table");
  REQUIRE(tbl != ck.views.end());
  CHECK(tbl->second.dtype == vt::DType::kF32);
  double table_max_abs = 0.0;
  for (int64_t i = 0; i < 2 * p.inner_dim(); ++i) {
    const double d = std::abs(static_cast<double>(tbl->second.Ptr<float>()[i]) -
                              TrueValue("scale_shift_table", static_cast<size_t>(i)));
    if (d > table_max_abs) table_max_abs = d;
  }
  INFO("table max abs = " << table_max_abs);
  CHECK(table_max_abs == 0.0);

  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: the NVFP4 DiT arm materializes onto the same contract") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kNvfp4, {});
  const std::string path = TmpPath("nvfp4");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);
  CHECK(ck.quant == Ltx2DitQuant::kNvfp4);
  int64_t checked = 0, bad = 0;
  std::string first_bad;
  for (const auto& kv : syn.expected) {
    auto it = ck.views.find(kv.first);
    REQUIRE(it != ck.views.end());
    const uint16_t* got = it->second.Ptr<uint16_t>();
    for (size_t i = 0; i < kv.second.size(); ++i) {
      ++checked;
      if (got[i] != kv.second[i]) {
        ++bad;
        if (first_bad.empty()) first_bad = kv.first;
      }
    }
  }
  const std::string count_msg = "checked=" + std::to_string(checked) + " bad=" +
                                std::to_string(bad) + " first=" + first_bad;
  INFO(count_msg);
  CHECK(checked > 0);
  CHECK(bad == 0);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: a missing tensor throws BY NAME and never reads as zeros") {
  const Ltx2DitParams p = TinyParams();
  SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string drop =
      std::string(vllm::kLtx2DitCheckpointPrefix) + "transformer_blocks.1.audio_attn2.to_v.bias";
  std::vector<StEntry> kept;
  bool found = false;
  for (const StEntry& e : syn.entries) {
    if (e.name == drop) {
      found = true;
      continue;
    }
    kept.push_back(e);
  }
  REQUIRE(found);
  const std::string path = TmpPath("missing");
  WriteSafetensors(kept, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  bool named = false;
  try {
    vllm::Ltx2LoadDitFromSafetensors(file);
  } catch (const std::exception& e) {
    const std::string what = e.what();
    const std::string what_msg = "what: " + what;
  INFO(what_msg);
    named = what.find("transformer_blocks.1.audio_attn2.to_v.bias") != std::string::npos;
  }
  CHECK(named);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: the unported families are refused by name, not absorbed") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(
      p, Ltx2DitQuant::kFp8,
      {"prompt_adaln_single.linear.weight", "keyframes_abs_pos_embedding",
       "video_embeddings_connector.learnable_registers"});
  const std::string path = TmpPath("unported");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  std::string what;
  try {
    vllm::Ltx2LoadDitFromSafetensors(file);
  } catch (const std::exception& e) {
    what = e.what();
  }
  const std::string what_msg = "what: " + what;
  INFO(what_msg);
  CHECK(what.find("prompt_adaln_single") != std::string::npos);
  CHECK(what.find("keyframes_abs_pos_embedding") != std::string::npos);
  CHECK(what.find("video_embeddings_connector") != std::string::npos);

  // The opt-in still REPORTS every one of them; it does not make them vanish.
  Ltx2DitLoadOptions options;
  options.allow_unported_modules = true;
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file, options);
  CHECK(ck.unported.size() == 3);
  CHECK(ck.checkpoint_params.use_prompt_adaln_single);
  CHECK_FALSE(ck.params.use_prompt_adaln_single);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: the f32 widening is OPT-IN and bit-exact over bf16") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("widen");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file, options);
  double max_abs = 0.0;
  for (const auto& kv : syn.expected) {
    auto it = ck.views.find(kv.first);
    REQUIRE(it != ck.views.end());
    REQUIRE(it->second.dtype == vt::DType::kF32);
    const float* got = it->second.Ptr<float>();
    for (size_t i = 0; i < kv.second.size(); ++i) {
      const double d = std::abs(static_cast<double>(got[i]) - Bf16ToF32(kv.second[i]));
      if (d > max_abs) max_abs = d;
    }
  }
  INFO("widen max abs = " << max_abs);
  CHECK(max_abs == 0.0);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: staging at load produces the same weights as the host load") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("stage");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  vt::Queue queue;
  const vllm::Ltx2DitCheckpoint staged = vllm::Ltx2StreamDitToDevice(queue, file);
  int64_t bad = 0, checked = 0;
  for (const auto& kv : syn.expected) {
    auto it = staged.views.find(kv.first);
    REQUIRE(it != staged.views.end());
    const uint16_t* got = it->second.Ptr<uint16_t>();
    for (size_t i = 0; i < kv.second.size(); ++i) {
      ++checked;
      if (got[i] != kv.second[i]) ++bad;
    }
  }
  INFO("staged checked=" << checked << " bad=" << bad);
  CHECK(checked > 0);
  CHECK(bad == 0);

  // Staging exists to avoid moving twice the bytes; widening while staging
  // would defeat it, so it is refused rather than silently honoured.
  Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;
  CHECK_THROWS(vllm::Ltx2StreamDitToDevice(queue, file, options));
  std::remove(path.c_str());
}

// ===========================================================================
// 7. The text-encoder loader, on a synthetic torchao file
// ===========================================================================

namespace {

// A torchao-NVFP4 text encoder with the shipped file's SHAPE RULES at reduced
// dimensions: the prefixless projections, a BF16 model.norm, the embedded
// tokenizer pack, and a vision module that must not choke the loader.
std::vector<StEntry> BuildSyntheticTe(int64_t hidden, int64_t layers, int64_t video_out,
                                      int64_t audio_out,
                                      std::map<std::string, std::vector<uint16_t>>* expected) {
  std::vector<StEntry> entries;
  const std::vector<float> norm(static_cast<size_t>(hidden), 1.0F);
  entries.push_back({"model.norm.weight", "BF16", {hidden}, PackBf16(norm)});
  for (int64_t l = 0; l < layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l);
    entries.push_back({b + ".input_layernorm.weight", "BF16", {hidden}, PackBf16(norm)});
  }
  const int64_t in_features = hidden * (layers + 1);
  const char* marker =
      R"({"format": "torchao_nvfp4", "block_size": 16, "scope": "full", )"
      R"("config": "NVFP4DynamicActivationNVFP4WeightConfig", "is_swizzled_scales": true, )"
      R"("use_triton_kernel": true, "use_dynamic_activation": true, )"
      R"("use_dynamic_per_tensor_scale": true})";

  struct P {
    const char* module;
    int64_t out;
  };
  const P projections[] = {
      {"text_embedding_projection.video_aggregate_embed", video_out},
      {"text_embedding_projection.audio_aggregate_embed", audio_out},
      {"vision_model.patch_dense", 128},  // present, out of scope, must not choke
  };
  for (const P& pr : projections) {
    const std::string m = pr.module;
    const int64_t groups = in_features / 16;
    const std::vector<uint8_t> packed =
        RandBytes(m + ".w", static_cast<size_t>(pr.out * in_features / 2));
    std::vector<uint8_t> lin(static_cast<size_t>(pr.out * groups));
    const std::vector<uint8_t> raw = RandBytes(m + ".s", lin.size());
    for (size_t i = 0; i < lin.size(); ++i) {
      lin[i] = (raw[i] == 0x7F || raw[i] == 0xFF) ? 0x38 : raw[i];
    }
    const std::vector<uint8_t> sw = SwizzleBlockScale(lin, pr.out, groups);
    const float scale2 = 0.0078125F;
    entries.push_back({m + ".weight", "U8", {pr.out, in_features / 2}, PackBytes(packed)});
    entries.push_back(
        {m + ".weight_scale", "F8_E4M3", {pr.out / 4, groups * 4}, PackBytes(sw)});
    entries.push_back({m + ".weight_scale_2", "F32", {},
                       std::string(reinterpret_cast<const char*>(&scale2), 4)});
    entries.push_back({m + ".torchao_nvfp4", "U8",
                       {static_cast<int64_t>(std::strlen(marker))}, std::string(marker)});
    if (expected != nullptr && m.rfind("text_embedding_projection", 0) == 0) {
      std::vector<uint16_t> want(static_cast<size_t>(pr.out * in_features));
      vllm::DequantNvfp4ToBf16(packed.data(), lin.data(), scale2, pr.out, in_features,
                               want.data());
      (*expected)[m] = want;
    }
    if (m.rfind("text_embedding_projection", 0) == 0) {
      std::vector<float> bias(static_cast<size_t>(pr.out));
      for (size_t i = 0; i < bias.size(); ++i) bias[i] = TrueValue(m + ".bias", i);
      entries.push_back({m + ".bias", "BF16", {pr.out}, PackBf16(bias)});
    }
  }
  entries.push_back({"tokenizer_json", "U8", {5}, std::string("{\"a\":")});
  entries.push_back({"hf_asset__tokenizer_config.json", "U8", {2}, std::string("{}")});
  entries.push_back({"hf_asset__processor_config.json", "U8", {2}, std::string("{}")});
  return entries;
}

}  // namespace

TEST_CASE("ltx2 loader: the torchao text encoder materializes onto the L3 contract") {
  const int64_t hidden = 128, layers = 3, video_out = 256, audio_out = 128;
  std::map<std::string, std::vector<uint16_t>> expected;
  const std::vector<StEntry> entries =
      BuildSyntheticTe(hidden, layers, video_out, audio_out, &expected);
  const std::string path = TmpPath("te");
  WriteSafetensors(entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);

  const vllm::Ltx2TextEncoderCheckpoint ck =
      vllm::Ltx2LoadTextEncoderFromSafetensors(file);
  CHECK(ck.gemma_hidden_size == hidden);
  CHECK(ck.gemma_num_hidden_layers == layers);
  CHECK(ck.video.out_features == video_out);
  CHECK(ck.video.in_features == hidden * (layers + 1));
  CHECK(ck.audio.out_features == audio_out);
  // The bias comes off a DIFFERENT dtype path and is the one a U8-only loader
  // drops; ltx2_text_encoder.h:264-269 names that failure exactly.
  CHECK(ck.video.bias_bf16.size() == static_cast<size_t>(video_out));
  CHECK(ck.audio.bias_bf16.size() == static_cast<size_t>(audio_out));
  // The multimodal tower is present and did NOT choke the loader.
  CHECK(ck.quantized_modules.size() == 3);
  // The tokenizer came out of the tensor, not a sibling file.
  CHECK(ck.assets.tokenizer_json.size() == 5);
  CHECK_FALSE(ck.assets.has_config);

  int64_t bad = 0, checked = 0;
  for (const auto& kv : expected) {
    const std::vector<uint16_t>& want = kv.second;
    const std::vector<uint16_t>& got =
        kv.first.find("video") != std::string::npos ? ck.video.weight_bf16
                                                    : ck.audio.weight_bf16;
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < want.size(); ++i) {
      ++checked;
      if (got[i] != want[i]) ++bad;
    }
  }
  INFO("te checked=" << checked << " bad=" << bad);
  CHECK(checked > 0);
  CHECK(bad == 0);

  // The f32 widening is opt-in and lands on L3's own contract.
  const vllm::Ltx2TextEncoderWeights w = vllm::Ltx2WidenTextProjectionsToF32(ck);
  CHECK(w.video.out_features == video_out);
  CHECK(w.video.in_features == hidden * (layers + 1));
  CHECK(w.video.bias.size() == static_cast<size_t>(video_out));
  double max_abs = 0.0;
  for (size_t i = 0; i < w.video.weight.size(); ++i) {
    const double d =
        std::abs(static_cast<double>(w.video.weight[i]) - Bf16ToF32(ck.video.weight_bf16[i]));
    if (d > max_abs) max_abs = d;
  }
  INFO("te widen max abs = " << max_abs);
  CHECK(max_abs == 0.0);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: an incomplete torchao module throws BY NAME") {
  std::vector<StEntry> entries = BuildSyntheticTe(128, 3, 256, 128, nullptr);
  std::vector<StEntry> kept;
  for (const StEntry& e : entries) {
    if (e.name == "vision_model.patch_dense.weight_scale") continue;
    kept.push_back(e);
  }
  const std::string path = TmpPath("te_incomplete");
  WriteSafetensors(kept, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);
  std::string what;
  try {
    vllm::Ltx2LoadTextEncoderFromSafetensors(file);
  } catch (const std::exception& e) {
    what = e.what();
  }
  const std::string what_msg = "what: " + what;
  INFO(what_msg);
  CHECK(what.find("vision_model.patch_dense") != std::string::npos);
  std::remove(path.c_str());
}

TEST_CASE("ltx2 loader: require_config mirrors upstream's refusal of a metadata-less pack") {
  const std::vector<StEntry> entries = BuildSyntheticTe(128, 3, 256, 128, nullptr);
  const std::string path = TmpPath("te_cfg");
  WriteSafetensors(entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);
  vllm::Ltx2TextEncoderLoadOptions options;
  options.require_config = true;
  CHECK_THROWS(vllm::Ltx2LoadTextEncoderFromSafetensors(file, options));
  std::remove(path.c_str());
}

// ===========================================================================
// 8. The contract's OWN refusal, which ltx2.h:228-232 promises and did not give
// ===========================================================================

TEST_CASE("ltx2 loader: BindLtx2DitWeights names the tensor it is missing") {
  const Ltx2DitParams p = TinyParams();
  const SyntheticDit syn = BuildSyntheticDit(p, Ltx2DitQuant::kFp8, {});
  const std::string path = TmpPath("bindname");
  WriteSafetensors(syn.entries, path);
  const SafetensorsFile file = SafetensorsFile::Open(path);
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);

  // ltx2.h:228-232 states that a name the contract requires and the map lacks
  // "throws BY NAME rather than reading as zeros". The loader above enforces
  // that before it binds, but the SEAM itself must too: a caller assembling its
  // own map — which is exactly what the L2 parity suite does — gets no such
  // pre-pass, and "a required tensor is missing" does not tell it which.
  std::map<std::string, vt::Tensor> views = ck.views;
  const std::string dropped = "transformer_blocks.1.audio_to_video_attn.to_out.0.weight";
  REQUIRE(views.erase(dropped) == 1);
  std::string what;
  try {
    vllm::BindLtx2DitWeights(ck.params, views);
  } catch (const std::exception& e) {
    what = e.what();
  }
  const std::string bind_msg = "what: " + what;
  INFO(bind_msg);
  CHECK(what.find(dropped) != std::string::npos);
  std::remove(path.c_str());
}
