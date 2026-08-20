// ENG-EXPERT-STREAM-DEVICE (#1299): is the MoE router's selection the plain
// lowest-index rank of the logits it was handed, and does the observer that
// MEASURES that get reached by a real forward?
//
// WHY THIS FILE EXISTS. `--device cpu` and `--device cuda` select different
// expert sets on `Qwen3.8-2.4T-A95B UD-Q1_0`, and a selection can differ for
// exactly two reasons: the two top-k implementations broke a tie differently,
// or the logits they ranked were not the same numbers. `VT_ROUTER_DUMP` was
// added to separate those on a leased GB10. An instrument nobody drives in CI
// rots against the code it measures, and a dump whose SHAPE is wrong reads as a
// measurement rather than as a broken instrument -- which is the shape
// `.agents/reachability.md` names -- so this drives it through
// `Qwen3_5Model::Forward`, the production paged forward the runner calls, and
// then reads the file back.
//
// THE ASSERTION THAT CARRIES THE INVESTIGATION is not "a file appeared". It is
// that the ids in each record are exactly the plain (value descending, index
// ascending) rank of the logits in the SAME record. That is the rule both arms
// claim to implement and the rule vLLM's own kernel states in words
// (`csrc/libtorch_stable/moe/topk_softmax_kernels.cu:536-537` @ 555967922 --
// "We want lower indices to 'win' in every thread so we break ties this way").
// If it holds, a differing selection cannot be a tie-break difference and the
// logits are where to look. Asserting it HERE, on the recorded bytes, is what
// makes the leased-run conclusion checkable without the lease.
//
// The tie case is separate and direct, because the synthetic model's random
// weights do not produce an exact bf16 tie at the boundary on demand. Ranking
// only proves the ORDER on distinct values; the tie case proves what happens
// when two are equal, which is the whole question.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "support/expert_stream_model.h"
#include "support/test_env.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using expert_stream_test::CachePool;
using expert_stream_test::MakeConfig;
using expert_stream_test::MakeWeights;
using expert_stream_test::PrefillAttnMeta;
using expert_stream_test::PrefillGdnMeta;
using expert_stream_test::Q;
using vllm::HfConfig;
using vllm::Qwen3_5MoeWeights;
using vllm::Qwen3_5Model;

namespace {

// The dump path is read ONCE into a function-local static on the first MoE
// block, so it has to be set before any test body runs -- setting it inside a
// case would work today and break the moment a case ordering changed. Same
// argument, and the same `vllm_test::SetEnv` shim (MSVC has no `setenv(3)`), as
// `test_expert_stream_wiring`'s `EnableExpertStreaming`.
std::string DumpPath() {
  static const std::string p = [] {
    const char* dir = std::getenv("TMPDIR");
    std::string d = (dir != nullptr && dir[0] != '\0') ? dir : "/tmp";
    return d + "/vllm_router_observer_test.bin";
  }();
  return p;
}

struct EnableRouterDump {
  EnableRouterDump() {
    std::remove(DumpPath().c_str());
    vllm_test::SetEnv("VT_ROUTER_DUMP", DumpPath().c_str());
    vllm_test::SetEnv("VT_ROUTER_DUMP_MAX_CALLS", "4");  // exactly one forward
  }
};
const EnableRouterDump kEnableRouterDump;

float Bf16ToF32(uint16_t u) {
  const uint32_t bits = static_cast<uint32_t>(u) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t F32ToBf16Bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);  // truncation is fine for the
                                             // exact values this test builds
}

struct Record {
  uint32_t call, t, e, k, h;
  uint32_t w_nk, w_dtype;
  uint64_t w_bytes, w_hash;
  std::vector<uint16_t> hidden, logits;
  std::vector<int32_t> ids;
  std::vector<float> weights;
};

// Read the whole dump. Returns false when the file is absent or truncated --
// which is the state the reachability mutation produces, and it must be
// distinguishable from "read a file with zero records".
bool ReadDump(const std::string& path, std::vector<Record>* out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  std::vector<uint8_t> b;
  uint8_t buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) b.insert(b.end(), buf, buf + n);
  std::fclose(f);
  size_t off = 0;
  auto take = [&](void* dst, size_t bytes) {
    if (off + bytes > b.size()) return false;
    std::memcpy(dst, b.data() + off, bytes);
    off += bytes;
    return true;
  };
  char magic[4];
  uint32_t version = 0;
  if (!take(magic, 4) || std::memcmp(magic, "VTRD", 4) != 0) return false;
  if (!take(&version, 4) || version != 2) return false;
  while (off < b.size()) {
    Record r;
    uint32_t hdr[5];
    if (!take(hdr, sizeof(hdr))) return false;
    r.call = hdr[0]; r.t = hdr[1]; r.e = hdr[2]; r.k = hdr[3]; r.h = hdr[4];
    uint32_t whdr[2];
    if (!take(whdr, sizeof(whdr))) return false;
    r.w_nk = whdr[0]; r.w_dtype = whdr[1];
    if (!take(&r.w_bytes, sizeof(r.w_bytes))) return false;
    if (!take(&r.w_hash, sizeof(r.w_hash))) return false;
    r.hidden.resize(static_cast<size_t>(r.t) * r.h);
    r.logits.resize(static_cast<size_t>(r.t) * r.e);
    r.ids.resize(static_cast<size_t>(r.t) * r.k);
    r.weights.resize(static_cast<size_t>(r.t) * r.k);
    if (!take(r.hidden.data(), r.hidden.size() * 2)) return false;
    if (!take(r.logits.data(), r.logits.size() * 2)) return false;
    if (!take(r.ids.data(), r.ids.size() * 4)) return false;
    if (!take(r.weights.data(), r.weights.size() * 4)) return false;
    out->push_back(std::move(r));
  }
  return true;
}

// The plain rule, written the slow obvious way: the k largest logits, ties going
// to the lower expert index. Deliberately NOT sharing an implementation with
// either kernel -- a gate that compares a kernel against a shared helper proves
// consistency and not correctness.
std::vector<int32_t> PlainRank(const uint16_t* row, int64_t e, int64_t k) {
  std::vector<char> taken(static_cast<size_t>(e), 0);
  std::vector<int32_t> out;
  for (int64_t r = 0; r < k; ++r) {
    int64_t best = -1;
    float best_v = 0.0f;
    for (int64_t j = 0; j < e; ++j) {
      if (taken[static_cast<size_t>(j)]) continue;
      const float v = Bf16ToF32(row[j]);
      if (best < 0 || v > best_v) {  // strict `>`, ascending j -> lowest index
        best_v = v;
        best = j;
      }
    }
    taken[static_cast<size_t>(best)] = 1;
    out.push_back(static_cast<int32_t>(best));
  }
  return out;
}

std::vector<float> OneForward(const HfConfig& c, const Qwen3_5MoeWeights& w,
                              const std::vector<int32_t>& ids) {
  vt::Queue q = Q();
  const int64_t T = static_cast<int64_t>(ids.size());
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/8);
  const std::vector<int32_t> blocks = {0};
  return Qwen3_5Model::Forward(ids, pos, PrefillAttnMeta(T, blocks, 8, 0),
                               PrefillGdnMeta(T, 0), pool.attn_kv, pool.gdn_state,
                               w, c, q, {});
}

}  // namespace

TEST_CASE("a real forward reaches the router observer, and its ids ARE the lowest-index rank") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};

  const std::vector<float> logits = OneForward(c, w, ids);
  REQUIRE(logits.size() ==
          static_cast<size_t>(ids.size()) * static_cast<size_t>(c.vocab_size));
  for (float v : logits) REQUIRE(std::isfinite(v));

  std::vector<Record> recs;
  // REACHABILITY. Deleting the production call site leaves no file at all, so
  // this REQUIRE is the assertion the mutation reds. "No file" and "a file with
  // no records" are different failures and ReadDump keeps them apart.
  REQUIRE(ReadDump(DumpPath(), &recs));
  REQUIRE(recs.size() == static_cast<size_t>(c.num_hidden_layers));

  for (size_t i = 0; i < recs.size(); ++i) {
    const Record& r = recs[i];
    CAPTURE(i);
    CHECK(r.call == static_cast<uint32_t>(i));
    CHECK(r.t == static_cast<uint32_t>(ids.size()));
    CHECK(r.e == static_cast<uint32_t>(c.num_experts));
    CHECK(r.k == static_cast<uint32_t>(c.num_experts_per_tok));
    CHECK(r.h == static_cast<uint32_t>(c.hidden_size));
    // The gate fingerprint is a real hash of real bytes, not a zero placeholder:
    // a host-released weight would report 0 and say nothing about the arms.
    CHECK(r.w_bytes == static_cast<uint64_t>(c.hidden_size) *
                           static_cast<uint64_t>(c.num_experts) * 2);
    CHECK(r.w_hash != 0);

    for (uint32_t t = 0; t < r.t; ++t) {
      CAPTURE(t);
      const std::vector<int32_t> want =
          PlainRank(r.logits.data() + static_cast<size_t>(t) * r.e, r.e, r.k);
      const std::vector<int32_t> got(r.ids.begin() + static_cast<size_t>(t) * r.k,
                                     r.ids.begin() + static_cast<size_t>(t + 1) * r.k);
      CHECK(got == want);
    }
  }
}

TEST_CASE("the router top-k gives an exact tie to the LOWER expert index") {
  // Ranking distinct values proves the order. It says nothing about the case the
  // whole #1299 question turns on: two experts whose bf16 logits are EQUAL. With
  // 512 experts in 8 mantissa bits that case is common, so the rule has to be
  // executed rather than read.
  const int64_t T = 1, E = 8, K = 3;
  // Experts 5 and 1 tie for the top; 6, 2 and 0 tie for the next place. The
  // lowest index must win each time, so the selection is 1, 5 (the loser of the
  // first tie is still the largest remaining), then 0.
  std::vector<uint16_t> row(static_cast<size_t>(E));
  row[0] = F32ToBf16Bits(2.0f);
  row[1] = F32ToBf16Bits(4.0f);
  row[2] = F32ToBf16Bits(2.0f);
  row[3] = F32ToBf16Bits(1.0f);
  row[4] = F32ToBf16Bits(0.5f);
  row[5] = F32ToBf16Bits(4.0f);
  row[6] = F32ToBf16Bits(2.0f);
  row[7] = F32ToBf16Bits(0.25f);

  vt::Queue q = Q();
  const vt::Device d{vt::DeviceType::kCPU, 0};
  std::vector<float> wts(static_cast<size_t>(T * K), 0.0f);
  std::vector<int32_t> sel(static_cast<size_t>(T * K), -1);
  vt::Tensor tl = vt::Tensor::Contiguous(row.data(), vt::DType::kBF16, d, {T, E});
  vt::Tensor tw = vt::Tensor::Contiguous(wts.data(), vt::DType::kF32, d, {T, K});
  vt::Tensor ti = vt::Tensor::Contiguous(sel.data(), vt::DType::kI32, d, {T, K});
  vt::MoeRouterTopKArgs args;
  args.top_k = static_cast<int>(K);
  args.renormalize = true;
  vt::MoeRouterTopK(q, tw, ti, tl, args, nullptr);

  const std::vector<int32_t> want = {1, 5, 0};
  CHECK(sel == want);
  // And the same rule, computed independently, agrees -- so a change that broke
  // BOTH would still have to break this literal too.
  CHECK(PlainRank(row.data(), E, K) == want);
}
