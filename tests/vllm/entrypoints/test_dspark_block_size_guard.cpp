// SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225) — the DSpark block floor, from the loader.
//
// The floor itself is `SpeculativeConfig::ResolveDspark`
// (include/vllm/config/speculative.h:179-185), ported under SPEC-DSPARK W1 from
// vllm/config/speculative.py:1003-1027 @ 555967922. It was already correct and
// already unit-tested at tests/vllm/config/test_speculative_dspark.cpp:99-107.
// It was also unreachable: both production call sites
// (src/vllm/entrypoints/model_loader.cpp:881-883 and :1675-1677) passed
// std::nullopt for n_predict AND for dspark_block_size, so no user could arrive
// at the check. .agents/reachability.md calls this the unpassed-parameter shape,
// and its rule is that the smallest failing test enters through the production
// entry point rather than constructing the value by hand.
//
// So these cases enter at LoadedEngine::ResolveSpecConfig, the function the
// LoadedEngine constructor calls (model_loader.cpp:1099) to finalize the
// entrypoint's speculative config against the checkpoint. They write a real
// draft config.json and let the loader read it, exactly as the production path
// does.
//
// RED before the fix: every THROWS case below returns a resolved config instead,
// because the loader hands ResolveDspark two std::nullopt values.
//
// Ported behavior, all @ 555967922:
//   * :1003-1027 — k below the block is a HARD error. Upstream's own comment
//     says a smaller k "produce[s] incorrect output", not merely lower
//     acceptance, because the block/Markov machinery gets an unsupported layout.
//   * :945-961  — the Gemma4 draft's block_size normalizes onto n_predict.
//   * :973-979  — k defaults to n_predict when the draft config carries one.
//   * :990-994  — k with no n_predict anywhere is an error.
//
// ONE DIVERGENCE, argued in .agents/specs/dspark-block-size-guard.md section 2:
// the floor falls back to `block_size` when `dspark_block_size` is absent.
// Upstream reads only `dspark_block_size`, an identifier that appears in no file
// of the pinned checkout except speculative.py, and NEITHER published Qwen3
// draft sets it — deepseek-ai/dspark_qwen3_4b_block7 and
// RadixArk/Qwen3.8-27B-DSpark @ 85ef153b both carry `block_size: 7` and no
// n_predict, and the :945-961 normalization is guarded by Gemma4DSparkModel. A
// literal port would therefore be keyed on a field no checkpoint we support
// sets. The "block_size supplies the floor" case below is that divergence, and
// it is the one that protects the lane we actually ship.
#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"

namespace fs = std::filesystem;
using vllm::SpeculativeConfig;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

namespace {

// A scratch draft checkpoint directory holding just a config.json. The loader
// resolves the draft directory with ResolveDflashDraftDir, which accepts any
// directory that contains a config.json, so this is the real production read.
class ScratchDraft {
 public:
  explicit ScratchDraft(const std::string& config_json) {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_dspark_guard_" + std::to_string(::getpid()) + "_" +
            std::to_string(counter++));
    fs::create_directories(dir_);
    std::ofstream out(dir_ / "config.json");
    out << config_json;
    out.close();
  }
  ~ScratchDraft() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  ScratchDraft(const ScratchDraft&) = delete;
  ScratchDraft& operator=(const ScratchDraft&) = delete;

  std::string path() const { return dir_.string(); }

 private:
  fs::path dir_;
};

// The CLI-side config the entrypoint builds before the checkpoint is read:
// method, the separate draft checkpoint, and the user's k.
EngineParams DsparkParams(const std::string& draft_path, std::optional<int> k) {
  EngineParams params;
  SpeculativeConfig cli;
  cli.method = "dspark";
  cli.draft_model_path = draft_path;
  cli.num_speculative_tokens = k;
  params.speculative_config = cli;
  return params;
}

// The 4B/27B published shape: a self-contained Qwen3 DSpark draft whose block
// depth is spelled `block_size`, with no n_predict and no dspark_block_size.
const char* kQwen3Block7 = R"({
  "architectures": ["Qwen3DSparkModel"],
  "model_type": "qwen3",
  "block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 151669,
  "target_layer_ids": [1, 9, 17, 25, 33]
})";

// The DeepSeek-V4 shape upstream's getattr actually reads.
const char* kDsv4Block7 = R"({
  "architectures": ["Qwen3DSparkModel"],
  "model_type": "qwen3",
  "dspark_block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 151669,
  "target_layer_ids": [1, 9, 17, 25, 33]
})";

// The Gemma4 draft whose block_size upstream normalizes onto n_predict.
const char* kGemma4Block7 = R"({
  "architectures": ["Gemma4DSparkModel"],
  "model_type": "gemma4",
  "block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 151669,
  "target_layer_ids": [1, 9, 17, 25, 33]
})";

}  // namespace

TEST_CASE("the loader refuses k below the draft's dspark_block_size") {
  // speculative.py:1003-1027, reached from the loader rather than by hand.
  const ScratchDraft draft(kDsv4Block7);
  const EngineParams params = DsparkParams(draft.path(), 6);
  CHECK_THROWS_AS(LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
                  std::invalid_argument);
}

TEST_CASE("the loader refuses k below the draft's block_size") {
  // THE DIVERGENCE, and the case that covers every published Qwen3 draft.
  // Upstream accepts this: n_predict is None and dspark_block_size is None, so
  // neither :973-988 nor :1003-1027 fires. Our draft block is sized by k alone
  // (spec_decode/dspark/speculator.h:56) and no weight is block-shaped, so k=6
  // against a block-7 checkpoint drafts a structurally wrong block in silence.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), 6);
  CHECK_THROWS_AS(LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
                  std::invalid_argument);
}

TEST_CASE("the refusal names the block and the k that was asked for") {
  // Upstream's message names both values, and so must ours: the user has to
  // learn which number to raise k to.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), 6);
  try {
    LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
    FAIL("expected a refusal for k=6 against a block-7 DSpark draft");
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    CHECK(what.find('7') != std::string::npos);
    CHECK(what.find('6') != std::string::npos);
  }
}

TEST_CASE("k at the block is accepted unchanged") {
  // The floor is >=, not >. This is the value both published drafts want.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), 7);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  CHECK(cfg->method == "dspark");
  CHECK(cfg->parallel_drafting);
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 7);
  REQUIRE(cfg->draft_model_path.has_value());
  CHECK(*cfg->draft_model_path == draft.path());
}

TEST_CASE("k above the block is accepted unchanged") {
  // Nothing this row does may narrow a value that works today.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), 14);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 14);
}

TEST_CASE("a Gemma4 draft's block_size defaults k") {
  // speculative.py:945-961 then :973-979. Before this row the loader threw
  // "requires num_speculative_tokens" before ResolveDspark could apply the
  // default, so the threaded n_predict would have been unreachable.
  const ScratchDraft draft(kGemma4Block7);
  const EngineParams params = DsparkParams(draft.path(), std::nullopt);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  CHECK(cfg->n_predict == 7);
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 7);
}

TEST_CASE("a native Qwen3 draft still requires k") {
  // speculative.py:990-994. The native config carries no n_predict, so there is
  // nothing to default from and the existing message must survive.
  const ScratchDraft draft(kQwen3Block7);
  const EngineParams params = DsparkParams(draft.path(), std::nullopt);
  CHECK_THROWS_AS(LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
                  std::invalid_argument);
}

TEST_CASE("a draft path with no config.json resolves as it did before") {
  // ResolveSpecConfig had no filesystem dependency before this row. Reading the
  // draft config must not turn a missing checkpoint into a config-time failure:
  // LoadDsparkDraft owns that message and names the path it looked in.
  const EngineParams params =
      DsparkParams("/nonexistent/dspark/draft/for/this/test", 7);
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  REQUIRE(cfg->num_speculative_tokens.has_value());
  CHECK(*cfg->num_speculative_tokens == 7);
}
