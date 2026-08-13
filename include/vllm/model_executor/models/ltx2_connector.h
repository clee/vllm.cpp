// LTX-2.5 EMBEDDINGS CONNECTOR — the 1-D transformer between the Gemma-4 text
// encoder and the DiT's cross-attention.
//
// Row: MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model. Spec:
// .agents/specs/ltx-2-5.md (phase L5). Issue #435.
//
// This brick sits BETWEEN two phases and was orphaned between them: L3 stops at
// the text encoder's output and L2 starts at the DiT's context input, so nothing
// owned the module that turns one into the other. It lands here.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream: Lightricks/LTX-2,
// packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py
//   OURS                          <-  UPSTREAM
//   Ltx2ConnectorForward          <-  :154-191 (Embeddings1DConnector.forward)
//   Ltx2ConnectorBlockForward     <-  :41-71  (_BasicTransformerBlock1D.forward)
//   Ltx2ConnectorReplaceRegisters <-  :139-152
//   Ltx2ConnectorConfig           <-  :95-137 + :194-256 (both configurators)
//
// ─── IT IS BUILT ON THE DiT's OWN PARTS, AND SO IS THIS PORT ─────────────────
// `_BasicTransformerBlock1D` imports the DiT's `Attention`, `FeedForward` and
// RoPE verbatim (:4-11). This TU therefore routes through `vllm::Ltx2Attention`,
// `vllm::Ltx2FeedForward`, `vllm::Ltx2PrecomputeFreqsCis` and
// `vllm::Ltx2ApplyRotaryEmb` from phase L2 rather than re-deriving them — a
// second attention implementation here would be a parallel path that could drift
// from the one the DiT is gated on.
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * `learnable_registers` IS A BFLOAT16 PARAMETER (:135-137 constructs it with
//    `dtype=torch.bfloat16`). Keeping those values in f32 is WIDER than upstream:
//    every register position then carries ~8 extra mantissa bits, the result is
//    still finite and still shaped right, and only the padded positions move.
//    AGENTS.md names this polarity exactly — a value gate cannot catch a dtype
//    that is too wide — so the rounding is explicit here and gated on its own
//    golden.
//  * THE REGISTER TABLE IS TILED, NOT INDEXED: `repeat(seq_len // num_registers, 1)`
//    (:146) walks 0..N-1, 0..N-1, …, so `seq_len` MUST be a multiple of
//    `num_learnable_registers` and upstream asserts it (:144).
//  * THE MASK IS REPLACED BY ZEROS once registers are substituted (:152). Every
//    position is attendable afterwards — including the ones that were padding. A
//    port that kept the original mask attends over fewer tokens and produces a
//    different, plausible conditioning.
//  * THE FINAL `rms_norm` (:189) has NO weight and torch's default eps of 1e-6
//    (utils.py:7-12). It is applied AFTER the last block, on top of the two the
//    block already applies.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// f32 for the activations, which is the parity dtype of this gate and matches
// what upstream computes when handed an f32 input (there is no per-layer dtype in
// the module). The ONE deliberate narrowing is `learnable_registers`, above,
// because upstream stores it narrow. The production bf16 arm is phase L6.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2.h"            // Ltx2RopeType and the DiT's parts
#include "vllm/model_executor/models/ltx2_audio_vae.h"  // Ltx2VaeWeights, the shared bag

namespace vllm {

// utils.py:7-12 — `torch.nn.functional.rms_norm`'s eps as `rms_norm` passes it.
// The connector uses the WEIGHTLESS form, so this is the only stabilizer in the
// residual path. NOT a member of the invisible-constant class, however near-zero
// the fixture's rows are: `rms_norm` adds the epsilon to the MEAN SQUARE, not to
// a row minimum, so it perturbs every row it normalizes. At the class's own 100x
// bar (1e-6 -> 1e-4) it REDS 5 of the arms in "ltx2 the Embeddings1DConnector
// reproduces upstream on every arm" — Split 0.0558581, Interleaved 0.104284,
// Float64 0.140343, NoRegisters 0.000542641, GatedNoBias 0.0892045.
// It is pinned as well as gated, in test_ltx2_pipeline.cpp, case "the constants
// the headers call pinned are actually pinned", which compares this against
// upstream's own `rms_norm` signature default rather than a retyped literal —
// the one check a regenerated golden cannot satisfy by moving with it.
inline constexpr double kLtx2ConnectorRmsNormEps = 1e-6;

// Embeddings1DConnector.__init__ defaults (:95-108), which are also both
// configurators' `transformer_config.get` fallbacks (:199-218, :227-255).
struct Ltx2ConnectorConfig {
  int64_t attention_head_dim = 128;
  int64_t num_attention_heads = 30;
  int64_t num_layers = 2;
  double positional_embedding_theta = 10000.0;
  // `[1]` upstream (:114-116). The RoPE grid is 1-D: one position axis over the
  // token index.
  std::vector<int64_t> positional_embedding_max_pos = {1};
  // 0 disables the substitution entirely, which is upstream's `None` (:103, :167).
  int64_t num_learnable_registers = 128;
  Ltx2RopeType rope_type = Ltx2RopeType::kSplit;
  bool double_precision_rope = false;
  bool apply_gated_attention = false;
  bool ff_bias = true;
  std::string prefix;

  int64_t inner_dim() const { return num_attention_heads * attention_head_dim; }
};

// The parameter contract, in `named_parameters()` order: `learnable_registers`
// first (a bare nn.Parameter precedes every submodule in torch's ordering), then
// each block's attention and feed-forward.
struct Ltx2ConnectorTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};
std::vector<Ltx2ConnectorTensorSpec> EnumerateLtx2ConnectorTensors(
    const Ltx2ConnectorConfig& config);

// The result of a connector forward: the conditioning the DiT's cross-attention
// consumes, plus the mask it must be paired with. Upstream returns both (:191)
// and a caller that keeps its ORIGINAL mask instead of this one re-masks
// positions the connector has just filled with registers.
struct Ltx2ConnectorOutput {
  std::vector<float> hidden_states;  // [batch, seq, inner_dim]
  std::vector<float> mask;           // [batch, 1, 1, seq], additive
};

// _replace_padded_with_learnable_registers (:139-152), exposed on its own because
// it is where the bf16 narrowing lives and a defect there is otherwise absorbed
// by two transformer blocks before anyone sees it.
Ltx2ConnectorOutput Ltx2ConnectorReplaceRegisters(const Ltx2ConnectorConfig& config,
                                                  const Ltx2VaeWeights& weights,
                                                  const float* hidden_states,
                                                  const float* additive_attention_mask,
                                                  int64_t batch, int64_t seq);

// Embeddings1DConnector.forward (:154-191). `additive_attention_mask` is
// [batch, 1, 1, seq] with 0 for a kept token and -finfo(f32).max for a padded
// one — the form `_prepare_attention_mask` produces (transformer_args.py:199-206)
// — and may be null only when the config carries no registers, which mirrors
// upstream's own unconditional dereference at :168-170.
Ltx2ConnectorOutput Ltx2ConnectorForward(const Ltx2ConnectorConfig& config,
                                         const Ltx2VaeWeights& weights,
                                         const float* hidden_states,
                                         const float* additive_attention_mask, int64_t batch,
                                         int64_t seq);

}  // namespace vllm
