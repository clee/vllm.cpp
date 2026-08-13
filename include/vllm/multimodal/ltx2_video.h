// LTX-2.5 behind the GENERALIZED video seam — the second family registered with
// `vllm::multimodal::VideoEngine`, and the driving loop that turns the L2-L6
// bricks into frames + a waveform.
//
// Row: MODEL-DIFFUSION-LTX25. Spec: .agents/specs/ltx-2-5.md phase L7. Issue #435.
//
// ─── WHAT THIS TU IS ─────────────────────────────────────────────────────────
//
// Phases L2-L6 shipped a DiT forward, a text feature extractor, two VAEs, a
// vocoder, an upsampler, a duration head, and a pipeline COMPONENT library —
// schedules, noisers, steppers, guiders, patchifiers, recipes. Nothing drove
// them: `Ltx2RefuseUnportedPipelineFeature(kVideoEngineWiring)` refused the
// composition BY NAME and named this phase as its owner. This TU is that
// composition and nothing else. It adds no numerics; every line either resolves
// a parameter, moves a buffer, or calls a brick that already has a golden.
//
// ─── WHAT IT IS A PORT OF (file:line on BOTH sides) ──────────────────────────
// Upstream: Lightricks/LTX-2 @ fd4ded7, packages/ltx-pipelines/src/ltx_pipelines/
//   OURS                              <-  UPSTREAM
//   Ltx2VideoEngine::Generate         <-  distilled.py:186-300 (DistilledPipeline.__call__)
//   the per-phase stage call          <-  utils/blocks.py:500-582 (DiffusionStage.__call__)
//   the latent state build            <-  utils/helpers.py:428-447 (create_noised_state)
//                                         + ltx-core tools.py:139-184 / :246-280
//   the denoise loop                  <-  utils/samplers.py:39-79 (euler_denoising_loop)
//                                         + :26-36 (_step_state)
//   the X0 conversion                 <-  ltx-core model/transformer/model.py:590-604
//                                         (X0Model.forward) + utils.py:38-50 (to_denoised)
//   the per-step Modality build       <-  utils/helpers.py:466-503
//                                         (modality_from_latent_state, timesteps_from_mask)
//   post_process_latent               <-  utils/helpers.py:462-464
//
// ─── WHAT THIS ENGINE REFUSES, AND WHY EACH WOULD RENDER ────────────────────
//
// 1. `device = 1` (CUDA) WITHOUT A CUDA BACKEND. Phase L7 refused every non-zero
//    device outright: L2's forward was f32-only by declaration and L6's
//    `Ltx2StreamDitToDevice` stages bf16 and refuses to widen, so no combination
//    put the DiT on a GPU. **Phase L8 closed that** — `Ltx2DitForwardDevice`
//    (ltx2_device.h) is the same graph with every activation in device memory and
//    the stream in the checkpoint's own bf16 — so a CUDA handle now denotes a
//    CUDA forward and the load succeeds.
//
//    What is still refused is the SUBSTITUTION. If the CUDA backend is not
//    registered in this build, the load is refused BY NAME rather than served the
//    CPU forward behind a CUDA-looking handle, because that substitution is what
//    would make every later timing and every "it ran on the GPU" claim false.
//
// 2. A PROMPT with no text tower. Phase L6 loads the caption projections and the
//    embedded tokenizer, and records the Gemma-4 TOWER itself as owed
//    (ltx2_loader.h:318-324) — so this engine cannot turn a prompt string into
//    hidden states. `has_encoder()` is therefore false and a prompt-carrying
//    request is refused BY NAME. Conditioning comes from prompt-embeds, which is
//    the seam's own documented fallback (video_engine.h:55-57).
//
//    WHAT PHASE L9c CHANGED, AND WHAT IT DID NOT. The link BELOW the tower is now
//    real: when the checkpoint carries the two `*_embeddings_connector` families
//    — both shipped LTX-2.5 DiTs do, 129 tensors each — the supplied prompt
//    embeds are run through the connector before the DiT sees them, with those
//    weights, under the checkpoint's own `connector_*` configuration. Before
//    L9c they went to cross-attention verbatim and the connector was reachable
//    only from a test. The TOWER is still owed, so what enters the connector is
//    still whatever the caller put in the file rather than an encoded prompt.
//
// 3. Any pipeline kind / model version the recipe table does not carry.
//    `ResolveLtx2PipelineRecipe` already throws rather than defaulting
//    (ltx2_pipeline.h:543-562); this engine passes the checkpoint's OWN
//    `model_version` through to it rather than assuming 2.5, because a checkpoint
//    of another generation resolved onto 2.5's sigmas renders confidently and
//    wrongly.
//
// ─── SCALE, STATED PLAINLY ───────────────────────────────────────────────────
//
// The f32 CPU forward is the parity forward, not a production one: at the shipped
// 21.00B geometry its weights alone are ~76 GB. The bf16 DEVICE forward (L8) is
// the production residency — ~42 GB staged tensor-by-tensor — and it is what
// `device = 1` runs. Neither changes the other number this row owes: a single
// denoise step over a 512x768x121 latent is ~2.6e14 FLOPs, and there is no
// production-configuration oracle to divide by, so no speed figure is claimed
// anywhere in this family (spec §0).
#pragma once

#include <memory>
#include <string>

#include "vllm/multimodal/video_engine.h"

namespace vllm {
struct Ltx2DitParams;
}  // namespace vllm

namespace vllm::multimodal {

// The stable registry name this family is reached under
// (VideoModelParams::family / vllm_video_model_params.family). It is the string
// `.agents/specs/ltx-2-5.md` and the L1 registry refusal test already print.
inline constexpr char kLtx2VideoFamily[] = "ltx-2.5";

// ── the family-specific LOAD extras (VideoModelParams::extras) ──────────────
// Every one of these is a knob upstream reads from somewhere this seam has no
// field for. An extra this family does not define is REFUSED, never ignored.

// The audio stream's prompt-embeds file, the twin of the seam's
// `prompt_embeds_path` (which carries the VIDEO stream). LTX-2.5 conditions two
// streams at two different widths — 4096 and 2048 — and one file cannot hold
// both, so the audio half rides here. Rows of `audio_cross_attention_dim`,
// little-endian f32, and the two files must agree on their ROW COUNT because
// upstream's two encodings come from one tokenization.
inline constexpr char kLtx2AudioPromptEmbedsExtra[] = "audio_prompt_embeds_path";

// `resolve_ltx_pipeline_recipe`'s first key (ltx2_recipes.py:161-175). Defaults
// to "distilled_two_stage", which is what the shipped
// `ltx-2.5-22b-distilled-transformer` is: the file NAMES itself distilled and
// `DistilledPipeline` is the entry point that loads it.
inline constexpr char kLtx2PipelineKindExtra[] = "pipeline_kind";

// Overrides the `model_version` the DiT checkpoint declares in its own
// `__metadata__`. Present for a checkpoint that carries none; it never silently
// replaces one that does, and a mismatch between the two is reported.
inline constexpr char kLtx2ModelVersionExtra[] = "model_version";

// The DiT's `{"transformer": {...}}` configuration, as a JSON FILE, for a
// checkpoint whose `__metadata__` carries none.
//
// MEASURED 2026-08-12, and it is why this extra exists: of the two shipped
// LTX-2.5 DiTs only the first-party NVFP4 one carries `__metadata__` at all.
// `vonkaiser/LTX-2.5-FP8-NVFP4`'s FP8 DiT — the copy every phase before L6 gated
// against and the one L8 ran on the GPU — has NO `__metadata__` key whatsoever.
// Without a config the geometry still resolves from SHAPES, but the values no
// shape encodes fall back to the parser's defaults: `double_precision_rope =
// false` and `av_ca_timestep_scale_multiplier = 1`, against LTX-2.5's declared
// `float64` and `1000`. Both move every RoPE angle and every audio<->video
// modulation, so a silent default is a DIFFERENT MODEL rendering confidently.
//
// So a DiT that declares no config is REFUSED unless this extra names one. The
// file holds the same object the shipped checkpoints put in
// `__metadata__["config"]` — `{"transformer": {...}}` — and it is adopted through
// the IDENTICAL weight-contract check the declared path uses, so a config
// belonging to another checkpoint is refused rather than bound.
inline constexpr char kLtx2DitConfigPathExtra[] = "dit_config_path";

// Proceed past the module families the L2 contract does not carry
// (`prompt_adaln_single`, `keyframes_abs_pos_embedding`, the two embeddings
// connectors — ltx2_loader.h:82-104). "1" opts in; anything else leaves the
// loader's refusal in place. The shipped DiTs all carry at least one of them, so
// this is the flag that says "gate the ported subset knowingly".
inline constexpr char kLtx2AllowUnportedExtra[] = "allow_unported_modules";

// Run only the phases up to and including this index of the resolved recipe
// (0-based). Absent runs every phase. This exists because the two-stage recipe's
// second phase needs the latent spatial upsampler, and a run without one must
// say so rather than skipping the phase silently.
inline constexpr char kLtx2MaxPhaseExtra[] = "max_phase";

// How many of the supplied prompt-embeds rows are REAL tokens; the rest are
// padding. Absent means every row is real.
//
// WHY A SEAM WITH NO TOKENIZER NEEDS THIS. The embeddings connector substitutes
// its `learnable_registers` table at PADDED positions
// (embeddings_connector.py:139-152), so the padding is not inert — it is what
// decides which of the connector's inputs are learned constants rather than
// caption features. Upstream always knows this, because the tokenizer produced
// the mask. This seam takes prompt embeds from a FILE, which carries no mask, so
// without this extra the padded tail would be conditioned on as if it were text
// and every register would go unused. Recorded as a knob rather than assumed,
// and it is the field the Gemma-4 tower will supply when it lands.
inline constexpr char kLtx2PromptValidRowsExtra[] = "prompt_embeds_valid_rows";

// A loaded LTX-2.5 checkpoint set. Construct through
// `vllm::multimodal::LoadVideoEngine` (detection) or by declaring
// `family = kLtx2VideoFamily`; this type is exposed so a test can name it.
class Ltx2VideoEngine : public VideoEngine {
 public:
  static std::unique_ptr<Ltx2VideoEngine> Load(const VideoModelParams& params);

  Ltx2VideoEngine(Ltx2VideoEngine&&) noexcept;
  Ltx2VideoEngine& operator=(Ltx2VideoEngine&&) noexcept;
  ~Ltx2VideoEngine() override;

  std::string family() const override;
  vt::Device device() const override;
  bool has_encoder() const override;
  bool has_prompt_embeds() const override;
  VideoResult Generate(const VideoGenParams& params) override;

  // The `model_version` this engine resolved its recipe with ("2.5"), and the
  // pipeline kind it resolved with. Exposed because "which recipe ran" is the
  // one thing a rendered clip cannot be inspected for.
  const std::string& model_version() const;
  const std::string& pipeline_kind() const;

  // The DiT parameters THIS ENGINE LOADED — the ones its forward actually runs
  // under, after the checkpoint's own declared config has been adopted.
  //
  // Exposed because the values that config decides are exactly the ones a
  // rendered clip cannot be inspected for and a SHAPE cannot see:
  // `double_precision_rope` and `av_ca_timestep_scale_multiplier` move every
  // RoPE angle and every audio<->video modulation while leaving the tensor set
  // byte-identical. A test that re-derives them from the file and asserts on its
  // own local copy proves nothing about what the engine bound; this accessor is
  // what lets it assert on the engine.
  const Ltx2DitParams& dit_params() const;

 private:
  Ltx2VideoEngine();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Does this checkpoint set hold an LTX-2.5 DiT? Exposed for the registry and for
// a test that wants the answer without a load. See the definition for the
// discriminator and for why it cannot collide with MiniMax-H3's.
bool DetectLtx2Video(const VideoModelParams& params);

}  // namespace vllm::multimodal
