#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_gemma_tower_goldens.inc — the Gemma-4 TOWER oracle.

Phase L3 recorded that this gate could not exist: the `transformers` on the box
had no `gemma4_unified` in `CONFIG_MAPPING`, so the tower could not be built at
reduced dimensions and there was nothing to compare against. That is no longer
true, and this generator is the proof — it BUILDS AND RUNS the tower.

What it gates, and why each piece is here rather than assumed:

  * `Gemma4Model::ForwardHiddenStates` against the REAL upstream forward. Until
    now the Gemma-4 port's own header said the forward was "grounded + compiles"
    with the end-to-end gate BLOCKED (gemma4.h, G1 HONEST STATUS). Compiling is
    not running, and this is the first execution-grounded comparison.

  * The MIXED per-layer attention geometry the LTX tower actually ships. The
    shipped 12B has 48 layers in an 8-fold `sliding, sliding, sliding, sliding,
    sliding, full` pattern, and the two kinds are NOT the same shape:

        sliding : 16 q heads x head_dim 256, 8 kv heads, v_proj PRESENT
        full    : 16 q heads x global_head_dim 512, 1 kv head, NO v_proj
                  (attention_k_eq_v: V aliases K), rope_type "proportional"
                  with partial_rotary_factor 0.25 and theta 1e6, against the
                  sliding layers' plain rope at theta 1e4

    A fixture with one uniform layer type cannot separate a port that handles
    both from a port that handles one and silently applies it twice, which is
    §7.0(c)'s "a fixture that cannot discriminate is the same defect as an
    unpinned constant". The reduced arch below keeps BOTH kinds and keeps the
    ratios (global head_dim = 2 x sliding head_dim, global kv heads = 1).

  * The LEFT-PADDING EQUIVALENCE, which is what makes a prompt affordable.
    Upstream pads every prompt to 1024 (gemma_assets.py:162,
    base_encoder.py:231-236, PaddingSide.LEFT) and runs all 1024 rows through a
    12B tower. Our port runs only the VALID tokens at their ORIGINAL absolute
    positions. That is equivalent -- pads are masked out of attention and are
    causally before every valid token, and the feature extractor zeroes their
    rows anyway -- but "is equivalent" is a claim, so section 3 emits the full
    left-padded oracle run and the C++ suite holds the short run's valid rows
    to the padded run's valid rows. If the equivalence is ever false, that gate
    is what says so, and a 100x cost claim stops resting on an argument.

  * ONE arithmetic width per state, both ways round. Section 2 is the oracle in
    float32 and section 4 the SAME oracle in bfloat16. Our forward carries the
    stream in bf16 and widens only on the way out (gemma4.h,
    Gemma4HiddenStatesResult), so bf16 is the dtype-MATCHED arm and f32 is the
    arm that would catch a reduction-order defect a bf16 store absorbs. Gating
    only one of them has burned this project before.

Both sides rebuild every weight from one deterministic FNV-1a + splitmix64
stream keyed by the parameter's own HuggingFace NAME, exactly as
scripts/gen-ltx2-text-goldens.py does, so no weight byte is checked in and the
weight-NAMING contract is itself part of the gate. A port that merges q/k/v in
the wrong order, or that reads `v_proj` on a full-attention layer that has none,
builds a different tensor from the same names and fails.

Upstream sources:
  transformers/models/gemma4_unified/modeling_gemma4_unified.py  -> the tower
  Lightricks/LTX-2 packages/ltx-core/src/ltx_core/text_encoders/gemma/
    encoders/base_encoder.py:68        -> model.model(..., output_hidden_states=True)
    encoders/encoder_configurator.py:68-73 -> AutoModelForImageTextToText.from_config
    encoders/base_encoder.py:231-236   -> max_length 1024, PaddingSide.LEFT
  diffusers src/diffusers/pipelines/ltx2/pipeline_ltx2.py:347-352 -> the same
    call and the stack/flatten pack, read as an INDEPENDENT second opinion

The real tower's `text_config` is not invented here: it is the one carried in
`__metadata__["gemma_config"]` of the OFFICIAL bf16 text encoder
(Lightricks/LTX-2.5, text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors),
read with an 84 KB HTTP range request over the safetensors header and committed
as `--real-config`. The SHIPPED vonkaiser NVFP4 file carries no `__metadata__`
at all, which is why the config has to come from somewhere and why it is not
guessed.

Usage:
    scripts/gen-ltx2-gemma-tower-goldens.py \
        --python /home/mudler/recon-cpu/venv/bin/python \
        --out tests/vllm/models/ltx2_gemma_tower_goldens.inc

Needs a `transformers` that registers `gemma4_unified` (>= 5.8; MEASURED working
at 5.12.1, MEASURED absent at 5.3.0) plus torch and numpy. CPU only. No
checkpoint, no download, no network.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# ORACLE IDENTITY — asserted, never assumed (spec 7.0(1)).
#
# A decoy once produced byte-identical goldens on this project, so the module
# that answers has to be the module we meant. `transformers` 5.3.0 is present on
# the same box under a different interpreter and does NOT register
# `gemma4_unified`; picking it up by accident reads exactly like "Gemma-4 is
# unsupported" rather than like a wrong environment.
# ---------------------------------------------------------------------------

MIN_TRANSFORMERS = (5, 8)


def assert_oracle_identity() -> dict:
    import transformers  # noqa: PLC0415
    from transformers import CONFIG_MAPPING  # noqa: PLC0415

    version = transformers.__version__
    parts = []
    for piece in version.split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts) < MIN_TRANSFORMERS:
        raise SystemExit(
            f"transformers {version} is below the {MIN_TRANSFORMERS[0]}."
            f"{MIN_TRANSFORMERS[1]} that first registers gemma4_unified. "
            "Point --python at an interpreter whose transformers has it; this "
            "is the exact blocker phase L3 recorded."
        )
    if "gemma4_unified" not in CONFIG_MAPPING:
        raise SystemExit(
            f"transformers {version} at {transformers.__file__} does not register "
            "'gemma4_unified' in CONFIG_MAPPING. Refusing to emit goldens from a "
            "tower this interpreter cannot build."
        )
    return {
        "transformers_version": version,
        "transformers_path": str(Path(transformers.__file__).resolve().parent),
        "torch_version": torch.__version__,
    }


def upstream_revision(root: Path) -> str:
    """HEAD, plus a DIRTY marker.

    `git rev-parse HEAD` reports the committed SHA whether or not the tree has
    uncommitted edits, so a revision recorded from it alone can name a commit
    that is not what ran (spec 7.0(2)).
    """
    if not root.is_dir():
        return "absent"
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a tarball checkout has no git metadata
        return "unknown"
    return head + ("-DIRTY" if status else "")


# ---------------------------------------------------------------------------
# The deterministic stream — byte-identical to scripts/gen-ltx2-text-goldens.py
# and to tests/vllm/models/test_ltx2_text_encoder.cpp :: Ltx2Rand.
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & _MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def ltx2_rand(name: str, count: int) -> np.ndarray:
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = (u >> 11) * (2.0**-53) * 2.0 - 1.0
    return out


def gemma_param_spec(name: str) -> tuple[float, float]:
    """(scale, offset) keyed ONLY by the HuggingFace parameter name.

    The offsets are not cosmetic. Gemma-4's RMSNorm is PLAIN — it multiplies by
    `self.weight` directly rather than by `1 + weight`
    (modeling_gemma4_unified.py:181-185), and its weights initialize to ones — so
    a norm weight centred on 0 would put every normalized activation near zero
    and hide a scale error. `layer_scalar` is a BUFFER initialized to ones
    (:501) and applied as `hidden_states *= self.layer_scalar` (:535); leaving it
    at exactly 1 would make a port that ignores it entirely pass.
    """
    if name.endswith("layer_scalar"):
        return 0.1, 1.0
    if name.endswith("_layernorm.weight") or name.endswith("norm.weight"):
        return 0.1, 1.0
    if name.endswith("embed_tokens.weight"):
        return 0.05, 0.0
    if name.endswith(".bias"):
        return 0.02, 0.0
    return 0.05, 0.0


def make_param(name: str, shape) -> torch.Tensor:
    scale, offset = gemma_param_spec(name)
    count = int(np.prod(shape)) if len(shape) else 1
    values = ltx2_rand(name, count) * scale + offset
    return torch.from_numpy(values.astype(np.float32)).reshape(tuple(shape))


# ---------------------------------------------------------------------------
# The reduced architecture.
#
# Every ratio the port branches on is preserved; only the magnitudes shrink.
# The layer_types pattern keeps the shipped tower's shape -- a run of sliding
# layers CLOSED by a full one, twice -- so a port that mixes the two head_dims
# up, or that resolves layer 0's geometry once and reuses it, diverges.
# ---------------------------------------------------------------------------

HIDDEN = 32                 # real 3840
NUM_LAYERS = 12             # real 48
HEAD_DIM = 8                # real 256
GLOBAL_HEAD_DIM = 16        # real 512
NUM_HEADS = 4               # real 16
NUM_KV_HEADS = 2            # real 8
NUM_GLOBAL_KV_HEADS = 1     # real 1  (unchanged: it IS one upstream)
INTERMEDIATE = 64           # real 15360
VOCAB = 64                  # real 262144
SLIDING_WINDOW = 6          # real 1024
MAX_POSITION = 128          # real 262144

# The shipped pattern is (sliding x 5, full) x 8. Two repeats is the smallest
# fixture in which a full layer is neither first nor last and a sliding layer
# FOLLOWS a full one -- the case a port that latches geometry once gets wrong.
LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 2
assert len(LAYER_TYPES) == NUM_LAYERS

# Section 1/2: an unpadded run. Section 3: the same VALID tokens left-padded.
TOKENS = [2, 17, 41, 5, 23, 9, 60, 33]
SEQ = len(TOKENS)
PADDED_SEQ = 20                       # > SLIDING_WINDOW, so the window is LIVE
PAD_ID = 0
NUM_PAD = PADDED_SEQ - SEQ


def reduced_text_config(real: dict) -> dict:
    """The REAL text_config with only the magnitudes reduced.

    Starting from the shipped config rather than from a hand-written dict is
    deliberate: every field this generator does not name -- `attention_k_eq_v`,
    `final_logit_softcapping`, `rms_norm_eps`, both `rope_parameters` entries,
    `tie_word_embeddings`, `hidden_activation` -- is carried over EXACTLY, so the
    fixture cannot quietly disagree with the checkpoint about a field nobody
    thought to reduce.
    """
    t = dict(real)
    t.update(
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        global_head_dim=GLOBAL_HEAD_DIM,
        num_global_key_value_heads=NUM_GLOBAL_KV_HEADS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
        sliding_window=SLIDING_WINDOW,
        max_position_embeddings=MAX_POSITION,
        layer_types=list(LAYER_TYPES),
    )
    return t


def build_tower(real_config: dict):
    """Exactly `GemmaTextEncoderConfigurator.from_metadata` at reduced dims."""
    from transformers import CONFIG_MAPPING, AutoModelForImageTextToText  # noqa: PLC0415

    wrapper = dict(real_config)
    wrapper["text_config"] = reduced_text_config(real_config["text_config"])
    # The vision and audio towers are not on the text-conditioning path and the
    # checkpoint's copies of them are not reduced here; dropping the sub-configs
    # makes transformers skip building them entirely.
    wrapper.pop("vision_config", None)
    wrapper.pop("audio_config", None)

    config_cls = CONFIG_MAPPING[real_config["model_type"]]
    config = config_cls.from_dict(wrapper)
    model = AutoModelForImageTextToText.from_config(config)
    inner = model.model  # base_encoder.py:68 -- the inner model, no lm_head

    filled = []
    with torch.no_grad():
        for name, param in inner.named_parameters():
            param.copy_(make_param(name, tuple(param.shape)))
            filled.append((name, tuple(param.shape)))
        # `layer_scalar` is a BUFFER, so named_parameters() misses it. It
        # multiplies every layer's output (:535) and a port that drops it is
        # invisible while it stays at its initialized 1.0.
        for name, buf in inner.named_buffers():
            if name.endswith("layer_scalar"):
                buf.copy_(make_param(name, tuple(buf.shape)))
                filled.append((name, tuple(buf.shape)))
    return model, inner, config, filled


def run_tower(inner, ids, mask, dtype, positions=None):
    m = inner.to(dtype).eval()
    kwargs = {}
    if positions is not None:
        # The absolute positions the tokens occupy in the padded batch. Left
        # padding does NOT renumber them: with no explicit `position_ids`
        # transformers derives them from `cache_position`, which counts the pad
        # rows, so a short run has to be told where its tokens really sit.
        kwargs["position_ids"] = torch.tensor([positions], dtype=torch.long)
    with torch.no_grad():
        out = m(
            input_ids=torch.tensor([ids], dtype=torch.long),
            attention_mask=torch.tensor([mask], dtype=torch.long),
            output_hidden_states=True,
            **kwargs,
        )
    states = [h[0].to(torch.float32).contiguous().numpy() for h in out.hidden_states]
    inner.to(torch.float32)
    return states


# ---------------------------------------------------------------------------
# Emission helpers (identical to scripts/gen-ltx2-text-goldens.py's)
# ---------------------------------------------------------------------------


def _cxx_float(value: float, digits: int) -> str:
    if value != value:
        return "NAN"
    if value == float("inf"):
        return "INFINITY"
    if value == float("-inf"):
        return "-INFINITY"
    return repr(float(f"%.{digits}g" % value))


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
    out.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        out.write("    " + ", ".join(_cxx_float(v, 9) + "f" for v in flat[i : i + 6]) + ",\n")
    out.write("};\n\n")


def emit_i32(out, name: str, values) -> None:
    flat = [int(v) for v in np.asarray(values).reshape(-1).tolist()]
    out.write(f"inline constexpr int32_t {name}[] = {{\n")
    for i in range(0, len(flat), 12):
        out.write("    " + ", ".join(str(v) for v in flat[i : i + 12]) + ",\n")
    out.write("};\n\n")


def emit_scalar(out, name: str, value) -> None:
    out.write(f"inline constexpr int64_t {name} = {int(value)};\n")


def emit_string(out, name: str, value: str) -> None:
    out.write(f'inline constexpr const char* {name} = R"JSON({value})JSON";\n\n')


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--real-config",
        type=Path,
        default=None,
        help="the gemma_config JSON from the official bf16 TE's __metadata__; "
        "defaults to the copy committed beside this script",
    )
    ap.add_argument("--ltx2", type=Path, default=Path.home() / "_git" / "LTX-2")
    ap.add_argument("--diffusers", type=Path, default=Path.home() / "_git" / "diffusers")
    args = ap.parse_args()

    identity = assert_oracle_identity()
    real_config_path = args.real_config or (
        Path(__file__).resolve().parent.parent
        / "tests" / "vllm" / "models" / "ltx2_gemma4_text_config.json"
    )
    real_config = json.loads(real_config_path.read_text())

    model, inner, config, filled = build_tower(real_config)

    padded_ids = [PAD_ID] * NUM_PAD + TOKENS
    padded_mask = [0] * NUM_PAD + [1] * SEQ

    plain_f32 = run_tower(inner, TOKENS, [1] * SEQ, torch.float32)
    plain_bf16 = run_tower(inner, TOKENS, [1] * SEQ, torch.bfloat16)
    padded_f32 = run_tower(inner, padded_ids, padded_mask, torch.float32)
    # The equivalence claim, isolated INSIDE the oracle and in f32 so no dtype
    # noise is mixed into it: the same valid tokens, told their absolute
    # positions, run WITHOUT the pads. If upstream's own two answers agree, the
    # claim "dropping the pads is free" is upstream's property; our port then
    # only has to inherit it, and the C++ gate can hold those two claims apart.
    short_abs_f32 = run_tower(
        inner, TOKENS, [1] * SEQ, torch.float32,
        positions=list(range(NUM_PAD, NUM_PAD + SEQ)),
    )
    equivalence = [
        float(np.abs(s - p[NUM_PAD:]).max())
        for s, p in zip(short_abs_f32, padded_f32)
    ]
    sys.stderr.write(
        "left-pad equivalence measured INSIDE the oracle (f32, per state): "
        f"max {max(equivalence):.3e}\n"
    )

    assert len(plain_f32) == NUM_LAYERS + 1, len(plain_f32)

    argv = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-ltx2-gemma-tower-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// The Gemma-4 TOWER oracle: the upstream HuggingFace implementation BUILT\n"
            "// AND RUN at reduced dimensions, which phase L3 recorded as impossible on\n"
            "// the transformers it had. Every weight is rebuilt on both sides from the\n"
            "// deterministic Ltx2Rand stream keyed by the HuggingFace parameter NAME, so\n"
            "// no weight byte is checked in and the naming contract is part of the gate.\n"
            "//\n"
            f"// Oracle: transformers {identity['transformers_version']} at\n"
            f"//         {identity['transformers_path']}\n"
            f"//         torch {identity['torch_version']}\n"
            f"// Lightricks/LTX-2 revision:  {upstream_revision(args.ltx2)}\n"
            f"// diffusers revision:         {upstream_revision(args.diffusers)}\n"
            "// Regenerate with:\n"
            f"//   {argv}\n"
            "#pragma once\n\n"
            "#include <cstdint>\n\n"
            "namespace vllm_test {\n\n"
        )

        out.write("// --- section 0: the reduced architecture ---\n")
        emit_scalar(out, "kLtxTowerHidden", HIDDEN)
        emit_scalar(out, "kLtxTowerNumLayers", NUM_LAYERS)
        emit_scalar(out, "kLtxTowerNumStates", NUM_LAYERS + 1)
        emit_scalar(out, "kLtxTowerHeadDim", HEAD_DIM)
        emit_scalar(out, "kLtxTowerGlobalHeadDim", GLOBAL_HEAD_DIM)
        emit_scalar(out, "kLtxTowerNumHeads", NUM_HEADS)
        emit_scalar(out, "kLtxTowerNumKvHeads", NUM_KV_HEADS)
        emit_scalar(out, "kLtxTowerNumGlobalKvHeads", NUM_GLOBAL_KV_HEADS)
        emit_scalar(out, "kLtxTowerIntermediate", INTERMEDIATE)
        emit_scalar(out, "kLtxTowerVocab", VOCAB)
        emit_scalar(out, "kLtxTowerSlidingWindow", SLIDING_WINDOW)
        emit_scalar(out, "kLtxTowerSeq", SEQ)
        emit_scalar(out, "kLtxTowerPaddedSeq", PADDED_SEQ)
        emit_scalar(out, "kLtxTowerNumPad", NUM_PAD)
        emit_scalar(out, "kLtxTowerPadId", PAD_ID)
        out.write("\n")
        # The exact reduced text_config, so the C++ side PARSES what ran rather
        # than reconstructing it field by field and drifting.
        emit_string(
            out,
            "kLtxTowerTextConfigJson",
            json.dumps(reduced_text_config(real_config["text_config"]), indent=1),
        )
        out.write("// The layer types, in order. 1 = full_attention, 0 = sliding.\n")
        emit_i32(out, "kLtxTowerLayerIsFull",
                 [1 if t == "full_attention" else 0 for t in LAYER_TYPES])

        out.write("// --- section 1: the inputs ---\n")
        emit_i32(out, "kLtxTowerTokens", TOKENS)
        emit_i32(out, "kLtxTowerPaddedTokens", padded_ids)
        emit_i32(out, "kLtxTowerPaddedMask", padded_mask)

        out.write(
            "// --- section 2: hidden states, oracle in FLOAT32 ---\n"
            "// [state][seq * hidden]. state 0 is the sqrt(hidden)-scaled embeddings;\n"
            "// state i is decoder layer i-1's output; the LAST state is model.norm of\n"
            "// the last layer, and the raw last-layer output never appears.\n"
        )
        for i, s in enumerate(plain_f32):
            emit_f32(out, f"kLtxTowerStateF32_{i}", s)

        out.write("// --- section 3: hidden states, oracle in BFLOAT16 (the SHIPPED dtype) ---\n")
        for i, s in enumerate(plain_bf16):
            emit_f32(out, f"kLtxTowerStateBf16_{i}", s)

        # THE NOISE FLOOR, MEASURED — the thing that makes the gate's tolerance a
        # measurement instead of a number somebody picked.
        #
        # Upstream's own answer moves by this much when the SAME code runs at
        # bf16 instead of f32, so it is the smallest difference this comparison
        # can possibly resolve. Holding our bf16 forward to it says something
        # precise: we are closer to upstream-in-bf16 than upstream-in-bf16 is to
        # upstream-in-f32. It also cannot be relaxed to rescue a failing port —
        # widening it means regenerating it, which means the oracle itself moved.
        noise = [float(np.abs(a - b).max()) for a, b in zip(plain_f32, plain_bf16)]
        scale = [float(np.abs(a).max()) for a in plain_f32]
        out.write(
            "// --- section 3b: the oracle's OWN f32-vs-bf16 spread, per state ---\n"
            "// The dtype noise floor, MEASURED rather than assumed. This is the\n"
            "// tolerance: a port whose bf16 answer sits inside it is indistinguishable\n"
            "// from upstream at upstream's own arithmetic width, and one that sits\n"
            "// outside it has a defect that bf16 rounding does not explain.\n"
        )
        emit_f32(out, "kLtxTowerDtypeNoise", noise)
        emit_f32(out, "kLtxTowerStateScale", scale)

        out.write(
            "// --- section 4: the LEFT-PADDED run, float32 ---\n"
            "// [state][padded_seq * hidden]. Rows 0..kLtxTowerNumPad-1 are pad rows and\n"
            "// their contents are upstream's garbage-but-masked values; the gate reads\n"
            "// only the VALID tail and holds section 2 to it.\n"
        )
        for i, s in enumerate(padded_f32):
            emit_f32(out, f"kLtxTowerPaddedStateF32_{i}", s)

        out.write(
            "// --- section 5: the equivalence, measured INSIDE the oracle ---\n"
            "// Per state, max|short-run-at-absolute-positions - padded-run's valid rows|,\n"
            "// both f32, so this number carries NO dtype noise. It is upstream's own\n"
            "// answer to 'is dropping the pads free?'. Our port inherits it; the C++\n"
            "// gate checks the two claims separately so a failure says which one broke.\n"
        )
        emit_f32(out, "kLtxTowerPadEquivalence", equivalence)

        out.write("}  // namespace vllm_test\n")

    sys.stderr.write(
        f"wrote {args.out} — {len(filled)} named tensors, "
        f"{NUM_LAYERS + 1} states x {SEQ} x {HIDDEN}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
