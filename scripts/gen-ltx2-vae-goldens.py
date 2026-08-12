#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_vae_goldens.inc — the LTX-2.5 VAE parity oracle.

LTX-2.5's decoders are pure-Python `ltx_core` modules (Lightricks/LTX-2,
`packages/ltx-core/src/ltx_core/model/`). The full checkpoint is ~30 GB and its
DiT does not fit this project's CI, but the VAE MATH is gateable exactly on any
CPU: this generator imports the upstream modules VERBATIM, builds them at REDUCED
dimensions with deterministic pseudo-random weights, runs them, and emits the
resulting tensors as C++ goldens. The C++ suite regenerates the identical weights
and inputs from the identical PRNG and must reproduce these outputs, so NO WEIGHT
BYTE is checked in. This mirrors the method that made the MiniMax-H3 VAE bricks
trustworthy (scripts/gen-minimax-h3-audio-vae-goldens.py).

Upstream sources (Lightricks/LTX-2, packages/ltx-core/src/ltx_core/):
  model/audio_vae/audio_vae.py          -> section 1 (audio decoder)
  model/audio_vae/vocoder.py            -> sections 2-4 (vocoder, legacy arm, BWE)
  model/video_vae/conv_video_decoder.py -> section 5 (Conv video decoder)

Usage:
    python3 scripts/gen-ltx2-vae-goldens.py \\
        --ltx2 ~/_git/LTX-2 \\
        --out tests/vllm/models/ltx2_vae_goldens.inc

Needs torch + numpy + einops (CPU only). `ltx_core` is imported with a single
sys.path insert; no checkpoint, venv, or gated download is involved.

UPSTREAM REVISION ANCHOR. The goldens are only interpretable against the exact
upstream tree that produced them: if Lightricks changes `video_vae/resnet.py` and
someone regenerates, the numbers move, and without a SHA nobody can tell whether
the PORT drifted or UPSTREAM did, nor bisect from anywhere. So this generator
resolves `git -C <--ltx2> rev-parse HEAD` at generation time and emits it into the
`.inc` twice: once as a header comment for a human, and once as
`kLtx2VaeUpstreamRevision`, which the C++ suite asserts against the SHA PINNED in
the test. That makes the anchor load-bearing rather than decorative — a
regeneration against a different checkout fails the gate instead of silently
replacing the oracle.

  Pinned revision: fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca

Advancing the pin is a deliberate, reviewable edit in BOTH places (here and
`kLtx2VaeUpstreamRevisionPin` in tests/vllm/models/test_ltx2_vae.cpp), never a
side effect of regenerating.

ORACLE IDENTITY, asserted rather than assumed. `sys.path.insert(0, ...)` normally
wins, but path precedence is not a guarantee this script should be staking the
oracle on: a `.pth` file, an editable install, a namespace-package layout or a
later refactor to `sys.path.append` all make a pip-installed `ltx_core` resolve
instead, and it would import silently and gate against the wrong source. So the
resolved `ltx_core.__file__` is checked to live under `--ltx2` before anything
runs. This mirrors scripts/gen-ltx2-goldens.py (phase L2).

Two harness adaptations, both recorded because they change nothing about the math:

  * `ConvVideoDecoder` injects Gaussian noise through `torch.randn` (decoder
    timestep conditioning, and `inject_noise` blocks). `torch.randn` is patched
    for the duration of the call to draw from the shared deterministic stream,
    keyed by CALL INDEX — exactly the ordering guarantee an upstream
    `torch.Generator` gives. The C++ side consumes an Ltx2NoiseStream in the same
    order.
  * Anti-aliasing filters (`*.filter`) are SKIPPED when filling weights: they are
    kaiser-sinc / hann-sinc windows COMPUTED at construction, never loaded, and
    both sides must build them independently. They are gated on their own
    (sections 2 and 4).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Deterministic weight/input stream, mirrored bit-for-bit by the C++ suite
# (tests/vllm/models/test_ltx2_vae.cpp :: Ltx2Rand). A per-tensor FNV-1a seed plus
# a splitmix64 counter makes every tensor independent of fill ORDER, so the two
# sides cannot silently drift by reordering their parameter construction. It is
# the same stream the MiniMax-H3 goldens use.
# ---------------------------------------------------------------------------


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


def ltx_rand(name: str, count: int) -> np.ndarray:
    """`count` values uniform in [-1, 1), reproducible from `name` alone."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = ((u >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


# ---------------------------------------------------------------------------
# The per-parameter role rule. Both sides implement it; the C++ side calls it
# with the same (name, rank), so a divergence here shows up as a golden mismatch
# rather than as a silently different tensor.
# ---------------------------------------------------------------------------


def param_values(name: str, shape) -> np.ndarray:
    count = int(np.prod(shape)) if len(shape) else 1
    rank = len(shape)
    if name == "timestep_scale_multiplier":
        # Upstream's own value (conv_video_decoder.py:257); not a random weight.
        return np.full(count, 1000.0)
    if name.endswith("mel_basis"):
        # A mel filterbank is NON-NEGATIVE. Signed values would push most bins
        # under log's 1e-5 clamp, and a saturated golden hides errors.
        basis = np.abs(ltx_rand(name, count)) * 0.2 + 0.05
        # ...EXCEPT for the one arm that exists precisely to saturate it. The
        # clamp `torch.clamp(mel, min=1e-5)` (vocoder.py:515) sets the floor of the
        # log-mel the bwe_generator consumes, and it is the member of the
        # invisible-constant class that BINDS IN PRODUCTION, because real silence
        # reaches it. The well-scaled basis above never can — measured: the raw
        # mel minimum is ~4.4e-3, and it stays there even for a zero input,
        # because the vocoder's conv biases keep the waveform off silence. Scaling
        # the basis by 1e-4 puts EVERY bin under the clamp, so on this arm the
        # constant alone decides the generator's input and a mutation of it moves
        # the golden.
        if ".bwequiet." in name:
            basis *= 1e-4
        return basis
    if name.endswith(".gamma"):
        return ltx_rand(name, count) * 0.1 + 1.0
    if name.endswith(".alpha") or name.endswith(".beta"):
        return ltx_rand(name, count) * 0.2
    if name.endswith("std-of-means"):
        return ltx_rand(name, count) * 0.1 + 1.0
    if name.endswith("mean-of-means"):
        return ltx_rand(name, count) * 0.1
    if name.endswith("scale_shift_table"):
        return ltx_rand(name, count) * 0.1
    if name.endswith("per_channel_scale1") or name.endswith("per_channel_scale2"):
        return ltx_rand(name, count) * 0.1
    if name.endswith(".bias"):
        return ltx_rand(name, count) * 0.05
    if rank == 1 and name.endswith(".weight"):
        # A 1-D `.weight` is an affine norm gain, initialized to ones upstream.
        return ltx_rand(name, count) * 0.1 + 1.0
    return ltx_rand(name, count) * 0.1


def fill_from_stream(module, prefix: str = "") -> list[tuple[str, int]]:
    """Overwrite every parameter/buffer from the shared stream.

    Returns the (name, count) manifest in state_dict order, which the C++ side
    asserts its own parameter bag matches EXACTLY — so a parameter one side
    builds and the other does not is a test failure, not a silent no-op.
    """
    import torch

    state = module.state_dict()
    manifest: list[tuple[str, int]] = []
    filled = {}
    for name, tensor in state.items():
        if name.endswith(".filter"):
            # kaiser-sinc / hann-sinc windows are COMPUTED, never loaded.
            filled[name] = tensor
            continue
        values = param_values(prefix + name, tuple(tensor.shape))
        filled[name] = torch.from_numpy(values.astype(np.float32)).reshape(tensor.shape)
        manifest.append((prefix + name, int(values.size)))
    module.load_state_dict(filled, strict=True)
    return manifest


def make_input(name: str, shape, scale: float):
    import torch

    count = int(np.prod(shape))
    values = ltx_rand(name, count) * scale
    return torch.from_numpy(values.astype(np.float32)).reshape(shape)


# ---------------------------------------------------------------------------
# Emit helpers
# ---------------------------------------------------------------------------


def _cxx_float(value: float, digits: int) -> str:
    if not math.isfinite(value):
        raise ValueError(f"refusing to emit non-finite golden value: {value}")
    text = f"{value:.{digits}g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
    out.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(_cxx_float(v, 9) + "f" for v in flat[i : i + 6])
        out.write("    " + chunk + ",\n")
    out.write("};\n\n")


def emit_scalar(out, name: str, value) -> None:
    out.write(f"inline constexpr int64_t {name} = {int(value)};\n")


def emit_manifest(out, name: str, manifest: list[tuple[str, int]]) -> None:
    out.write(f"inline constexpr const char* {name}Names[] = {{\n")
    for key, _ in manifest:
        out.write(f'    "{key}",\n')
    out.write("};\n")
    out.write(f"inline constexpr int64_t {name}Counts[] = {{\n")
    for i in range(0, len(manifest), 10):
        out.write("    " + ", ".join(str(c) for _, c in manifest[i : i + 10]) + ",\n")
    out.write("};\n\n")


# ---------------------------------------------------------------------------
# Reduced-dimension architectures. Every structural ratio the port branches on is
# preserved; only the magnitudes shrink.
# ---------------------------------------------------------------------------

# Section 1 — audio decoder. attn_resolutions={8} makes the DEEPEST up level carry
# attention (curr_res = resolution // 2**(levels-1) = 8), so both the mid-block
# attention and the per-level attention list are exercised. causality_axis=height
# is the shipped default (model_configurator.py:134): dim 2 is TIME, so all its
# padding must land on the LEFT.
AUDIO_DEC = dict(
    ch=8,
    out_ch=2,
    ch_mult=(1, 2, 4),
    num_res_blocks=1,
    attn_resolutions={8},
    resolution=32,
    z_channels=4,
    dropout=0.0,
    mid_block_add_attention=True,
    sample_rate=16000,
    mel_hop_length=160,
    is_causal=True,
)
AUDIO_DEC_LATENT_T = 3
AUDIO_DEC_LATENT_F = 2  # z_channels * mel_bins(latent) must equal `ch` (patchified width)

# Section 2 — BigVGAN v2 vocoder (resblock "AMP1", snakebeta). conv_pre's input is
# hardcoded to 128 upstream (2 stereo channels x 64 mel bins), so the reduced arm
# keeps 64 mel bins and shrinks everything else.
VOC = dict(
    resblock_kernel_sizes=[3, 7],
    upsample_rates=[2, 2],
    upsample_kernel_sizes=[4, 4],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
    upsample_initial_channel=16,
    resblock="AMP1",
    output_sampling_rate=16000,
    activation="snakebeta",
    use_tanh_at_final=True,
    apply_final_activation=True,
    use_bias_at_final=True,
)
VOC_FRAMES = 5
VOC_MEL_BINS = 64
# Large enough that the convolution path, not conv_post's bias, dominates the
# golden: a bias-dominated golden is nearly constant and gates almost nothing.
VOC_INPUT_SCALE = 1.0

# Section 3 — the LEGACY resblock "1" arm (ResBlock1 + leaky ReLU, no anti-aliased
# activation). Pre-2.3 checkpoints select it (model_configurator.py:53).
VOC_LEGACY = dict(
    resblock_kernel_sizes=[3],
    upsample_rates=[2],
    upsample_kernel_sizes=[4],
    resblock_dilation_sizes=[[1, 3, 5]],
    upsample_initial_channel=8,
    resblock="1",
    output_sampling_rate=16000,
)

# Section 4 — VocoderWithBWE. hop_length 8 against a 20-sample vocoder output
# leaves a remainder, so the pad-to-a-multiple-of-hop branch is exercised, and the
# bwe upsample product (16) x mel frames (3) matches 2 x padded length (48).
BWE_GEN = dict(
    resblock_kernel_sizes=[3],
    upsample_rates=[4, 4],
    upsample_kernel_sizes=[8, 8],
    resblock_dilation_sizes=[[1, 3, 5]],
    upsample_initial_channel=16,
    resblock="AMP1",
    output_sampling_rate=32000,
    activation="snakebeta",
    apply_final_activation=False,
)
BWE = dict(filter_length=16, hop_length=8, win_length=16, n_mel_channels=64,
           input_sampling_rate=16000, output_sampling_rate=32000)

# Section 5 — Conv video decoder. The block list covers every block kind the
# decoder can build: res_x (with inject_noise), compress_all (residual),
# res_x_y (channel-halving, so norm3 + conv_shortcut are live), compress_space,
# attn, compress_time. `attn_res_x` is deliberately absent: upstream passes
# `attention_head_dim` to UNetMidBlock3D, which does not accept it, so the block
# cannot be constructed at this revision (see the C++ refusal).
VIDEO_BLOCKS = [
    ("res_x", {"num_layers": 1, "inject_noise": True}),
    ("compress_all", {"multiplier": 2, "residual": True}),
    ("res_x_y", {"num_layers": 1, "multiplier": 2}),
    ("compress_space", {"multiplier": 1}),
    ("attn", {"num_layers": 1}),
    ("compress_time", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
]
VIDEO_DEC = dict(
    convolution_dimensions=3,
    in_channels=6,
    out_channels=3,
    patch_size=2,
    causal=True,
    timestep_conditioning=True,
    base_channels=8,
)
VIDEO_LATENT = (1, 6, 3, 2, 2)


def section_audio_decoder(out) -> None:
    from ltx_core.model.audio_vae.audio_vae import AudioDecoder
    from ltx_core.model.audio_vae.causality_axis import CausalityAxis
    from ltx_core.model.common.normalization import NormType

    mel_bins = AUDIO_DEC["ch"] // AUDIO_DEC["z_channels"] * 4  # 8 output mel bins
    decoder = AudioDecoder(
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=mel_bins,
        **AUDIO_DEC,
    ).eval()
    manifest = fill_from_stream(decoder, prefix="ltx2.audiodec.")
    latent = make_input(
        "ltx2.audiodec.input",
        (1, AUDIO_DEC["z_channels"], AUDIO_DEC_LATENT_T, AUDIO_DEC_LATENT_F),
        1.0,
    )
    y = decoder(latent)

    out.write("// --- section 1: AudioDecoder (audio_vae.py:277-494) ---\n")
    emit_scalar(out, "kLtx2AudioDecLatentT", AUDIO_DEC_LATENT_T)
    emit_scalar(out, "kLtx2AudioDecLatentF", AUDIO_DEC_LATENT_F)
    emit_scalar(out, "kLtx2AudioDecOutFrames", y.shape[2])
    emit_scalar(out, "kLtx2AudioDecOutMelBins", y.shape[3])
    out.write("\n")
    emit_manifest(out, "kLtx2AudioDecParam", manifest)
    emit_f32(out, "kLtx2AudioDecGolden", y.numpy())

    # The same decoder asked for MORE mel bins than the network produces: upstream
    # zero-pads on the right of the frequency axis (audio_vae.py:458-467). A port
    # that silently returns the unpadded tensor passes every other assertion.
    decoder_pad = AudioDecoder(
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=mel_bins + 3,
        **AUDIO_DEC,
    ).eval()
    fill_from_stream(decoder_pad, prefix="ltx2.audiodec.")
    y_pad = decoder_pad(latent)
    emit_scalar(out, "kLtx2AudioDecPadOutMelBins", y_pad.shape[3])
    out.write("\n")
    emit_f32(out, "kLtx2AudioDecPadGolden", y_pad.numpy())

    # --- section 1b: what "causal" actually reaches ---
    # Measured, not assumed. The decoder's AttnBlocks attend over the WHOLE
    # (time, mel) map (attention.py:31-55), so with the shipped attention on, a
    # change anywhere reaches every output frame and the causality claim is only
    # about the CONVOLUTIONS. Turning attention off isolates exactly the trap this
    # port must not fall into — a symmetric temporal pad instead of a one-sided
    # one — and upstream itself says which frames may move.
    causal = AudioDecoder(
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=mel_bins,
        **{**AUDIO_DEC, "attn_resolutions": set(), "mid_block_add_attention": False},
    ).eval()
    causal_manifest = fill_from_stream(causal, prefix="ltx2.audiodeccausal.")
    import torch

    bumped = latent.clone()
    bumped[:, :, -1, :] += 3.0
    base_out = causal(latent)
    bumped_out = causal(bumped)
    moved = [
        t
        for t in range(base_out.shape[2])
        if not torch.equal(base_out[:, :, t], bumped_out[:, :, t])
    ]
    assert moved, "the causality probe must move SOMETHING or it gates nothing"
    out.write("// --- section 1b: convolution-only causal reach (attention off) ---\n")
    emit_scalar(out, "kLtx2AudioDecCausalFirstMoved", moved[0])
    emit_scalar(out, "kLtx2AudioDecCausalLastMoved", moved[-1])
    # With the SHIPPED attention on, upstream moves EVERY frame; the C++ side
    # asserts that too, so the port cannot claim a causality it does not have.
    out.write("\n")
    emit_manifest(out, "kLtx2AudioDecCausalParam", causal_manifest)

    # --- section 1c: the OTHER THREE causality axes ---
    # Everything above runs `causality_axis=height`, the shipped default. The
    # remaining three arms of `CausalConv2d`'s padding switch (causal_conv_2d.py,
    # via causality_axis.py:4-10) were never executed, so the port's pad split for
    # them was an untested claim. They are cheap to gate and one of them is subtle:
    # WIDTH_COMPATIBILITY pads the width one-sidedly like WIDTH, but the
    # UPSAMPLER treats it differently (upsample.py:44-48 does NOT drop the first
    # element for that axis, while WIDTH and HEIGHT do), so the two axes are not
    # interchangeable however similar their convolution padding looks.
    for label, axis in (
        ("None", CausalityAxis.NONE),
        ("Width", CausalityAxis.WIDTH),
        ("WidthCompat", CausalityAxis.WIDTH_COMPATIBILITY),
    ):
        arm = AudioDecoder(
            norm_type=NormType.PIXEL,
            causality_axis=axis,
            mel_bins=mel_bins,
            **AUDIO_DEC,
        ).eval()
        arm_manifest = fill_from_stream(arm, prefix="ltx2.audiodec.")
        y_axis = arm(latent)
        out.write(f"// --- section 1c: causality_axis = {axis.name} ---\n")
        emit_scalar(out, f"kLtx2AudioDec{label}OutFrames", y_axis.shape[2])
        emit_scalar(out, f"kLtx2AudioDec{label}OutMelBins", y_axis.shape[3])
        out.write("\n")
        emit_manifest(out, f"kLtx2AudioDec{label}Param", arm_manifest)
        emit_f32(out, f"kLtx2AudioDec{label}Golden", y_axis.numpy())


def _vocoder(cfg):
    from ltx_core.model.audio_vae.vocoder import Vocoder

    return Vocoder(**cfg).eval()


def section_vocoder(out) -> None:
    from ltx_core.model.audio_vae.vocoder import kaiser_sinc_filter1d

    voc = _vocoder(VOC)
    manifest = fill_from_stream(voc, prefix="ltx2.voc.")
    mel = make_input("ltx2.voc.input", (1, 2, VOC_FRAMES, VOC_MEL_BINS), VOC_INPUT_SCALE)
    y = voc(mel)

    out.write("// --- section 2: Vocoder, BigVGAN v2 arm (vocoder.py:293-438) ---\n")
    emit_scalar(out, "kLtx2VocFrames", VOC_FRAMES)
    emit_scalar(out, "kLtx2VocMelBins", VOC_MEL_BINS)
    emit_scalar(out, "kLtx2VocOutSamples", y.shape[2])
    out.write("\n")
    # The anti-aliased activation's kaiser-sinc filter is COMPUTED, never loaded.
    # Gate it first: a wrong filter makes every SnakeBeta wrong and the decoder
    # mismatch impossible to localize.
    emit_f32(out, "kLtx2VocUpFilterGolden",
             kaiser_sinc_filter1d(cutoff=0.5 / 2, half_width=0.6 / 2, kernel_size=12).reshape(-1))
    emit_manifest(out, "kLtx2VocParam", manifest)
    emit_f32(out, "kLtx2VocGolden", y.numpy())

    # --- section 2b: resblock "AMP1" with activation "snake" ---
    # The arm that proves act_post is NOT governed by `activation`. Upstream builds
    # `self.act_post = Activation1d(SnakeBeta(final_channels))` (vocoder.py:388)
    # inside `if self.is_amp` and passes it no `activation=` argument, unlike the
    # resblocks one line earlier (vocoder.py:376). So on THIS arm every resblock
    # activation is plain Snake — which reuses ALPHA as its reciprocal scale
    # (vocoder.py:198) — while act_post still reads `.beta`. A port that keys
    # act_post off `activation` produces a plausible waveform from the wrong
    # scale, and no other arm can tell, because on the shipped snakebeta arm the
    # two agree.
    voc_snake = _vocoder({**VOC, "activation": "snake"})
    snake_manifest = fill_from_stream(voc_snake, prefix="ltx2.vocsnake.")
    y_snake = voc_snake(mel)
    out.write('// --- section 2b: AMP1 with activation "snake" (act_post stays SnakeBeta) ---\n')
    emit_scalar(out, "kLtx2VocSnakeOutSamples", y_snake.shape[2])
    out.write("\n")
    emit_manifest(out, "kLtx2VocSnakeParam", snake_manifest)
    emit_f32(out, "kLtx2VocSnakeGolden", y_snake.numpy())


def section_vocoder_legacy(out) -> None:
    voc = _vocoder(VOC_LEGACY)
    manifest = fill_from_stream(voc, prefix="ltx2.vocleg.")
    mel = make_input("ltx2.voc.input", (1, 2, VOC_FRAMES, VOC_MEL_BINS), VOC_INPUT_SCALE)
    y = voc(mel)

    out.write('// --- section 3: Vocoder, legacy resblock "1" arm (resnet.py:12-80) ---\n')
    emit_scalar(out, "kLtx2VocLegacyOutSamples", y.shape[2])
    out.write("\n")
    emit_manifest(out, "kLtx2VocLegacyParam", manifest)
    emit_f32(out, "kLtx2VocLegacyGolden", y.numpy())


def section_bwe(out) -> None:
    from ltx_core.model.audio_vae.vocoder import MelSTFT, UpSample1d, VocoderWithBWE

    voc = _vocoder(VOC)
    gen = _vocoder(BWE_GEN)
    mel_stft = MelSTFT(
        filter_length=BWE["filter_length"],
        hop_length=BWE["hop_length"],
        win_length=BWE["win_length"],
        n_mel_channels=BWE["n_mel_channels"],
    )
    bwe = VocoderWithBWE(
        vocoder=voc,
        bwe_generator=gen,
        mel_stft=mel_stft,
        input_sampling_rate=BWE["input_sampling_rate"],
        output_sampling_rate=BWE["output_sampling_rate"],
        hop_length=BWE["hop_length"],
    ).eval()
    manifest = fill_from_stream(bwe, prefix="ltx2.bwe.")
    mel = make_input("ltx2.voc.input", (1, 2, VOC_FRAMES, VOC_MEL_BINS), VOC_INPUT_SCALE)
    y = bwe(mel)

    out.write("// --- section 4: VocoderWithBWE (vocoder.py:519-630) ---\n")
    emit_scalar(out, "kLtx2BweOutSamples", y.shape[2])
    out.write("\n")
    # The hann-sinc resampler filter is persistent=False: it is COMPUTED, never in
    # the checkpoint, and it is NOT the kaiser filter the activations use.
    resampler = UpSample1d(ratio=2, persistent=False, window_type="hann")
    emit_scalar(out, "kLtx2BweResamplerKernel", resampler.kernel_size)
    out.write("\n")
    emit_f32(out, "kLtx2BweResamplerFilterGolden", resampler.filter.reshape(-1).numpy())
    emit_manifest(out, "kLtx2BweParam", manifest)
    emit_f32(out, "kLtx2BweGolden", y.numpy())

    # --- section 4b: the arm where the mel log CLAMP actually binds ---
    # Everything above leaves `torch.clamp(mel, min=1e-5)` (vocoder.py:515) inert:
    # the raw mel minimum is ~4.4e-3. So the clamp is an INVISIBLE CONSTANT there,
    # and mutation confirms 1e-5 -> 1e-8 changes nothing. This arm attenuates
    # mel_basis by 1e-4 (see param_values) so every bin lands under the clamp and
    # the constant alone decides what the bwe_generator sees — which is what real
    # silence does in production. The saturation COUNT is emitted too, so the C++
    # side asserts the probe is actually saturated rather than trusting it.
    quiet_voc = _vocoder(VOC)
    quiet_gen = _vocoder(BWE_GEN)
    quiet_stft = MelSTFT(
        filter_length=BWE["filter_length"],
        hop_length=BWE["hop_length"],
        win_length=BWE["win_length"],
        n_mel_channels=BWE["n_mel_channels"],
    )
    quiet = VocoderWithBWE(
        vocoder=quiet_voc,
        bwe_generator=quiet_gen,
        mel_stft=quiet_stft,
        input_sampling_rate=BWE["input_sampling_rate"],
        output_sampling_rate=BWE["output_sampling_rate"],
        hop_length=BWE["hop_length"],
    ).eval()
    quiet_manifest = fill_from_stream(quiet, prefix="ltx2.bwequiet.")
    y_quiet = quiet(mel)

    # Recompute the RAW (pre-clamp) mel exactly as VocoderWithBWE.forward does, to
    # count how many values the clamp is holding up.
    import torch

    low = quiet_voc(mel)
    pad = (-low.shape[-1]) % BWE["hop_length"]
    padded = torch.nn.functional.pad(low, (0, pad))
    magnitude, _ = quiet_stft.stft_fn(padded.reshape(-1, padded.shape[-1]))
    raw_mel = torch.matmul(quiet_stft.mel_basis.to(magnitude.dtype), magnitude)
    saturated = int((raw_mel < 1e-5).sum())
    assert saturated == raw_mel.numel(), (
        f"the saturating probe must saturate EVERY bin or it gates nothing: "
        f"{saturated}/{raw_mel.numel()}, min={float(raw_mel.min()):g}"
    )
    out.write("// --- section 4b: the BWE mel log clamp, SATURATED (vocoder.py:515) ---\n")
    emit_scalar(out, "kLtx2BweQuietSaturatedBins", saturated)
    emit_scalar(out, "kLtx2BweQuietOutSamples", y_quiet.shape[2])
    out.write("\n")
    emit_manifest(out, "kLtx2BweQuietParam", quiet_manifest)
    emit_f32(out, "kLtx2BweQuietGolden", y_quiet.numpy())


def section_conv_video_decoder(out) -> None:
    import torch

    from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder
    from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType

    decoder = ConvVideoDecoder(
        decoder_blocks=VIDEO_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **VIDEO_DEC,
    ).eval()
    manifest = fill_from_stream(decoder, prefix="ltx2.videodec.")
    latent = make_input("ltx2.videodec.input", VIDEO_LATENT, 1.0)

    # Patch torch.randn to the shared stream, keyed by CALL INDEX. Upstream's own
    # determinism knob is a torch.Generator, which is likewise call-ordered; the
    # C++ Ltx2NoiseStream consumes the identical sequence.
    draws: list[int] = []
    real_randn = torch.randn

    def patched_randn(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)
        count = int(np.prod(shape)) if shape else 1
        values = ltx_rand(f"ltx2.videodec.noise.{len(draws)}", count)
        draws.append(count)
        return torch.from_numpy(values.astype(np.float32)).reshape(shape)

    torch.randn = patched_randn
    try:
        y = decoder(latent)
    finally:
        torch.randn = real_randn

    out.write("// --- section 5: ConvVideoDecoder (conv_video_decoder.py:146-357) ---\n")
    emit_scalar(out, "kLtx2VideoDecLatentC", VIDEO_LATENT[1])
    emit_scalar(out, "kLtx2VideoDecLatentT", VIDEO_LATENT[2])
    emit_scalar(out, "kLtx2VideoDecLatentH", VIDEO_LATENT[3])
    emit_scalar(out, "kLtx2VideoDecLatentW", VIDEO_LATENT[4])
    emit_scalar(out, "kLtx2VideoDecOutC", y.shape[1])
    emit_scalar(out, "kLtx2VideoDecOutT", y.shape[2])
    emit_scalar(out, "kLtx2VideoDecOutH", y.shape[3])
    emit_scalar(out, "kLtx2VideoDecOutW", y.shape[4])
    emit_scalar(out, "kLtx2VideoDecNoiseDraws", len(draws))
    out.write("inline constexpr int64_t kLtx2VideoDecNoiseCounts[] = {\n    "
              + ", ".join(str(c) for c in draws) + ",\n};\n\n")
    emit_manifest(out, "kLtx2VideoDecParam", manifest)
    emit_f32(out, "kLtx2VideoDecGolden", y.numpy())

    # --- section 5b: what "causal" actually reaches ---
    # Measured, not assumed, exactly like section 1b. `res_x_y`'s shortcut norm is
    # a GroupNorm with ONE group over (C, T, H, W) (resnet.py:93-97), whose
    # statistics span TIME, so the shipped block list is not end-to-end causal
    # however correct the padding is. Stripping it leaves a decoder whose reach is
    # decided by the causal padding alone — which is the trap worth gating.
    causal_blocks = [
        ("res_x", {"num_layers": 1}),
        ("compress_all", {"multiplier": 1, "residual": False}),
        ("res_x", {"num_layers": 1}),
    ]
    causal_dec = ConvVideoDecoder(
        decoder_blocks=causal_blocks,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **{**VIDEO_DEC, "timestep_conditioning": False},
    ).eval()
    causal_manifest = fill_from_stream(causal_dec, prefix="ltx2.videodeccausal.")
    bumped = latent.clone()
    bumped[:, :, -1] += 5.0
    base_out = causal_dec(latent)
    bumped_out = causal_dec(bumped)
    moved = [
        t
        for t in range(base_out.shape[2])
        if not torch.equal(base_out[:, :, t], bumped_out[:, :, t])
    ]
    assert moved, "the causality probe must move SOMETHING or it gates nothing"
    out.write("// --- section 5b: convolution-only causal reach (no res_x_y, no timestep) ---\n")
    emit_scalar(out, "kLtx2VideoDecCausalOutT", base_out.shape[2])
    emit_scalar(out, "kLtx2VideoDecCausalFirstMoved", moved[0])
    emit_scalar(out, "kLtx2VideoDecCausalLastMoved", moved[-1])
    out.write("\n")
    emit_manifest(out, "kLtx2VideoDecCausalParam", causal_manifest)

    # --- section 5c: causal=False, which is UPSTREAM'S OWN DEFAULT ---
    # `ConvVideoDecoder.__init__` declares `causal: bool = False`
    # (conv_video_decoder.py:184) and its docstring calls that the standard
    # decoder, yet every arm above runs causal=True. The non-causal branch is a
    # DIFFERENT padding rule, not a disabled one: CausalConv3d replicates the
    # first AND last frame (kernel-1)//2 times each instead of putting kernel-1
    # copies of frame 0 on the left (convolution.py:266-317), which is why the
    # frame count comes out the same either way and why getting it wrong shifts
    # the whole clip without changing a single shape.
    # It shares the causal arm's WEIGHTS, INPUT and NOISE stream deliberately, so
    # the only difference between the two goldens is the padding rule — which lets
    # the C++ side assert the two arms actually diverge.
    noncausal_draws: list[int] = []

    def noncausal_randn(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)
        count = int(np.prod(shape)) if shape else 1
        values = ltx_rand(f"ltx2.videodec.noise.{len(noncausal_draws)}", count)
        noncausal_draws.append(count)
        return torch.from_numpy(values.astype(np.float32)).reshape(shape)

    noncausal = ConvVideoDecoder(
        decoder_blocks=VIDEO_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **{**VIDEO_DEC, "causal": False},
    ).eval()
    noncausal_manifest = fill_from_stream(noncausal, prefix="ltx2.videodec.")
    torch.randn = noncausal_randn
    try:
        y_nc = noncausal(latent)
    finally:
        torch.randn = real_randn
    assert not torch.equal(y, y_nc), (
        "causal and non-causal must DIFFER on identical weights, or the arm gates nothing"
    )

    out.write("// --- section 5c: causal=False, upstream's default arm ---\n")
    emit_scalar(out, "kLtx2VideoDecNcOutT", y_nc.shape[2])
    emit_scalar(out, "kLtx2VideoDecNcOutH", y_nc.shape[3])
    emit_scalar(out, "kLtx2VideoDecNcOutW", y_nc.shape[4])
    emit_scalar(out, "kLtx2VideoDecNcNoiseDraws", len(noncausal_draws))
    out.write("inline constexpr int64_t kLtx2VideoDecNcNoiseCounts[] = {\n    "
              + ", ".join(str(c) for c in noncausal_draws) + ",\n};\n\n")
    emit_manifest(out, "kLtx2VideoDecNcParam", noncausal_manifest)
    emit_f32(out, "kLtx2VideoDecNcGolden", y_nc.numpy())


def load_upstream(root: Path) -> Path:
    """Import `ltx_core` BY PATH from `root`, and prove that is what resolved."""
    src = root / "packages" / "ltx-core" / "src"
    if not (src / "ltx_core" / "model" / "audio_vae" / "audio_vae.py").is_file():
        raise SystemExit(f"no ltx_core under {src}; point --ltx2 at a Lightricks/LTX-2 checkout")
    sys.path.insert(0, str(src))
    import ltx_core  # noqa: PLC0415

    # ORACLE IDENTITY, asserted rather than assumed: a pip-installed or editable
    # `ltx_core` that wins path resolution would import silently and gate every
    # golden below against the wrong source. See the module docstring.
    resolved = Path(ltx_core.__file__).resolve()
    if not resolved.is_relative_to(src.resolve()):
        raise SystemExit(
            f"ltx_core resolved to {resolved}, which is NOT under the checkout at {src}. "
            "Refusing to generate goldens from an oracle this script did not choose."
        )
    return src


def upstream_revision(root: Path) -> str:
    """The exact upstream tree these goldens were produced from."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001 - a tarball checkout carries no git metadata
        return "unknown"
    return done.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx2", required=True, type=Path,
                        help="a checkout of Lightricks/LTX-2 (the repo root)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.ltx2.expanduser().resolve()
    load_upstream(root)
    revision = upstream_revision(root)

    import torch

    torch.set_grad_enabled(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-ltx2-vae-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// LTX-2.5 VAE goldens, produced by executing the UPSTREAM ltx_core modules\n"
            "// (Lightricks/LTX-2, packages/ltx-core/src/ltx_core/model/) at reduced\n"
            "// dimensions on CPU. Weights and inputs come from the shared deterministic\n"
            "// stream, so no weight byte is checked in. Regenerate with:\n"
            "//   python3 scripts/gen-ltx2-vae-goldens.py --ltx2 <LTX-2 checkout>\n"
            "//       --out tests/vllm/models/ltx2_vae_goldens.inc\n"
            "//\n"
            f"// Upstream revision: {revision}\n"
            "//\n"
            "// See .agents/specs/ltx-2-5.md section 7 for why this is the gate.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
            "// The upstream tree these numbers came from. The suite asserts this equals\n"
            "// the SHA it pins, so regenerating against a DIFFERENT checkout fails the\n"
            "// gate instead of silently replacing the oracle.\n"
            f'inline constexpr const char* kLtx2VaeUpstreamRevision = "{revision}";\n\n'
        )
        section_audio_decoder(out)
        section_vocoder(out)
        section_vocoder_legacy(out)
        section_bwe(out)
        section_conv_video_decoder(out)
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
