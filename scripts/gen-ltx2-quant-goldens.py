#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_quant_goldens.inc — the LTX-2.5 phase-L6 quant oracle.

Two things need gating and they need DIFFERENT oracles, so they get different
sections and their provenance is recorded separately.

1. THE SCALE SWIZZLE (`kLtx2Blocked*`). torchao writes NVFP4 group scales in the
   cuBLAS "block scaling factors layout"
   (https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout).
   The producer is `to_blocked` in
   vllm/model_executor/layers/quantization/qutlass_utils.py:165-180, whose own
   header records that it was copied from
   https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats — i.e.
   from the exact torchao module that quantized our checkpoint. vLLM writes the
   same permutation a second time as `swizzle_blockscale`
   (vllm/model_executor/layers/quantization/utils/nvfp4_utils.py:44-49).

   `to_blocked_torch` below is that function's `backend="torch"` body,
   TRANSCRIBED VERBATIM, and `--vllm` pins the source it was transcribed from:
   the generator diffs its own transcription against the checkout's live text
   and REFUSES on any drift, so the copy cannot silently rot.

   HONEST LIMIT, recorded rather than papered over: vLLM is not IMPORTABLE on
   this host (no installed package, `import vllm.*` dies in `vllm.distributed`
   on a missing `zmq`), and `swizzle_blockscale` calls `.cuda()` unconditionally,
   so neither producer can be EXECUTED here. This section is therefore gated
   against a pinned transcription, not against a running oracle. Executing
   `to_blocked` on a host where vLLM imports is OWED.

2. THE REAL CHECKPOINT BYTES (`kLtx2Real*`). These come off the SHIPPED files on
   $CHECKPOINT_ROOT — a few hundred bytes read at their own offsets, never a
   payload download — and the expected values are decoded with TORCH, which is a
   genuinely independent implementation of fp8-e4m3: `torch.uint8 ->
   view(float8_e4m3fn) -> float()`. So the fp8 half of both dequant paths is
   gated against something that is not ours. The e2m1 nibble LUT is not decoded
   by torch (no fp4 dtype); it is the one already gated by the modelopt path in
   tests/vllm/test_nvfp4_dequant.cpp and is reused, not re-derived.

Usage:
    python3 scripts/gen-ltx2-quant-goldens.py \\
        --vllm ~/_git/vllm \\
        --checkpoint-root /mnt/nas_share/checkpoints \\
        --out tests/vllm/models/ltx2_quant_goldens.inc

Needs torch + numpy (CPU only).
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Provenance. AGENTS.md wants the upstream revision anchor; L4's decoy
# experiment (spec section 7.0(b)) showed the anchor is worthless unless the
# tree it names is the tree that ran, so a DIRTY checkout is refused outright —
# `git rev-parse HEAD` reports the committed SHA whatever the worktree holds,
# which is precisely how a clean anchor gets stamped onto drifted goldens.
# ---------------------------------------------------------------------------


def pinned_revision(root: Path, label: str, paths: list[str]) -> tuple[str, bool]:
    """(sha, whole_tree_clean). REFUSES when any of `paths` is dirty.

    The anchor is scoped to what was actually read. A dirty file among `paths`
    is fatal — that is exactly the case where `rev-parse` stamps a clean SHA onto
    goldens the committed tree cannot reproduce. A dirty file ELSEWHERE is not
    fatal, but it is recorded in the emitted header, because an anchor that
    quietly implies more than it checked is the same defect one step removed.
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001 - a tarball checkout has no git metadata
        raise SystemExit(f"{label}: cannot read a revision from {root}: {exc}") from exc
    dirty_paths = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--"] + paths, text=True
    ).strip()
    if dirty_paths:
        raise SystemExit(
            f"{label}: the sources this generator reads are DIRTY at {sha}.\n"
            "  Refusing: rev-parse would stamp a CLEAN anchor onto goldens produced "
            "by an edited tree, which is the exact failure .agents/specs/ltx-2-5.md "
            "section 7.0(b) records. Commit or stash first.\n"
            f"  {dirty_paths}"
        )
    whole = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    return sha, not whole


# ---------------------------------------------------------------------------
# The transcription, and the pin that keeps it honest
# ---------------------------------------------------------------------------

# Transcribed VERBATIM from vllm/model_executor/layers/quantization/
# qutlass_utils.py:174-180 (the `backend="torch"` body of `to_blocked`). The
# `cdiv` calls are inlined as the assert below already forces exact multiples.
_TO_BLOCKED_SOURCE_ANCHOR = """    rows, cols = input_matrix.shape
    n_row_blocks = cdiv(rows, 128)
    n_col_blocks = cdiv(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    assert (rows, cols) == (padded_rows, padded_cols)

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()"""


# The only vLLM files this generator reads. The revision anchor is scoped to
# exactly these, and they are the ones a dirty tree is refused over.
_PINNED_VLLM_PATHS = [
    "vllm/model_executor/layers/quantization/qutlass_utils.py",
    "vllm/model_executor/layers/quantization/utils/nvfp4_utils.py",
]


def check_transcription(vllm_root: Path) -> None:
    """Fail if the transcription above no longer matches the pinned checkout."""
    path = vllm_root / "vllm/model_executor/layers/quantization/qutlass_utils.py"
    if not path.is_file():
        raise SystemExit(f"not a vLLM checkout: {path} is missing")
    text = path.read_text(encoding="utf-8")
    if _TO_BLOCKED_SOURCE_ANCHOR not in text:
        raise SystemExit(
            f"{path}: the transcribed `to_blocked` body no longer appears verbatim.\n"
            "  The swizzle this port inverts has MOVED. Re-read it and re-transcribe; "
            "do not relax this check."
        )
    # The second writing of the same permutation, pinned so a divergence between
    # vLLM's two producers cannot pass unnoticed either.
    other = vllm_root / "vllm/model_executor/layers/quantization/utils/nvfp4_utils.py"
    if not other.is_file():
        raise SystemExit(f"not a vLLM checkout: {other} is missing")
    otext = other.read_text(encoding="utf-8")
    for fragment in (
        "padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)",
        "swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()",
    ):
        if fragment not in otext:
            raise SystemExit(
                f"{other}: `swizzle_blockscale` no longer contains {fragment!r}; "
                "the two vLLM producers may have diverged. Re-read both."
            )


def to_blocked_torch(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = input_matrix
    assert (rows, cols) == (padded_rows, padded_cols)
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


# ---------------------------------------------------------------------------
# Deterministic byte stream, mirrored bit-for-bit by the C++ suite
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def fnv1a64(data) -> int:
    h = 0xCBF29CE484222325
    if isinstance(data, str):
        data = data.encode("utf-8")
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & _MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def rand_bytes(name: str, count: int) -> np.ndarray:
    """`count` bytes reproducible from `name` alone. The C++ suite rebuilds these."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.uint8)
    for i in range(count):
        out[i] = (splitmix64((seed + i) & _MASK64) >> 24) & 0xFF
    return out


# ---------------------------------------------------------------------------
# Safetensors header reading — no payload beyond the named byte ranges
# ---------------------------------------------------------------------------


def open_header(path: Path):
    fh = path.open("rb")
    length = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(length))
    header.pop("__metadata__", None)
    return fh, header, 8 + length


def read_slice(fh, header, base: int, name: str, nbytes: int) -> bytes:
    info = header[name]
    begin, end = info["data_offsets"]
    take = min(nbytes, end - begin)
    fh.seek(base + begin)
    raw = fh.read(take)
    if len(raw) != take:
        raise SystemExit(f"{name}: short read ({len(raw)} of {take})")
    return raw


def f8e4m3_to_f32(raw: bytes) -> np.ndarray:
    """Decode fp8-e4m3fn with TORCH — an implementation that is not ours."""
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float8_e4m3fn)
    return t.float().numpy()


_E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1_nibbles_to_f32(raw: bytes) -> np.ndarray:
    """Low-nibble-first E2M1 decode, the LUT nvfp4_dequant.h:37-38 already gates."""
    b = np.frombuffer(raw, dtype=np.uint8)
    nib = np.empty(b.size * 2, dtype=np.uint8)
    nib[0::2] = b & 0x0F
    nib[1::2] = b >> 4
    mag = _E2M1_LUT[nib & 0x7]
    return np.where((nib & 0x8) != 0, -mag, mag).astype(np.float32)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def cxx_f32(v: float) -> str:
    # `%.9g` of an integral value emits `64`, and `64F` is not a float literal —
    # it is an integer with an unknown user-defined suffix, which is a hard
    # compile error rather than a silent narrowing. Force a decimal point.
    text = f"{float(v):.9g}"
    if not any(c in text for c in ".eE"):
        text += ".0"
    return text + "F"


def emit_bytes(out, name: str, data) -> None:
    values = ", ".join(str(int(b)) for b in data)
    out.write(f"inline constexpr uint8_t {name}[] = {{{values}}};\n")
    out.write(f"inline constexpr int64_t {name}Count = {len(data)};\n\n")


def emit_f32(out, name: str, data) -> None:
    values = ", ".join(cxx_f32(v) for v in data)
    out.write(f"inline constexpr float {name}[] = {{{values}}};\n")
    out.write(f"inline constexpr int64_t {name}Count = {len(data)};\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm", required=True, type=Path, help="the pinned vLLM checkout")
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    check_transcription(args.vllm)
    vllm_sha, vllm_clean = pinned_revision(args.vllm, "vllm", _PINNED_VLLM_PATHS)

    root = args.checkpoint_root / "ltx-2.5/vonkaiser-fp8-nvfp4"
    dit_path = root / "transformer/ltx-2.5-22b-distilled-fp8.safetensors"
    te_path = root / "text_encoders/gemma4-12b-with-proj-nvfp4-torchao.safetensors"
    for p in (dit_path, te_path):
        if not p.is_file():
            raise SystemExit(f"missing shipped checkpoint: {p}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out.open("w", encoding="utf-8")
    out.write(
        "// GENERATED by scripts/gen-ltx2-quant-goldens.py — DO NOT EDIT BY HAND.\n"
        "//\n"
        "// LTX-2.5 phase L6 (.agents/specs/ltx-2-5.md, issue #435): the torchao-NVFP4\n"
        "// scale swizzle and the two shipped checkpoints' own bytes.\n"
        "//\n"
        f"// vLLM revision (swizzle transcription pinned against it): {vllm_sha}\n"
        + (
            "// Both pinned vLLM sources are clean at that revision; the rest of that\n"
            "// checkout's worktree is NOT, so the anchor covers those two files only.\n"
            if not vllm_clean
            else "// That vLLM checkout was entirely clean at generation time.\n"
        )
        +
        "// The swizzle oracle is a PINNED TRANSCRIPTION of vLLM's `to_blocked`, not a\n"
        "// running one: vLLM is not importable on the generating host. See the script.\n"
        "//\n"
        "// Regenerate (one line; a trailing backslash in a // comment is a\n"
        "// -Werror=comment line continuation):\n"
        "//   python3 scripts/gen-ltx2-quant-goldens.py --vllm <vllm>"
        " --checkpoint-root <root> --out tests/vllm/models/ltx2_quant_goldens.inc\n"
        "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
    )
    out.write(f'inline constexpr const char* kLtx2QuantVllmRevision = "{vllm_sha}";\n\n')

    # --- section 1: the swizzle, over shapes that exercise every tile boundary -
    cases = [(128, 4), (128, 16), (256, 4), (384, 12), (128, 240)]
    out.write(f"inline constexpr int64_t kLtx2BlockedCaseCount = {len(cases)};\n")
    out.write("struct Ltx2BlockedCase { int64_t rows, cols; };\n")
    out.write("inline constexpr Ltx2BlockedCase kLtx2BlockedCases[] = {\n")
    for rows, cols in cases:
        out.write(f"    {{{rows}, {cols}}},\n")
    out.write("};\n\n")
    for idx, (rows, cols) in enumerate(cases):
        src = rand_bytes(f"blocked.{rows}x{cols}", rows * cols)
        t = torch.from_numpy(src.reshape(rows, cols).copy())
        blocked = to_blocked_torch(t).numpy()
        assert blocked.size == rows * cols, (blocked.size, rows * cols)
        emit_bytes(out, f"kLtx2BlockedLinear{idx}", src)
        emit_bytes(out, f"kLtx2BlockedSwizzled{idx}", blocked)

    # --- section 2: the SHIPPED text encoder's own swizzled scale ---------------
    #
    # The first 512 bytes of a swizzled buffer are exactly the (rt=0, ct=0) tile,
    # i.e. logical rows 0..127 x cols 0..3 — self-contained, so a 512-byte read
    # pins the real layout without materializing a 7.4 GB file.
    te_fh, te_hdr, te_base = open_header(te_path)
    te_module = "text_embedding_projection.video_aggregate_embed"
    scale_name = te_module + ".weight_scale"
    scale_info = te_hdr[scale_name]
    real_tile = read_slice(te_fh, te_hdr, te_base, scale_name, 512)
    out.write(
        "// The (row-tile 0, col-tile 0) block of the SHIPPED text encoder's\n"
        f"// `{scale_name}` — logical rows 0..127, cols 0..3.\n"
        f"// Stored shape {scale_info['shape']} (= [out/4, (in/16)*4]).\n"
    )
    emit_bytes(out, "kLtx2RealTeScaleTileSwizzled", real_tile)
    # Unswizzle by inverting the transcription: run the SAME permute chain over an
    # index map (int32, because a 512-entry arange does not fit in uint8) to get
    # linear->swizzled, then scatter through it.
    idx = torch.arange(128 * 4, dtype=torch.int32).reshape(128, 4)
    blocks = idx.view(1, 128, 1, 4).permute(0, 2, 1, 3)
    fwd = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).numpy()
    linear_tile = np.empty(128 * 4, dtype=np.uint8)
    linear_tile[fwd] = np.frombuffer(real_tile, dtype=np.uint8)
    emit_bytes(out, "kLtx2RealTeScaleTileLinear", linear_tile)
    out.write(
        "// Decoded with TORCH's own fp8-e4m3fn, so the byte->value half of this\n"
        "// gate is not our implementation checking itself.\n"
    )
    emit_f32(out, "kLtx2RealTeScaleTileLinearF32", f8e4m3_to_f32(bytes(linear_tile)))

    scale2_raw = read_slice(te_fh, te_hdr, te_base, te_module + ".weight_scale_2", 4)
    out.write(
        f"inline constexpr float kLtx2RealTeScale2 = "
        f"{cxx_f32(struct.unpack('<f', scale2_raw)[0])};\n\n"
    )
    packed = read_slice(te_fh, te_hdr, te_base, te_module + ".weight", 32)
    emit_bytes(out, "kLtx2RealTePackedHead", packed)
    # weight[0, 0:64] = e2m1(nibbles) * f8(scale[0, 0..3]) * scale2, group 16.
    nib = e2m1_nibbles_to_f32(bytes(packed))
    grp = f8e4m3_to_f32(bytes(linear_tile[:4]))
    scale2 = struct.unpack("<f", scale2_raw)[0]
    expect = np.array(
        [nib[i] * grp[i // 16] * scale2 for i in range(64)], dtype=np.float32
    )
    emit_f32(out, "kLtx2RealTeWeightHeadF32", expect)

    marker = te_hdr[te_module + ".torchao_nvfp4"]
    marker_raw = read_slice(
        te_fh, te_hdr, te_base, te_module + ".torchao_nvfp4",
        marker["data_offsets"][1] - marker["data_offsets"][0],
    )
    out.write(
        "// The shipped marker, verbatim — what makes this torchao and not\n"
        "// compressed-tensors, read rather than assumed.\n"
        f"inline constexpr const char* kLtx2RealTeMarkerJson =\n"
        f"    {json.dumps(marker_raw.decode('utf-8'))};\n\n"
    )
    te_fh.close()

    # --- section 3: the SHIPPED DiT's own FP8 bytes ----------------------------
    dit_fh, dit_hdr, dit_base = open_header(dit_path)
    dit_module = "model.diffusion_model.proj_out"
    w_raw = read_slice(dit_fh, dit_hdr, dit_base, dit_module + ".weight", 32)
    s_raw = read_slice(dit_fh, dit_hdr, dit_base, dit_module + ".weight_scale", 4)
    dit_scale = struct.unpack("<f", s_raw)[0]
    out.write(
        f"// The SHIPPED FP8 DiT's `{dit_module}` head: raw E4M3 bytes, its per-tensor\n"
        "// F32 scale, and the product TORCH computes for them.\n"
    )
    emit_bytes(out, "kLtx2RealDitFp8Head", w_raw)
    out.write(f"inline constexpr float kLtx2RealDitFp8Scale = {cxx_f32(dit_scale)};\n\n")
    emit_f32(out, "kLtx2RealDitFp8HeadF32", f8e4m3_to_f32(bytes(w_raw)) * dit_scale)
    dit_fh.close()

    out.write("}  // namespace vllm_test\n")
    out.close()
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
