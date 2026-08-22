#!/usr/bin/env python3
"""Compare two LTX-2.5 renders pixel-for-pixel and sample-for-sample.

A diffusion render has no token gate. There is no discrete output to hold
against a reference, so the correctness net every other model in this tree
leans on does not exist here. This is the substitute: two renders of the same
prompt, seed, geometry, checkpoint and binary, differing only in the knob under
test, compared on the bytes they actually wrote.

It exists because `LTX25-DIT-ATTN-FLASH` (#1549) moved the DiT self-attention
from `vt::Attention` to `vt::AttentionDenseFlash`, the two are NOT bit-identical
on CUDA, and nothing measured what that did to a picture (#1612). The same
question is owed for the FA-2 arm (#1551), whose divergence is larger, so this
tool takes the arm labels as arguments and hard-codes neither.

WHAT IT MEASURES, and why each one is here rather than a fourth statistic:

  identity     Byte equality of the frame files, then array equality. If the
               two arms are bit-identical there is nothing further to argue
               and every threshold below is vacuous. Report it FIRST so a
               reader never mistakes a passing bound for an unread one.

  |delta|      max, mean and the full histogram over 8-bit RGB. The histogram
               is not decoration: "within numerical noise" predicts a mass
               concentrated at 0 and 1, and a bimodal tail is the shape of a
               structural difference wearing a small mean.

  PSNR         The video-coding convention. 40 dB on 8-bit is the usual
               "visually lossless" line and it is a threshold this experiment
               did not choose for itself.

  SSIM         Wang et al. 2004, the ORIGINAL 11x11 Gaussian sigma=1.5 window
               on luma, not scikit-image's 7x7 uniform default. Stated because
               the two disagree in the third decimal and this gate reads that
               far.

  temporal     The self-calibrating one, and the only bound here derived from
               the render rather than from a convention. Mean absolute
               difference between ADJACENT FRAMES of arm A is the video's own
               frame-to-frame step. An arm-to-arm difference far below it is
               smaller than the motion the render is made of. A ratio, unlike
               a constant, does not need to be re-argued at another geometry.

  audio        The DiT drives both streams, so a video-only comparison would
               leave half the change unmeasured.

USAGE
    ltx25-render-compare.py --a <dir> --b <dir> [--control <dir>] \
        [--label-a naive] [--label-b flash] [--json out.json]

`--control` is a THIRD render of arm A's own configuration. It measures the
noise floor: run-to-run nondeterminism of the same binary and the same knob.
Without it, an arm-to-arm delta cannot be attributed to the knob rather than to
the machine. With it, the attribution is arithmetic -- if the control is zero,
every bit of the A-vs-B delta is the knob; if the control is the same size as
the delta, the knob changed nothing the box does not change on its own.

Exit 0 when every threshold passes, 1 when one fails, 2 when the inputs cannot
be read. A missing input is never a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import wave

import numpy as np

# --- the registered thresholds ------------------------------------------------
# These are DEFAULTS, and they are written here rather than passed at the call
# site so that the criterion is committed to the repository before any number is
# read against it. `.agents/specs/ltx25-dit-attn-flash.md` section 10.4 derives
# each one. Overriding one on the command line is legitimate for a different
# arm pair (#1551) and is recorded in the JSON as an override.
DEFAULT_MAX_MEAN_ABS = 1.0        # 8-bit levels, mean over every pixel/channel
DEFAULT_MIN_PSNR_DB = 40.0        # visually-lossless convention
DEFAULT_MIN_SSIM = 0.99           # per-frame minimum, not the mean
DEFAULT_MAX_TEMPORAL_RATIO = 0.10 # arm delta vs the render's own motion step
DEFAULT_MIN_AUDIO_PSNR_DB = 40.0
DEFAULT_MIN_AUDIO_CORR = 0.999


# --- PPM ----------------------------------------------------------------------
def read_ppm(path: str) -> np.ndarray:
    """Read a binary P6 PPM into an (H, W, 3) uint8 array.

    Written out rather than delegated because the only image library certain to
    be present in a leased worker is the one that ships with numpy, which is
    none. The parser is strict: a maxval other than 255 changes the meaning of
    every threshold below, so it refuses instead of rescaling silently.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(b"P6"):
        raise ValueError(f"{path}: not a binary P6 PPM (starts {data[:2]!r})")
    # Header tokens: P6 width height maxval, with '#' comments allowed anywhere.
    tokens: list[bytes] = []
    i = 2
    while len(tokens) < 3:
        while i < len(data) and data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while i < len(data) and data[i : i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        j = i
        while j < len(data) and not data[j : j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    i += 1  # exactly one whitespace byte after maxval, per the format
    w, h, maxval = (int(t) for t in tokens)
    if maxval != 255:
        raise ValueError(f"{path}: maxval {maxval}, expected 255")
    need = w * h * 3
    px = data[i : i + need]
    if len(px) != need:
        raise ValueError(f"{path}: truncated, {len(px)} of {need} pixel bytes")
    return np.frombuffer(px, dtype=np.uint8).reshape(h, w, 3)


def frame_paths(d: str) -> list[str]:
    names = sorted(n for n in os.listdir(d) if n.startswith("frame_") and n.endswith(".ppm"))
    return [os.path.join(d, n) for n in names]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- SSIM ---------------------------------------------------------------------
def _gauss1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    r = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    k = np.exp(-(r ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _blur(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable convolution with reflect padding, numpy only."""
    pad = len(k) // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(x)
    for i, w in enumerate(k):
        out += w * xp[i : i + x.shape[0], :]
    xp = np.pad(out, ((0, 0), (pad, pad)), mode="reflect")
    out2 = np.zeros_like(x)
    for i, w in enumerate(k):
        out2 += w * xp[:, i : i + x.shape[1]]
    return out2


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma, the plane SSIM is conventionally computed on."""
    f = rgb.astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Wang et al. 2004 mean SSIM on luma, 11x11 Gaussian sigma=1.5, L=255."""
    k = _gauss1d()
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a, mu_b = _blur(a, k), _blur(b, k)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    s_a = _blur(a * a, k) - mu_a2
    s_b = _blur(b * b, k) - mu_b2
    s_ab = _blur(a * b, k) - mu_ab
    num = (2 * mu_ab + c1) * (2 * s_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (s_a + s_b + c2)
    return float(np.mean(num / den))


def psnr_from_mse(mse: float, peak: float = 255.0) -> float:
    if mse <= 0.0:
        return math.inf
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


# --- video --------------------------------------------------------------------
def compare_video(dir_a: str, dir_b: str, label_a: str, label_b: str) -> dict:
    pa, pb = frame_paths(dir_a), frame_paths(dir_b)
    if not pa or not pb:
        raise SystemExit(f"FATAL: no frames ({label_a}: {len(pa)}, {label_b}: {len(pb)})")
    if len(pa) != len(pb):
        raise SystemExit(f"FATAL: frame count differs ({len(pa)} vs {len(pb)})")

    res: dict = {"label_a": label_a, "label_b": label_b, "frames": len(pa), "per_frame": []}

    identical_files = 0
    total_sq = 0.0
    total_abs = 0.0
    total_n = 0
    hist = np.zeros(256, dtype=np.int64)
    prev_a: np.ndarray | None = None
    adjacent_mads: list[float] = []
    global_max = 0

    for idx, (fa, fb) in enumerate(zip(pa, pb)):
        ha, hb = sha256_file(fa), sha256_file(fb)
        same_file = ha == hb
        identical_files += int(same_file)
        A, B = read_ppm(fa), read_ppm(fb)
        if A.shape != B.shape:
            raise SystemExit(f"FATAL: frame {idx} shape {A.shape} vs {B.shape}")
        d = np.abs(A.astype(np.int16) - B.astype(np.int16))
        hist += np.bincount(d.reshape(-1), minlength=256).astype(np.int64)
        mx = int(d.max())
        global_max = max(global_max, mx)
        mean_abs = float(d.mean())
        mse = float((d.astype(np.float64) ** 2).mean())
        total_sq += mse * d.size
        total_abs += float(d.sum())
        total_n += d.size
        la, lb = luma(A), luma(B)
        s = ssim(la, lb)
        if prev_a is not None:
            adjacent_mads.append(float(np.abs(la - prev_a).mean()))
        prev_a = la
        res["per_frame"].append(
            {
                "index": idx,
                "file_a": os.path.basename(fa),
                "sha_equal": same_file,
                "max_abs": mx,
                "mean_abs": mean_abs,
                "psnr_db": psnr_from_mse(mse),
                "ssim": s,
                "differing_pixels": int((d.sum(axis=2) > 0).sum()),
                "pixels": int(d.shape[0] * d.shape[1]),
            }
        )

    agg_mse = total_sq / total_n
    res["identical_frame_files"] = identical_files
    res["bit_identical"] = identical_files == len(pa)
    res["max_abs"] = global_max
    res["mean_abs"] = total_abs / total_n
    res["psnr_db"] = psnr_from_mse(agg_mse)
    res["rmse"] = math.sqrt(agg_mse)
    res["ssim_mean"] = float(np.mean([f["ssim"] for f in res["per_frame"]]))
    res["ssim_min"] = float(np.min([f["ssim"] for f in res["per_frame"]]))
    res["psnr_min_db"] = float(np.min([f["psnr_db"] for f in res["per_frame"]]))
    res["delta_histogram"] = {str(v): int(c) for v, c in enumerate(hist) if c}
    res["samples"] = int(total_n)
    # The self-calibrating denominator: arm A's own frame-to-frame step, on luma,
    # in the same 8-bit units as mean_abs above.
    res["adjacent_frame_mad_a"] = float(np.mean(adjacent_mads)) if adjacent_mads else None
    if res["adjacent_frame_mad_a"]:
        # mean_abs is over RGB, adjacent MAD over luma; recompute the numerator on
        # luma so the ratio divides like with like rather than nearly-like.
        res["temporal_ratio"] = None  # filled by the caller, which has the luma delta
    return res


def compare_video_luma_delta(dir_a: str, dir_b: str) -> float:
    """Mean |delta| on LUMA, the numerator of the temporal ratio."""
    pa, pb = frame_paths(dir_a), frame_paths(dir_b)
    tot, n = 0.0, 0
    for fa, fb in zip(pa, pb):
        la, lb = luma(read_ppm(fa)), luma(read_ppm(fb))
        tot += float(np.abs(la - lb).sum())
        n += la.size
    return tot / n


# --- audio --------------------------------------------------------------------
def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: sample width {sw}, expected 2 (16-bit PCM)")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return a.reshape(-1, ch), sr


def compare_audio(a_path: str, b_path: str) -> dict:
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return {"present": False, "reason": "one or both wav files absent"}
    A, sr_a = read_wav(a_path)
    B, sr_b = read_wav(b_path)
    out: dict = {"present": True, "sample_rate_a": sr_a, "sample_rate_b": sr_b,
                 "frames_a": int(A.shape[0]), "frames_b": int(B.shape[0]),
                 "sha_equal": sha256_file(a_path) == sha256_file(b_path)}
    if A.shape != B.shape or sr_a != sr_b:
        out["comparable"] = False
        return out
    out["comparable"] = True
    d = np.abs(A - B)
    peak = 32768.0
    mse = float((d ** 2).mean())
    out["max_abs_lsb"] = float(d.max())
    out["mean_abs_lsb"] = float(d.mean())
    out["max_abs_fs"] = float(d.max() / peak)
    out["rms_diff_fs"] = float(math.sqrt(mse) / peak)
    out["psnr_db"] = psnr_from_mse(mse, peak=peak)
    out["bit_identical"] = bool(out["sha_equal"] and d.max() == 0)
    fa, fb = A.reshape(-1), B.reshape(-1)
    if fa.std() > 0 and fb.std() > 0:
        out["pearson_r"] = float(np.corrcoef(fa, fb)[0, 1])
    else:
        out["pearson_r"] = None
    return out


# --- main ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="arm A render directory (the reference)")
    ap.add_argument("--b", required=True, help="arm B render directory (the change under test)")
    ap.add_argument("--control", default=None, help="a repeat of arm A: the run-to-run noise floor")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--label-control", default="control")
    ap.add_argument("--audio-name", default="audio.wav")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-mean-abs", type=float, default=DEFAULT_MAX_MEAN_ABS)
    ap.add_argument("--min-psnr-db", type=float, default=DEFAULT_MIN_PSNR_DB)
    ap.add_argument("--min-ssim", type=float, default=DEFAULT_MIN_SSIM)
    ap.add_argument("--max-temporal-ratio", type=float, default=DEFAULT_MAX_TEMPORAL_RATIO)
    ap.add_argument("--min-audio-psnr-db", type=float, default=DEFAULT_MIN_AUDIO_PSNR_DB)
    ap.add_argument("--min-audio-corr", type=float, default=DEFAULT_MIN_AUDIO_CORR)
    args = ap.parse_args()

    for d in (args.a, args.b) + ((args.control,) if args.control else ()):
        if not os.path.isdir(d):
            print(f"FATAL: not a directory: {d}", file=sys.stderr)
            return 2

    report: dict = {
        "thresholds": {
            "max_mean_abs": args.max_mean_abs,
            "min_psnr_db": args.min_psnr_db,
            "min_ssim": args.min_ssim,
            "max_temporal_ratio": args.max_temporal_ratio,
            "min_audio_psnr_db": args.min_audio_psnr_db,
            "min_audio_corr": args.min_audio_corr,
        },
        "inputs": {"a": os.path.abspath(args.a), "b": os.path.abspath(args.b),
                   "control": os.path.abspath(args.control) if args.control else None},
    }

    v = compare_video(args.a, args.b, args.label_a, args.label_b)
    luma_delta = compare_video_luma_delta(args.a, args.b)
    v["mean_abs_luma"] = luma_delta
    if v["adjacent_frame_mad_a"]:
        v["temporal_ratio"] = luma_delta / v["adjacent_frame_mad_a"]
    report["video"] = v

    report["audio"] = compare_audio(
        os.path.join(args.a, args.audio_name), os.path.join(args.b, args.audio_name)
    )

    if args.control:
        c = compare_video(args.a, args.control, args.label_a, args.label_control)
        c["mean_abs_luma"] = compare_video_luma_delta(args.a, args.control)
        report["control_video"] = c
        report["control_audio"] = compare_audio(
            os.path.join(args.a, args.audio_name), os.path.join(args.control, args.audio_name)
        )

    # --- verdict --------------------------------------------------------------
    checks: list[tuple[str, bool, str]] = []
    if v["bit_identical"]:
        checks.append(("video.bit_identical", True, "every frame file sha256-equal"))
    else:
        checks.append(
            ("video.mean_abs", v["mean_abs"] <= args.max_mean_abs,
             f"{v['mean_abs']:.6f} <= {args.max_mean_abs}")
        )
        checks.append(
            ("video.psnr_min_db", v["psnr_min_db"] >= args.min_psnr_db,
             f"{v['psnr_min_db']:.3f} >= {args.min_psnr_db}")
        )
        checks.append(
            ("video.ssim_min", v["ssim_min"] >= args.min_ssim,
             f"{v['ssim_min']:.6f} >= {args.min_ssim}")
        )
        if v.get("temporal_ratio") is not None:
            checks.append(
                ("video.temporal_ratio", v["temporal_ratio"] <= args.max_temporal_ratio,
                 f"{v['temporal_ratio']:.6f} <= {args.max_temporal_ratio}")
            )
        else:
            checks.append(("video.temporal_ratio", False, "no adjacent-frame denominator"))

    a = report["audio"]
    if not a.get("present"):
        checks.append(("audio.present", False, a.get("reason", "absent")))
    elif not a.get("comparable"):
        checks.append(("audio.comparable", False, "shape or sample rate differs"))
    elif a.get("bit_identical"):
        checks.append(("audio.bit_identical", True, "wav sha256-equal"))
    else:
        checks.append(("audio.psnr_db", a["psnr_db"] >= args.min_audio_psnr_db,
                       f"{a['psnr_db']:.3f} >= {args.min_audio_psnr_db}"))
        checks.append(("audio.pearson_r", (a["pearson_r"] or 0.0) >= args.min_audio_corr,
                       f"{a['pearson_r']} >= {args.min_audio_corr}"))

    ok = all(c[1] for c in checks)
    report["checks"] = [{"name": n, "pass": p, "detail": d} for n, p, d in checks]
    report["verdict"] = "PASS" if ok else "FAIL"

    # --- print ----------------------------------------------------------------
    print(f"=== {args.label_a} vs {args.label_b} ===")
    print(f"frames                 {v['frames']}")
    print(f"bit-identical frames   {v['identical_frame_files']}/{v['frames']}")
    print(f"max |delta| (8-bit)    {v['max_abs']}")
    print(f"mean |delta| RGB       {v['mean_abs']:.6f}")
    print(f"mean |delta| luma      {v['mean_abs_luma']:.6f}")
    print(f"RMSE                   {v['rmse']:.6f}")
    print(f"PSNR aggregate         {v['psnr_db']:.3f} dB   (worst frame {v['psnr_min_db']:.3f} dB)")
    print(f"SSIM mean              {v['ssim_mean']:.6f}   (worst frame {v['ssim_min']:.6f})")
    print(f"adjacent-frame MAD (A) {v['adjacent_frame_mad_a']}")
    print(f"temporal ratio         {v.get('temporal_ratio')}")
    print(f"|delta| histogram      {dict(list(report['video']['delta_histogram'].items())[:12])}")
    if "control_video" in report:
        c = report["control_video"]
        print(f"--- control ({args.label_a} vs {args.label_control}): the noise floor ---")
        print(f"bit-identical frames   {c['identical_frame_files']}/{c['frames']}")
        print(f"max |delta|            {c['max_abs']}   mean {c['mean_abs']:.6f}")
        print(f"PSNR                   {c['psnr_db']:.3f} dB   SSIM min {c['ssim_min']:.6f}")
    print("--- audio ---")
    for k in ("present", "comparable", "bit_identical", "max_abs_lsb", "max_abs_fs",
              "rms_diff_fs", "psnr_db", "pearson_r"):
        if k in a:
            print(f"{k:22s} {a[k]}")
    print("--- checks ---")
    for n, p, d in checks:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}: {d}")
    print(f"VERDICT {report['verdict']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
