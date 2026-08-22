#!/usr/bin/env python3
"""The pixel comparison's own discrimination proof.

`.agents/specs/ltx25-dit-attn-flash.md` section 10, #1612.

`scripts/ltx25-render-compare.py` is the substitute for a token gate on a model
that cannot have one: LTX-2.5 renders pixels, not symbols, so "the output is
the same" has to be a measurement rather than an equality. A tool that answers
that question is only worth its verdict if it FAILS on a difference that matters
and PASSES on one that does not, and neither half is provable by reading it.

So both halves are pinned here, on fabricated frames, with no NAS and no GPU:

  Identity        two identical renders read as bit-identical, and every
                  threshold is then reported as vacuous rather than as passed.
  Dither          +/-1 on 3% of samples -- the shape section 10.2 predicts from
                  bf16 rounding -- passes all four video checks with headroom.
  Structure       ONE PIXEL of global horizontal shift fails ALL FOUR. That is
                  the calibration the section 10.4 threshold table quotes, and a
                  criterion that admitted it would not be a criterion.
  Audio           a scaled waveform fails, because the DiT drives both streams
                  and a video-only comparison leaves half the change unmeasured.
  Refusal         an unreadable input exits 2. A missing input is never a pass.

The fixtures are TEXTURED rather than flat. A flat image makes SSIM degenerate
and makes a one-pixel shift invisible, so a test built on one would pass while
proving nothing -- the shape this file exists to refuse.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/ltx25-render-compare.py"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNREADABLE = 2

W, H, FRAMES = 96, 64, 6


def write_ppm(path: Path, arr: np.ndarray) -> None:
    h, w, _ = arr.shape
    path.write_bytes(b"P6\n%d %d\n255\n" % (w, h) + arr.astype(np.uint8).tobytes())


def write_wav(path: Path, samples: np.ndarray, rate: int = 48000) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(samples.astype("<i2").tobytes())


def make_render(d: Path, rng: np.random.Generator, motion: int = 3) -> list[np.ndarray]:
    """A textured, MOVING sequence: SSIM and the temporal denominator both need one."""
    d.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:H, 0:W]
    base = (
        127
        + 60 * np.sin(xx / 4.0)
        + 40 * np.cos(yy / 3.0)
        + rng.integers(-20, 21, (H, W))
    )
    frames = []
    for i in range(FRAMES):
        shifted = np.roll(base, motion * i, axis=1)
        rgb = np.stack(
            [shifted, np.roll(shifted, 5, axis=0), np.roll(shifted, -5, axis=1)], axis=2
        )
        arr = np.clip(rgb, 0, 255).astype(np.uint8)
        write_ppm(d / f"frame_{i:06d}.ppm", arr)
        frames.append(arr)
    t = np.arange(4800)
    wav = np.stack([8000 * np.sin(t / 20.0), 8000 * np.sin(t / 31.0)], axis=1)
    write_wav(d / "audio.wav", wav)
    return frames


def run(*args: str) -> tuple[int, str, dict | None]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
        jpath = jf.name
    p = subprocess.run(
        [sys.executable, str(TOOL), *args, "--json", jpath],
        capture_output=True,
        text=True,
    )
    try:
        report = json.loads(Path(jpath).read_text())
    except (OSError, json.JSONDecodeError):
        report = None
    return p.returncode, p.stdout + p.stderr, report


def checks_of(report: dict) -> dict[str, bool]:
    return {c["name"]: c["pass"] for c in report["checks"]}


class Discrimination(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        rng = np.random.default_rng(20260822)
        self.a = root / "a"
        self.frames = make_render(self.a, rng)

        # SAME bytes, a second directory: two renders that agree.
        self.same = root / "same"
        make_render(self.same, np.random.default_rng(20260822))

        # DITHER: +/-1 on 3% of samples. Section 10.2 predicts single-ULP bf16
        # flips on 8.6e-05 to 3.7e-04 of attention outputs; 3% at the 8-bit
        # artefact is deliberately far MORE perturbation than that, so a pass
        # here is a pass with room.
        self.dither = root / "dither"
        self.dither.mkdir()
        drng = np.random.default_rng(7)
        for i, arr in enumerate(self.frames):
            noise = (drng.random(arr.shape) < 0.03) * drng.integers(-1, 2, arr.shape)
            write_ppm(self.dither / f"frame_{i:06d}.ppm",
                      np.clip(arr.astype(np.int16) + noise, 0, 255))
        (self.dither / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())

        # STRUCTURE: one pixel of global horizontal shift.
        self.shift = root / "shift"
        self.shift.mkdir()
        for i, arr in enumerate(self.frames):
            write_ppm(self.shift / f"frame_{i:06d}.ppm", np.roll(arr, 1, axis=1))
        (self.shift / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_identical_renders_read_as_bit_identical(self) -> None:
        rc, out, rep = run("--a", str(self.a), "--b", str(self.same))
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertTrue(rep["video"]["bit_identical"], out)
        self.assertEqual(rep["video"]["max_abs"], 0)
        self.assertEqual(rep["video"]["identical_frame_files"], FRAMES)
        # A bit-identical pair reports THAT, and does not report four thresholds
        # as passed. A reader must never mistake a vacuous bound for a read one.
        names = checks_of(rep)
        self.assertIn("video.bit_identical", names)
        self.assertNotIn("video.psnr_min_db", names)
        self.assertTrue(rep["audio"]["bit_identical"])
        self.assertEqual(rep["verdict"], "PASS")

    def test_dither_passes_every_video_check(self) -> None:
        rc, out, rep = run("--a", str(self.a), "--b", str(self.dither))
        self.assertEqual(rc, EXIT_PASS, out)
        v, names = rep["video"], checks_of(rep)
        self.assertFalse(v["bit_identical"])
        for name in ("video.mean_abs", "video.psnr_min_db", "video.ssim_min",
                     "video.temporal_ratio"):
            self.assertTrue(names[name], f"{name} failed on a dither: {out}")
        # The delta is confined to 0 and 1: the shape "numerical noise" predicts.
        self.assertEqual(set(v["delta_histogram"]) - {"0", "1"}, set())
        self.assertLess(v["mean_abs"], 0.1)
        self.assertGreater(v["psnr_min_db"], 50.0)
        self.assertGreater(v["ssim_min"], 0.99)

    def test_one_pixel_shift_fails_all_four_video_checks(self) -> None:
        """The discrimination proof. All four, not one: a criterion that caught
        a global shift on only one axis would be one threshold with three
        decorations."""
        rc, out, rep = run("--a", str(self.a), "--b", str(self.shift))
        self.assertEqual(rc, EXIT_FAIL, out)
        names = checks_of(rep)
        for name in ("video.mean_abs", "video.psnr_min_db", "video.ssim_min",
                     "video.temporal_ratio"):
            self.assertFalse(names[name], f"{name} PASSED on a one-pixel shift: {out}")
        self.assertEqual(rep["verdict"], "FAIL")

    def test_temporal_ratio_is_normalised_by_arm_a_motion(self) -> None:
        """V4's denominator is the render's own frame-to-frame step, so the same
        absolute difference must read SMALLER against a faster-moving render.
        Without that, V4 is a constant wearing a ratio's name."""
        root = Path(self.tmp.name)
        slow, fast = root / "slow", root / "fast"
        make_render(slow, np.random.default_rng(3), motion=1)
        make_render(fast, np.random.default_rng(3), motion=12)
        for src, dst in ((slow, root / "slow_s"), (fast, root / "fast_s")):
            dst.mkdir()
            for p in sorted(src.glob("frame_*.ppm")):
                arr = np.frombuffer(
                    p.read_bytes().split(b"255\n", 1)[1], dtype=np.uint8
                ).reshape(H, W, 3)
                write_ppm(dst / p.name, np.clip(arr.astype(np.int16) + 2, 0, 255))
            (dst / "audio.wav").write_bytes((src / "audio.wav").read_bytes())
        _, _, slow_rep = run("--a", str(slow), "--b", str(root / "slow_s"))
        _, _, fast_rep = run("--a", str(fast), "--b", str(root / "fast_s"))
        self.assertAlmostEqual(slow_rep["video"]["mean_abs"],
                               fast_rep["video"]["mean_abs"], places=6)
        self.assertGreater(slow_rep["video"]["temporal_ratio"],
                           fast_rep["video"]["temporal_ratio"])

    def test_audio_divergence_fails_even_when_the_video_matches(self) -> None:
        """The DiT drives both streams. A comparison that only reads pixels
        would call a broken audio path identical."""
        root = Path(self.tmp.name)
        bad = root / "bad_audio"
        bad.mkdir()
        for p in sorted(self.a.glob("frame_*.ppm")):
            (bad / p.name).write_bytes(p.read_bytes())
        with wave.open(str(self.a / "audio.wav"), "rb") as f:
            n, rate = f.getnframes(), f.getframerate()
            raw = f.readframes(n)
        s = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, 2)
        write_wav(bad / "audio.wav", np.clip(s * 0.5, -32768, 32767), rate)
        rc, out, rep = run("--a", str(self.a), "--b", str(bad))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertTrue(rep["video"]["bit_identical"], out)
        self.assertFalse(checks_of(rep)["audio.psnr_db"], out)

    def test_control_arm_is_reported_separately(self) -> None:
        """The control never enters the verdict: it is the scale the verdict is
        read against, and a tool that folded it into the pass/fail would hide
        exactly the attribution it exists to supply."""
        rc, out, rep = run("--a", str(self.a), "--b", str(self.dither),
                           "--control", str(self.same))
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertIn("control_video", rep)
        self.assertTrue(rep["control_video"]["bit_identical"], out)
        self.assertNotIn("control", " ".join(c["name"] for c in rep["checks"]))


class IdenticallyBroken(unittest.TestCase):
    """The hole every difference-only comparison has.

    Two all-black renders differ by zero, score infinite PSNR and SSIM 1.0, and
    would read as the strongest possible pass. A run that exited 0 having
    written frames that were all one colour has happened in this repository, so
    this is a recorded failure mode and not a hypothetical. Each arm is
    therefore judged on its own content BEFORE anything is subtracted.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _flat(self, d: Path, value: int = 0) -> None:
        d.mkdir(parents=True, exist_ok=True)
        arr = np.full((H, W, 3), value, dtype=np.uint8)
        for i in range(FRAMES):
            write_ppm(d / f"frame_{i:06d}.ppm", arr)
        t = np.arange(4800)
        write_wav(d / "audio.wav", np.stack([8000 * np.sin(t / 20.0)] * 2, axis=1))

    def test_two_all_black_renders_do_not_read_as_a_perfect_match(self) -> None:
        a, b = self.root / "a", self.root / "b"
        self._flat(a)
        self._flat(b)
        rc, out, rep = run("--a", str(a), "--b", str(b), "--label-a", "x", "--label-b", "y")
        # The difference really is nothing, and the tool says so honestly.
        self.assertTrue(rep["video"]["bit_identical"], out)
        # And it still FAILS, because neither arm rendered a picture.
        self.assertEqual(rc, EXIT_FAIL, out)
        names = checks_of(rep)
        self.assertFalse(names["content.x.not_uniform"], out)
        self.assertFalse(names["content.y.not_uniform"], out)
        self.assertFalse(names["content.x.motion"], out)
        self.assertFalse(names["content.x.distinct_frames"], out)

    def test_a_frozen_render_fails_on_motion_even_when_it_has_a_picture(self) -> None:
        """Textured but identical frames: a picture with nothing moving. The
        variance check passes and the motion check is what catches it, which is
        why both exist."""
        a = self.root / "a"
        make_render(a, np.random.default_rng(11))
        frozen = self.root / "frozen"
        frozen.mkdir()
        first = sorted(a.glob("frame_*.ppm"))[0].read_bytes()
        for i in range(FRAMES):
            (frozen / f"frame_{i:06d}.ppm").write_bytes(first)
        (frozen / "audio.wav").write_bytes((a / "audio.wav").read_bytes())
        rc, out, rep = run("--a", str(a), "--b", str(frozen),
                           "--label-a", "good", "--label-b", "frozen")
        self.assertEqual(rc, EXIT_FAIL, out)
        names = checks_of(rep)
        self.assertTrue(names["content.frozen.not_uniform"], out)
        self.assertFalse(names["content.frozen.motion"], out)
        self.assertFalse(names["content.frozen.distinct_frames"], out)
        self.assertTrue(names["content.good.motion"], out)

    def test_a_healthy_pair_passes_every_content_check(self) -> None:
        a, b = self.root / "a", self.root / "b"
        make_render(a, np.random.default_rng(13))
        make_render(b, np.random.default_rng(13))
        rc, out, rep = run("--a", str(a), "--b", str(b), "--label-a", "p", "--label-b", "q")
        self.assertEqual(rc, EXIT_PASS, out)
        names = checks_of(rep)
        for label in ("p", "q"):
            for check in ("frames", "not_uniform", "distinct_frames", "motion"):
                self.assertTrue(names[f"content.{label}.{check}"],
                                f"content.{label}.{check} failed on a healthy render: {out}")


class Refusal(unittest.TestCase):
    def test_missing_directory_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a = Path(t) / "a"
            make_render(a, np.random.default_rng(1))
            rc, out, _ = run("--a", str(a), "--b", str(Path(t) / "absent"))
        self.assertEqual(rc, EXIT_UNREADABLE, out)

    def test_frame_count_mismatch_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            next(iter(sorted(b.glob("frame_*.ppm")))).unlink()
            rc, out, _ = run("--a", str(a), "--b", str(b))
        self.assertNotEqual(rc, EXIT_PASS, out)
        self.assertIn("frame count differs", out)

    def test_ppm_reader_refuses_a_maxval_it_cannot_scale(self) -> None:
        """Every threshold below is in 8-bit levels. A 16-bit PPM would make
        each one mean something else, so it is refused rather than rescaled."""
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.ppm"
            p.write_bytes(b"P6\n2 2\n65535\n" + b"\0" * 24)
            mod = _load_tool()
            with self.assertRaises(ValueError):
                mod.read_ppm(str(p))

    def test_ppm_reader_refuses_a_truncated_file(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.ppm"
            p.write_bytes(b"P6\n8 8\n255\n" + b"\0" * 10)
            mod = _load_tool()
            with self.assertRaises(ValueError):
                mod.read_ppm(str(p))


def _load_tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ltx25_render_compare", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Metrics(unittest.TestCase):
    def test_ssim_of_a_frame_with_itself_is_one(self) -> None:
        mod = _load_tool()
        rng = np.random.default_rng(5)
        a = rng.integers(0, 256, (H, W)).astype(np.float64)
        self.assertAlmostEqual(mod.ssim(a, a), 1.0, places=9)

    def test_psnr_of_zero_error_is_infinite_not_a_large_number(self) -> None:
        mod = _load_tool()
        self.assertEqual(mod.psnr_from_mse(0.0), float("inf"))

    def test_psnr_matches_its_definition(self) -> None:
        mod = _load_tool()
        # A uniform error of exactly 1 level: 20*log10(255) = 48.1308 dB.
        self.assertAlmostEqual(mod.psnr_from_mse(1.0), 48.13080361, places=6)


if __name__ == "__main__":
    unittest.main()
