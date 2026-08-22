#!/usr/bin/env python3
"""The pixel comparison's own discrimination proof.

`.agents/specs/ltx25-dit-attn-flash.md` section 10, #1612.

`scripts/ltx25-render-compare.py` is the substitute for a token gate on a model
that cannot have one: LTX-2.5 renders pixels, not symbols, so "the output is
the same" has to be a measurement rather than an equality. A tool that answers
that question is only worth its verdict if it FAILS on a difference that matters
and PASSES on one that does not, and neither half is provable by reading it.

So both halves are pinned here, on fabricated frames, with no NAS and no GPU:

  Discrimination     two identical renders read as bit-identical and every
                     threshold is then vacuous rather than passed; a +/-1
                     dither on 3% of samples -- the shape section 10.2 predicts
                     from bf16 rounding -- passes all four video checks with
                     headroom; ONE PIXEL of global horizontal shift fails ALL
                     FOUR, which is the calibration section 10.4 quotes and a
                     criterion that admitted it would not be a criterion.
  IdenticallyBroken  C0: two all-black renders differ by zero and score the
                     strongest pass every V check can give, so each arm is
                     judged on its own content first.
  Refusal            an input that cannot be read exits 2, and a threshold
                     failure still exits 1. A missing input is never a pass and
                     never a difference either.
  Audio              A1 and A2 disagree on a waveform that drifted in time,
                     which is the case A2 exists for and the only case that can
                     read it; absent audio is a failed check and never a pass.
  Control            which arm the control repeats is an argument, and the
                     control-to-treatment ratio section 10.5 selects on is
                     computed rather than eyeballed.
  SsimIsTheOneItNames  V3 names Wang et al. 2004 at 11x11 sigma=1.5 on Rec.601
                     luma, and every part of that name is pinned by a property
                     or a value: scikit-image is absent here and stays absent.

The fixtures are TEXTURED rather than flat. A flat image makes SSIM degenerate
and makes a one-pixel shift invisible, so a test built on one would pass while
proving nothing -- the shape this file exists to refuse.
"""

from __future__ import annotations

import json
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
            for check in ("not_uniform", "distinct_frames", "motion"):
                self.assertTrue(names[f"content.{label}.{check}"],
                                f"content.{label}.{check} failed on a healthy render: {out}")
        # C0 registers THREE checks per arm, not four. "frames written" was a
        # fourth that could never be False -- an arm with no frames is refused
        # at exit 2 before any check is built -- and a check that cannot fail is
        # a decoration in a table whose whole value is that each row can.
        self.assertNotIn("content.p.frames", names)
        self.assertNotIn("content.q.frames", names)


class Refusal(unittest.TestCase):
    """Exit 2 is a SEPARATE verdict from exit 1, and the separation is the point.

    Exit 1 means "the pixels differ", which is a statement about a render that
    happened. Exit 0 and exit 1 are both readings of a completed experiment. An
    input that cannot be read is not a reading at all, and a broken render that
    reported the same status as a divergent one would be indistinguishable from
    the finding this tool exists to make -- at exactly the moment nobody is
    looking closely, because a 1 is the expected answer for a while.

    The docstring promised this and only a missing top-level DIRECTORY did it.
    An empty arm, a frame-count mismatch, frames that do not correspond, a
    truncated frame and a 16-bit frame all exited 1 and wrote no JSON.
    """

    def test_missing_directory_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a = Path(t) / "a"
            make_render(a, np.random.default_rng(1))
            rc, out, _ = run("--a", str(a), "--b", str(Path(t) / "absent"))
        self.assertEqual(rc, EXIT_UNREADABLE, out)

    def test_an_empty_arm_directory_exits_two(self) -> None:
        """A render that wrote nothing. Every video statistic is undefined, so
        there is no comparison to fail: this is a refusal, not a verdict."""
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            b.mkdir()
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("no frame", out.lower())
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_frame_count_mismatch_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            next(iter(sorted(b.glob("frame_*.ppm")))).unlink()
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("frame count differs", out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_frames_that_do_not_correspond_by_name_exit_two(self) -> None:
        """The pairing is `sorted()` against `sorted()`. Equal counts are not
        equal frames: an arm that dropped frame 000000 and gained a frame 000049
        pairs every index against its neighbour and reports the render's own
        motion as the arm-to-arm delta -- a large, plausible, entirely spurious
        number."""
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            first = sorted(b.glob("frame_*.ppm"))[0]
            first.rename(b / "frame_000099.ppm")
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("frame_000099.ppm", out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_a_truncated_frame_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            p = sorted(b.glob("frame_*.ppm"))[2]
            p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2])
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("truncated", out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_a_sixteen_bit_frame_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            p = sorted(b.glob("frame_*.ppm"))[1]
            p.write_bytes(b"P6\n%d %d\n65535\n" % (W, H) + b"\0" * (W * H * 6))
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("maxval", out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_a_geometry_mismatch_between_arms_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            p = sorted(b.glob("frame_*.ppm"))[0]
            write_ppm(p, np.zeros((H // 2, W, 3), dtype=np.uint8) + 40)
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_a_threshold_failure_is_still_exit_one(self) -> None:
        """The other half of the separation. Widening exit 2 to cover a
        divergent render would erase the finding instead of the ambiguity."""
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            frames = make_render(a, np.random.default_rng(1))
            b.mkdir()
            for i, arr in enumerate(frames):
                write_ppm(b / f"frame_{i:06d}.ppm", np.roll(arr, 1, axis=1))
            (b / "audio.wav").write_bytes((a / "audio.wav").read_bytes())
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertEqual(rep["verdict"], "FAIL")

    def test_ppm_reader_reads_width_before_height(self) -> None:
        """A square fixture cannot see this, and every fixture in this file was
        square-ish. The header is `P6 W H maxval` and the array is (H, W, 3)."""
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.ppm"
            px = bytes(range(7 * 3 * 3))
            p.write_bytes(b"P6\n7 3\n255\n" + px)
            arr = _load_tool().read_ppm(str(p))
        self.assertEqual(arr.shape, (3, 7, 3))
        self.assertEqual(tuple(int(v) for v in arr[0, 0]), (0, 1, 2))
        self.assertEqual(tuple(int(v) for v in arr[1, 0]), (21, 22, 23))

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


class Audio(unittest.TestCase):
    """A1 and A2 are two checks and they were pinned by one fixture.

    The only divergent-audio fixture was a 0.5x scale, and Pearson r of a
    waveform against a scaled copy of itself is EXACTLY 1.0. So A2 could not
    fail on anything this suite owned: `DEFAULT_MIN_AUDIO_CORR` moved from 0.999
    to -1.0 stayed green, and deleting the check entirely stayed green. Section
    10.4 says A2 exists for "a waveform that has drifted in time", so the
    fixture that reads it is a waveform that has drifted in time.
    """

    def _video_only_pair(self, root: Path) -> tuple[Path, Path]:
        a, b = root / "a", root / "b"
        make_render(a, np.random.default_rng(17))
        b.mkdir()
        for p in sorted(a.glob("frame_*.ppm")):
            (b / p.name).write_bytes(p.read_bytes())
        return a, b

    def test_a_time_shifted_waveform_fails_correlation_while_psnr_passes(self) -> None:
        """The two checks DISAGREE on this fixture, and that is the measured
        fact rather than a hoped-for one: one sample of shift on a 20-sample
        radian scale is 0.05 rad, which is 41.28 dB (A1 passes, floor 40) and
        r = 0.99875 (A2 fails, floor 0.999). A tool holding only A1 would call
        this stream identical."""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            a, b = self._video_only_pair(root)
            n = np.arange(4800)
            write_wav(a / "audio.wav",
                      np.stack([8000 * np.sin(n / 20.0), 8000 * np.cos(n / 20.0)], axis=1))
            write_wav(b / "audio.wav",
                      np.stack([8000 * np.sin((n + 1) / 20.0),
                                8000 * np.cos((n + 1) / 20.0)], axis=1))
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertTrue(rep["video"]["bit_identical"], out)
        names = checks_of(rep)
        self.assertTrue(names["audio.psnr_db"],
                        f"A1 was expected to PASS on a one-sample drift: {out}")
        self.assertFalse(names["audio.pearson_r"],
                         f"A2 was expected to FAIL on a one-sample drift: {out}")
        self.assertGreater(rep["audio"]["psnr_db"], 40.0)
        self.assertLess(rep["audio"]["psnr_db"], 42.0)
        self.assertLess(rep["audio"]["pearson_r"], 0.999)
        self.assertGreater(rep["audio"]["pearson_r"], 0.998)

    def test_absent_audio_is_a_failed_check_and_never_a_pass(self) -> None:
        """Two arms whose video agrees bit-for-bit and that wrote no audio at
        all. `present` must be False and the verdict must be FAIL: the DiT
        drives both streams, so a silent half is an unmeasured half."""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            a, b = self._video_only_pair(root)
            (a / "audio.wav").unlink()
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertTrue(rep["video"]["bit_identical"], out)
        self.assertFalse(rep["audio"]["present"], out)
        self.assertNotIn("comparable", rep["audio"])
        self.assertNotIn("bit_identical", rep["audio"])
        self.assertFalse(checks_of(rep)["audio.present"], out)
        self.assertEqual(rep["verdict"], "FAIL")


class Control(unittest.TestCase):
    """WHICH ARM the control repeats is now stated, not conventional.

    The tool computed the control as `compare_video(a, control, ...)` and said
    so only in a docstring. The harness passed `--a naive --b flash --control
    flash-ctl`, so the "run-to-run noise floor" was a SECOND naive-vs-flash
    comparison. It necessarily read about the same size as the treatment, and
    section 10.5's second branch would then have published "indistinguishable
    from run-to-run nondeterminism" whatever the kernel did. A silent convention
    a caller can invert is not a convention.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.a = root / "a"
        frames = make_render(self.a, np.random.default_rng(21))
        # b differs from a everywhere by ONE level, which every threshold
        # admits, so the pair is a PASS and the control is read on a passing
        # result -- the only case section 10.5 has two readings for. `ctl` is a
        # byte-for-byte repeat of b.
        self.b, self.ctl = root / "b", root / "ctl"
        for d in (self.b, self.ctl):
            d.mkdir()
            for i, arr in enumerate(frames):
                write_ppm(d / f"frame_{i:06d}.ppm", np.clip(arr.astype(np.int16) + 1, 0, 255))
            (d / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_control_of_b_compares_the_control_against_arm_b(self) -> None:
        rc, out, rep = run("--a", str(self.a), "--b", str(self.b),
                           "--control", str(self.ctl), "--control-of", "b",
                           "--label-a", "naive", "--label-b", "flash",
                           "--label-control", "flash-ctl")
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertEqual(rep["control_of"], "b")
        self.assertEqual(rep["control_video"]["label_a"], "flash")
        self.assertTrue(rep["control_video"]["bit_identical"],
                        f"the control repeats FLASH and must be compared to it: {out}")
        self.assertIn("flash-ctl", out)
        # The block says, in words, which arm it was read against.
        self.assertRegex(out, r"control .*flash-ctl.* repeats arm B \(flash\)")

    def test_control_of_defaults_to_a_and_says_so(self) -> None:
        rc, out, rep = run("--a", str(self.a), "--b", str(self.b),
                           "--control", str(self.ctl),
                           "--label-a", "naive", "--label-b", "flash",
                           "--label-control", "flash-ctl")
        self.assertEqual(rep["control_of"], "a")
        self.assertEqual(rep["control_video"]["label_a"], "naive")
        self.assertFalse(rep["control_video"]["bit_identical"], out)
        self.assertRegex(out, r"control .*flash-ctl.* repeats arm A \(naive\)")

    def test_the_control_to_treatment_ratio_is_computed_not_eyeballed(self) -> None:
        """Section 10.5 selects between two OPPOSITE published verdicts on
        "control is 0" against "control comparable to the delta", and nothing
        computed that comparison. It is reported, never checked: it chooses
        between two readings, not between pass and fail."""
        rc, out, rep = run("--a", str(self.a), "--b", str(self.b),
                           "--control", str(self.ctl), "--control-of", "b")
        self.assertEqual(rc, EXIT_PASS, out)
        r = rep["control_ratio"]
        # The control repeats B and IS B, so its delta against B is zero and the
        # ratio is exactly 0: the "noise floor is zero" branch, arithmetically.
        self.assertEqual(r["control_mean_abs_luma"], 0.0)
        self.assertAlmostEqual(r["treatment_mean_abs_luma"],
                               rep["video"]["mean_abs_luma"], places=12)
        self.assertEqual(r["ratio_mean_abs_luma"], 0.0)
        self.assertEqual(r["ratio_mean_abs_rgb"], 0.0)
        self.assertIsNone(r["undefined"])
        self.assertIn("control/treatment", out)
        # And it is REPORTED, not gated: no check carries the ratio.
        self.assertNotIn("control", " ".join(c["name"] for c in rep["checks"]))

    def test_the_ratio_is_undefined_rather_than_a_division_by_zero(self) -> None:
        """The expected case for a bit-identical treatment, which is exactly
        what section 10.2 predicts this experiment will NOT see -- so it is the
        case a reader would meet only when something else went wrong."""
        root = Path(self.tmp.name)
        same = root / "same"
        same.mkdir()
        for p in sorted(self.a.glob("*")):
            (same / p.name).write_bytes(p.read_bytes())
        rc, out, rep = run("--a", str(self.a), "--b", str(same),
                           "--control", str(self.ctl))
        self.assertEqual(rc, EXIT_PASS, out)
        r = rep["control_ratio"]
        self.assertIsNone(r["ratio_mean_abs_luma"])
        self.assertIsNotNone(r["undefined"])
        self.assertGreater(r["control_mean_abs_luma"], 0.0)
        self.assertIn("undefined", out)


def _load_tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ltx25_render_compare", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _texture(seed: int = 20260822) -> np.ndarray:
    """One textured plane, as float64 luma levels. The SSIM fixtures below are
    all derived from this one array so that every pinned number is reproducible
    from the seed printed here and nothing else."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 127 + 60 * np.sin(xx / 4.0) + 40 * np.cos(yy / 3.0) + rng.integers(-20, 21, (H, W))
    return np.clip(base, 0, 255).astype(np.uint8).astype(np.float64)


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


class SsimIsTheOneItNames(unittest.TestCase):
    """V3 names a specific SSIM, and the name is load-bearing.

    Section 10.4 says "Wang et al. 2004, 11x11 Gaussian sigma=1.5 on luma", and
    says it because scikit-image's 7x7 uniform default disagrees in the third
    decimal while a `0.99` gate reads to the sixth. That sentence was a claim: a
    test that only asked "is SSIM(a, a) == 1" stayed green with sigma at 3.0 or
    0.5, the window at 7 or 3, either stabilising constant at 0.5 or at 0, the
    second separable pass blurring the wrong axis or deleted outright, and the
    luma weights replaced by red-only or a flat third. Every one of those is a
    different metric wearing the same name.

    scikit-image is NOT available here and must stay absent, so the criterion is
    pinned by PROPERTIES and by values computed at this revision:

      the taps themselves      catches sigma and the window size directly
      the impulse response     catches either separable pass, because a
                               separable blur of a unit impulse IS outer(k, k)
      its two profiles         catches a second pass on the WRONG axis, which
                               leaves the response summing to 1 and asymmetric
      three fixture SSIMs      catches C1 and C2, which change no property above
                               and only move the number
    """

    def test_gaussian_taps_are_the_eleven_of_sigma_1_5(self) -> None:
        mod = _load_tool()
        k = mod._gauss1d()
        self.assertEqual(len(k), 11, "the window is 11 wide, not scikit-image's 7")
        expected = [
            0.00102838008447911, 0.007598758135239185, 0.03600077212843083,
            0.10936068950970002, 0.2130055377112537, 0.26601172486179436,
            0.2130055377112537, 0.10936068950970002, 0.03600077212843083,
            0.007598758135239185, 0.00102838008447911,
        ]
        np.testing.assert_allclose(k, expected, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(float(k.sum()), 1.0, places=15)

    def test_the_taps_fit_the_sigma_they_claim(self) -> None:
        """The second moment of the discrete taps. A truncated 11-tap Gaussian
        loses a little of the tail, so it reads 1.4978 rather than 1.5000 -- and
        that residual is itself the evidence the window is 11 rather than
        infinite."""
        mod = _load_tool()
        k = mod._gauss1d()
        r = np.arange(len(k), dtype=np.float64) - (len(k) - 1) / 2.0
        fitted = float(np.sqrt((r ** 2 * k).sum()))
        self.assertAlmostEqual(fitted, 1.4978283460942616, places=12)
        self.assertLess(abs(fitted - 1.5), 3e-3)

    def test_blur_of_a_unit_impulse_is_the_separable_outer_product(self) -> None:
        mod = _load_tool()
        k = mod._gauss1d()
        n = 31
        imp = np.zeros((n, n))
        imp[n // 2, n // 2] = 1.0
        out = mod._blur(imp, k)
        lo, hi = n // 2 - 5, n // 2 + 6
        np.testing.assert_allclose(out[lo:hi, lo:hi], np.outer(k, k), rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=12)
        # Nothing outside the 11x11 support: a wider window would spill here.
        support = np.zeros_like(out, dtype=bool)
        support[lo:hi, lo:hi] = True
        self.assertEqual(float(np.abs(out[~support]).max()), 0.0)

    def test_the_impulse_response_is_identical_along_both_axes(self) -> None:
        """The mutation this exists for: a second separable pass that blurs axis
        0 again instead of axis 1, or is deleted. Both leave the response
        summing to 1.0 and both leave SSIM(a, a) == 1, and both make the metric
        anisotropic -- which is what this reads."""
        mod = _load_tool()
        k = mod._gauss1d()
        n = 31
        imp = np.zeros((n, n))
        imp[n // 2, n // 2] = 1.0
        out = mod._blur(imp, k)
        np.testing.assert_allclose(out[n // 2, :], out[:, n // 2], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(out[n // 2, :], out[n // 2, ::-1], rtol=0.0, atol=1e-15)

    def test_ssim_of_a_one_pixel_shift_is_pinned(self) -> None:
        """The value section 10.4's fixture row quotes. It moves under every one
        of the ten mutations in this class's docstring."""
        mod = _load_tool()
        a = _texture()
        self.assertAlmostEqual(mod.ssim(a, np.roll(a, 1, axis=1)),
                               0.7029544035134992, places=8)

    def test_ssim_of_a_five_level_luminance_shift_is_pinned(self) -> None:
        """A pure luminance offset: the variance terms are identical on both
        sides, so the contrast and structure factors are exactly 1 and this
        number is the LUMINANCE factor alone. It is what pins C1."""
        mod = _load_tool()
        a = _texture()
        self.assertAlmostEqual(mod.ssim(a, np.clip(a + 5, 0, 255)),
                               0.9987425512854893, places=8)

    def test_ssim_of_a_contrast_scaled_pair_is_pinned(self) -> None:
        """The mirror of the test above: the means are identical and the
        variances are not, so this number moves with C2 and not with C1."""
        mod = _load_tool()
        a = _texture()
        c = np.clip(127 + 0.9 * (a - 127), 0, 255)
        self.assertAlmostEqual(mod.ssim(a, c), 0.9931050937084597, places=8)

    def test_luma_is_the_rec_601_triple(self) -> None:
        mod = _load_tool()
        px = lambda r, g, b: float(mod.luma(np.array([[[r, g, b]]], dtype=np.uint8))[0, 0])
        # Each primary on its own reads back its own weight, which is the triple
        # asserted rather than described: red-only, a flat third, and an R/B swap
        # each fail a different line here.
        self.assertAlmostEqual(px(255, 0, 0) / 255.0, 0.299, places=12)
        self.assertAlmostEqual(px(0, 255, 0) / 255.0, 0.587, places=12)
        self.assertAlmostEqual(px(0, 0, 255) / 255.0, 0.114, places=12)
        self.assertAlmostEqual(px(10, 20, 30), 0.299 * 10 + 0.587 * 20 + 0.114 * 30,
                               places=12)
        self.assertAlmostEqual(px(10, 20, 30), 18.15, places=12)


if __name__ == "__main__":
    unittest.main()
