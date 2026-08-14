#!/usr/bin/env python3
"""Guarantees for ``scripts/gpu-lock.sh`` -- the one sanctioned way to take the GPU.

Issue #587: GPU serialisation on ``dgx.casa`` was split across ``$HOME/gpu.lock``
and ``/tmp/gpu.lock``, so two jobs each holding a *different* file ran
concurrently while each believed it owned the box. The wrapper exists so the
path stops being a thing every script re-decides, and so the path it *actually*
took is stamped into the evidence rather than assumed.

Two properties are load-bearing and both are claims about an instrument, which
is exactly the class that quietly stops being true:

1. **It refuses, never falls back.** A wrapper that proceeds without the lock it
   failed to take is a measurement that cannot report its own invalidity.
2. **It records.** The lock path actually taken, the wait, the box health, and
   -- first, because it is the field that says *which* thing killed the run --
   the wrapped command's exit code. ``137`` (someone else's ``pkill``) and a
   compiler diagnostic and an ENOSPC line are three different diagnoses.

The ``Mutations`` class at the bottom proves these tests catch their defect
rather than merely describing it: each mutant breaks one guarantee in a scratch
copy of the script and the corresponding check must go red.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gpu-lock.sh"

# Exit statuses the wrapper reserves for its own failures. They must stay
# distinct from each other; a wrapped command that happens to exit 75 is told
# apart by the stamped `outcome=` field, which only a RUN emits.
#
# EXIT_USAGE is pinned by VALUE, not merely as "not zero". The review of #596
# found `EXIT_USAGE=2 -> 78` survived every usage check, which would have made a
# malformed command line indistinguishable from a lock the wrapper could not
# take -- the one distinction a dispatched job reads to decide whether to retry.
EXIT_REFUSED = 78
EXIT_TIMEOUT = 75
EXIT_USAGE = 2

FIELD = re.compile(r"^gpu-lock: ([a-z0-9-]+)=(.*)$")


def has_flock() -> bool:
    return shutil.which("flock") is not None


def run(args, *, env=None, cwd=None, timeout=60, base_env=True):
    """Invoke the wrapper. Never inherits a real GPU_LOCK/HOME by accident."""
    full = dict(os.environ) if base_env else {}
    full.pop("GPU_LOCK", None)
    full.pop("GPU_LOCK_TIMEOUT", None)
    full.pop("VLLM_CPP_GPU_LOCK_HELD", None)
    full.pop("VLLM_CPP_GPU_LOCK_PID", None)
    if env:
        full.update(env)
    return subprocess.run(
        [str(SCRIPT), *args],
        env=full,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def fields(stderr: str) -> "list[tuple[str, str]]":
    """Every ``gpu-lock: key=value`` line, in emission order."""
    out = []
    for line in stderr.splitlines():
        m = FIELD.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def block(stderr: str, name: str) -> "list[tuple[str, str]]":
    """Fields between ``=== NAME ===`` and the next ``===`` banner."""
    lines = stderr.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"gpu-lock: === {name} ===":
            start = i + 1
            break
    if start is None:
        return []
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("gpu-lock: ==="):
            end = j
            break
    return fields("\n".join(lines[start:end]))


def value(pairs, key):
    for k, v in pairs:
        if k == key:
            return v
    return None


def replace_once(source: str, old: str, new: str) -> str:
    """Mutate exactly one anchor, or fail loudly.

    A mutation that matched a near-duplicate is worse than no mutation: it
    reports a caught defect somewhere the test never claimed to cover. Count
    first, replace second.
    """
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"anchor {old!r} occurs {count} times, expected 1")
    return source.replace(old, new, 1)


def extract_region(source: str, name: str) -> str:
    """The ``# BEGIN <name>`` .. ``# END <name>`` block, markers included."""
    begin, end = f"# BEGIN {name}\n", f"# END {name}\n"
    for marker in (begin, end):
        if source.count(marker) != 1:
            raise AssertionError(
                f"marker {marker.strip()!r} occurs {source.count(marker)} times")
    i = source.index(begin)
    j = source.index(end, i) + len(end)
    return source[i:j]


def dead_pid() -> int:
    """A PID that is certainly not running: reaped before we return it."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def lock_is_free(lock_path: str) -> bool:
    """Ask the kernel, not the record: can `flock -n` take this file right now?"""
    return subprocess.run(
        ["flock", "-n", lock_path, "-c", "true"],
        capture_output=True,
    ).returncode == 0


@contextlib.contextmanager
def other_holder(lock_path: str, seconds: int = 30):
    """A second job holding the SAME lock file, as #587's two jobs did not.

    Yields the holder's PID: it is a LIVE process that is deliberately *not* an
    ancestor of anything the wrapper runs, which is what the pass-through
    integrity checks need to tell "a lock is held" from "held by my parent".
    """
    marker = lock_path + ".held"
    proc = subprocess.Popen(
        ["flock", lock_path, "-c", f"touch {shlex.quote(marker)}; sleep {seconds}"],
        start_new_session=True,
    )
    try:
        deadline = time.time() + 15
        while not os.path.exists(marker):
            if time.time() > deadline:
                raise AssertionError("background holder never acquired the lock")
            time.sleep(0.02)
        yield proc.pid
    finally:
        # Kill the GROUP: killing only the shell orphans `sleep`, which keeps the
        # inherited descriptor open and the lock held for the rest of the suite.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(marker)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not has_flock():
            raise unittest.SkipTest("flock(1) is required")
        if os.geteuid() == 0:
            raise unittest.SkipTest("root ignores the mode bits the refusal rests on")

    def setUp(self):
        # realpath: the wrapper stamps the RESOLVED path, and on some hosts the
        # temp root is itself a symlink.
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="gpulock-"))
        self.addCleanup(self._cleanup)
        self.lock = os.path.join(self.tmp, "gpu.lock")
        self.marker = os.path.join(self.tmp, "ran.marker")
        # The holder sidecar sits beside the lock file it describes, so a
        # blocked reader finds it from the one path it already knows.
        self.sidecar = self.lock + ".holder"

    def _cleanup(self):
        for root, dirs, _ in os.walk(self.tmp):
            for d in dirs:
                with contextlib.suppress(OSError):
                    os.chmod(os.path.join(root, d), 0o755)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def touch_cmd(self):
        return ["sh", "-c", f"touch {shlex.quote(self.marker)}"]

    def assertNotRun(self, res):
        self.assertFalse(
            os.path.exists(self.marker),
            f"the command RAN despite {res.returncode}; stderr:\n{res.stderr}",
        )


# --------------------------------------------------------------------------
# 1. It refuses, never falls back.
# --------------------------------------------------------------------------
class Refusal(Base):
    def test_unwritable_lock_directory_refuses(self):
        ro = os.path.join(self.tmp, "ro")
        os.mkdir(ro)
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        res = run(["--lock", os.path.join(ro, "gpu.lock"), "--", *self.touch_cmd()])
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertNotRun(res)

    def test_unwritable_existing_lock_file_refuses(self):
        Path(self.lock).write_text("")
        os.chmod(self.lock, stat.S_IRUSR)
        res = run(["--lock", self.lock, "--", *self.touch_cmd()])
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertNotRun(res)

    def test_missing_lock_parent_refuses(self):
        res = run(["--lock", os.path.join(self.tmp, "nope", "gpu.lock"),
                   "--", *self.touch_cmd()])
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertNotRun(res)

    def test_unwritable_record_file_refuses(self):
        ro = os.path.join(self.tmp, "ro")
        os.mkdir(ro)
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        res = run(["--lock", self.lock, "--record", os.path.join(ro, "ev.txt"),
                   "--", *self.touch_cmd()])
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertNotRun(res)

    def test_refusal_is_loud_and_names_the_path(self):
        ro = os.path.join(self.tmp, "ro")
        os.mkdir(ro)
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        target = os.path.join(ro, "gpu.lock")
        res = run(["--lock", target, "--", *self.touch_cmd()])
        self.assertIn("gpu-lock: === REFUSED ===", res.stderr)
        pairs = block(res.stderr, "REFUSED")
        self.assertEqual(value(pairs, "outcome"), "REFUSED")
        self.assertEqual(value(pairs, "lock-path"), target)
        self.assertTrue(value(pairs, "reason"), "refusal carried no reason")

    def test_empty_command_is_a_usage_error_not_a_silent_success(self):
        res = run(["--lock", self.lock, "--"])
        self.assertEqual(res.returncode, EXIT_USAGE, res.stderr)

    def test_nonnumeric_timeout_is_rejected(self):
        res = run(["--lock", self.lock, "--timeout", "soon", "--", *self.touch_cmd()])
        self.assertEqual(res.returncode, EXIT_USAGE, res.stderr)
        self.assertNotRun(res)

    def test_usage_status_is_distinct_from_refusal_and_timeout(self):
        # A malformed command line is the caller's bug and a taken lock is the
        # box's state. A dispatched job retries the second and never the first,
        # so the two statuses may not collapse into each other.
        self.assertNotIn(EXIT_USAGE, (EXIT_REFUSED, EXIT_TIMEOUT))
        for args in (["--lock", self.lock, "--"],
                     ["--lock", self.lock, "--timeout", "soon", "--", "true"],
                     ["--lock", self.lock, "--timeout", "0", "--", "true"],
                     ["--nonsense", "--", "true"],
                     ["--lock"]):
            res = run(args)
            self.assertEqual(res.returncode, EXIT_USAGE,
                             f"{args} exited {res.returncode}:\n{res.stderr}")

    # -- F8: a wrapper that never falls back may not fall back SILENTLY -------
    def test_empty_lock_flag_is_a_usage_error_not_a_silent_fallback(self):
        # `--lock ''` used to resolve $HOME/gpu.lock. A caller who asked for a
        # path and supplied none gets told, not quietly redirected at the
        # canonical file -- which on a gate host is the one everyone else holds.
        res = run(["--lock", "", "--", *self.touch_cmd()], env={"HOME": self.tmp})
        self.assertEqual(res.returncode, EXIT_USAGE, res.stderr)
        self.assertNotRun(res)

    def test_empty_lock_flag_with_equals_is_also_a_usage_error(self):
        res = run(["--lock=", "--", *self.touch_cmd()], env={"HOME": self.tmp})
        self.assertEqual(res.returncode, EXIT_USAGE, res.stderr)
        self.assertNotRun(res)

    # -- F5: a REFUSED block must reach --record ------------------------------
    def test_lock_refusal_reaches_the_record_file(self):
        # `.env.example` ships `GPU_LOCK=` empty and a dispatched job discards
        # stderr, so the commonest refusal used to leave NO artefact at all: the
        # record file was not even created, because the lock was validated
        # before the record was opened.
        ro = os.path.join(self.tmp, "ro")
        os.mkdir(ro)
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        rec = os.path.join(self.tmp, "evidence.txt")
        res = run(["--lock", os.path.join(ro, "gpu.lock"), "--record", rec,
                   "--", *self.touch_cmd()])
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertTrue(os.path.exists(rec),
                        f"--record was never created on a lock refusal:\n{res.stderr}")
        text = Path(rec).read_text()
        self.assertIn("=== REFUSED ===", text)
        self.assertIn("outcome=REFUSED", text)
        self.assertIn("reason=", text)

    def test_timeout_reaches_the_record_file(self):
        rec = os.path.join(self.tmp, "evidence.txt")
        with other_holder(self.lock):
            res = run(["--lock", self.lock, "--timeout", "0.5", "--record", rec,
                       "--", *self.touch_cmd()], timeout=30)
        self.assertEqual(res.returncode, EXIT_TIMEOUT, res.stderr)
        self.assertIn("outcome=TIMEOUT", Path(rec).read_text())

    def test_a_refusal_before_the_path_resolves_still_reaches_the_record(self):
        # HOME unset and no --lock: the earliest refusal there is. It is still
        # an outcome, and an outcome with no artefact is the instrument that
        # cannot report its own failure.
        rec = os.path.join(self.tmp, "evidence.txt")
        res = run(["--record", rec, "--", *self.touch_cmd()],
                  env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                  base_env=False)
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertTrue(os.path.exists(rec), res.stderr)
        self.assertIn("outcome=REFUSED", Path(rec).read_text())


# --------------------------------------------------------------------------
# 2. The wait is bounded and its expiry has its own status.
# --------------------------------------------------------------------------
class Timeout(Base):
    def test_contended_lock_times_out_with_distinct_status(self):
        with other_holder(self.lock):
            res = run(["--lock", self.lock, "--timeout", "0.5",
                       "--", *self.touch_cmd()], timeout=30)
        self.assertEqual(res.returncode, EXIT_TIMEOUT, res.stderr)
        self.assertNotRun(res)
        self.assertIn("gpu-lock: === TIMEOUT ===", res.stderr)
        self.assertEqual(value(block(res.stderr, "TIMEOUT"), "outcome"), "TIMEOUT")

    def test_timeout_status_differs_from_refusal(self):
        self.assertNotEqual(EXIT_TIMEOUT, EXIT_REFUSED)

    def test_wait_is_bounded_not_indefinite(self):
        # A holder outliving the wrapper's patience by 20x. An unbounded flock
        # here is how a job silently sits forever behind a dead holder.
        with other_holder(self.lock, seconds=30):
            start = time.time()
            res = run(["--lock", self.lock, "--timeout", "1",
                       "--", *self.touch_cmd()], timeout=20)
            waited = time.time() - start
        self.assertEqual(res.returncode, EXIT_TIMEOUT, res.stderr)
        self.assertLess(waited, 15, "the wrapper waited far past its timeout")

    def test_lock_is_actually_taken_so_a_holder_excludes_us(self):
        # The #587 defect in miniature: if the wrapper does not really take the
        # lock it will run happily beside a job that holds it.
        with other_holder(self.lock):
            res = run(["--lock", self.lock, "--timeout", "0.5",
                       "--", *self.touch_cmd()], timeout=30)
        self.assertNotRun(res)

    def test_uncontended_lock_is_acquired_immediately(self):
        res = run(["--lock", self.lock, "--timeout", "5", "--", *self.touch_cmd()])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(os.path.exists(self.marker))


# --------------------------------------------------------------------------
# 3. It records: the stamped evidence.
# --------------------------------------------------------------------------
class Stamp(Base):
    def acquire(self, *extra, env=None):
        res = run(["--lock", self.lock, *extra, "--", "true"], env=env)
        self.assertEqual(res.returncode, 0, res.stderr)
        return res

    def test_acquire_block_carries_every_required_field(self):
        # The list is EXHAUSTIVE on purpose. It shipped naming eight of the
        # thirteen stamped fields, so `requested-lock`, `acquired-utc`, `host`
        # and `label` could all be deleted with the suite still green -- and
        # `requested-lock` is the single field that makes a lock-path divergence
        # legible afterwards, which is the whole of #587.
        res = self.acquire("--timeout", "5", "--label", "gate")
        pairs = block(res.stderr, "ACQUIRE")
        keys = [k for k, _ in pairs]
        for required in ("mode", "lock-path", "requested-lock", "lock-source",
                         "holder-pid", "waited-seconds", "timeout-seconds",
                         "acquired-utc", "host", "label", "loadavg", "df-root",
                         "command"):
            self.assertIn(required, keys, f"missing {required}; got {keys}")

    def test_requested_lock_records_the_spelling_that_was_ASKED_for(self):
        # `lock-path` is resolved and `requested-lock` is verbatim. Two agents
        # who resolve the same file by different spellings look identical in the
        # first field and are told apart only by the second.
        real = os.path.join(self.tmp, "real")
        os.mkdir(real)
        link = os.path.join(self.tmp, "link")
        os.symlink(real, link)
        asked = os.path.join(link, "gpu.lock")
        res = run(["--lock", asked, "--", "true"])
        self.assertEqual(res.returncode, 0, res.stderr)
        pairs = block(res.stderr, "ACQUIRE")
        self.assertEqual(value(pairs, "requested-lock"), asked)
        self.assertEqual(value(pairs, "lock-path"), os.path.join(real, "gpu.lock"))

    def test_requested_lock_is_stamped_on_a_refusal_too(self):
        ro = os.path.join(self.tmp, "ro")
        os.mkdir(ro)
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        target = os.path.join(ro, "gpu.lock")
        res = run(["--lock", target, "--", "true"])
        self.assertEqual(value(block(res.stderr, "REFUSED"), "requested-lock"), target)

    # -- F8: the fallback happens, but it is never silent ---------------------
    def test_lock_source_names_the_flag_when_lock_is_given(self):
        res = self.acquire()
        self.assertIn("--lock", value(block(res.stderr, "ACQUIRE"), "lock-source"))

    def test_lock_source_names_the_env_when_it_supplied_the_path(self):
        other = os.path.join(self.tmp, "elsewhere.lock")
        res = run(["--", "true"], env={"HOME": self.tmp, "GPU_LOCK": other})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("GPU_LOCK", value(block(res.stderr, "ACQUIRE"), "lock-source"))

    def test_an_empty_gpu_lock_env_falls_back_LOUDLY(self):
        # `.env.example` ships `GPU_LOCK=` empty and documents `set -a; . ./.env`,
        # so an empty value is the live default on any host that copied it. The
        # fallback is correct; being unable to see it in the record is not.
        res = run(["--", "true"], env={"HOME": self.tmp, "GPU_LOCK": ""})
        self.assertEqual(res.returncode, 0, res.stderr)
        pairs = block(res.stderr, "ACQUIRE")
        self.assertEqual(value(pairs, "lock-path"), self.lock)
        source = value(pairs, "lock-source")
        self.assertIn("default", source)
        self.assertIn("empty", source.lower(),
                      f"an EMPTY GPU_LOCK fell back indistinguishably from an "
                      f"unset one: lock-source={source!r}")

    def test_an_unset_gpu_lock_env_is_a_plain_default(self):
        res = run(["--", "true"], env={"HOME": self.tmp})
        self.assertEqual(res.returncode, 0, res.stderr)
        source = value(block(res.stderr, "ACQUIRE"), "lock-source")
        self.assertIn("default", source)
        self.assertNotIn("empty", source.lower())

    def test_lock_path_is_absolute(self):
        res = self.acquire()
        self.assertTrue(value(block(res.stderr, "ACQUIRE"), "lock-path").startswith("/"))

    def test_relative_lock_path_is_stamped_absolute(self):
        res = run(["--lock", "gpu.lock", "--", "true"], cwd=self.tmp)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(value(block(res.stderr, "ACQUIRE"), "lock-path"), self.lock)

    def test_symlinked_lock_path_is_stamped_resolved(self):
        # #587 made visible: two spellings of one file, or one spelling of two
        # files, are only distinguishable once the RESOLVED path is in the record.
        real = os.path.join(self.tmp, "real")
        os.mkdir(real)
        link = os.path.join(self.tmp, "link")
        os.symlink(real, link)
        res = run(["--lock", os.path.join(link, "gpu.lock"), "--", "true"])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(value(block(res.stderr, "ACQUIRE"), "lock-path"),
                         os.path.join(real, "gpu.lock"))

    def test_default_lock_path_is_home_gpu_lock(self):
        res = run(["--", "true"], env={"HOME": self.tmp})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(value(block(res.stderr, "ACQUIRE"), "lock-path"), self.lock)

    def test_gpu_lock_env_overrides_the_default(self):
        other = os.path.join(self.tmp, "elsewhere.lock")
        res = run(["--", "true"], env={"HOME": self.tmp, "GPU_LOCK": other})
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(value(block(res.stderr, "ACQUIRE"), "lock-path"), other)

    def test_explicit_lock_flag_overrides_the_env(self):
        other = os.path.join(self.tmp, "elsewhere.lock")
        res = run(["--lock", self.lock, "--", "true"],
                  env={"HOME": self.tmp, "GPU_LOCK": other})
        self.assertEqual(value(block(res.stderr, "ACQUIRE"), "lock-path"), self.lock)

    def test_holder_pid_is_a_live_process(self):
        res = self.acquire()
        pid = value(block(res.stderr, "ACQUIRE"), "holder-pid")
        self.assertRegex(pid, r"^\d+$")
        self.assertGreater(int(pid), 0)

    def test_waited_seconds_is_numeric(self):
        res = self.acquire()
        self.assertRegex(value(block(res.stderr, "ACQUIRE"), "waited-seconds"),
                         r"^\d+\.\d+$")

    def test_waited_seconds_reflects_a_real_wait(self):
        with other_holder(self.lock, seconds=2):
            res = run(["--lock", self.lock, "--timeout", "20", "--", "true"],
                      timeout=40)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertGreater(float(value(block(res.stderr, "ACQUIRE"),
                                       "waited-seconds")), 0.5)

    def test_df_root_is_the_real_root_filesystem_row(self):
        res = self.acquire()
        df = value(block(res.stderr, "ACQUIRE"), "df-root")
        self.assertTrue(df.endswith(" /"), f"not a `df -h /` data row: {df!r}")
        self.assertRegex(df, r"\d+%")

    def test_loadavg_has_three_numbers(self):
        res = self.acquire()
        nums = value(block(res.stderr, "ACQUIRE"), "loadavg").split()
        self.assertEqual(len(nums), 3)
        for n in nums:
            float(n)

    def test_command_is_recorded(self):
        res = run(["--lock", self.lock, "--", "sh", "-c", "exit 0"])
        self.assertIn("sh", value(block(res.stderr, "ACQUIRE"), "command"))

    def test_record_file_receives_the_same_stamp(self):
        rec = os.path.join(self.tmp, "evidence.txt")
        res = run(["--lock", self.lock, "--record", rec, "--", "true"])
        self.assertEqual(res.returncode, 0, res.stderr)
        text = Path(rec).read_text()
        self.assertIn("=== ACQUIRE ===", text)
        self.assertIn("=== RELEASE ===", text)
        self.assertIn(f"lock-path={self.lock}", text)
        self.assertIn("exit-code=0", text)

    def test_record_file_is_appended_not_truncated(self):
        rec = os.path.join(self.tmp, "evidence.txt")
        Path(rec).write_text("EARLIER-LEG\n")
        run(["--lock", self.lock, "--record", rec, "--", "true"])
        self.assertIn("EARLIER-LEG", Path(rec).read_text())

    def test_stdout_of_the_wrapped_command_is_not_polluted(self):
        res = run(["--lock", self.lock, "--", "sh", "-c", "echo PAYLOAD"])
        self.assertEqual(res.stdout, "PAYLOAD\n")


# --------------------------------------------------------------------------
# 4. The exit code comes FIRST, and comes through unchanged.
# --------------------------------------------------------------------------
class ExitCode(Base):
    def release(self, res):
        return block(res.stderr, "RELEASE")

    def test_zero_propagates(self):
        res = run(["--lock", self.lock, "--", "sh", "-c", "exit 0"])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(value(self.release(res), "exit-code"), "0")

    def test_nonzero_propagates_unchanged(self):
        res = run(["--lock", self.lock, "--", "sh", "-c", "exit 42"])
        self.assertEqual(res.returncode, 42, res.stderr)
        self.assertEqual(value(self.release(res), "exit-code"), "42")

    def test_sigkill_137_propagates_unchanged(self):
        # The case that matters most: 137 says another agent's `pkill` killed the
        # run, not that the code is wrong. A wrapper that flattens it to 1 turns
        # "not mine" into "mine".
        res = run(["--lock", self.lock, "--", "sh", "-c", "kill -9 $$"])
        self.assertEqual(res.returncode, 137, res.stderr)
        self.assertEqual(value(self.release(res), "exit-code"), "137")

    def test_sigkill_is_named_in_the_record(self):
        res = run(["--lock", self.lock, "--", "sh", "-c", "kill -9 $$"])
        reason = value(self.release(res), "exit-reason") or ""
        self.assertIn("9", reason)
        self.assertIn("signal", reason.lower())

    def test_missing_executable_propagates_127(self):
        res = run(["--lock", self.lock, "--", "definitely-not-a-real-binary-587"])
        self.assertEqual(res.returncode, 127, res.stderr)
        self.assertEqual(value(self.release(res), "exit-code"), "127")

    def test_exit_code_is_the_first_field_of_the_release_block(self):
        # Order is the requirement, not just presence: disk and load say the box
        # was unhealthy, the exit code says WHICH thing killed the run.
        res = run(["--lock", self.lock, "--", "sh", "-c", "exit 3"])
        pairs = self.release(res)
        self.assertTrue(pairs, f"no RELEASE block:\n{res.stderr}")
        self.assertEqual(pairs[0][0], "exit-code",
                         f"first RELEASE field was {pairs[0][0]}")

    def test_release_block_repeats_the_lock_path_and_health(self):
        res = run(["--lock", self.lock, "--", "true"])
        keys = [k for k, _ in self.release(res)]
        for required in ("exit-code", "outcome", "lock-path", "elapsed-seconds",
                         "loadavg-at-exit", "df-root-at-exit"):
            self.assertIn(required, keys, f"missing {required}; got {keys}")

    def test_ran_outcome_is_distinguishable_from_a_wrapper_failure(self):
        # A wrapped command exiting 75 must not read as the wrapper timing out.
        res = run(["--lock", self.lock, "--", "sh", "-c", "exit 75"])
        self.assertEqual(res.returncode, EXIT_TIMEOUT)
        self.assertEqual(value(self.release(res), "outcome"), "RAN")
        self.assertNotIn("=== TIMEOUT ===", res.stderr)


# --------------------------------------------------------------------------
# 5. It does not self-deadlock.
# --------------------------------------------------------------------------
class Nesting(Base):
    def nested(self, *extra):
        inner = [str(SCRIPT), "--lock", self.lock, "--timeout", "2",
                 "--", *self.touch_cmd()]
        return run(["--lock", self.lock, "--timeout", "2", *extra,
                    "--", *inner], timeout=45)

    def test_nested_invocation_does_not_deadlock(self):
        res = self.nested()
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(os.path.exists(self.marker), res.stderr)

    def test_nested_invocation_is_stamped_as_a_pass_through(self):
        res = self.nested()
        self.assertIn("PASS-THROUGH", res.stderr)

    def test_outer_invocation_is_stamped_as_an_acquire(self):
        res = run(["--lock", self.lock, "--", "true"])
        self.assertEqual(value(block(res.stderr, "ACQUIRE"), "mode"), "ACQUIRED")

    def test_a_different_lock_still_acquires(self):
        # Pass-through keys on the RESOLVED path, so an unrelated second lock is
        # taken for real rather than waved through.
        other = os.path.join(self.tmp, "other.lock")
        inner = [str(SCRIPT), "--lock", other, "--timeout", "2", "--", "true"]
        res = run(["--lock", self.lock, "--timeout", "2", "--", *inner], timeout=45)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn(f"lock-path={other}", res.stderr)

    def test_a_different_lock_is_ACQUIRED_not_merely_stamped(self):
        # F3. `test_a_different_lock_still_acquires` above asserts the STAMP,
        # and a guard keyed on `[ -n "$VLLM_CPP_GPU_LOCK_HELD" ]` instead of
        # path equality survives it: the mutant waves the inner command through
        # and still prints `lock-path=<other>`, because the field is printed
        # either way. That is this repository's own "a gate on a shared helper
        # proves consistency, not correctness" trap. So hold the inner lock from
        # OUTSIDE and require the inner invocation to be excluded by it.
        other = os.path.join(self.tmp, "other.lock")
        with other_holder(other):
            inner = [str(SCRIPT), "--lock", other, "--timeout", "0.5",
                     "--", *self.touch_cmd()]
            res = run(["--lock", self.lock, "--timeout", "2", "--", *inner],
                      timeout=45)
        # The inner wrapper times out; the outer propagates that status.
        self.assertEqual(res.returncode, EXIT_TIMEOUT, res.stderr)
        self.assertNotRun(res)

    def test_the_outer_lock_is_ACQUIRED_not_merely_stamped(self):
        # The same trap one level up: an outer invocation whose own lock is held
        # by a third party must not run at all, however its mode is stamped.
        with other_holder(self.lock):
            inner = [str(SCRIPT), "--lock", self.lock, "--timeout", "0.5",
                     "--", *self.touch_cmd()]
            res = run(["--lock", self.lock, "--timeout", "0.5", "--", *inner],
                      timeout=45)
        self.assertEqual(res.returncode, EXIT_TIMEOUT, res.stderr)
        self.assertNotRun(res)


# --------------------------------------------------------------------------
# 6. Pass-through is a proof, not a claim in an environment variable.
# --------------------------------------------------------------------------
class PassThroughIntegrity(Base):
    """F1. ``VLLM_CPP_GPU_LOCK_HELD`` proves the variable *names* the path.

    It never proved the lock was still *held*, and #587 reproduces straight
    through the wrapper because of it: Python's ``subprocess`` defaults to
    ``close_fds=True``, so a detached job inherits the exported variable and not
    the locked descriptor. The outer wrapper exits, the lock is genuinely free,
    an unrelated job legitimately takes it -- and the detached job then invokes
    the wrapper, stamps ``mode=PASS-THROUGH`` and runs beside the holder. Two
    jobs, one lock file, each believing it owns the GPU.

    The data to settle it was already exported and never read back:
    ``VLLM_CPP_GPU_LOCK_PID``. A pass-through is legitimate exactly when that
    PID is a live ancestor of this process, because only then is the descriptor
    that holds the lock demonstrably still open in a process we descend from.
    """

    def _kill_detached(self, pidfile):
        """Never leave the detached job behind, however the test ended."""
        with contextlib.suppress(OSError, ValueError):
            os.killpg(int(Path(pidfile).read_text()), signal.SIGKILL)

    def held_env(self, **extra):
        env = {"VLLM_CPP_GPU_LOCK_HELD": self.lock}
        env.update(extra)
        return env

    def test_env_naming_the_path_with_no_pid_does_not_pass_through(self):
        # Defeated by any environment that merely names the path.
        with other_holder(self.lock):
            res = run(["--lock", self.lock, "--timeout", "0.5",
                       "--", *self.touch_cmd()],
                      env=self.held_env(), timeout=30)
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertNotRun(res)

    def test_env_naming_a_dead_holder_does_not_pass_through(self):
        with other_holder(self.lock):
            res = run(["--lock", self.lock, "--timeout", "0.5",
                       "--", *self.touch_cmd()],
                      env=self.held_env(VLLM_CPP_GPU_LOCK_PID=str(dead_pid())),
                      timeout=30)
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertNotRun(res)

    def test_env_naming_a_live_NON_ancestor_does_not_pass_through(self):
        # The holder is alive -- and is a sibling, not an ancestor. Liveness
        # alone is not the property; descent is.
        with other_holder(self.lock) as holder_pid:
            res = run(["--lock", self.lock, "--timeout", "0.5",
                       "--", *self.touch_cmd()],
                      env=self.held_env(VLLM_CPP_GPU_LOCK_PID=str(holder_pid)),
                      timeout=30)
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertNotRun(res)

    def test_a_non_numeric_holder_pid_does_not_pass_through(self):
        with other_holder(self.lock):
            res = run(["--lock", self.lock, "--timeout", "0.5",
                       "--", *self.touch_cmd()],
                      env=self.held_env(VLLM_CPP_GPU_LOCK_PID="not-a-pid"),
                      timeout=30)
        self.assertEqual(res.returncode, EXIT_REFUSED, res.stderr)
        self.assertNotRun(res)

    def test_the_refusal_names_the_unverifiable_claim(self):
        res = run(["--lock", self.lock, "--timeout", "0.5", "--", "true"],
                  env=self.held_env(VLLM_CPP_GPU_LOCK_PID=str(dead_pid())))
        reason = value(block(res.stderr, "REFUSED"), "reason") or ""
        self.assertIn("VLLM_CPP_GPU_LOCK_PID", reason, res.stderr)

    def test_a_genuine_nested_invocation_still_passes_through(self):
        # The positive control. The fix must not turn every pass-through into a
        # refusal: real nesting is what the pass-through exists for, and without
        # it a nested call deadlocks until its own timeout.
        inner = [str(SCRIPT), "--lock", self.lock, "--timeout", "2",
                 "--", *self.touch_cmd()]
        res = run(["--lock", self.lock, "--timeout", "2", "--", *inner],
                  timeout=45)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(os.path.exists(self.marker), res.stderr)
        self.assertIn("PASS-THROUGH", res.stderr)

    def test_a_detached_job_that_lost_the_descriptor_does_not_pass_through(self):
        """#587 reproduced THROUGH the wrapper, non-adversarially.

        Nothing here is hostile: the harness is an ordinary Python job that
        spawns background work, and ``close_fds=True`` is the documented default.
        """
        go = os.path.join(self.tmp, "go")
        rec = os.path.join(self.tmp, "detached.txt")
        pidfile = os.path.join(self.tmp, "detached.pid")
        log = os.path.join(self.tmp, "detached.log")
        harness = os.path.join(self.tmp, "harness.py")
        # BOUNDED, so a job whose temp directory is swept away cannot poll for
        # ever: a leaked spinner outliving its own test is a defect in the test,
        # not an acceptable cost of it.
        detached = (
            f"i=0; while [ ! -e {shlex.quote(go)} ] && [ $i -lt 1200 ]; do "
            f"sleep 0.05; i=$((i+1)); done; "
            f"[ -e {shlex.quote(go)} ] || exit 0; "
            f"exec {shlex.quote(str(SCRIPT))} --lock {shlex.quote(self.lock)} "
            f"--timeout 5 --record {shlex.quote(rec)} "
            f"-- touch {shlex.quote(self.marker)}"
        )
        Path(harness).write_text(textwrap.dedent(f"""\
            import subprocess
            # close_fds defaults to True, so the LOCKED descriptor does not
            # reach this job -- while the exported variable does. stdout and
            # stderr go to a FILE rather than the inherited pipes: a detached
            # job holding the parent's capture pipes open would block the
            # harness's own reader until it exits.
            log = open({log!r}, "wb")
            p = subprocess.Popen(["sh", "-c", {detached!r}],
                                 start_new_session=True,
                                 stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log)
            open({pidfile!r}, "w").write(str(p.pid))
            """))
        res = run(["--lock", self.lock, "--timeout", "5",
                   "--", sys.executable, harness])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.addCleanup(self._kill_detached, pidfile)

        # The outer wrapper has exited, so the lock really is free: an unrelated
        # job takes it, exactly as one legitimately would.
        self.assertTrue(lock_is_free(self.lock),
                        "the outer wrapper did not release the lock on exit")
        with other_holder(self.lock, seconds=25) as _:
            Path(go).write_text("")
            deadline = time.time() + 25
            refused = False
            while time.time() < deadline:
                if os.path.exists(self.marker):
                    break
                if os.path.exists(rec) and "REFUSED" in Path(rec).read_text():
                    refused = True
                    break
                time.sleep(0.05)
        record = Path(rec).read_text() if os.path.exists(rec) else "<no record>"
        self.assertFalse(
            os.path.exists(self.marker),
            f"the detached job RAN beside the lock holder; its record:\n{record}")
        self.assertTrue(refused, f"detached wrapper never refused; record:\n{record}")


# --------------------------------------------------------------------------
# 7. The wrapper's own death releases the lock and says so.
# --------------------------------------------------------------------------
class Lifecycle(Base):
    """F6 and the holder sidecar.

    Without a trap, SIGTERM to the wrapper emits ACQUIRE with no RELEASE, the
    wrapped job keeps running orphaned through the descriptor it inherited, and
    `flock -n` reports the lock STILL HELD while the stamped `holder-pid` names
    a process that no longer exists. The sidecar is only trustworthy on top of
    that: without the trap it goes stale exactly when a blocked reader consults
    it.
    """

    def start(self, *extra, cmd=None, record=None):
        args = ["--lock", self.lock, "--timeout", "10", *extra]
        if record:
            args += ["--record", record]
        args += ["--", *(cmd or ["sh", "-c", f"touch {shlex.quote(self.marker)}; sleep 60"])]
        proc = subprocess.Popen(
            [str(SCRIPT), *args],
            env={k: v for k, v in os.environ.items()
                 if k not in ("GPU_LOCK", "GPU_LOCK_TIMEOUT",
                              "VLLM_CPP_GPU_LOCK_HELD", "VLLM_CPP_GPU_LOCK_PID")},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.addCleanup(self._reap, proc)
        return proc

    def _reap(self, proc):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                with contextlib.suppress(Exception):
                    pipe.close()

    def await_marker(self, path, seconds=20):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if os.path.exists(path):
                return True
            time.sleep(0.02)
        return False

    def await_text(self, path, needle, seconds=20):
        """Wait for CONTENT, not merely for the file.

        `command-pid` is appended once the job exists, a moment after the
        sidecar is created; waiting on the path alone races that append.
        """
        deadline = time.time() + seconds
        while time.time() < deadline:
            with contextlib.suppress(OSError):
                if needle in Path(path).read_text():
                    return True
            time.sleep(0.02)
        return False

    def test_stdin_reaches_the_wrapped_command(self):
        # Guards the F6 repair itself: running the command in the background so
        # a signal can be caught mid-run reassigns stdin to /dev/null unless the
        # descriptor is re-attached explicitly. A wrapper that silently eats
        # stdin breaks every piped gate.
        res = subprocess.run(
            [str(SCRIPT), "--lock", self.lock, "--", "cat"],
            input="PAYLOAD\n", capture_output=True, text=True, timeout=60,
            env={k: v for k, v in os.environ.items() if k != "GPU_LOCK"},
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout, "PAYLOAD\n")

    def test_sigterm_to_the_wrapper_emits_a_release_block(self):
        rec = os.path.join(self.tmp, "evidence.txt")
        proc = self.start(record=rec)
        self.assertTrue(self.await_marker(self.marker), "wrapped job never started")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        text = Path(rec).read_text()
        self.assertIn("=== ACQUIRE ===", text)
        self.assertIn("=== RELEASE ===", text,
                      f"a signalled wrapper left ACQUIRE with no RELEASE:\n{text}")

    def test_sigterm_to_the_wrapper_releases_the_lock(self):
        proc = self.start()
        self.assertTrue(self.await_marker(self.marker), "wrapped job never started")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        deadline = time.time() + 10
        while time.time() < deadline and not lock_is_free(self.lock):
            time.sleep(0.05)
        self.assertTrue(lock_is_free(self.lock),
                        "the lock is still held after the wrapper died")

    def test_sigterm_to_the_wrapper_does_not_orphan_the_job(self):
        gone = os.path.join(self.tmp, "child.gone")
        proc = self.start(cmd=["sh", "-c",
                               f"trap 'touch {shlex.quote(gone)}; exit 143' TERM; "
                               f"touch {shlex.quote(self.marker)}; "
                               "while :; do sleep 0.1; done"])
        self.assertTrue(self.await_marker(self.marker), "wrapped job never started")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        self.assertTrue(self.await_marker(gone, seconds=10),
                        "the wrapped job kept running after the wrapper died")

    def test_the_wrapper_signal_is_named_in_the_record(self):
        rec = os.path.join(self.tmp, "evidence.txt")
        proc = self.start(record=rec)
        self.assertTrue(self.await_marker(self.marker))
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        self.assertIn("wrapper-signal=", Path(rec).read_text())

    # -- the holder sidecar ---------------------------------------------------
    def test_sidecar_exists_while_the_lock_is_held(self):
        proc = self.start()
        self.assertTrue(self.await_marker(self.marker))
        self.assertTrue(self.await_text(self.sidecar, "command-pid=", seconds=10),
                        "no holder sidecar beside a held lock")
        text = Path(self.sidecar).read_text()
        for key in ("lock-path=", "label=", "holder-pid=", "command-pid=",
                    "started-utc=", "command="):
            self.assertIn(key, text, f"sidecar missing {key}:\n{text}")
        proc.kill()

    def test_sidecar_carries_the_label_that_says_WHY(self):
        proc = self.start("--label", "27b-decode-grid")
        self.assertTrue(self.await_marker(self.sidecar, seconds=15))
        self.assertIn("label=27b-decode-grid", Path(self.sidecar).read_text())
        proc.kill()

    def test_sidecar_pids_are_live_while_the_hold_is_live(self):
        # The taxonomy the sidecar exists for turns on this: a LIVE pid at 0%
        # utilisation for hours is a conversation, a DEAD one is a crashed hold
        # that is safe to break. Both PIDs are recorded because a SIGKILLed
        # wrapper leaves the descendant holding the lock.
        proc = self.start()
        self.assertTrue(self.await_text(self.sidecar, "command-pid=", seconds=15))
        text = Path(self.sidecar).read_text()
        pids = dict(re.findall(r"^(holder-pid|command-pid)=(\d+)$", text, re.M))
        self.assertEqual(set(pids), {"holder-pid", "command-pid"}, text)
        for name, pid in pids.items():
            os.kill(int(pid), 0)  # raises if not live
        self.assertEqual(pids["holder-pid"], str(proc.pid))
        proc.kill()

    def test_sidecar_is_removed_on_release(self):
        res = run(["--lock", self.lock, "--", "true"])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertFalse(os.path.exists(self.sidecar),
                         "a released lock left a sidecar naming a dead holder")

    def test_sidecar_is_removed_when_the_wrapper_is_signalled(self):
        proc = self.start()
        self.assertTrue(self.await_text(self.sidecar, "command-pid=", seconds=15))
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        self.assertFalse(os.path.exists(self.sidecar),
                         "a signalled wrapper left its sidecar behind")

    def test_a_signal_arriving_before_the_job_starts_is_not_swallowed(self):
        """The window between the acquire and the job existing.

        A trap firing there has no job to forward to, and swallowing the signal
        means the wrapper sits out the entire run it was just told to stop --
        `sleep 30` here, a four-hour benchmark on a gate host.

        The window is ~1 ms wide, so ONE attempt would be a coin toss and a
        flaky test is worse than no test. The sidecar appears immediately before
        the job starts, which puts the signal on the right side of the acquire;
        REPEATING makes the defect certain to be hit while the repair makes
        every attempt pass regardless of where the signal lands.
        """
        for attempt in range(12):
            self.marker = os.path.join(self.tmp, f"early-{attempt}.marker")
            self.sidecar = self.lock + ".holder"
            proc = self.start(cmd=["sleep", "30"])
            self.assertTrue(self.await_marker(self.sidecar, seconds=15),
                            "the wrapper never acquired")
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.fail(f"attempt {attempt}: the wrapper swallowed a signal "
                          "that arrived before its job started")
            deadline = time.time() + 5
            while time.time() < deadline and not lock_is_free(self.lock):
                time.sleep(0.02)
            self.assertTrue(lock_is_free(self.lock),
                            f"attempt {attempt}: the lock is still held")

    def test_a_crashed_hold_leaves_the_sidecar_as_evidence(self):
        # SIGKILL cannot be trapped. The sidecar surviving is the POINT: it is
        # how "crashed hold, safe to break" is told from "a pre-wrapper holder,
        # treat as opaque", which before this was `nvidia-smi` archaeology.
        proc = self.start()
        self.assertTrue(self.await_marker(self.sidecar, seconds=15))
        proc.kill()
        proc.wait(timeout=10)
        self.assertTrue(os.path.exists(self.sidecar))
        self.assertIn(f"holder-pid={proc.pid}", Path(self.sidecar).read_text())

    def test_a_pass_through_does_not_disturb_the_ancestors_sidecar(self):
        # A nested invocation neither rewrites the sidecar nor removes it when
        # it finishes: the ancestor still holds the lock it describes.
        inner = shlex.join([str(SCRIPT), "--lock", self.lock, "--timeout", "2",
                            "--", "cat", self.sidecar])
        script = (f"{inner}; "
                  f"test -f {shlex.quote(self.sidecar)} && echo SIDECAR-SURVIVED")
        res = run(["--lock", self.lock, "--timeout", "5", "--label", "outer",
                   "--", "sh", "-c", script], timeout=45)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("label=outer", res.stdout,
                      "the nested invocation overwrote its ancestor's sidecar")
        self.assertIn("SIDECAR-SURVIVED", res.stdout,
                      "the nested invocation removed its ancestor's sidecar")


# --------------------------------------------------------------------------
# 8. Mutations: prove the checks above catch their defect.
# --------------------------------------------------------------------------
class Mutations(unittest.TestCase):
    """Break one guarantee at a time in a scratch copy; the check must go red.

    A test that merely describes a property is not evidence the property is
    enforced. Restoring the tree is free here because nothing outside the
    temporary directory is written.
    """

    @classmethod
    def setUpClass(cls):
        if not has_flock():
            raise unittest.SkipTest("flock(1) is required")
        if os.geteuid() == 0:
            raise unittest.SkipTest("root ignores the mode bits the refusal rests on")
        cls.source = SCRIPT.read_text()

    def mutant(self, transform):
        text = transform(self.source)
        self.assertNotEqual(text, self.source, "mutation matched nothing")
        d = tempfile.mkdtemp(prefix="gpulock-mut-")
        self.addCleanup(shutil.rmtree, d, True)
        path = Path(d) / "gpu-lock.sh"
        path.write_text(text)
        path.chmod(0o755)
        return path

    def check(self, path, cls, *names):
        """Run named tests from a guarantee class against a mutant wrapper."""
        global SCRIPT
        saved = SCRIPT
        SCRIPT = path
        try:
            suite = unittest.TestSuite(cls(n) for n in names)
            with open(os.devnull, "w") as quiet:
                result = unittest.TextTestRunner(stream=quiet, verbosity=0).run(suite)
            return len(result.failures) + len(result.errors)
        finally:
            SCRIPT = saved

    # -- M1: the refusal falls through instead of aborting -------------------
    def test_M1_refusal_that_returns_instead_of_exiting_is_caught(self):
        m = self.mutant(lambda s: s.replace('exit "$EXIT_REFUSED"', 'return 0', 1))
        self.assertGreater(
            self.check(m, Refusal,
                       "test_unwritable_lock_directory_refuses",
                       "test_missing_lock_parent_refuses"), 0)

    # -- M2: the lock actually taken is not recorded -------------------------
    def test_M2_dropping_the_lock_path_field_is_caught(self):
        m = self.mutant(lambda s: "\n".join(
            l for l in s.splitlines() if "lock-path=" not in l) + "\n")
        self.assertGreater(
            self.check(m, Stamp,
                       "test_acquire_block_carries_every_required_field",
                       "test_symlinked_lock_path_is_stamped_resolved"), 0)

    # -- M3: the exit code is not recorded -----------------------------------
    def test_M3_dropping_the_exit_code_field_is_caught(self):
        m = self.mutant(lambda s: "\n".join(
            l for l in s.splitlines() if "exit-code=" not in l) + "\n")
        self.assertGreater(
            self.check(m, ExitCode,
                       "test_exit_code_is_the_first_field_of_the_release_block",
                       "test_nonzero_propagates_unchanged"), 0)

    # -- M4: the exit code is swallowed --------------------------------------
    def test_M4_swallowing_the_exit_code_is_caught(self):
        m = self.mutant(lambda s: s.replace('exit "$RC"', "exit 0", 1))
        self.assertGreater(
            self.check(m, ExitCode,
                       "test_nonzero_propagates_unchanged",
                       "test_sigkill_137_propagates_unchanged"), 0)

    # -- M5: timeout stops being a distinct status ---------------------------
    def test_M5_collapsing_the_timeout_status_is_caught(self):
        m = self.mutant(lambda s: s.replace("EXIT_TIMEOUT=75", "EXIT_TIMEOUT=0", 1))
        self.assertGreater(
            self.check(m, Timeout,
                       "test_contended_lock_times_out_with_distinct_status"), 0)

    # -- M6: the lock is never actually taken --------------------------------
    def test_M6_skipping_the_acquisition_is_caught(self):
        m = self.mutant(lambda s: s.replace(
            'flock -w "$TIMEOUT" "$LOCK_FD"', "true", 1))
        self.assertGreater(
            self.check(m, Timeout,
                       "test_lock_is_actually_taken_so_a_holder_excludes_us",
                       "test_contended_lock_times_out_with_distinct_status"), 0)

    # -- M7: the wait becomes unbounded --------------------------------------
    def test_M7_unbounded_wait_is_caught(self):
        m = self.mutant(lambda s: s.replace(
            'flock -w "$TIMEOUT" "$LOCK_FD"', 'flock "$LOCK_FD"', 1))
        self.assertGreater(
            self.check(m, Timeout, "test_wait_is_bounded_not_indefinite"), 0)

    # -- M8: nesting self-deadlocks ------------------------------------------
    def test_M8_removing_the_pass_through_is_caught(self):
        m = self.mutant(lambda s: s.replace(
            '[ "${VLLM_CPP_GPU_LOCK_HELD:-}" = "$LOCK_PATH" ]', "false", 1))
        self.assertGreater(
            self.check(m, Nesting, "test_nested_invocation_does_not_deadlock"), 0)

    # -- M9: the pass-through trusts the environment variable alone ----------
    def test_M9_unvalidated_pass_through_is_caught(self):
        """The shipped #596 behaviour, and the whole of F1.

        Accepting the claim without proving the recorded PID is a live ancestor
        is what let a detached job stamp `mode=PASS-THROUGH` beside an unrelated
        holder.
        """
        m = self.mutant(lambda s: s.replace(
            'if lock_pid_is_live_ancestor "${VLLM_CPP_GPU_LOCK_PID:-}"; then',
            "if true; then", 1))
        self.assertGreater(
            self.check(m, PassThroughIntegrity,
                       "test_env_naming_the_path_with_no_pid_does_not_pass_through",
                       "test_env_naming_a_dead_holder_does_not_pass_through",
                       "test_a_detached_job_that_lost_the_descriptor_does_not_pass_through"), 0)

    # -- M10: the guard keys on presence, not on the path (reviewer's R7) ----
    def test_M10_guard_keyed_on_mere_presence_is_caught(self):
        m = self.mutant(lambda s: s.replace(
            '[ "${VLLM_CPP_GPU_LOCK_HELD:-}" = "$LOCK_PATH" ]',
            '[ -n "${VLLM_CPP_GPU_LOCK_HELD:-}" ]', 1))
        self.assertGreater(
            self.check(m, Nesting,
                       "test_a_different_lock_is_ACQUIRED_not_merely_stamped"), 0)

    # -- M11: the requested spelling is not recorded -------------------------
    def test_M11_dropping_the_requested_lock_field_is_caught(self):
        m = self.mutant(lambda s: "\n".join(
            l for l in s.splitlines() if "requested-lock=" not in l) + "\n")
        self.assertGreater(
            self.check(m, Stamp,
                       "test_acquire_block_carries_every_required_field",
                       "test_requested_lock_records_the_spelling_that_was_ASKED_for"), 0)

    # -- M12: the fallback becomes silent again ------------------------------
    def test_M12_dropping_the_lock_source_field_is_caught(self):
        m = self.mutant(lambda s: "\n".join(
            l for l in s.splitlines() if "lock-source=" not in l) + "\n")
        self.assertGreater(
            self.check(m, Stamp,
                       "test_acquire_block_carries_every_required_field",
                       "test_an_empty_gpu_lock_env_falls_back_LOUDLY"), 0)

    # -- M13: an empty --lock falls back instead of refusing -----------------
    def test_M13_empty_lock_flag_falling_back_is_caught(self):
        m = self.mutant(lambda s: s.replace(
            'die_usage "--lock was given an empty path"', "true", 1))
        self.assertGreater(
            self.check(m, Refusal,
                       "test_empty_lock_flag_is_a_usage_error_not_a_silent_fallback",
                       "test_empty_lock_flag_with_equals_is_also_a_usage_error"), 0)

    # -- M14: the usage status collapses into a refusal ----------------------
    def test_M14_collapsing_the_usage_status_is_caught(self):
        m = self.mutant(lambda s: s.replace("EXIT_USAGE=2", "EXIT_USAGE=78", 1))
        self.assertGreater(
            self.check(m, Refusal,
                       "test_usage_status_is_distinct_from_refusal_and_timeout"), 0)

    # -- M15: the record is opened AFTER the lock is validated ---------------
    def test_M15_validating_the_lock_before_opening_the_record_is_caught(self):
        """Restores the shipped order rather than deleting anything.

        A mutation that simply drops the record block would redden every record
        test and prove nothing about ORDER, which is the whole finding: on a
        lock-path refusal the record file was never created, while a TIMEOUT
        reached it -- so the commonest refusal left no artefact and stderr is
        what a dispatched job discards.
        """
        def swap(source: str) -> str:
            record = extract_region(source, "RECORD-OPEN")
            validate = extract_region(source, "LOCK-VALIDATE")
            without = source.replace(record, "", 1)
            return without.replace(validate, validate + record, 1)

        m = self.mutant(swap)
        self.assertGreater(
            self.check(m, Refusal,
                       "test_lock_refusal_reaches_the_record_file",
                       "test_a_refusal_before_the_path_resolves_still_reaches_the_record"), 0)

    # -- M16: the wrapper's own death leaves no RELEASE ----------------------
    def test_M16_removing_the_signal_traps_is_caught(self):
        m = self.mutant(lambda s: "\n".join(
            l for l in s.splitlines()
            if not l.startswith("trap ")) + "\n")
        self.assertGreater(
            self.check(m, Lifecycle,
                       "test_sigterm_to_the_wrapper_emits_a_release_block",
                       "test_sigterm_to_the_wrapper_releases_the_lock",
                       "test_sigterm_to_the_wrapper_does_not_orphan_the_job"), 0)

    # -- M17: the wrapped job stops receiving stdin --------------------------
    def test_M17_losing_stdin_on_the_wrapped_command_is_caught(self):
        # `&` without an explicit stdin redirection reassigns /dev/null in a
        # non-interactive shell -- the trap the F6 repair walks into.
        m = self.mutant(lambda s: replace_once(s, "} <&0 &", "} &"))
        self.assertGreater(
            self.check(m, Lifecycle, "test_stdin_reaches_the_wrapped_command"), 0)

    # -- M20: the forwarded signal reaches a middleman, not the job ----------
    def test_M20_losing_the_exec_orphans_the_job(self):
        """Dropping `exec` reintroduces F6 one layer down.

        Without it bash forks a subshell that forks the command, `$!` names the
        subshell, and the forwarded signal kills the middleman while the real
        job keeps running -- measured, not theorised.
        """
        m = self.mutant(lambda s: replace_once(
            s, '{ exec ${LAUNCH[@]+"${LAUNCH[@]}"} "$@"; } <&0 &',
            '{ ${LAUNCH[@]+"${LAUNCH[@]}"} "$@"; } <&0 &'))
        self.assertGreater(
            self.check(m, Lifecycle,
                       "test_sigterm_to_the_wrapper_does_not_orphan_the_job",
                       "test_sigterm_to_the_wrapper_releases_the_lock"), 0)

    # -- M21: the job gets no process group of its own -----------------------
    def test_M21_signalling_only_the_job_leaves_its_children_running(self):
        # A gate command is usually a script whose real work is a grandchild.
        m = self.mutant(lambda s: replace_once(
            s, "if command -v setsid >/dev/null 2>&1; then", "if false; then"))
        self.assertGreater(
            self.check(m, Lifecycle,
                       "test_sigterm_to_the_wrapper_releases_the_lock"), 0)

    # -- M22: a signal in the pre-job window is swallowed --------------------
    def test_M22_swallowing_an_early_signal_is_caught(self):
        m = self.mutant(lambda s: replace_once(
            s, '  on_signal "$WRAPPER_SIGNAL"\n', "  :\n"))
        self.assertGreater(
            self.check(m, Lifecycle,
                       "test_a_signal_arriving_before_the_job_starts_is_not_swallowed"), 0)

    # -- M18: the sidecar is never written -----------------------------------
    def test_M18_dropping_the_sidecar_is_caught(self):
        m = self.mutant(lambda s: replace_once(s, "\n  sidecar_open\n", "\n  :\n"))
        self.assertGreater(
            self.check(m, Lifecycle,
                       "test_sidecar_exists_while_the_lock_is_held"), 0)

    # -- M19: the sidecar outlives the hold it describes ---------------------
    def test_M19_leaving_the_sidecar_behind_is_caught(self):
        m = self.mutant(lambda s: replace_once(s, "\n  sidecar_close\n", "\n  :\n"))
        self.assertGreater(
            self.check(m, Lifecycle,
                       "test_sidecar_is_removed_on_release",
                       "test_sidecar_is_removed_when_the_wrapper_is_signalled"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
