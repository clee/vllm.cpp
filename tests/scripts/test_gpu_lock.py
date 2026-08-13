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
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gpu-lock.sh"

# Exit statuses the wrapper reserves for its own failures. They must stay
# distinct from each other; a wrapped command that happens to exit 75 is told
# apart by the stamped `outcome=` field, which only a RUN emits.
EXIT_REFUSED = 78
EXIT_TIMEOUT = 75

FIELD = re.compile(r"^gpu-lock: ([a-z0-9-]+)=(.*)$")


def has_flock() -> bool:
    return shutil.which("flock") is not None


def run(args, *, env=None, cwd=None, timeout=60, base_env=True):
    """Invoke the wrapper. Never inherits a real GPU_LOCK/HOME by accident."""
    full = dict(os.environ) if base_env else {}
    full.pop("GPU_LOCK", None)
    full.pop("GPU_LOCK_TIMEOUT", None)
    full.pop("VLLM_CPP_GPU_LOCK_HELD", None)
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


@contextlib.contextmanager
def other_holder(lock_path: str, seconds: int = 30):
    """A second job holding the SAME lock file, as #587's two jobs did not."""
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
        yield
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
        self.assertNotEqual(res.returncode, 0)

    def test_nonnumeric_timeout_is_rejected(self):
        res = run(["--lock", self.lock, "--timeout", "soon", "--", *self.touch_cmd()])
        self.assertNotEqual(res.returncode, 0)
        self.assertNotRun(res)


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
        res = self.acquire("--timeout", "5")
        pairs = block(res.stderr, "ACQUIRE")
        keys = [k for k, _ in pairs]
        for required in ("mode", "lock-path", "holder-pid", "waited-seconds",
                         "timeout-seconds", "loadavg", "df-root", "command"):
            self.assertIn(required, keys, f"missing {required}; got {keys}")

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


# --------------------------------------------------------------------------
# 6. Mutations: prove the checks above catch their defect.
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
