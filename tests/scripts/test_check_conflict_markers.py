#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-conflict-markers.py (#1417).

Four record gates return rc 0 on a document carrying literal conflict markers,
so a half-merged keyed row satisfies every budget it is measured against. The
checker under test is the one place that reads a tracked file and asks whether a
merge tool wrote into it.

TWO DIRECTIONS IN EVERY CASE. A gate that refuses a marker is only half the
contract; the other half is that it stays silent on ordinary text. A bare row of
seven `=` characters is a setext heading underline and a horizontal rule, and
five shipped files already carry lines of eight or more `=`. So the separator
cases below assert exit 0 as hard as the marker cases assert exit 1.

NO MARKER LITERAL LIVES IN THIS FILE. Every pattern is built by character
repetition, so this suite never contains a marker at the start of a line and
therefore never needs an exclusion from the checker it tests. That is the whole
reason the tree needs no allowlist: an allowlist is a file every change must
append to, and AGENTS.md forbids that surface.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-conflict-markers.py"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
CI = ROOT / ".github/workflows/ci.yml"

# Built, never written. See the module docstring.
START = "<" * 7
SEPARATOR = "=" * 7
END = ">" * 7

EXAMINED = re.compile(r"examined ([0-9]+) tracked text files")


class CheckerCase(unittest.TestCase):
    """Base case: build a scratch git repository and run the shipped checker."""

    def run_checker(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )

    def scratch_repo(self, files: dict[str, bytes]) -> Path:
        """A temporary repository whose index tracks `files`.

        The files are staged and never committed. `git ls-files` reads the
        index, so a commit would add an identity requirement and prove nothing.
        """
        root = Path(tempfile.mkdtemp(prefix="vllm-conflict-markers-"))
        subprocess.run(
            ["git", "init", "-q", str(root)], check=True, capture_output=True
        )
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if files:
            subprocess.run(
                ["git", "-C", str(root), "add", "-A"],
                check=True,
                capture_output=True,
            )
        return root

    def examined_count(self, output: str) -> int:
        match = EXAMINED.search(output)
        self.assertIsNotNone(
            match, f"report names no examined count:\n{output}"
        )
        assert match is not None
        return int(match.group(1))


class MarkerRefusalTests(CheckerCase):
    """The red-before direction: a spliced marker must exit NON-ZERO."""

    def test_a_full_conflict_hunk_is_refused(self) -> None:
        body = (
            "| Capability | State |\n"
            "|---|---|\n"
            f"{START} HEAD\n"
            "| a | ours |\n"
            f"{SEPARATOR}\n"
            "| a | theirs |\n"
            f"{END} origin/main\n"
        )
        root = self.scratch_repo({"docs/STATUS.md": body.encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        # Each of the three lines named, by number, so the reader sees the hunk
        # and not only its first line.
        self.assertIn("docs/STATUS.md:3", result.stdout)
        self.assertIn("docs/STATUS.md:5", result.stdout)
        self.assertIn("docs/STATUS.md:7", result.stdout)

    def test_a_lone_start_marker_is_refused(self) -> None:
        root = self.scratch_repo({"a.md": f"text\n{START} HEAD\nmore\n".encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("a.md:2", result.stdout)

    def test_a_lone_end_marker_is_refused(self) -> None:
        root = self.scratch_repo({"a.md": f"text\n{END} branch\n".encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("a.md:2", result.stdout)

    def test_crlf_line_endings_are_refused(self) -> None:
        body = f"text\r\n{START} HEAD\r\nours\r\n{SEPARATOR}\r\ntheirs\r\n{END} b\r\n"
        root = self.scratch_repo({"a.md": body.encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("a.md:2", result.stdout)
        # The separator too: a rstrip that forgets the carriage return would
        # report the two markers and silently drop the middle of the hunk.
        self.assertIn("a.md:4", result.stdout)

    def test_a_separator_after_the_hunk_closes_is_not_named(self) -> None:
        """The adjacency guard, made load-bearing.

        Written because the first mutation run found it was not. Dropping
        `open_at is not None` left every other case green: a file with no marker
        exits the scan early, so the guard never decided anything there. It only
        decides in a file that HAS a marker and also carries a legal rule of
        seven `=` outside the hunk, and the observable is the REPORT rather than
        the exit code, which is 1 either way.
        """
        body = (
            "text\n"
            f"{START} HEAD\n"
            "ours\n"
            f"{SEPARATOR}\n"
            "theirs\n"
            f"{END} other\n"
            "A heading\n"
            f"{SEPARATOR}\n"
        )
        root = self.scratch_repo({"a.md": body.encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        # Positive control first: the report does name lines, so a later
        # assertion of absence is about this line and not about a broken match.
        self.assertIn("a.md:2", result.stdout)
        self.assertIn("a.md:4", result.stdout)
        self.assertIn("a.md:6", result.stdout)
        self.assertNotIn("a.md:8", result.stdout, result.stdout)

    def test_the_failure_report_names_the_remedy(self) -> None:
        root = self.scratch_repo({"a.md": f"{START} HEAD\n".encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("indent", result.stdout.lower())


class OrdinaryTextTests(CheckerCase):
    """The other direction: the gate must not fire on legal markdown."""

    def test_a_bare_separator_alone_is_not_refused(self) -> None:
        # A setext heading underline. Exactly seven `=`, no marker anywhere.
        body = f"A heading\n{SEPARATOR}\n\nbody text\n"
        root = self.scratch_repo({"a.md": body.encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_long_rule_of_equals_is_not_refused(self) -> None:
        # The shape docs/bench-evidence/*.log and the tokenizer corpora carry.
        body = "=" * 40 + "\nrun output\n" + "=" * 8 + "\n"
        root = self.scratch_repo({"a.log": body.encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_separator_outside_an_open_hunk_is_not_reported(self) -> None:
        # Separator, then text, then separator: no marker opened a hunk, so
        # neither line is a finding and the file is clean.
        body = f"one\n{SEPARATOR}\ntwo\n{SEPARATOR}\n"
        root = self.scratch_repo({"a.md": body.encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_marker_that_is_not_at_column_zero_is_not_refused(self) -> None:
        # The documented remedy for a document that quotes a marker on purpose.
        body = f"    {START} HEAD\n    {SEPARATOR}\n    {END} other\n"
        root = self.scratch_repo({"guide.md": body.encode()})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class SkipTests(CheckerCase):
    """What is not read, and the counts that say so."""

    def test_a_binary_file_with_marker_bytes_is_skipped(self) -> None:
        blob = b"\x00\x01\x02" + f"\n{START} HEAD\n".encode() + b"\x00" * 16
        root = self.scratch_repo({"t.bin": blob, "a.md": b"clean\n"})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 binary", result.stdout)
        self.assertEqual(self.examined_count(result.stdout), 1)

    def test_a_symlink_is_skipped(self) -> None:
        root = self.scratch_repo({"a.md": b"clean\n"})
        os.symlink("a.md", root / "link.md")
        subprocess.run(
            ["git", "-C", str(root), "add", "-A"], check=True, capture_output=True
        )
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 symlink", result.stdout)

    def test_a_tracked_path_absent_from_the_worktree_is_skipped(self) -> None:
        root = self.scratch_repo({"a.md": b"clean\n", "gone.md": b"clean\n"})
        (root / "gone.md").unlink()
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 absent", result.stdout)

    def test_an_untracked_file_is_not_examined(self) -> None:
        root = self.scratch_repo({"a.md": b"clean\n"})
        (root / "scratch.md").write_bytes(f"{START} HEAD\n".encode())
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.examined_count(result.stdout), 1)


class ReportingTests(CheckerCase):
    """A gate that cannot say how much it examined has not reported."""

    def test_a_tree_with_no_text_files_exits_two(self) -> None:
        root = self.scratch_repo({})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(self.examined_count(result.stdout + result.stderr), 0)

    def test_the_report_names_the_resolved_root(self) -> None:
        root = self.scratch_repo({"a.md": b"clean\n"})
        result = self.run_checker(root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # A checker resolves its root from its own path, so a run from a linked
        # worktree can read the shared checkout while printing OK. The report
        # says which tree the verdict is about.
        self.assertIn(str(root.resolve()), result.stdout)

    def test_the_shipped_tree_is_clean(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CHECKER)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        examined = self.examined_count(result.stdout)
        # Not a stored count and not a ratchet: a lower bound far below the real
        # number, which only fails when the scan collapsed to nothing. The tree
        # carried 4806 tracked paths when this was written.
        self.assertGreater(examined, 1000, result.stdout)


class RegistrationTests(unittest.TestCase):
    """A gate registered nowhere runs nowhere."""

    def test_the_checker_is_registered_in_preflight_and_ci(self) -> None:
        preflight = PREFLIGHT.read_text()
        self.assertIn("check-conflict-markers", preflight)
        self.assertIn("test_check_conflict_markers", preflight)
        self.assertIn("check-conflict-markers", CI.read_text())


if __name__ == "__main__":
    # unittest.main() alone would exit before this file could assert that
    # anything ran. A suite that ran zero cases prints OK and proves nothing.
    outcome = unittest.main(exit=False, verbosity=2)
    ran = outcome.result.testsRun
    print(f"cases run: {ran}")
    if ran == 0:
        print("FAIL: zero cases ran, so this suite reported nothing")
        raise SystemExit(2)
    raise SystemExit(0 if outcome.result.wasSuccessful() else 1)
