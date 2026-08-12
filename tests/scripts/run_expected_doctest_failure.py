#!/usr/bin/env python3
"""Require one intentional doctest assertion failure without process abort."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: run_expected_doctest_failure.py TEST EXPECTED_TEXT")
        return 2

    completed = subprocess.run(
        [sys.argv[1]],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="")
    if completed.returncode != 1:
        print(f"expected doctest exit 1, got {completed.returncode}")
        return 1
    if sys.argv[2] not in completed.stdout:
        print(f"missing expected doctest assertion: {sys.argv[2]}")
        return 1
    if "[doctest] Status: FAILURE!" not in completed.stdout:
        print("doctest failure summary was not emitted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
