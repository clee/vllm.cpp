#!/usr/bin/env python3
"""Derive the per-output-token time of one A2-Q1 A3 gate run, WITH its terms.

`examples/nemotron_h_gen` reports neither a rate nor a duration: it prints the
engine load time and a `TOKEN MATCH: m/n` line and nothing else. So the per-token
number this row is measured on has to be derived, and a derived number that hides
its terms is how a rate over an unknown denominator comes to be quoted as if it
were measured.

This prints the wall time, the load it subtracts, and the token count it divides
by, on separate lines, and it REFUSES rather than printing 0 when either term is
missing from the log.

    nemotron-h-a2q1-per-token.py <a3.log> <label> <t0_epoch> <t1_epoch>
"""

from __future__ import annotations

import re
import sys

# vLLM 0.26.0.dev0 on the same GB10 workload, the denominator this row's gap is
# quoted against. Carried as a constant so the ratio below cannot be computed
# against a number nobody can find the origin of.
VLLM_PER_TOKEN_S = 0.014369


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__.strip())
        return 2
    log, label, t0, t1 = argv[1], argv[2], float(argv[3]), float(argv[4])
    with open(log, errors="replace") as fh:
        text = fh.read()
    wall = t1 - t0
    print(f"{label}: wall {wall:.3f} s")

    load = re.search(r"engine loaded in ([0-9.]+)s", text)
    match = re.search(r"TOKEN MATCH: (\d+)/(\d+) over (\d+) prompt", text)
    if load is None:
        print(f"{label}: NO LOAD LINE -- per-token time NOT derivable, not 0")
        return 0
    if match is None:
        print(f"{label}: NO TOKEN MATCH LINE -- per-token time NOT derivable, not 0")
        return 0

    load_s = float(load.group(1))
    matched, compared = int(match.group(1)), int(match.group(2))
    print(f"{label}: engine load {load_s:.1f} s ; tokens compared {compared} ; "
          f"matched {matched}")
    if compared <= 0:
        print(f"{label}: ZERO tokens compared -- a rate here would divide by nothing")
        return 0
    decode_s = wall - load_s
    if decode_s <= 0.0:
        # The wall clock cannot be shorter than the load it contains. When it
        # reads that way the two terms came from different runs, or the sampler
        # wrapper mismeasured, and the quotient is a NEGATIVE rate -- a number
        # that looks like a measurement and would be quoted as one. Refuse.
        print(f"{label}: wall {wall:.3f} s is not greater than the load "
              f"{load_s:.1f} s -- the terms disagree, so NO per-token time is "
              f"reported rather than a negative rate")
        return 0
    per_token = decode_s / compared
    print(f"{label}: per output token {per_token:.6f} s "
          f"(vLLM {VLLM_PER_TOKEN_S} s; ratio {per_token / VLLM_PER_TOKEN_S:.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
