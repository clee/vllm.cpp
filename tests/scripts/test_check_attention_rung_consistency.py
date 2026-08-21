#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-attention-rung-consistency.py.

The mutation cases below are the point of the file. A checker that reports zero
drift on a green tree proves nothing on its own: it reports zero drift when its
regex matches nothing at all, which is exactly how #1544's defect went unseen for
nine call sites. Each `MutationTests` case makes the tree carry the regression the
checker exists to catch and requires the checker to go RED.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-attention-rung-consistency.py"
SPEC = importlib.util.spec_from_file_location("check_attention_rung", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

MODELS = "src/vllm/model_executor/models"
ALLOWLIST = ROOT / "scripts/attention-rung-allowlist.txt"

MARKED = """\
void Forward() {
  // VT-ATTN-NAIVE: reference arm of the dense/paged equivalence gate.
  vt::Attention(q, out, qq, kk, vv, args);
}
"""

UNMARKED = """\
void Forward() {
  vt::Attention(q, out, qq, kk, vv, args);
}
"""


class CallDetectionTests(unittest.TestCase):
    def test_unmarked_site_is_found(self) -> None:
        self.assertEqual(mod.scan_file(UNMARKED), [(2, False)])

    def test_marked_site_is_found_and_credited(self) -> None:
        self.assertEqual(mod.scan_file(MARKED), [(3, True)])

    def test_fast_rungs_are_never_sites(self) -> None:
        # Without the word boundary in _NAIVE_CALL every one of these matches, and
        # the checker would demand a marker beside exactly the calls it wants.
        for fast in (
            "vt::AttentionDenseFlash(q, o, a, b, c, args);",
            "vt::AttentionDenseFast(q, o, a, b, c, args);",
            "vt::AttentionDenseFa2(q, o, a, b, c, args);",
            "vt::AttentionCross(q, o, a, b, c, args);",
            "vt::PagedAttention(q, o, a, b, c, args);",
        ):
            self.assertEqual(mod.scan_file(fast), [], fast)

    def test_a_commented_out_call_is_not_a_site(self) -> None:
        self.assertEqual(mod.scan_file("// vt::Attention(q, o, a, b, c, args);\n"), [])

    def test_a_block_commented_call_is_not_a_site(self) -> None:
        text = "/* legacy:\n vt::Attention(q, o, a, b, c, args);\n*/\n"
        self.assertEqual(mod.scan_file(text), [])

    def test_an_if_zero_call_is_not_a_site(self) -> None:
        text = "#if 0\nvt::Attention(q, o, a, b, c, args);\n#endif\n"
        self.assertEqual(mod.scan_file(text), [])

    def test_line_numbers_survive_normalization(self) -> None:
        # normalize_source is position-preserving; a checker that reported a line
        # from the normalized text would drift by every stripped block comment.
        text = "/* a\n b\n c */\n\nvt::Attention(q, o, a, b, c, args);\n"
        self.assertEqual(mod.scan_file(text), [(5, False)])
        self.assertEqual(text.splitlines()[4].strip()[:14], "vt::Attention(")

    def test_two_sites_are_reported_independently(self) -> None:
        text = MARKED + "\n" * 40 + UNMARKED
        sites = mod.scan_file(text)
        self.assertEqual([marked for _, marked in sites], [True, False])


class MarkerTests(unittest.TestCase):
    def test_reason_must_be_substantive(self) -> None:
        self.assertIsNone(mod.marker_reason("  // nothing to see here"))
        self.assertEqual(mod.marker_reason("// VT-ATTN-NAIVE: x"), "x")
        # ...but a one-character reason does not satisfy the site.
        self.assertFalse(mod.has_marker(["// VT-ATTN-NAIVE: x", "vt::Attention(a);"], 2))

    def test_marker_must_be_a_comment(self) -> None:
        # A string literal naming the marker is not a record.
        self.assertIsNone(mod.marker_reason('const char* s = "VT-ATTN-NAIVE: nope at all";'))

    def test_marker_window_is_bounded(self) -> None:
        lines = ["// VT-ATTN-NAIVE: a genuine recorded reason"] + [""] * 40
        lines.append("vt::Attention(a);")
        self.assertFalse(mod.has_marker(lines, len(lines)))
        near = ["// VT-ATTN-NAIVE: a genuine recorded reason"] + [""] * 5
        near.append("vt::Attention(a);")
        self.assertTrue(mod.has_marker(near, len(near)))

    def test_marker_on_the_call_line_counts(self) -> None:
        line = "vt::Attention(a);  // VT-ATTN-NAIVE: the eager rung of the A/B"
        self.assertTrue(mod.has_marker([line], 1))


class DriftTests(unittest.TestCase):
    def test_marked_site_never_drifts(self) -> None:
        self.assertEqual(
            mod.drift_sites({f"{MODELS}/nemotron_h.cpp": [(675, True)]}, set()), []
        )

    def test_unmarked_site_drifts(self) -> None:
        self.assertEqual(
            mod.drift_sites({f"{MODELS}/muse_glimmer_vision.cpp": [(639, False)]}, set()),
            [(f"{MODELS}/muse_glimmer_vision.cpp", 639)],
        )

    def test_allowlisted_stem_passes(self) -> None:
        self.assertEqual(
            mod.drift_sites({f"{MODELS}/ltx2.cpp": [(959, False)]}, {"ltx2"}),
            [],
        )

    def test_mixed_reports_only_the_unmarked(self) -> None:
        self.assertEqual(
            mod.drift_sites(
                {
                    f"{MODELS}/whisper_audio.cpp": [(324, True)],
                    f"{MODELS}/qwen3_5.cpp": [(5279, True)],
                    f"{MODELS}/muse_glimmer_vision.cpp": [(639, False)],
                    f"{MODELS}/ltx2.cpp": [(959, False)],
                },
                allowlisted={"ltx2"},
            ),
            [(f"{MODELS}/muse_glimmer_vision.cpp", 639)],
        )

    def test_stale_entry_is_reported_and_not_fatal(self) -> None:
        scanned = {f"{MODELS}/ltx2.cpp": [(959, True)]}
        self.assertEqual(mod.stale_allowlist_entries(scanned, {"ltx2"}), ["ltx2"])
        self.assertEqual(mod.drift_sites(scanned, {"ltx2"}), [])

    def test_a_header_and_a_cpp_sharing_a_stem_do_not_collide(self) -> None:
        # Keyed on the PATH: keyed on the stem, ltx2.h would overwrite ltx2.cpp and
        # the checker would silently scan one file instead of two.
        scanned = {
            f"{MODELS}/ltx2.cpp": [(959, False)],
            "include/vllm/model_executor/models/ltx2.h": [(31, False)],
        }
        self.assertEqual(len(scanned), 2)
        self.assertEqual(len(mod.drift_sites(scanned, set())), 2)
        self.assertEqual(mod.drift_sites(scanned, {"ltx2"}), [])

    def test_allowlist_parsing(self) -> None:
        text = "# comment\nltx2  # trailing reason\nltx2_device\n\n"
        self.assertEqual(mod.allowlisted_names(text), {"ltx2", "ltx2_device"})


class ShippedTreeTests(unittest.TestCase):
    def scan(self):
        return mod.scan_models(), mod.allowlisted_names(
            ALLOWLIST.read_text(encoding="utf-8")
        )

    def test_shipped_tree_is_green(self) -> None:
        scanned, allowed = self.scan()
        self.assertEqual(mod.drift_sites(scanned, allowed), [])

    def test_the_population_is_not_empty(self) -> None:
        # A checker whose scan finds nothing is green for the wrong reason. This is
        # the guard against a regex that stops matching after a rename.
        scanned, _ = self.scan()
        self.assertGreaterEqual(sum(len(v) for v in scanned.values()), 9)

    def test_the_six_deliberate_sites_carry_a_marker(self) -> None:
        scanned, _ = self.scan()
        for stem in (
            "whisper_audio",
            "qwen3_vl_vision",
            "kimi_linear_device",
            "qwen3_5",
            "nemotron_h",
            "nemotron_h_device",
        ):
            path = f"{MODELS}/{stem}.cpp"
            self.assertIn(path, scanned, path)
            self.assertTrue(all(m for _, m in scanned[path]), path)

    def test_allowlist_holds_only_the_in_flight_stems(self) -> None:
        # It is not a parking lot. Growth is a review decision, and this pins the
        # set so growth is visible in a diff of this file.
        _, allowed = self.scan()
        self.assertEqual(allowed, {"muse_glimmer_vision", "ltx2", "ltx2_device"})


class MutationTests(unittest.TestCase):
    """Each case injects the regression the checker exists to catch."""

    def setUp(self) -> None:
        self.scanned, self.allowed = mod.scan_models(), mod.allowlisted_names(
            ALLOWLIST.read_text(encoding="utf-8")
        )

    def test_a_new_unmarked_model_goes_red(self) -> None:
        mutated = dict(self.scanned)
        new = f"{MODELS}/some_new_vision_tower.cpp"
        mutated[new] = [(412, False)]
        self.assertEqual(mod.drift_sites(mutated, self.allowed), [(new, 412)])

    def test_a_new_unmarked_call_in_a_HEADER_goes_red(self) -> None:
        # The bypass the .h glob closes: a call moved into an inline function.
        mutated = dict(self.scanned)
        hdr = "include/vllm/model_executor/models/some_new_tower.h"
        mutated[hdr] = [(88, False)]
        self.assertEqual(mod.drift_sites(mutated, self.allowed), [(hdr, 88)])

    def test_deleting_a_marker_goes_red(self) -> None:
        mutated = dict(self.scanned)
        path = f"{MODELS}/whisper_audio.cpp"
        mutated[path] = [(line, False) for line, _ in self.scanned[path]]
        self.assertTrue(mod.drift_sites(mutated, self.allowed))

    def test_a_second_unmarked_call_in_a_marked_file_goes_red(self) -> None:
        # Per-SITE, not per-file: a file that already records one reason must not
        # launder a new naive call added elsewhere in it.
        mutated = dict(self.scanned)
        path = f"{MODELS}/qwen3_5.cpp"
        mutated[path] = list(self.scanned[path]) + [(9999, False)]
        self.assertIn((path, 9999), mod.drift_sites(mutated, self.allowed))

    def test_a_stub_reason_goes_red(self) -> None:
        lines = ["// VT-ATTN-NAIVE: todo", "vt::Attention(a);"]
        self.assertFalse(mod.has_marker(lines, 2))

    def test_widening_the_regex_to_the_fast_rungs_is_visible(self) -> None:
        # If _NAIVE_CALL ever loses its word boundary, every fast-rung call becomes
        # a site and the shipped tree turns red. Pinning it here means the widening
        # is caught in this suite instead of as an unexplained mass failure.
        self.assertIsNone(mod._NAIVE_CALL.search("vt::AttentionDenseFlash(a);"))
        self.assertIsNotNone(mod._NAIVE_CALL.search("vt::Attention (a);"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
