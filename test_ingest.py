"""Unit tests for sliding windows, extractive summary, and KnowledgePyramid (stdlib only)."""

from __future__ import annotations

import unittest

from ingest import (
    KnowledgePyramid,
    PyramidLevel,
    extractive_summary,
    sliding_window_chunks,
)


class TestSlidingWindowChunks(unittest.TestCase):
    def test_empty_and_whitespace(self) -> None:
        self.assertEqual(sliding_window_chunks("", 4000, 2000), [])
        self.assertEqual(sliding_window_chunks("   \n", 4000, 2000), [])

    def test_invalid_args(self) -> None:
        with self.assertRaises(ValueError):
            sliding_window_chunks("a", 0, 0)
        with self.assertRaises(ValueError):
            sliding_window_chunks("a", 100, -1)
        with self.assertRaises(ValueError):
            sliding_window_chunks("a", 100, 100)

    def test_single_window_when_short(self) -> None:
        self.assertEqual(sliding_window_chunks("hello world", 4000, 2000), ["hello world"])

    def test_overlap_stride(self) -> None:
        # 10 chars, window 4, overlap 1 -> step 3 -> positions 0,3,6,9
        s = "0123456789"
        chunks = sliding_window_chunks(s, window_chars=4, overlap_chars=1)
        self.assertEqual(chunks, ["0123", "3456", "6789", "9"])


class TestExtractiveSummary(unittest.TestCase):
    def test_short_returns_full(self) -> None:
        t = "Short text."
        self.assertEqual(extractive_summary(t, max_chars=100), t)

    def test_prefers_content_rich_sentence(self) -> None:
        text = (
            "The cat sat. "
            "Machine learning models require data and careful training pipelines. "
            "A random filler sentence about nothing special."
        )
        out = extractive_summary(text, max_chars=120)
        self.assertIn("Machine learning", out)
        self.assertLessEqual(len(out), 120)


class TestKnowledgePyramid(unittest.TestCase):
    def test_ingest_retrieve_finance(self) -> None:
        kp = KnowledgePyramid(window_chars=500, overlap_chars=0, keyword_top_k=5)
        kp.ingest_text("Our revenue grew this quarter. Finance teams reviewed the balance sheet.")
        hit = kp.retrieve("revenue profit")
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.window_index, 0)
        self.assertIn(hit.level, (PyramidLevel.LEVEL_1_RAW, PyramidLevel.LEVEL_4_KEYWORDS))

    def test_category_tech(self) -> None:
        kp = KnowledgePyramid(window_chars=200, overlap_chars=0)
        kp.ingest_text("We deploy the model on kubernetes and use python.")
        w0 = kp.windows[0]
        self.assertEqual(w0.category, "Tech")

    def test_retrieve_top_k_dedup(self) -> None:
        kp = KnowledgePyramid(window_chars=300, overlap_chars=0)
        kp.ingest_text("alpha beta gamma delta epsilon zeta eta theta.")
        hits = kp.retrieve_top_k("alpha", k=3)
        self.assertGreaterEqual(len(hits), 1)
        keys = {(h.window_index, h.level) for h in hits}
        self.assertEqual(len(keys), len(hits))


if __name__ == "__main__":
    unittest.main()
