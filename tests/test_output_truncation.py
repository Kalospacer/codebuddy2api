#!/usr/bin/env python3
"""Test byte-safe middle truncation and its exact projections."""
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Allow direct execution.

from app.output_truncation import TruncationResult, truncate_middle_bytes


class OutputTruncationTests(unittest.TestCase):
    def test_public_result_shape(self):
        self.assertTrue(is_dataclass(TruncationResult))
        self.assertEqual(
            [field.name for field in fields(TruncationResult)],
            [
                "text",
                "truncated",
                "original_bytes",
                "projected_bytes",
                "original_estimated_tokens",
                "projected_estimated_tokens",
                "total_lines",
                "omitted_estimated_tokens",
            ],
        )

    def test_zero_limit_and_fitting_ascii_are_returned_unchanged(self):
        text = "first line\nsecond line\nthird line"
        byte_count = len(text.encode("utf-8"))
        expected_tokens = (byte_count + 3) // 4
        for limit in (0, 256, 512):
            with self.subTest(limit=limit):
                result = truncate_middle_bytes(text, limit)
                self.assertEqual(result.text, text)
                self.assertFalse(result.truncated)
                self.assertEqual(result.original_bytes, byte_count)
                self.assertEqual(result.projected_bytes, byte_count)
                self.assertEqual(result.original_estimated_tokens, expected_tokens)
                self.assertEqual(result.projected_estimated_tokens, expected_tokens)
                self.assertEqual(result.total_lines, 3)
                self.assertEqual(result.omitted_estimated_tokens, 0)

    def test_ascii_multiline_keeps_head_and_tail_with_warning(self):
        text = "HEAD-MARKER\n" + ("middle line\n" * 30) + "TAIL-MARKER"
        result = truncate_middle_bytes(text, 256)
        self.assertTrue(result.truncated)
        self.assertTrue(result.text.startswith("HEAD-MARKER\n"))
        self.assertTrue(result.text.endswith("TAIL-MARKER"))
        marker_start = result.text.index("[Warning:")
        marker_end = result.text.index("]\n", marker_start) + 2
        head = result.text[:marker_start].removesuffix("\n")
        tail = result.text[marker_end:]
        self.assertTrue(text.startswith(head))
        self.assertTrue(text.endswith(tail))
        self.assertLess(len(head) + len(tail), len(text))
        self.assertIn("Warning: middle omitted", result.text)
        self.assertIn(f"original bytes: {len(text.encode('utf-8'))}", result.text)
        self.assertIn("4 bytes/token", result.text)
        self.assertIn("total lines: 32", result.text)

    def test_single_line_and_tight_budget(self):
        text = "HEAD-middle-content-TAIL" * 20
        result = truncate_middle_bytes(text, 256)
        self.assertTrue(result.truncated)
        self.assertLessEqual(result.projected_bytes, 256)
        self.assertTrue(result.text.startswith("HEAD-"))
        self.assertTrue(result.text.endswith("TAIL"))
        self.assertIn("middle omitted", result.text)
        self.assertEqual(result.total_lines, 1)

    def test_chinese_and_emoji_boundaries_remain_valid_utf8(self):
        text = "开头-中文-🙂-" + "中间内容🙂" * 30 + "-结尾-🚀"
        before = text
        result = truncate_middle_bytes(text, 256)
        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.text.encode("utf-8")), 256)
        self.assertTrue(result.text.startswith("开头-中文-🙂-"))
        self.assertTrue(result.text.endswith("-结尾-🚀"))
        self.assertIn("middle omitted", result.text)
        self.assertEqual(text, before)
        result.text.encode("utf-8").decode("utf-8")

    def test_projection_statistics_are_exact(self):
        text = "é\n" + "🙂漢" * 50 + "\n終"
        original_bytes = len(text.encode("utf-8"))
        result = truncate_middle_bytes(text, 256)
        self.assertEqual(result.projected_bytes, len(result.text.encode("utf-8")))
        self.assertEqual(result.original_bytes, original_bytes)
        self.assertEqual(result.original_estimated_tokens, (original_bytes + 3) // 4)
        self.assertEqual(result.projected_estimated_tokens, (result.projected_bytes + 3) // 4)
        self.assertEqual(
            result.omitted_estimated_tokens,
            result.original_estimated_tokens - result.projected_estimated_tokens,
        )
        self.assertEqual(result.total_lines, 3)

    def test_empty_text_statistics(self):
        result = truncate_middle_bytes("", 0)
        self.assertEqual(result.text, "")
        self.assertEqual(result.total_lines, 0)
        self.assertEqual(result.original_estimated_tokens, 0)
        self.assertEqual(result.projected_estimated_tokens, 0)


    def test_newline_flood_line_count_does_not_require_split_list(self):
        text = "\n" * 1_000_000
        result = truncate_middle_bytes(text, 256)
        self.assertEqual(result.total_lines, 1_000_000)
        self.assertLessEqual(result.projected_bytes, 256)
    def test_small_or_invalid_limits_raise(self):
        for limit in (-1, 1, 128, 255):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                truncate_middle_bytes("text", limit)


if __name__ == "__main__":
    unittest.main()
