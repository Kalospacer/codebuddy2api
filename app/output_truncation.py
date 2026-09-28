"""Truncate oversized text while preserving its head and tail."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TruncationResult", "count_text_lines", "truncate_middle_bytes"]

_BYTES_PER_TOKEN = 4
_MIN_ACTIVE_LIMIT = 256


@dataclass(frozen=True)
class TruncationResult:
    """Text projection and its byte and token estimates."""

    text: str
    truncated: bool
    original_bytes: int
    projected_bytes: int
    original_estimated_tokens: int
    projected_estimated_tokens: int
    total_lines: int
    omitted_estimated_tokens: int


def _estimate_tokens(byte_count: int) -> int:
    return (byte_count + _BYTES_PER_TOKEN - 1) // _BYTES_PER_TOKEN


def count_text_lines(text: str) -> int:
    """Count newline-delimited lines without materializing a split list."""
    if not text:
        return 0
    return text.count("\n") + (not text.endswith("\n"))


def _warning(original_bytes: int, estimated_tokens: int, total_lines: int) -> str:
    return (
        f"\n[Warning: middle omitted; original bytes: {original_bytes}, "
        f"estimated tokens: ~{estimated_tokens} (4 bytes/token), "
        f"total lines: {total_lines}.]\n"
    )


def _decode_prefix(raw: bytes, limit: int) -> str:
    end = min(limit, len(raw))
    while end > 0:
        try:
            return raw[:end].decode("utf-8")
        except UnicodeDecodeError:
            end -= 1
    return ""


def _decode_suffix(raw: bytes, limit: int) -> str:
    start = max(len(raw) - limit, 0)
    while start < len(raw):
        try:
            return raw[start:].decode("utf-8")
        except UnicodeDecodeError:
            start += 1
    return ""


def truncate_middle_bytes(text: str, max_bytes: int) -> TruncationResult:
    """Return text within max_bytes, keeping valid UTF-8 from both ends."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    if 0 < max_bytes < _MIN_ACTIVE_LIMIT:
        raise ValueError("max_bytes must be 0 or at least 256")

    raw = text.encode("utf-8")
    original_bytes = len(raw)
    original_estimated_tokens = _estimate_tokens(original_bytes)
    total_lines = count_text_lines(text)

    if max_bytes == 0 or original_bytes <= max_bytes:
        return TruncationResult(
            text=text,
            truncated=False,
            original_bytes=original_bytes,
            projected_bytes=original_bytes,
            original_estimated_tokens=original_estimated_tokens,
            projected_estimated_tokens=original_estimated_tokens,
            total_lines=total_lines,
            omitted_estimated_tokens=0,
        )

    warning = _warning(original_bytes, original_estimated_tokens, total_lines)
    content_budget = max_bytes - len(warning.encode("utf-8"))
    if content_budget <= 0:
        raise ValueError("max_bytes is too small for the truncation warning")

    head_budget = content_budget // 2
    tail_budget = content_budget - head_budget
    head = _decode_prefix(raw, head_budget)
    tail = _decode_suffix(raw, tail_budget)
    projected = head + warning + tail
    projected_bytes = len(projected.encode("utf-8"))
    projected_estimated_tokens = _estimate_tokens(projected_bytes)

    return TruncationResult(
        text=projected,
        truncated=True,
        original_bytes=original_bytes,
        projected_bytes=projected_bytes,
        original_estimated_tokens=original_estimated_tokens,
        projected_estimated_tokens=projected_estimated_tokens,
        total_lines=total_lines,
        omitted_estimated_tokens=original_estimated_tokens - projected_estimated_tokens,
    )
