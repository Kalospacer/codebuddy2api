"""Apply optional Responses context projection without changing real user text."""

from __future__ import annotations

import json
import re
from typing import Any

from app.harness_context import parse_harness_text
from app.output_truncation import TruncationResult, count_text_lines, truncate_middle_bytes


PROJECTION_MODES = ("balanced", "passthrough")
_TEXT_TYPES = {"text", "input_text", "output_text"}

_JSON_OVERHEAD_RESERVE = 256
_MAX_TOOL_ARGUMENTS_PARSE_BYTES = 1024 * 1024
_MAX_WRAPPER_EDGE_BYTES = 8192
_JSON_QUOTE_OR_NONSTANDARD = re.compile(r'"|NaN|-?Infinity')


class _InvalidJsonConstant(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise _InvalidJsonConstant(f"unsupported JSON constant: {value}")


def _find_unescaped_quote(text: str, start: int) -> int:
    """Find the next unescaped quote without copying the string."""
    cursor = start
    while True:
        quote = text.find('"', cursor)
        if quote < 0:
            return -1
        slash = quote - 1
        backslashes = 0
        while slash >= start and text[slash] == "\\":
            backslashes += 1
            slash -= 1
        if backslashes % 2 == 0:
            return quote
        cursor = quote + 1


def _contains_nonstandard_json_constant(text: str) -> bool:
    """Detect bare non-standard JSON constants without materializing parsed data."""
    cursor = 0
    while True:
        match = _JSON_QUOTE_OR_NONSTANDARD.search(text, cursor)
        if match is None:
            return False
        if text[match.start()] != '"':
            return True
        quote = _find_unescaped_quote(text, match.end())
        if quote < 0:
            return False
        cursor = quote + 1


def project_responses_chat_body(
    body: dict,
    *,
    mode: str = "balanced",
    max_item_bytes: int = 40000,
) -> tuple[dict, dict]:
    """Project a Responses-derived Chat body or return it unchanged."""
    if mode not in PROJECTION_MODES:
        raise ValueError("invalid Responses projection mode")
    if isinstance(max_item_bytes, bool) or not isinstance(max_item_bytes, int) or max_item_bytes < 0:
        raise ValueError("max_item_bytes must be a non-negative integer")
    if 0 < max_item_bytes < 256:
        raise ValueError("max_item_bytes must be 0 or at least 256")

    messages = list(body.get("messages") or [])
    tools = list(body.get("tools") or [])
    counters = {
        "harness_messages_projected": 0,
        "truncated_items": 0,
        "truncated_original_bytes": 0,
        "truncated_projected_bytes": 0,
    }
    projected = dict(body)

    if mode == "passthrough":
        projected_messages = messages
    else:
        projected_messages = [
            _project_message(message, max_item_bytes, counters) for message in messages
        ]

    if "messages" in body:
        projected["messages"] = projected_messages
    if "tools" in body:
        projected["tools"] = tools

    return projected, {
        "mode": mode,
        "max_item_bytes": max_item_bytes,
        "original_messages": len(messages),
        "projected_messages": len(projected_messages),
        "original_message_chars": _messages_size(messages),
        "projected_message_chars": _messages_size(projected_messages),
        "original_tools": len(tools),
        "projected_tools": len(tools),
        "original_tool_chars": _tools_size(tools),
        "projected_tool_chars": _tools_size(tools),
        **counters,
    }


def _project_message(message: Any, max_item_bytes: int, counters: dict[str, int]) -> Any:
    if not isinstance(message, dict):
        return message

    role = message.get("role")
    projected = dict(message)
    if role in {"system", "user"}:
        content, changed = _map_text_content(
            message.get("content", ""),
            lambda text: _project_instruction_text(text),
        )
        projected["content"] = content
        counters["harness_messages_projected"] += int(changed)
    elif role == "assistant":
        projected["content"] = _project_generated_content(
            message.get("content", ""), max_item_bytes, counters
        )
        if "tool_calls" in message:
            projected["tool_calls"] = [
                _project_tool_call(call, max_item_bytes, counters)
                for call in message.get("tool_calls") or []
            ]
    elif role == "tool":
        projected["content"] = _project_generated_content(
            message.get("content", ""), max_item_bytes, counters
        )
    return projected


def _project_instruction_text(text: str) -> str:
    parsed = parse_harness_text(text)
    return parsed.render() if parsed.matched else text


def _project_generated_content(content: Any, max_item_bytes: int, counters: dict[str, int]) -> Any:
    transformed, _ = _map_text_content(
        content,
        lambda text: _truncate_generated_text(text, max_item_bytes, counters),
    )
    return transformed


def _truncate_generated_text(text: str, max_item_bytes: int, counters: dict[str, int]) -> str:
    result = truncate_middle_bytes(text, max_item_bytes)
    if result.truncated:
        _record_truncation(result, counters)
    return result.text


def _map_text_content(content: Any, transform) -> tuple[Any, bool]:
    if isinstance(content, str):
        projected = transform(content)
        return projected, projected != content
    if not isinstance(content, list):
        return content, False

    projected = []
    changed = False
    for block in content:
        replacement = block
        if isinstance(block, str):
            replacement = transform(block)
        elif isinstance(block, dict) and block.get("type") in _TEXT_TYPES and isinstance(block.get("text"), str):
            replacement = {**block, "text": transform(block["text"])}
        changed = changed or replacement is not block and replacement != block
        projected.append(replacement)
    return (projected if changed else content), changed


def _project_tool_call(tool_call: Any, max_item_bytes: int, counters: dict[str, int]) -> Any:
    if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
        return tool_call

    projected = dict(tool_call)
    function = dict(tool_call["function"])
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("tool call arguments must be a JSON string")
    if max_item_bytes == 0:
        return projected
    argument_bytes = len(arguments.encode("utf-8"))
    if argument_bytes > _MAX_TOOL_ARGUMENTS_PARSE_BYTES:
        if (argument_bytes <= max_item_bytes
                and not _contains_nonstandard_json_constant(arguments)):
            return projected
        function["arguments"] = _truncate_json_argument_text(arguments, max_item_bytes, counters)
        projected["function"] = function
        return projected
    try:
        decoded = json.loads(arguments, parse_constant=_reject_json_constant)
    except _InvalidJsonConstant:
        function["arguments"] = _truncate_json_argument_text(arguments, max_item_bytes, counters)
        projected["function"] = function
        return projected
    except (TypeError, ValueError, RecursionError):
        if argument_bytes <= max_item_bytes:
            return projected
        function["arguments"] = _truncate_json_argument_text(arguments, max_item_bytes, counters)
        projected["function"] = function
        return projected
    if _compact_json(decoded) is None:
        function["arguments"] = _truncate_json_argument_text(arguments, max_item_bytes, counters)
        projected["function"] = function
        return projected
    if argument_bytes <= max_item_bytes:
        return projected
    string_limit = max(256, max_item_bytes - _JSON_OVERHEAD_RESERVE)
    trial_counters = {
        "truncated_items": 0,
        "truncated_original_bytes": 0,
        "truncated_projected_bytes": 0,
    }
    projected_value, changed = _truncate_json_strings(decoded, string_limit, trial_counters)
    serialized = _compact_json(projected_value)
    if serialized is not None and len(serialized.encode("utf-8")) <= max_item_bytes:
        if changed or serialized != arguments:
            function["arguments"] = serialized
            projected["function"] = function
            _merge_counters(trial_counters, counters)
        return projected

    function["arguments"] = _truncate_json_argument_text(arguments, max_item_bytes, counters)
    projected["function"] = function
    return projected


def _truncate_json_argument_text(text: str, max_bytes: int, counters: dict[str, int]) -> str:
    """Return bounded valid JSON with the original head, tail and size metadata."""
    raw = text.encode("utf-8")
    original_bytes = len(raw)
    original_tokens = (original_bytes + 3) // 4
    total_lines = count_text_lines(text)
    wrapper = {
        "_truncated": {
            "warning": "middle omitted; head and tail retained",
            "original_bytes": original_bytes,
            "estimated_tokens": original_tokens,
            "total_lines": total_lines,
        },
        "head": "",
        "tail": "",
    }
    empty = _compact_json(wrapper) or "{}"
    remaining = max(0, max_bytes - len(empty.encode("utf-8")))
    edge_limit = min(remaining // 2, _MAX_WRAPPER_EDGE_BYTES)
    wrapper["head"] = _fit_json_prefix(raw, edge_limit)
    wrapper["tail"] = _fit_json_suffix(raw, min(remaining - edge_limit, _MAX_WRAPPER_EDGE_BYTES))
    payload = _compact_json(wrapper) or empty
    if len(payload.encode("utf-8")) > max_bytes:
        payload = _compact_json({"truncated": True, "original_bytes": original_bytes}) or "{}"
    projected_bytes = len(payload.encode("utf-8"))
    projected_tokens = (projected_bytes + 3) // 4
    result = TruncationResult(
        text=payload,
        truncated=True,
        original_bytes=original_bytes,
        projected_bytes=projected_bytes,
        original_estimated_tokens=original_tokens,
        projected_estimated_tokens=projected_tokens,
        total_lines=total_lines,
        omitted_estimated_tokens=max(0, original_tokens - projected_tokens),
    )
    _record_truncation(result, counters)
    return payload


def _compact_json(value: Any) -> str | None:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return None


def _json_escape_size(value: str) -> int:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return len(encoded[1:-1].encode("utf-8"))


def _fit_json_prefix(raw: bytes, limit: int) -> str:
    return _fit_json_edge(raw, limit, from_end=False)


def _fit_json_suffix(raw: bytes, limit: int) -> str:
    return _fit_json_edge(raw, limit, from_end=True)


def _fit_json_edge(raw: bytes, limit: int, *, from_end: bool) -> str:
    if limit <= 0:
        return ""
    low = 0
    high = min(len(raw), max(limit, limit * 6))
    while low < high:
        middle = (low + high + 1) // 2
        candidate = raw[-middle:] if from_end else raw[:middle]
        value = candidate.decode("utf-8", "ignore")
        if _json_escape_size(value) <= limit:
            low = middle
        else:
            high = middle - 1
    candidate = raw[-low:] if from_end else raw[:low]
    return candidate.decode("utf-8", "ignore") if low else ""


def _merge_counters(source: dict[str, int], target: dict[str, int]) -> None:
    for key in ("truncated_items", "truncated_original_bytes", "truncated_projected_bytes"):
        target[key] += source.get(key, 0)


def _truncate_json_strings(value: Any, max_item_bytes: int, counters: dict[str, int], depth: int = 0) -> tuple[Any, bool]:
    if depth > 64:
        return value, False
    if isinstance(value, str):
        projected = _truncate_generated_text(value, max_item_bytes, counters)
        return projected, projected != value
    if isinstance(value, list):
        projected = []
        changed = False
        for item in value:
            projected_item, item_changed = _truncate_json_strings(
                item, max_item_bytes, counters, depth + 1
            )
            projected.append(projected_item)
            changed = changed or item_changed
        return projected, changed
    if isinstance(value, dict):
        projected = {}
        changed = False
        for key, item in value.items():
            projected_item, item_changed = _truncate_json_strings(
                item, max_item_bytes, counters, depth + 1
            )
            projected[key] = projected_item
            changed = changed or item_changed
        return projected, changed
    return value, False


def _record_truncation(result, counters: dict[str, int]) -> None:
    counters["truncated_items"] += 1
    counters["truncated_original_bytes"] += result.original_bytes
    counters["truncated_projected_bytes"] += result.projected_bytes


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            for key in ("text", "output"):
                if key in block:
                    parts.append(str(block.get(key) or ""))
                    break
    return "".join(parts)


def _message_cost(message: Any) -> int:
    if not isinstance(message, dict):
        return 0
    cost = len(_content_to_text(message.get("content", "")))
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        cost += len(str(function.get("name") or ""))
        cost += len(str(function.get("arguments") or ""))
    return cost


def _messages_size(messages: list[Any]) -> int:
    return sum(_message_cost(message) + len(message.get("role", "")) for message in messages if isinstance(message, dict))


def _tools_size(tools: list[Any]) -> int:
    try:
        return len(json.dumps(tools, ensure_ascii=False))
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return 0
