"""Responses projection keeps complete tool metadata in every supported mode."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.responses_projection import project_responses_chat_body


SCHEMA = {
    "type": "object",
    "title": "Tool inputs",
    "description": "Read sandbox inputs.",
    "properties": {
        "path": {
            "type": "string", "title": "Path", "description": "A path.",
            "enum": ["sandbox", "local"],
        },
        "list": {"type": "array", "items": {"type": "string", "description": "Item"}},
        "choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "mapping": {"type": "object", "additionalProperties": {"type": "string"}},
        "description": {"type": "string", "title": "A description property"},
    },
    "required": ["path"],
    "additionalProperties": False,
    "x-vendor": {"deep": {"schema": "must remain"}},
}


def tool_body(agentic=False):
    return {
        "model": "auto",
        "messages": [
            {"role": "system", "content": (
                "You are a coding agent running in the Codex CLI."
                if agentic else "You are a helpful assistant."
            )},
            {"role": "user", "content": "Read a file."},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_data",
                "description": "Read sandbox data.",
                "title": "Data reader",
                "parameters": SCHEMA,
                "strict": True,
            },
        }],
    }


class ToolMetadataTests(unittest.TestCase):
    def test_balanced_and_passthrough_preserve_complete_tool_schema(self):
        for agentic, mode in ((False, "balanced"), (True, "balanced"), (False, "passthrough"), (True, "passthrough")):
            with self.subTest(agentic=agentic, mode=mode):
                body = tool_body(agentic)
                before = copy.deepcopy(body)
                result, stats = project_responses_chat_body(body, mode=mode)
                self.assertEqual(body, before)
                self.assertEqual(result["tools"], body["tools"])
                self.assertEqual(result["messages"], body["messages"])
                self.assertEqual(result["tools"][0]["function"]["parameters"], SCHEMA)
                self.assertEqual(result["tools"][0]["function"]["description"], "Read sandbox data.")
                self.assertEqual(result["tools"][0]["function"]["title"], "Data reader")
                self.assertIs(result["tools"][0]["function"]["strict"], True)
                self.assertEqual(stats["mode"], mode)
                self.assertEqual(stats["original_tools"], stats["projected_tools"])
                self.assertEqual(stats["original_tool_chars"], stats["projected_tool_chars"])

    def test_projection_does_not_mutate_nested_tool_structures(self):
        body = tool_body(agentic=True)
        before = json.dumps(body, sort_keys=True, ensure_ascii=False)
        result, _ = project_responses_chat_body(body, max_item_bytes=0)
        self.assertEqual(json.dumps(body, sort_keys=True, ensure_ascii=False), before)
        self.assertEqual(result["tools"], body["tools"])
        self.assertEqual(result["tools"][0]["function"]["parameters"], SCHEMA)

    def test_stats_expose_tool_preservation_in_official_shape(self):
        body = tool_body()
        _, stats = project_responses_chat_body(body, mode="balanced", max_item_bytes=40000)
        self.assertEqual(
            set(stats),
            {
                "mode", "max_item_bytes", "original_messages", "projected_messages",
                "original_message_chars", "projected_message_chars", "original_tools",
                "projected_tools", "original_tool_chars", "projected_tool_chars",
                "harness_messages_projected", "truncated_items", "truncated_original_bytes",
                "truncated_projected_bytes",
            },
        )
        self.assertEqual(stats["mode"], "balanced")
        self.assertEqual(stats["max_item_bytes"], 40000)
        self.assertEqual(stats["original_tools"], 1)
        self.assertEqual(stats["projected_tools"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
