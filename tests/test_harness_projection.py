"""Responses projection regressions for harness blocks, generated content, and limits."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi.testclient import TestClient

import converter
from app import upstream_io
from app.desensitize import desensitize_body

from app.adapters.responses_projection import project_responses_chat_body
from app.harness_context import parse_harness_text


TOOL = {
    "type": "function",
    "function": {
        "name": "exec_command",
        "description": "Run a command",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string", "description": "Command"}},
            "additionalProperties": False,
        },
    },
}


def body(messages, tools=None):
    return {"model": "auto", "messages": deepcopy(messages), "tools": deepcopy(tools or [TOOL])}


class ProjectionTests(unittest.TestCase):
    def test_balanced_replaces_only_recognized_harness_blocks(self):
        messages = [
            {"role": "system", "content": "SYSTEM POLICY\n<environment_context>volatile</environment_context>\nSYSTEM TAIL"},
            {"role": "user", "content": "# AGENTS.md instructions\n<INSTRUCTIONS>Use tabs.</INSTRUCTIONS>\nUSER TASK"},
            {"role": "user", "content": "<permissions instructions>old rules</permissions instructions>\nLATEST TASK"},
        ]
        payload = body(messages)
        before = deepcopy(payload)
        result, stats = project_responses_chat_body(payload)
        self.assertEqual(payload, before)
        self.assertEqual(result["messages"][0]["content"], "SYSTEM POLICY\n\n\nEnvironment context is provided by the harness.\n\n\nSYSTEM TAIL")
        self.assertIn("Repository instructions and durable user context are provided.", result["messages"][1]["content"])
        self.assertIn("Use tabs.", result["messages"][1]["content"])
        self.assertIn("Runtime permissions apply", result["messages"][2]["content"])
        self.assertIn("LATEST TASK", result["messages"][2]["content"])
        self.assertNotIn("# AGENTS.md instructions", result["messages"][1]["content"])
        self.assertEqual(result["tools"], payload["tools"])
        self.assertEqual(stats["mode"], "balanced")
        self.assertEqual(stats["harness_messages_projected"], 3)

    def test_real_user_task_in_legacy_harness_message_is_retained(self):
        task = "MUST_KEEP_TASK: inspect only and do not modify files."
        text = "# AGENTS.md instructions\n<skills_instructions>old skills</skills_instructions>\n" + task
        payload = body([{"role": "user", "content": text}])
        parsed = parse_harness_text(text)
        self.assertTrue(parsed.matched)
        self.assertIn(task, parsed.user_text)
        result, _ = project_responses_chat_body(payload)
        self.assertIn(task, result["messages"][0]["content"])
        self.assertNotIn("old skills", result["messages"][0]["content"])


    def test_real_text_survives_balanced_and_optional_desensitization(self):
        system = "CUSTOM SYSTEM: keep every rule. " + "Detail. " * 200
        task = "请解释 exploit development 和 sandbox 的含义，不执行任何操作。"
        user = "# AGENTS.md instructions\n<system-reminder>old runtime state</system-reminder>\n" + task
        result, _ = project_responses_chat_body(body([
            {"role": "system", "content": system}, {"role": "user", "content": user},
        ]))
        self.assertEqual(result["messages"][0]["content"], system)
        self.assertIn(task, result["messages"][1]["content"])
        for compact in (False, True):
            processed = desensitize_body(
                result, roles=("system", "developer"), desensitize_harness_user=True,
                compact_harness=compact,
            )
            self.assertEqual(processed["messages"][0]["content"].replace("\u200b", ""), system)
            self.assertIn(task, processed["messages"][1]["content"].replace("\u200b", ""))

    def test_unclosed_harness_markup_stays_literal(self):
        task = "- Inspect only\nDo not modify files."
        text = "<environment_context>\nUnclosed metadata-looking text.\n" + task
        parsed = parse_harness_text(text)
        self.assertFalse(parsed.matched)
        self.assertEqual(parsed.user_text, text)
        result, stats = project_responses_chat_body(body([{"role": "user", "content": text}]))
        self.assertEqual(result["messages"][0]["content"], text)
        self.assertEqual(stats["harness_messages_projected"], 0)
    def test_passthrough_does_not_change_messages_or_tools(self):
        messages = [
            {"role": "user", "content": "# AGENTS.md instructions\n<environment_context>volatile</environment_context>"},
            {"role": "assistant", "content": "x" * 500, "tool_calls": [{
                "id": "call", "type": "function",
                "function": {"name": "exec_command", "arguments": '{"cmd":"echo hi"}'},
            }]},
            {"role": "tool", "tool_call_id": "call", "content": "y" * 500},
        ]
        payload = body(messages)
        before = deepcopy(payload)
        result, stats = project_responses_chat_body(payload, mode="passthrough", max_item_bytes=256)
        self.assertEqual(result, before)
        self.assertEqual(payload, before)
        self.assertEqual(stats["mode"], "passthrough")
        self.assertEqual(stats["truncated_items"], 0)
        self.assertEqual(stats["harness_messages_projected"], 0)

    def test_generated_content_keeps_utf8_head_tail_and_metadata(self):
        assistant_text = "HEAD\n" + "中" * 180 + "\nTAIL"
        tool_output = "OUTPUT\n" + "输出" * 180 + "\nEND"
        messages = [
            {"role": "assistant", "content": assistant_text},
            {"role": "tool", "tool_call_id": "call", "content": tool_output},
        ]
        result, stats = project_responses_chat_body(body(messages, []), max_item_bytes=256)
        assistant = result["messages"][0]["content"]
        output = result["messages"][1]["content"]
        self.assertTrue(assistant.startswith("HEAD"))
        self.assertTrue(assistant.endswith("TAIL"))
        self.assertTrue(output.startswith("OUTPUT"))
        self.assertTrue(output.endswith("END"))
        for value in (assistant, output):
            self.assertIn("original bytes:", value)
            self.assertIn("estimated tokens:", value)
            self.assertIn("total lines:", value)
            self.assertLessEqual(len(value.encode("utf-8")), 256)
        self.assertEqual(stats["truncated_items"], 2)

    def test_json_tool_arguments_remain_valid_and_apply_patch_is_not_omitted(self):
        arguments = json.dumps({"cmd": "echo " + "x" * 1500, "workdir": "/tmp"})
        patch = json.dumps({"patch": "*** Begin Patch\n" + "+" * 1500 + "*** End Patch"})
        messages = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "command", "type": "function", "function": {
                "name": "exec_command", "arguments": arguments,
            }},
            {"id": "patch", "type": "function", "function": {
                "name": "apply_patch", "arguments": patch,
            }},
        ]}]
        result, stats = project_responses_chat_body(body(messages, []), max_item_bytes=1024)
        command = json.loads(result["messages"][0]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(command["workdir"], "/tmp")
        self.assertTrue(command["cmd"].startswith("echo "))
        self.assertTrue(command["cmd"].endswith("x"))
        self.assertIn("middle omitted", command["cmd"])
        self.assertIn("original bytes:", command["cmd"])
        self.assertIn("estimated tokens:", command["cmd"])
        self.assertIn("total lines:", command["cmd"])
        patch_args = json.loads(result["messages"][0]["tool_calls"][1]["function"]["arguments"])
        self.assertTrue(patch_args["patch"].startswith("*** Begin Patch"))
        self.assertTrue(patch_args["patch"].endswith("*** End Patch"))
        self.assertEqual(stats["truncated_items"], 2)


    def test_non_string_json_arguments_use_bounded_valid_wrapper(self):
        arguments = json.dumps({"values": list(range(10000))})
        messages = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "numbers", "type": "function",
            "function": {"name": "numbers", "arguments": arguments},
        }]}]
        result, stats = project_responses_chat_body(body(messages, []), max_item_bytes=256)
        wire = result["messages"][0]["tool_calls"][0]["function"]["arguments"]
        parsed = json.loads(wire)
        self.assertLessEqual(len(wire.encode("utf-8")), 256)
        self.assertEqual(parsed["_truncated"]["original_bytes"], len(arguments.encode("utf-8")))
        self.assertIn("values", parsed["head"])
        self.assertTrue(parsed["tail"].endswith("]}"))
        self.assertEqual(stats["truncated_items"], 1)
        self.assertEqual(stats["truncated_original_bytes"], len(arguments.encode("utf-8")))
        self.assertEqual(stats["truncated_projected_bytes"], len(wire.encode("utf-8")))

    def test_large_tool_arguments_skip_full_json_materialization(self):
        arguments = "[" + ("0," * 600000) + "0]"
        messages = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "large", "type": "function",
            "function": {"name": "large", "arguments": arguments},
        }]}]
        with patch("app.adapters.responses_projection.json.loads") as loads:
            result, stats = project_responses_chat_body(body(messages, []), max_item_bytes=40000)
        loads.assert_not_called()
        wire = result["messages"][0]["tool_calls"][0]["function"]["arguments"]
        self.assertLessEqual(len(wire.encode("utf-8")), 40000)
        self.assertEqual(json.loads(wire)["_truncated"]["original_bytes"], len(arguments.encode("utf-8")))
        self.assertEqual(stats["truncated_items"], 1)

    def test_large_fitting_tool_arguments_are_unchanged(self):
        arguments = json.dumps({"items": ['escaped " quote and NaN'] + ["x" * 1000] * 1099}, separators=(",", ":"))
        argument_bytes = len(arguments.encode("utf-8"))
        self.assertGreater(argument_bytes, 1024 * 1024)
        self.assertLessEqual(argument_bytes, 2_000_000)
        messages = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "large-fitting", "type": "function",
            "function": {"name": "large", "arguments": arguments},
        }]}]
        payload = body(messages, [])
        before = deepcopy(payload)
        with patch("app.adapters.responses_projection.json.loads") as loads:
            result, stats = project_responses_chat_body(payload, max_item_bytes=2_000_000)
        loads.assert_not_called()
        self.assertEqual(payload, before)
        wire = result["messages"][0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(wire, arguments)
        self.assertEqual(stats["truncated_items"], 0)

    def test_large_fitting_arguments_reject_nonstandard_constants_without_parsing(self):
        arguments = '{"prefix":"' + ("x" * 1_100_000) + '","bad":NaN}'
        self.assertGreater(len(arguments.encode("utf-8")), 1024 * 1024)
        messages = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "large-nan", "type": "function",
            "function": {"name": "large", "arguments": arguments},
        }]}]
        with patch("app.adapters.responses_projection.json.loads") as loads:
            result, stats = project_responses_chat_body(body(messages, []), max_item_bytes=2_000_000)
        loads.assert_not_called()
        wire = result["messages"][0]["tool_calls"][0]["function"]["arguments"]

        def reject_constant(value):
            raise ValueError(f"unexpected constant: {value}")

        parsed = json.loads(wire, parse_constant=reject_constant)
        self.assertIn("_truncated", parsed)
        self.assertEqual(stats["truncated_items"], 1)

    def test_escape_dense_wrapper_keeps_both_edges(self):
        arguments = json.dumps({"x": "\\" * 60000}, separators=(",", ":"))
        messages = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "escaped", "type": "function",
            "function": {"name": "escaped", "arguments": arguments},
        }]}]
        result, _ = project_responses_chat_body(body(messages, []), max_item_bytes=40000)
        wire = result["messages"][0]["tool_calls"][0]["function"]["arguments"]
        parsed = json.loads(wire)
        self.assertLessEqual(len(wire.encode("utf-8")), 40000)
        self.assertIn("_truncated", parsed)
        self.assertTrue(parsed["head"])
        self.assertTrue(parsed["tail"])

    def test_nonstandard_json_constants_fall_back_to_strict_wrapper(self):
        arguments = '{"x":NaN,"s":"' + ("x" * 500) + '"}'
        messages = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "nan", "type": "function",
            "function": {"name": "nan", "arguments": arguments},
        }]}]
        result, _ = project_responses_chat_body(body(messages, []), max_item_bytes=1024)
        wire = result["messages"][0]["tool_calls"][0]["function"]["arguments"]

        def reject_constant(value):
            raise ValueError(f"unexpected constant: {value}")

        parsed = json.loads(wire, parse_constant=reject_constant)
        self.assertIn("_truncated", parsed)
        self.assertLessEqual(len(wire.encode("utf-8")), 1024)

    def test_zero_limit_disables_generated_clipping(self):
        messages = [
            {"role": "assistant", "content": "a" * 1000},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call", "type": "function",
                "function": {"name": "tool", "arguments": '{"value":"' + "x" * 1000 + '"}'},
            }]},
        ]
        payload = body(messages, [])
        before = deepcopy(payload)
        result, stats = project_responses_chat_body(payload, max_item_bytes=0)
        self.assertEqual(result, before)
        self.assertEqual(payload, before)
        self.assertEqual(stats["truncated_items"], 0)

    def test_history_tool_chain_images_and_system_text_are_preserved(self):
        image = {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}
        call = {"id": "call-old", "type": "function", "function": {"name": "view", "arguments": "{}"}}
        messages = [
            {"role": "system", "content": "CUSTOM SYSTEM POLICY"},
            {"role": "user", "content": [{"type": "text", "text": "TASK"}, image]},
            {"role": "assistant", "content": "inspect", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call-old", "content": "old result"},
        ]
        payload = body(messages)
        result, stats = project_responses_chat_body(payload)
        self.assertEqual(result["messages"], payload["messages"])
        self.assertEqual(result["messages"][1]["content"][1], image)
        self.assertEqual(result["messages"][2]["tool_calls"][0]["id"], "call-old")
        self.assertEqual(result["messages"][3]["tool_call_id"], "call-old")
        self.assertIn("CUSTOM SYSTEM POLICY", result["messages"][0]["content"])
        self.assertEqual(stats["projected_messages"], stats["original_messages"])
        self.assertEqual(stats["harness_messages_projected"], 0)

    def test_invalid_mode_and_limit_are_rejected(self):
        for mode in ("aggressive", "conservative", "", None, True):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    project_responses_chat_body({"messages": []}, mode=mode)
        for limit in (-1, 1, 128, 255, True, 1.5, "256", None, [], {}):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    project_responses_chat_body({"messages": []}, max_item_bytes=limit)
        self.assertEqual(project_responses_chat_body({"messages": []}, max_item_bytes=256)[1]["max_item_bytes"], 256)

    def test_stats_have_current_official_shape(self):
        payload = body([{"role": "user", "content": "hello"}])
        _, stats = project_responses_chat_body(payload, mode="balanced", max_item_bytes=40000)
        self.assertEqual(set(stats), {
            "mode", "max_item_bytes", "original_messages", "projected_messages",
            "original_message_chars", "projected_message_chars", "original_tools",
            "projected_tools", "original_tool_chars", "projected_tool_chars",
            "harness_messages_projected", "truncated_items", "truncated_original_bytes",
            "truncated_projected_bytes",
        })
        self.assertEqual(stats["mode"], "balanced")
        self.assertEqual(stats["max_item_bytes"], 40000)
        self.assertEqual(stats["original_messages"], stats["projected_messages"])
        self.assertEqual(stats["original_tools"], stats["projected_tools"])


class ResponsesEndpointTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(converter.CONFIG, {
            "api_key": "", "cred": None, "cred_pool": None, "model_guard": False,
            "max_images": 16, "image_policy": "truncate", "max_request_bytes": 32 * 1024 * 1024,
            "log_body_limit": 0, "log_path": None, "desensitize": False, "no_compact": False,
            "responses_projection_mode": "balanced", "responses_projection_max_bytes": 512,
        }))
        self.enterContext(patch.object(converter, "_cred_for", return_value=(None, {})))
        self.enterContext(patch.object(converter, "_log"))
        self.captured = []
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(self.handle)
        self.enterContext(patch.object(upstream_io.httpx, "AsyncClient",
                                      side_effect=lambda **kw: real_client(transport=transport, **kw)))
        self.client = self.enterContext(TestClient(converter.app))

    def handle(self, request):
        self.captured.append(json.loads(request.content))
        chunk = {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}
        return httpx.Response(200, content=("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())

    def request_payload(self, assistant_text, arguments, tool_output):
        return {
            "model": "auto", "stream": False,
            "input": [
                {"role": "user", "content": "inspect"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": assistant_text},
                ]},
                {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": arguments},
                {"type": "function_call_output", "call_id": "call_1", "output": tool_output},
            ],
            "tools": [{
                "type": "function", "name": "exec_command", "description": "Run command",
                "parameters": {"type": "object", "properties": {
                    "cmd": {"type": "string", "description": "Command"},
                }, "additionalProperties": False},
            }],
        }

    def test_projection_mode_header_and_upstream_payload(self):
        assistant_text = "HEAD\n" + "中" * 180 + "\nTAIL"
        arguments = json.dumps({"cmd": "echo " + "x" * 500})
        tool_output = "OUTPUT\n" + "输出" * 180 + "\nEND"
        for mode in ("balanced", "passthrough"):
            with self.subTest(mode=mode):
                converter.CONFIG["responses_projection_mode"] = mode
                self.captured.clear()
                payload = self.request_payload(assistant_text, arguments, tool_output)
                original = deepcopy(payload)
                response = self.client.post("/v1/responses", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["X-CodeBuddy-Responses-Projection"], mode)
                self.assertEqual(payload, original)
                self.assertEqual(len(self.captured), 1)
                captured = self.captured[0]
                self.assertEqual(captured["tools"][0]["function"]["parameters"], payload["tools"][0]["parameters"])
                assistant = next(message for message in captured["messages"] if message["role"] == "assistant")
                tool = next(message for message in captured["messages"] if message["role"] == "tool")
                if mode == "passthrough":
                    self.assertEqual(assistant["content"], assistant_text)
                    self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], arguments)
                    self.assertEqual(tool["content"], tool_output)
                else:
                    self.assertTrue(assistant["content"].startswith("HEAD"))
                    self.assertTrue(assistant["content"].endswith("TAIL"))
                    self.assertIn("original bytes:", assistant["content"])
                    self.assertTrue(tool["content"].startswith("OUTPUT"))
                    self.assertTrue(tool["content"].endswith("END"))
                    projected_args = json.loads(assistant["tool_calls"][0]["function"]["arguments"])
                    self.assertTrue(projected_args["cmd"].startswith("echo "))
                    self.assertTrue(projected_args["cmd"].endswith("x"))
                    self.assertIn("middle omitted", projected_args["cmd"])


    def test_invalid_unicode_is_rejected_before_upstream(self):
        payload = {"model": "auto", "stream": False, "input": [
            {"type": "function_call", "name": "tool", "arguments": '{"x":"\ud800"}'},
        ]}
        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.captured.clear()
        response = self.client.post(
            "/v1/responses", content=raw, headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["code"], "invalid_unicode")
        self.assertEqual(self.captured, [])

    def test_non_string_tool_arguments_are_rejected_before_upstream(self):
        payload = {"model": "auto", "stream": False, "input": [
            {"type": "function_call", "name": "tool", "arguments": {"x": 1}},
        ]}
        self.captured.clear()
        response = self.client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("JSON string", response.json()["error"]["message"])
        self.assertEqual(self.captured, [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
