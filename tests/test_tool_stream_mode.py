"""Tool-bearing Chat streams must expose the first frame before generation completes."""
import asyncio
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import converter


class ToolStreamModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_passthrough_delivers_before_upstream_completion(self):
        release = asyncio.Event()
        first = 'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{"}}]},"finish_reason":null}]}'

        class Response:
            status_code = 200

            async def aiter_lines(self):
                yield first
                yield ""
                await release.wait()
                yield 'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":"tool_calls"}]}'
                yield ""
                yield "data: [DONE]"
                yield ""

        @asynccontextmanager
        async def backend(*args, **kwargs):
            yield Response()

        with patch.dict(converter.CONFIG, tool_stream_passthrough=True, log_path=None), \
                patch.object(converter, "_backend_stream", backend):
            stream = converter._stream_upstream("https://synthetic.invalid", {},
                                               {"model": "synthetic", "tools": [{}]}, t0=time.time())
            try:
                frame = await asyncio.wait_for(anext(stream), timeout=1)
                self.assertIn(b"call-1", frame)
                self.assertFalse(release.is_set())
                release.set()
                remaining = b"".join([chunk async for chunk in stream])
                self.assertIn(b"[DONE]", remaining)
            finally:
                release.set()
                await stream.aclose()

    async def test_aggregate_mode_uses_checked_generation(self):
        result = {"choices": [{"message": {"role": "assistant", "content": "done"},
                                "finish_reason": "stop"}], "usage": {}}
        with patch.dict(converter.CONFIG, tool_stream_passthrough=False), \
                patch.object(converter, "_fetch_checked_chat", return_value=result) as checked:
            chunks = [chunk async for chunk in converter._stream_upstream(
                "https://synthetic.invalid", {}, {"tools": [{}]}, t0=time.time())]
        checked.assert_awaited_once()
        self.assertIn(b"done", b"".join(chunks))


if __name__ == "__main__":
    unittest.main()
