import json
import unittest
from unittest.mock import patch

import httpx

from app.mimo_client import MimoWebClient, _ReasoningStreamParser, MIMO_MODELS
from app.utils import parse_curl_command, parse_xml_tool_call


class CurlParserTests(unittest.TestCase):
    def test_parses_chromium_cookie_flag(self):
        result = parse_curl_command(
            "curl 'https://aistudio.xiaomimimo.com/open-apis/bot/chat' "
            "-H 'user-agent: TestUA' "
            "-b 'serviceToken=abc123; foo=bar' "
            "--data-raw '{\"query\":\"Hi\"}'"
        )

        self.assertEqual(
            result["url"],
            "https://aistudio.xiaomimimo.com/open-apis/bot/chat",
        )
        self.assertEqual(result["cookies"], "serviceToken=abc123; foo=bar")
        self.assertEqual(result["user_agent"], "TestUA")
        self.assertEqual(result["body"], {"query": "Hi"})


class ToolCallParserTests(unittest.TestCase):
    def test_parses_mimo_xml_tool_call(self):
        tool_calls = parse_xml_tool_call(
            '<tool_call>\n'
            '<function=bash>\n'
            '<parameter=command>whoami</parameter>\n'
            '<parameter=description>Get current user</parameter>\n'
            '</function>\n'
            '</tool_call>'
        )

        self.assertIsNotNone(tool_calls)
        tool_call = tool_calls[0]
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "bash")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {"command": "whoami", "description": "Get current user"},
        )

    def test_parses_embedded_mimo_xml_tool_call(self):
        tool_calls = parse_xml_tool_call(
            "I'll run a simple command to verify the tool system is working."
            "<tool_call>\n"
            "<function=bash>\n"
            '<parameter=command>echo "Tool system is working"</parameter>\n'
            "<parameter=description>Test tool call</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {
                "command": 'echo "Tool system is working"',
                "description": "Test tool call",
            },
        )

    def test_parses_named_mimo_xml_tool_call(self):
        tool_calls = parse_xml_tool_call(
            'I will run the command now.\n\n'
            '<toolcall>\n'
            '<tool_call>\n'
            '<name>bash</name>\n'
            '<arguments>\n'
            '<command>echo "Tool system is working"</command>\n'
            '<description>Test tool call</description>\n'
            '</arguments>\n'
            '</tool_call>\n'
            '</toolcall>'
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "bash")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {
                "command": 'echo "Tool system is working"',
                "description": "Test tool call",
            },
        )

    def test_parses_json_wrapped_mimo_tool_call(self):
        tool_calls = parse_xml_tool_call(
            'I will run the command now.\n\n'
            '<toolcall>\n'
            '{"name": "bash", "arguments": {"command": "echo \\"Tool system is working\\"", "description": "Test tool call"}}\n'
            '</toolcall>'
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "bash")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {
                "command": 'echo "Tool system is working"',
                "description": "Test tool call",
            },
        )

    def test_parses_json_wrapped_mimo_tool_call_with_underscore(self):
        tool_calls = parse_xml_tool_call(
            '<tool_call>\n'
            '{"name": "bash", "arguments": {"command": "echo ok"}}\n'
            '</tool_call>'
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"command": "echo ok"},
        )

    def test_ignores_regular_content(self):
        self.assertIsNone(parse_xml_tool_call("Hello"))


class ReasoningParserTests(unittest.TestCase):
    def test_handles_markers_split_across_chunks(self):
        parser = _ReasoningStreamParser()
        output = []
        for chunk in ("<thi", "nk>\x00secret", "</think", ">\x00answer"):
            output.extend(parser.feed(chunk))
        output.extend(parser.feed("", final=True))

        reasoning = "".join(text for text, is_reasoning in output if is_reasoning)
        content = "".join(text for text, is_reasoning in output if not is_reasoning)
        self.assertEqual(reasoning, "secret")
        self.assertEqual(content, "answer")


class MimoClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.usage_patch = patch("app.mimo_client.usage_tracker.record")
        self.mock_usage_record = self.usage_patch.start()
        self.client = MimoWebClient({
            "cookies": (
                'serviceToken="test"; '
                'xiaomichatbot_ph="browser-fingerprint=="'
            )
        })
        await self.client._client.aclose()

        sse = b"".join([
            b'event: dialogId\ndata: {"content":"42"}\n\n',
            (
                b'event: message\n'
                b'data: {"content":"<think>\\u0000why</think>\\u0000Hello"}\n\n'
            ),
            (
                b'event: usage\n'
                b'data: {"promptTokens":2,"completionTokens":3,"totalTokens":5}\n\n'
            ),
            b"event: finish\ndata: {}\n\n",
        ])

        async def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(request.url.path, "/open-apis/bot/chat")
            self.assertEqual(
                request.url.params["xiaomichatbot_ph"],
                "browser-fingerprint==",
            )
            self.assertEqual(payload["query"], "Hi")
            self.assertEqual(payload["modelConfig"]["model"], "mimo-v2-flash")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse,
            )

        self.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

    async def asyncTearDown(self):
        await self.client.close()
        self.usage_patch.stop()

    async def test_non_stream_response(self):
        result = await self.client.chat_completion(
            [{"role": "user", "content": "Hi"}],
            model="mimo-v2-flash",
        )

        message = result["choices"][0]["message"]
        self.assertEqual(message["content"], "Hello")
        self.assertEqual(message["reasoning_content"], "why")
        self.assertEqual(result["usage"]["total_tokens"], 5)
        self.mock_usage_record.assert_called_with(
            self.client.name,
            "mimo-v2-flash",
            {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
            False,
        )

    async def test_stream_response(self):
        stream = await self.client.chat_completion(
            [{"role": "user", "content": "Hi"}],
            model="mimo-v2-flash",
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

        deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
        self.assertIn({"role": "assistant", "reasoning_content": "why"}, deltas)
        self.assertIn({"role": "assistant", "content": "Hello"}, deltas)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.mock_usage_record.assert_called_with(
            self.client.name,
            "mimo-v2-flash",
            {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
            True,
        )

    async def test_non_stream_response_executes_xml_tool_call(self):
        first_sse = b"".join([
            (
                b'event: message\n'
                b'data: {"content":"<tool_call>\\n<function=bash>\\n'
                b'<parameter=command>printf executed</parameter>\\n'
                b'<parameter=description>Run direct command</parameter>\\n'
                b'</function>\\n</tool_call>"}\n\n'
            ),
            b"event: finish\ndata: {}\n\n",
        ])
        second_sse = b"".join([
            b'event: message\ndata: {"content":"Tool finished: executed"}\n\n',
            b"event: finish\ndata: {}\n\n",
        ])
        seen_payloads = []

        async def handler(request):
            payload = json.loads(request.content)
            seen_payloads.append(payload)
            content = first_sse if len(seen_payloads) == 1 else second_sse
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
            )

        await self.client._client.aclose()
        self.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        result = await self.client.chat_completion(
            [{"role": "user", "content": "Hi"}],
            model="mimo-v2-flash",
        )

        choice = result["choices"][0]
        message = choice["message"]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(message["content"], "Tool finished: executed")
        self.assertEqual(len(seen_payloads), 2)
        self.assertIn("$ printf executed", seen_payloads[1]["query"])
        self.assertIn("executed", seen_payloads[1]["query"])


class UsageNormalizationTests(unittest.TestCase):
    def test_normalizes_openai_cached_tokens(self):
        usage = MimoWebClient._normalize_usage({
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 6},
        })
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(usage["completion_tokens"], 4)
        self.assertEqual(usage["cache_tokens"], 6)

    def test_normalizes_mimo_usage_fields(self):
        usage = MimoWebClient._normalize_usage({
            "promptTokens": 7,
            "completionTokens": 3,
            "totalTokens": 10,
            "cacheTokens": 2,
        })
        self.assertEqual(usage["prompt_tokens"], 7)
        self.assertEqual(usage["completion_tokens"], 3)
        self.assertEqual(usage["total_tokens"], 10)
        self.assertEqual(usage["cache_tokens"], 2)


class BuildPayloadTests(unittest.TestCase):
    def setUp(self):
        self.client = MimoWebClient({
            "cookies": 'serviceToken="test"; xiaomichatbot_ph="fp=="'
        })

    def test_web_search_disabled_by_default(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro", {}
        )
        self.assertEqual(payload["modelConfig"]["webSearchStatus"], "disabled")

    def test_web_search_enabled(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro",
            {"web_search": True},
        )
        self.assertEqual(payload["modelConfig"]["webSearchStatus"], "enabled")

    def test_thinking_false_by_default(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro", {}
        )
        self.assertFalse(payload["modelConfig"]["enableThinking"])

    def test_thinking_bool_true(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro",
            {"thinking": True},
        )
        self.assertTrue(payload["modelConfig"]["enableThinking"])

    def test_thinking_reasoning_effort(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro",
            {"reasoning_effort": "high"},
        )
        self.assertTrue(payload["modelConfig"]["enableThinking"])

    def test_stop_string(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro",
            {"stop": "STOP"},
        )
        self.assertEqual(payload["stopSequences"], ["STOP"])

    def test_stop_list(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro",
            {"stop": ["A", "B"]},
        )
        self.assertEqual(payload["stopSequences"], ["A", "B"])

    def test_no_stop_when_not_set(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro", {}
        )
        self.assertNotIn("stopSequences", payload)

    def test_multi_medias_empty_by_default(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}], "mimo-v2.5-pro", {}
        )
        self.assertEqual(payload["multiMedias"], [])

    def test_image_parts_are_ignored(self):
        payload = self.client._build_payload(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ],
            "mimo-v2.5-pro",
            {},
        )
        self.assertEqual(payload["query"], "describe this")
        self.assertEqual(payload["multiMedias"], [])

if __name__ == "__main__":
    unittest.main()
