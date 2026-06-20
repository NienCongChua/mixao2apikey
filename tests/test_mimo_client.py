import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.mimo_client import MimoWebClient, _ReasoningStreamParser, MIMO_MODELS
from app.utils import parse_curl_command, parse_xml_tool_call, strip_xml_tool_calls


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

    def test_parses_multiple_adjacent_mimo_xml_tool_calls(self):
        tool_calls = parse_xml_tool_call(
            '<tool_call>\n'
            '<function=bash>\n'
            '<parameter=command>ls -la /tmp</parameter>\n'
            '<parameter=description>List temp</parameter>\n'
            '</function>\n'
            '</tool_call>'
            '<tool_call>\n'
            '<function=bash>\n'
            '<parameter=command>find /tmp -maxdepth 1 -type f | head</parameter>\n'
            '<parameter=description>Find temp files</parameter>\n'
            '</function>\n'
            '</tool_call>'
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"command": "ls -la /tmp", "description": "List temp"},
        )
        self.assertEqual(
            json.loads(tool_calls[1]["function"]["arguments"]),
            {
                "command": "find /tmp -maxdepth 1 -type f | head",
                "description": "Find temp files",
            },
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

    def test_parses_empty_direct_tool_call(self):
        tool_calls = parse_xml_tool_call(
            '<tool_call>\n'
            '<function=pwd>\n'
            '</function>\n'
            '</tool_call>'
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(tool_calls[0]["function"]["name"], "pwd")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {},
        )

    def test_parses_direct_tool_tags(self):
        tool_calls = parse_xml_tool_call(
            'Tool:\n'
            '<webfetch>{"url": "https://httpbin.org/get", "format": "json"}</webfetch>\n'
            '<task>{"operation": {"action": "list"}}</task>\n'
            '<question>{"questions": [{"question": "Pick?", "options": [{"label": "A"}]}]}</question>'
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(
            [tool_call["function"]["name"] for tool_call in tool_calls],
            ["webfetch", "task", "question"],
        )
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"url": "https://httpbin.org/get", "format": "json"},
        )
        self.assertEqual(
            json.loads(tool_calls[1]["function"]["arguments"]),
            {"operation": {"action": "list"}},
        )

    def test_ignores_regular_content(self):
        self.assertIsNone(parse_xml_tool_call("Hello"))

    def test_strips_tool_calls_from_visible_content(self):
        content = (
            "I will inspect this.\n"
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>pwd</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "Then I will summarize."
        )

        self.assertNotIn("<tool_call>", strip_xml_tool_calls(content))
        self.assertIn("I will inspect this.", strip_xml_tool_calls(content))
        self.assertIn("Then I will summarize.", strip_xml_tool_calls(content))

    def test_strips_direct_tool_tags_from_visible_content(self):
        content = (
            "Fetch this.\n"
            '<webfetch>{"url": "https://example.com"}</webfetch>\n'
            "Done."
        )

        stripped = strip_xml_tool_calls(content)
        self.assertNotIn("<webfetch>", stripped)
        self.assertIn("Fetch this.", stripped)
        self.assertIn("Done.", stripped)


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

    async def test_stream_response_executes_tool_call_after_preamble(self):
        first_sse = b"".join([
            (
                b'event: message\n'
                b'data: {"content":"I will inspect this.\\n<tool_call>\\n'
                b'<function=bash>\\n"}\n\n'
            ),
            (
                b'event: message\n'
                b'data: {"content":"<parameter=command>printf streamed</parameter>\\n'
                b'<parameter=description>Run streamed command</parameter>\\n'
                b'</function>\\n</tool_call>"}\n\n'
            ),
            b"event: finish\ndata: {}\n\n",
        ])
        second_sse = b"".join([
            b'event: message\ndata: {"content":"Tool finished: streamed"}\n\n',
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

        stream = await self.client.chat_completion(
            [{"role": "user", "content": "Hi"}],
            model="mimo-v2-flash",
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

        contents = [
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
            if chunk["choices"][0]["delta"].get("content")
        ]
        joined = "".join(contents)
        self.assertIn("I will inspect this.", joined)
        self.assertIn("Tool finished: streamed", joined)
        self.assertNotIn("<tool_call>", joined)
        self.assertEqual(len(seen_payloads), 2)
        self.assertIn("$ printf streamed", seen_payloads[1]["query"])
        self.assertNotIn("<tool_call>", seen_payloads[1]["query"])

    async def test_stream_response_emits_incremental_content(self):
        sse = b"".join([
            b'event: message\ndata: {"content":"abcdefghij"}\n\n',
            b'event: message\ndata: {"content":"klmnopqrst"}\n\n',
            b"event: finish\ndata: {}\n\n",
        ])

        async def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse,
            )

        await self.client._client.aclose()
        self.client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        stream = await self.client.chat_completion(
            [{"role": "user", "content": "Hi"}],
            model="mimo-v2-flash",
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

        contents = [
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
            if chunk["choices"][0]["delta"].get("content")
        ]
        self.assertGreater(len(contents), 1)
        self.assertEqual("".join(contents), "abcdefghijklmnopqrst")

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

    async def test_non_stream_response_executes_multiple_xml_tool_calls(self):
        first_sse = b"".join([
            (
                b'event: message\n'
                b'data: {"content":"<tool_call>\\n<function=bash>\\n'
                b'<parameter=command>printf first</parameter>\\n'
                b'<parameter=description>Run first command</parameter>\\n'
                b'</function>\\n</tool_call>'
                b'<tool_call>\\n<function=bash>\\n'
                b'<parameter=command>printf second</parameter>\\n'
                b'<parameter=description>Run second command</parameter>\\n'
                b'</function>\\n</tool_call>"}\n\n'
            ),
            b"event: finish\ndata: {}\n\n",
        ])
        second_sse = b"".join([
            b'event: message\ndata: {"content":"Both tools finished"}\n\n',
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

        self.assertEqual(
            result["choices"][0]["message"]["content"],
            "Both tools finished",
        )
        self.assertEqual(len(seen_payloads), 2)
        self.assertIn("$ printf first", seen_payloads[1]["query"])
        self.assertIn("$ printf second", seen_payloads[1]["query"])
        self.assertNotIn("<tool_call>", seen_payloads[1]["query"])

    async def test_execute_uppercase_bash_tool_call(self):
        result = await self.client._execute_tool_call({
            "id": "call_upper",
            "function": {
                "name": "Bash",
                "arguments": json.dumps({"command": "printf upper"}),
            },
        })

        self.assertIn("$ printf upper", result["content"])
        self.assertIn("upper", result["content"])
        self.assertNotIn("unsupported tool", result["content"])

    async def test_execute_direct_shell_tool_without_arguments(self):
        result = await self.client._execute_tool_call({
            "id": "call_pwd",
            "function": {
                "name": "pwd",
                "arguments": "{}",
            },
        })

        self.assertIn("$ pwd", result["content"])
        self.assertNotIn("unsupported tool", result["content"])
        self.assertNotIn("missing command", result["content"])

    async def test_execute_read_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("alpha\nbeta\n", encoding="utf-8")

            result = await self.client._execute_tool_call({
                "id": "call_read",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"path": str(path)}),
                },
            })

        self.assertIn("$ read", result["content"])
        self.assertIn("alpha", result["content"])
        self.assertIn("beta", result["content"])
        self.assertNotIn("unsupported tool", result["content"])

    async def test_execute_read_tool_accepts_file_path_parameter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("alpha\nbeta\n", encoding="utf-8")

            result = await self.client._execute_tool_call({
                "id": "call_read_filepath",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({
                        "filePath": str(path),
                        "startLine": 2,
                        "lineLimit": 1,
                    }),
                },
            })

        self.assertIn("$ read", result["content"])
        self.assertIn("beta", result["content"])
        self.assertNotIn("alpha", result["content"])
        self.assertNotIn("missing filePath", result["content"])

    async def test_execute_glob_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "one.py").write_text("", encoding="utf-8")
            (root / "two.txt").write_text("", encoding="utf-8")

            result = await self.client._execute_tool_call({
                "id": "call_glob",
                "function": {
                    "name": "glob",
                    "arguments": json.dumps({"pattern": str(root / "*.py")}),
                },
            })

        self.assertIn("$ glob", result["content"])
        self.assertIn("one.py", result["content"])
        self.assertNotIn("two.txt", result["content"])
        self.assertNotIn("unsupported tool", result["content"])

    async def test_execute_grep_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "one.py").write_text("alpha\nneedle\n", encoding="utf-8")
            (root / "two.py").write_text("nothing\n", encoding="utf-8")

            result = await self.client._execute_tool_call({
                "id": "call_grep",
                "function": {
                    "name": "grep",
                    "arguments": json.dumps({
                        "pattern": "needle",
                        "path": str(root),
                    }),
                },
            })

        self.assertIn("$ grep needle", result["content"])
        self.assertIn("one.py", result["content"])
        self.assertIn("needle", result["content"])
        self.assertNotIn("two.py", result["content"])
        self.assertNotIn("unsupported tool", result["content"])

    async def test_execute_memory_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = str(Path(temp_dir) / "memory.json")
            with patch("app.mimo_client.MEMORY_PATH", memory_path):
                save_result = await self.client._execute_tool_call({
                    "id": "call_memory_save",
                    "function": {
                        "name": "memory",
                        "arguments": json.dumps({
                            "action": "remember",
                            "key": "project",
                            "content": "tool calling is enabled",
                        }),
                    },
                })
                recall_result = await self.client._execute_tool_call({
                    "id": "call_memory_recall",
                    "function": {
                        "name": "memory",
                        "arguments": json.dumps({
                            "action": "recall",
                            "query": "tool calling",
                        }),
                    },
                })

        self.assertIn("Saved memory project", save_result["content"])
        self.assertIn("project: tool calling is enabled", recall_result["content"])
        self.assertNotIn("unsupported tool", recall_result["content"])

    async def test_execute_webfetch_tool(self):
        class FakeResponse:
            status_code = 200
            headers = {"content-type": "application/json"}
            text = '{"ok": true}'

            def json(self):
                return {"ok": True}

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def request(self, method, url, **kwargs):
                self.method = method
                self.url = url
                return FakeResponse()

        with patch("app.mimo_client.httpx.AsyncClient", FakeAsyncClient):
            result = await self.client._execute_tool_call({
                "id": "call_webfetch",
                "function": {
                    "name": "webfetch",
                    "arguments": json.dumps({
                        "url": "https://example.com/api",
                        "format": "json",
                    }),
                },
            })

        self.assertIn("$ webfetch https://example.com/api", result["content"])
        self.assertIn("HTTP 200", result["content"])
        self.assertIn('"ok": true', result["content"].lower())
        self.assertNotIn("unsupported tool", result["content"])

    async def test_execute_write_and_edit_tools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "note.txt"
            write_result = await self.client._execute_tool_call({
                "id": "call_write",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({
                        "path": str(path),
                        "content": "alpha beta",
                    }),
                },
            })
            edit_result = await self.client._execute_tool_call({
                "id": "call_edit",
                "function": {
                    "name": "edit",
                    "arguments": json.dumps({
                        "path": str(path),
                        "old_string": "beta",
                        "new_string": "gamma",
                    }),
                },
            })
            final_content = path.read_text(encoding="utf-8")

        self.assertIn("wrote", write_result["content"])
        self.assertIn("replaced 1 occurrence", edit_result["content"])
        self.assertEqual(final_content, "alpha gamma")

    async def test_execute_capability_stub_tools(self):
        for name in ("task", "question", "actor", "workflow", "skill", "history"):
            result = await self.client._execute_tool_call({
                "id": f"call_{name}",
                "function": {
                    "name": name,
                    "arguments": json.dumps({"action": "list"}),
                },
            })
            self.assertIn(f"$ {name}", result["content"])
            self.assertNotIn("unsupported tool", result["content"])

    async def test_execute_task_tool_uses_operation_object(self):
        result = await self.client._execute_tool_call({
            "id": "call_task",
            "function": {
                "name": "task",
                "arguments": json.dumps({
                    "operation": {
                        "action": "create",
                        "title": "Fix tool calls",
                    }
                }),
            },
        })

        self.assertIn("$ task create", result["content"])
        self.assertIn('"action": "create"', result["content"])
        self.assertIn('"title": "Fix tool calls"', result["content"])
        self.assertNotIn("unsupported tool", result["content"])


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

    def test_tool_instructions_are_added_when_tools_are_provided(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "commit and push"}],
            "mimo-v2.5-pro",
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run a shell command",
                        },
                    }
                ],
                "parallel_tool_calls": True,
            },
        )

        self.assertIn("System tool-calling instructions:", payload["query"])
        self.assertIn("<tool_call>", payload["query"])
        self.assertIn("<function=bash>", payload["query"])
        self.assertIn("git status --short", payload["query"])
        self.assertIn("commit and push", payload["query"])

    def test_tool_instructions_are_not_added_without_tools(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "hi"}],
            "mimo-v2.5-pro",
            {},
        )

        self.assertEqual(payload["query"], "hi")

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
