import asyncio
import glob as globlib
import json
import logging
import re
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from app.config import MEMORY_PATH, load_credentials, save_credentials
from app.usage import usage_tracker
from app.utils import (
    build_non_stream_response,
    parse_cookies_from_string,
    parse_stream_chunk,
    parse_xml_tool_call,
    strip_xml_tool_calls,
)


logger = logging.getLogger(__name__)

# Chat-capable model IDs currently exposed by MiMo Studio.
MIMO_MODELS = {
    "mimo-v2.5-pro": "mimo-v2.5-pro",
    "mimo-v2.5": "mimo-v2.5",
    "mimo-v2.1-pro": "mimo-v2.1-pro",
    "mimo-v2-pro": "mimo-v2-pro",
    "mimo-v2-flash": "mimo-v2-flash",
}

THINK_START = "<think>\x00"
THINK_END = "</think>\x00"
MAX_AUTO_TOOL_ROUNDS = 3
TOOL_TIMEOUT_SECONDS = 120
TOOL_OUTPUT_MAX_CHARS = 20000
SHELL_TOOL_NAMES = {"bash", "sh", "shell", "terminal", "run_command"}
DIRECT_SHELL_TOOLS = {"whoami", "date", "pwd", "ls", "uname", "git"}
TIME_TOOL_NAMES = {"time", "get_time", "get_current_time", "current_time"}
READ_TOOL_NAMES = {"read", "read_file", "file_read", "cat"}
GLOB_TOOL_NAMES = {"glob", "glob_files", "find_files"}
GREP_TOOL_NAMES = {"grep", "search", "search_files"}
MEMORY_TOOL_NAMES = {"memory", "remember", "recall"}
WEBFETCH_TOOL_NAMES = {"webfetch", "web_fetch"}
WRITE_TOOL_NAMES = {"write", "write_file"}
EDIT_TOOL_NAMES = {"edit", "edit_file"}
HISTORY_TOOL_NAMES = {"history"}
QUESTION_TOOL_NAMES = {"question"}
TASK_TOOL_NAMES = {"task"}
CAPABILITY_STUB_TOOL_NAMES = {"actor", "workflow", "skill"}
DIRECT_TAG_TOOL_NAMES = sorted(
    SHELL_TOOL_NAMES
    | DIRECT_SHELL_TOOLS
    | TIME_TOOL_NAMES
    | READ_TOOL_NAMES
    | GLOB_TOOL_NAMES
    | GREP_TOOL_NAMES
    | MEMORY_TOOL_NAMES
    | WEBFETCH_TOOL_NAMES
    | WRITE_TOOL_NAMES
    | EDIT_TOOL_NAMES
    | HISTORY_TOOL_NAMES
    | QUESTION_TOOL_NAMES
    | TASK_TOOL_NAMES
    | CAPABILITY_STUB_TOOL_NAMES
)
DIRECT_TAG_TOOL_PATTERN = "|".join(re.escape(name) for name in DIRECT_TAG_TOOL_NAMES)
SEARCH_EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
}
SEARCH_MAX_FILES = 5000


class _ReasoningStreamParser:
    """Split MiMo thinking markers even when a marker spans HTTP chunks."""

    def __init__(self):
        self.buffer = ""
        self.in_reasoning = False

    def feed(self, text: str, final: bool = False) -> list[tuple[str, bool]]:
        self.buffer += text
        output: list[tuple[str, bool]] = []

        while self.buffer:
            marker = THINK_END if self.in_reasoning else THINK_START
            marker_index = self.buffer.find(marker)

            if marker_index >= 0:
                if marker_index:
                    output.append((self.buffer[:marker_index], self.in_reasoning))
                self.buffer = self.buffer[marker_index + len(marker):]
                self.in_reasoning = not self.in_reasoning
                continue

            if final:
                output.append((self.buffer.replace("\x00", ""), self.in_reasoning))
                self.buffer = ""
                break

            # Retain enough trailing characters to recognize a split marker.
            keep = min(len(self.buffer), len(marker) - 1)
            emit_length = len(self.buffer) - keep
            if emit_length:
                output.append((
                    self.buffer[:emit_length].replace("\x00", ""),
                    self.in_reasoning,
                ))
                self.buffer = self.buffer[emit_length:]
            break

        return [(part, reasoning) for part, reasoning in output if part]


_TOOL_CALL_TAG_RE = re.compile(
    rf"</?(?:tool_?call|{DIRECT_TAG_TOOL_PATTERN})\b[^>]*>",
    re.IGNORECASE,
)
_TOOL_CALL_START_RE = re.compile(
    rf"<(?:tool_?call|{DIRECT_TAG_TOOL_PATTERN})\b[^>]*>",
    re.IGNORECASE,
)
_TOOL_CALL_PREFIXES = tuple(
    ["<tool_call", "<toolcall"] + [f"<{name}" for name in DIRECT_TAG_TOOL_NAMES]
)
_TOOL_CALL_TAG_TAIL_LENGTH = max(
    [len("</tool_call>")] + [len(f"</{name}>") for name in DIRECT_TAG_TOOL_NAMES]
) - 1


class _ToolCallStreamFilter:
    """Hide XML-like tool-call blocks while preserving normal streaming text."""

    def __init__(self):
        self.buffer = ""
        self.depth = 0
        self.emitted = False

    def feed(self, text: str, final: bool = False) -> list[str]:
        self.buffer += text
        output: list[str] = []

        while self.buffer:
            if self.depth:
                match = _TOOL_CALL_TAG_RE.search(self.buffer)
                if not match:
                    if final:
                        self.buffer = ""
                    else:
                        keep = min(len(self.buffer), _TOOL_CALL_TAG_TAIL_LENGTH)
                        self.buffer = self.buffer[-keep:]
                    break

                if match.group(0).startswith("</"):
                    self.depth = max(0, self.depth - 1)
                else:
                    self.depth += 1
                self.buffer = self.buffer[match.end():]
                continue

            start_match = _TOOL_CALL_START_RE.search(self.buffer)
            if start_match:
                self._emit(output, self.buffer[:start_match.start()])
                self.depth = 1
                self.buffer = self.buffer[start_match.end():]
                continue

            if final:
                self._emit(output, self.buffer)
                self.buffer = ""
                break

            keep = self._tool_prefix_suffix_length(self.buffer)
            emit_length = len(self.buffer) - keep
            if emit_length:
                self._emit(output, self.buffer[:emit_length])
                self.buffer = self.buffer[emit_length:]
            break

        return output

    def _emit(self, output: list[str], text: str) -> None:
        if text:
            output.append(text)
            self.emitted = True

    @staticmethod
    def _tool_prefix_suffix_length(text: str) -> int:
        lowered = text.lower()
        max_length = 0
        for prefix in _TOOL_CALL_PREFIXES:
            limit = min(len(prefix), len(lowered))
            for length in range(1, limit + 1):
                if lowered.endswith(prefix[:length]):
                    max_length = max(max_length, length)
        return max_length


class MimoWebClient:
    """
    Client for the MiMo Studio web endpoint using a browser login cookie.
    """

    WEB_CHAT_URL = "https://aistudio.xiaomimimo.com/open-apis/bot/chat"
    def __init__(self, credentials: dict):
        self.name = credentials.get("name", "default")
        self.cookie_string = credentials.get("cookies", "")
        self.credential_index: Optional[int] = None
        self.cookies = parse_cookies_from_string(credentials.get("cookies", ""))
        self.chat_params = self._build_chat_params(credentials)
        self.user_agent = credentials.get(
            "user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        self._client = self._build_client()

    @staticmethod
    def _unquote_cookie_value(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            return value[1:-1]
        return value

    def _build_chat_params(self, credentials: dict) -> dict:
        """
        MiMo's frontend appends ``xiaomichatbot_ph`` to every POST request.
        Firefox exposes it in both the copied URL and the Cookie header.
        """
        chat_url = credentials.get("chat_url") or credentials.get("url") or ""
        query = parse_qs(urlparse(chat_url).query)
        fingerprint = (
            credentials.get("xiaomichatbot_ph")
            or (query.get("xiaomichatbot_ph") or [""])[0]
            or self.cookies.get("xiaomichatbot_ph", "")
        )
        fingerprint = self._unquote_cookie_value(str(fingerprint))
        return {"xiaomichatbot_ph": fingerprint} if fingerprint else {}

    def _build_client(self) -> httpx.AsyncClient:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "*/*",
            "Accept-Language": (
                "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,zh;q=0.5"
            ),
            "Content-Type": "application/json",
            "Origin": "https://aistudio.xiaomimimo.com",
            "Referer": "https://aistudio.xiaomimimo.com/",
            "x-timeZone": "Asia/Ho_Chi_Minh",
        }
        return httpx.AsyncClient(
            headers=headers,
            cookies=self.cookies,
            timeout=httpx.Timeout(120.0, connect=30.0),
            follow_redirects=True,
        )

    @staticmethod
    def _content_to_text(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") in (
                    "text",
                    "input_text",
                ):
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part)
        return str(content)

    @staticmethod
    def _tool_schema_name(tool: dict) -> str:
        if not isinstance(tool, dict):
            return ""
        function = tool.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "").strip()
        return str(tool.get("name") or "").strip()

    @staticmethod
    def _tool_schema_description(tool: dict) -> str:
        if not isinstance(tool, dict):
            return ""
        function = tool.get("function")
        if isinstance(function, dict):
            return str(function.get("description") or "").strip()
        return str(tool.get("description") or "").strip()

    @classmethod
    def _format_tool_instructions(
        cls,
        tools: Optional[list[dict]],
        tool_choice: object = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> str:
        if not tools and tool_choice in (None, "none"):
            return ""

        tool_lines = []
        seen = set()
        for tool in tools or []:
            name = cls._tool_schema_name(tool)
            if not name or name in seen:
                continue
            seen.add(name)
            description = cls._tool_schema_description(tool)
            if description:
                tool_lines.append(f"- {name}: {description}")
            else:
                tool_lines.append(f"- {name}")

        fallback_tools = [
            "bash",
            "read",
            "glob",
            "grep",
            "memory",
            "webfetch",
            "write",
            "edit",
            "history",
            "task",
            "question",
            "actor",
            "workflow",
            "whoami",
            "date",
            "pwd",
            "ls",
            "uname",
            "git",
        ]
        for name in fallback_tools:
            if name not in seen:
                tool_lines.append(f"- {name}")
                seen.add(name)

        parallel_text = (
            "You may emit multiple adjacent tool_call blocks when independent "
            "tool calls can run in parallel."
            if parallel_tool_calls is not False
            else "Emit only one tool_call block at a time."
        )

        return (
            "System tool-calling instructions:\n"
            "When you need external information, filesystem access, shell "
            "commands, git, time, or environment inspection, do not write the "
            "command as plain text. Instead output tool calls only, using this "
            "exact XML-like format:\n"
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>git status --short</parameter>\n"
            "<parameter=description>Check repository status</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "For direct tools with no arguments, use an empty function body, "
            "for example:\n"
            "<tool_call>\n"
            "<function=pwd>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "For file reads and globbing, use parameters like:\n"
            "<tool_call><function=read><parameter=path>app/main.py</parameter>"
            "</function></tool_call>\n"
            "<tool_call><function=glob><parameter=pattern>**/*.py</parameter>"
            "</function></tool_call>\n\n"
            "For grep and memory, use parameters like:\n"
            "<tool_call><function=grep><parameter=pattern>TODO</parameter>"
            "<parameter=path>app</parameter></function></tool_call>\n"
            "<tool_call><function=memory><parameter=action>remember</parameter>"
            "<parameter=content>Important project fact</parameter></function>"
            "</tool_call>\n\n"
            "Direct tags are also accepted for compatibility, for example:\n"
            "<webfetch>{\"url\":\"https://example.com\",\"format\":\"text\"}"
            "</webfetch>\n"
            "<question>{\"questions\":[{\"question\":\"Pick one\"}]}"
            "</question>\n\n"
            "Do not wrap tool calls in Markdown fences. Do not explain the "
            "tool call before emitting it. After tool results are returned, "
            "continue from the results.\n"
            f"{parallel_text}\n"
            "Available tools:\n"
            + "\n".join(tool_lines)
        )

    @classmethod
    def _messages_to_query(
        cls,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: object = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> str:
        """
        MiMo accepts one query and stores history server-side. OpenAI callers
        send their full history, so serialize it into a fresh, stateless MiMo
        conversation for each request.
        """
        rendered = []
        tool_instructions = cls._format_tool_instructions(
            tools,
            tool_choice,
            parallel_tool_calls,
        )
        if tool_instructions:
            rendered.append(f"System:\n{tool_instructions}")

        labels = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
        }
        for message in messages:
            content = cls._content_to_text(message.get("content"))
            if not content:
                continue
            label = labels.get(
                message.get("role"),
                str(message.get("role", "Message")),
            )
            rendered.append(f"{label}:\n{content}")

        if not rendered:
            raise ValueError("At least one non-empty message is required")

        if not tool_instructions and len(rendered) == 1 and messages[-1].get("role") == "user":
            return cls._content_to_text(messages[-1].get("content"))

        return "\n\n".join(rendered) + "\n\nAssistant:"

    def _build_payload(
        self,
        messages: list[dict],
        model: str,
        params: dict,
    ) -> dict:
        enable_thinking = params.get("thinking") is True or bool(
            params.get("reasoning_effort")
        )

        web_search = params.get("web_search")
        web_search_status = "enabled" if web_search else "disabled"

        model_config = {
            "enableThinking": enable_thinking,
            "webSearchStatus": web_search_status,
            "model": model,
        }
        if params.get("temperature") is not None:
            model_config["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            model_config["topP"] = params["top_p"]

        payload = {
            "msgId": str(uuid.uuid4()),
            "conversationId": str(uuid.uuid4()),
            "query": self._messages_to_query(
                messages,
                params.get("tools"),
                params.get("tool_choice"),
                params.get("parallel_tool_calls"),
            ),
            "isEditedQuery": False,
            "modelConfig": model_config,
            "multiMedias": [],
        }

        if params.get("stop"):
            stop = params["stop"]
            payload["stopSequences"] = [stop] if isinstance(stop, str) else stop

        return payload

    @staticmethod
    def _event_content(data: object) -> str:
        if isinstance(data, dict):
            value = data.get("content", "")
            return value if isinstance(value, str) else str(value)
        return str(data or "")

    @staticmethod
    async def _iter_mimo_events(
        response: httpx.Response,
    ) -> AsyncIterator[tuple[str, object]]:
        """Parse named SSE events from the MiMo response."""
        event_name = "message"
        data_lines: list[str] = []

        async for line in response.aiter_lines():
            if line == "":
                if data_lines:
                    raw_data = "\n".join(data_lines)
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        data = raw_data
                    yield event_name, data
                event_name = "message"
                data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines:
            raw_data = "\n".join(data_lines)
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                data = raw_data
            yield event_name, data

    @staticmethod
    def _normalize_usage(data: object) -> dict:
        def as_int(value) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        empty = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not isinstance(data, dict):
            return empty

        source = data.get("nativeUsage") if isinstance(data.get("nativeUsage"), dict) else data
        prompt_details = (
            source.get("prompt_tokens_details")
            or source.get("promptTokensDetails")
            or {}
        )
        if not isinstance(prompt_details, dict):
            prompt_details = {}

        prompt_tokens = as_int(
            source.get("prompt_tokens")
            or source.get("input_tokens")
            or source.get("promptTokens")
            or source.get("inputTokens")
            or 0
        )
        completion_tokens = as_int(
            source.get("completion_tokens")
            or source.get("output_tokens")
            or source.get("completionTokens")
            or source.get("outputTokens")
            or 0
        )
        total_tokens = as_int(
            source.get("total_tokens")
            or source.get("totalTokens")
            or 0
        )
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        cached_tokens = as_int(
            source.get("cache_tokens")
            or source.get("cached_tokens")
            or source.get("cacheTokens")
            or source.get("cachedTokens")
            or prompt_details.get("cached_tokens")
            or prompt_details.get("cachedTokens")
            or 0
        )

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if cached_tokens:
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
            usage["cache_tokens"] = cached_tokens
        return usage


    async def chat_completion(
        self,
        messages: list[dict],
        model: str = "mimo-v2.5-pro",
        stream: bool = False,
        **kwargs,
    ) -> dict | AsyncGenerator[dict, None]:
        payload = self._build_payload(messages, model, kwargs)
        if stream:
            return self._stream_chat(payload, messages, kwargs)
        return await self._non_stream_chat(payload, messages, kwargs)

    @staticmethod
    def _tool_call_arguments(tool_call: dict) -> dict:
        raw_args = tool_call.get("function", {}).get("arguments", "{}")
        if isinstance(raw_args, dict):
            return raw_args
        try:
            value = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _tool_arg(args: dict, *names: str) -> str:
        for name in names:
            value = args.get(name)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _tool_int_arg(args: dict, default: int, *names: str) -> int:
        text = MimoWebClient._tool_arg(args, *names)
        if not text:
            return default
        try:
            return int(text)
        except ValueError:
            return default

    @staticmethod
    def _tool_bool_arg(args: dict, default: bool, *names: str) -> bool:
        text = MimoWebClient._tool_arg(args, *names).lower()
        if not text:
            return default
        return text in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _tool_list_arg(args: dict, *names: str) -> list[str]:
        for name in names:
            value = args.get(name)
            if value is None:
                continue
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str):
                parts = [part.strip() for part in value.split(",")]
                return [part for part in parts if part]
            text = str(value).strip()
            if text:
                return [text]
        return []

    @staticmethod
    def _truncate_tool_output(content: str) -> str:
        if len(content) <= TOOL_OUTPUT_MAX_CHARS:
            return content
        omitted = len(content) - TOOL_OUTPUT_MAX_CHARS
        return (
            content[:TOOL_OUTPUT_MAX_CHARS]
            + f"\n[tool output truncated: {omitted} characters omitted]"
        )

    @classmethod
    def _build_shell_command(cls, name: str, args: dict) -> str:
        normalized = name.strip().lower()
        if normalized in TIME_TOOL_NAMES:
            return "date"

        if normalized in SHELL_TOOL_NAMES:
            return cls._tool_arg(
                args,
                "command",
                "cmd",
                "script",
                "code",
                "input",
                "query",
            )

        if normalized not in DIRECT_SHELL_TOOLS:
            return ""

        command_arg = cls._tool_arg(args, "command", "cmd")
        if command_arg:
            if command_arg.lower().startswith(normalized):
                return command_arg
            if normalized == "git":
                return f"git {command_arg}"
            return f"{normalized} {command_arg}"

        parts = [normalized]
        option_text = cls._tool_arg(args, "args", "arguments", "flags", "options")
        if option_text:
            parts.append(option_text)

        path = cls._tool_arg(args, "path", "dir", "directory", "file", "file_path")
        if path and normalized in {"ls"}:
            parts.append(shlex.quote(path))

        return " ".join(parts)

    @classmethod
    def _read_tool_content(cls, args: dict) -> str:
        file_path = cls._tool_arg(args, "path", "file", "file_path", "filename")
        if not file_path:
            return "Tool execution failed: missing path parameter."

        path = Path(file_path).expanduser()
        start_line = max(1, cls._tool_int_arg(args, 1, "start_line", "offset", "line"))
        line_limit = max(1, cls._tool_int_arg(args, 2000, "limit", "lines", "line_limit"))

        lines = [f"$ read {shlex.quote(str(path))}"]
        try:
            with path.open("r", encoding="utf-8", errors="replace") as file:
                emitted = 0
                for line_number, line in enumerate(file, start=1):
                    if line_number < start_line:
                        continue
                    if emitted >= line_limit:
                        lines.append(
                            f"[read output truncated after {line_limit} lines]"
                        )
                        break
                    lines.append(line.rstrip("\n"))
                    emitted += 1
        except OSError as exc:
            lines.append(f"Tool execution failed: {exc}")

        return cls._truncate_tool_output("\n".join(lines))

    @classmethod
    def _glob_tool_content(cls, args: dict) -> str:
        pattern = cls._tool_arg(args, "pattern", "glob", "path", "query")
        if not pattern:
            return "Tool execution failed: missing pattern parameter."

        root = cls._tool_arg(args, "cwd", "root", "directory", "dir")
        if root and not Path(pattern).is_absolute():
            pattern = str(Path(root).expanduser() / pattern)
        else:
            pattern = str(Path(pattern).expanduser())

        limit = max(1, cls._tool_int_arg(args, 200, "limit", "max_results"))
        matches = sorted(globlib.glob(pattern, recursive=True))
        visible_matches = matches[:limit]

        lines = [f"$ glob {shlex.quote(pattern)}"]
        lines.extend(visible_matches)
        if len(matches) > limit:
            lines.append(
                f"[glob output truncated: {len(matches) - limit} matches omitted]"
            )
        if not matches:
            lines.append("[no matches]")
        return cls._truncate_tool_output("\n".join(lines))

    @classmethod
    def _grep_tool_content(cls, args: dict) -> str:
        pattern = cls._tool_arg(args, "pattern", "regex", "query", "search")
        if not pattern:
            return "Tool execution failed: missing pattern parameter."

        root = cls._tool_arg(args, "cwd", "root", "directory", "dir")
        path_args = cls._tool_list_arg(args, "path", "paths", "file", "files")
        include_patterns = cls._tool_list_arg(args, "include", "glob", "file_glob")
        if not path_args:
            path_args = include_patterns or ["."]

        case_sensitive = cls._tool_bool_arg(args, True, "case_sensitive")
        if cls._tool_bool_arg(args, False, "ignore_case"):
            case_sensitive = False
        limit = max(1, cls._tool_int_arg(args, 200, "limit", "max_results"))
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"Tool execution failed: invalid regex: {exc}"

        files = cls._grep_candidate_files(path_args, root)
        lines = [f"$ grep {shlex.quote(pattern)}"]
        matches = 0
        scanned = 0
        for path in files:
            if scanned >= SEARCH_MAX_FILES or matches >= limit:
                break
            scanned += 1
            try:
                with path.open("r", encoding="utf-8", errors="replace") as file:
                    for line_number, line in enumerate(file, start=1):
                        if regex.search(line):
                            lines.append(
                                f"{path}:{line_number}:{line.rstrip()}"
                            )
                            matches += 1
                            if matches >= limit:
                                break
            except (OSError, UnicodeError):
                continue

        if not matches:
            lines.append("[no matches]")
        if matches >= limit:
            lines.append(f"[grep output truncated after {limit} matches]")
        if scanned >= SEARCH_MAX_FILES:
            lines.append(f"[grep scan truncated after {SEARCH_MAX_FILES} files]")
        return cls._truncate_tool_output("\n".join(lines))

    @classmethod
    def _grep_candidate_files(cls, path_args: list[str], root: str = "") -> list[Path]:
        root_path = Path(root).expanduser() if root else None
        files: list[Path] = []
        for raw_path in path_args:
            path = Path(raw_path).expanduser()
            if root_path and not path.is_absolute():
                path = root_path / path

            matches = globlib.glob(str(path), recursive=True)
            if matches:
                for match in matches:
                    files.extend(cls._walk_search_path(Path(match)))
            else:
                files.extend(cls._walk_search_path(path))

        unique = {}
        for path in files:
            try:
                unique[str(path.resolve())] = path
            except OSError:
                unique[str(path)] = path
        return sorted(unique.values(), key=lambda item: str(item))

    @classmethod
    def _walk_search_path(cls, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if not path.is_dir():
            return []

        files = []
        for child in path.rglob("*"):
            if any(part in SEARCH_EXCLUDED_DIRS for part in child.parts):
                continue
            if child.is_file():
                files.append(child)
        return files

    @classmethod
    def _memory_tool_content(cls, args: dict) -> str:
        action = cls._tool_arg(args, "action", "operation", "op", "mode").lower()
        key = cls._tool_arg(args, "key", "id", "name")
        content = cls._tool_arg(args, "content", "text", "value", "memory")
        query = cls._tool_arg(args, "query", "search", "pattern")

        if not action:
            if content:
                action = "remember"
            elif query or key:
                action = "search"
            else:
                action = "list"

        records = cls._load_memory_records()
        if action in {"remember", "save", "add", "set", "write"}:
            if not content:
                return "Tool execution failed: missing content parameter."
            record = {
                "id": key or f"mem_{uuid.uuid4().hex[:12]}",
                "content": content,
                "created_at": int(time.time()),
            }
            records = [item for item in records if item.get("id") != record["id"]]
            records.append(record)
            cls._save_memory_records(records)
            return f"$ memory remember\nSaved memory {record['id']}: {content}"

        if action in {"clear", "delete", "remove"}:
            if key:
                kept = [item for item in records if item.get("id") != key]
                cls._save_memory_records(kept)
                return f"$ memory delete\nDeleted {len(records) - len(kept)} record(s)."
            cls._save_memory_records([])
            return f"$ memory clear\nDeleted {len(records)} record(s)."

        limit = max(1, cls._tool_int_arg(args, 20, "limit", "max_results"))
        if action in {"search", "recall", "get", "read", "list"}:
            needle = (query or key).lower()
            if needle:
                selected = [
                    item for item in records
                    if needle in str(item.get("id", "")).lower()
                    or needle in str(item.get("content", "")).lower()
                ]
            else:
                selected = records
            selected = selected[-limit:]
            lines = [f"$ memory {action}"]
            if not selected:
                lines.append("[no memories]")
            for item in selected:
                lines.append(f"{item.get('id')}: {item.get('content', '')}")
            return cls._truncate_tool_output("\n".join(lines))

        return f"Tool execution failed: unsupported memory action '{action}'."

    @staticmethod
    def _load_memory_records() -> list[dict]:
        path = Path(MEMORY_PATH)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    @staticmethod
    def _save_memory_records(records: list[dict]) -> None:
        path = Path(MEMORY_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    async def _webfetch_tool_content(cls, args: dict) -> str:
        url = cls._tool_arg(args, "url", "uri", "href")
        if not url:
            return "Tool execution failed: missing url parameter."

        method = cls._tool_arg(args, "method") or "GET"
        method = method.upper()
        output_format = (cls._tool_arg(args, "format", "type") or "text").lower()
        headers = args.get("headers")
        if isinstance(headers, str):
            try:
                headers = json.loads(headers)
            except json.JSONDecodeError:
                headers = {}
        if not isinstance(headers, dict):
            headers = {}

        request_kwargs = {"headers": {str(k): str(v) for k, v in headers.items()}}
        if args.get("json") is not None:
            request_kwargs["json"] = args["json"]
        elif args.get("body") is not None:
            request_kwargs["content"] = str(args["body"])
        elif args.get("data") is not None:
            request_kwargs["content"] = str(args["data"])

        lines = [f"$ webfetch {url}"]
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
            ) as client:
                response = await client.request(method, url, **request_kwargs)
        except Exception as exc:
            lines.append(f"Tool execution failed: {exc}")
            return cls._truncate_tool_output("\n".join(lines))

        lines.append(f"HTTP {response.status_code}")
        content_type = response.headers.get("content-type")
        if content_type:
            lines.append(f"content-type: {content_type}")

        text = response.text
        if output_format == "json":
            try:
                text = json.dumps(response.json(), ensure_ascii=False, indent=2)
            except ValueError:
                pass
        lines.append(text)
        return cls._truncate_tool_output("\n".join(lines))

    @classmethod
    def _write_tool_content(cls, args: dict) -> str:
        file_path = cls._tool_arg(args, "path", "file", "file_path", "filename")
        content = cls._tool_arg(args, "content", "text", "value", "input")
        if not file_path:
            return "Tool execution failed: missing path parameter."
        if content == "":
            return "Tool execution failed: missing content parameter."

        path = Path(file_path).expanduser()
        append = cls._tool_bool_arg(args, False, "append")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with path.open(mode, encoding="utf-8") as file:
                file.write(content)
        except OSError as exc:
            return f"Tool execution failed: {exc}"

        action = "appended" if append else "wrote"
        return f"$ write {shlex.quote(str(path))}\n{action} {len(content)} characters"

    @classmethod
    def _edit_tool_content(cls, args: dict) -> str:
        file_path = cls._tool_arg(args, "path", "file", "file_path", "filename")
        old_text = cls._tool_arg(args, "old_string", "old", "search", "target")
        new_text = cls._tool_arg(args, "new_string", "new", "replace", "replacement")
        if not file_path:
            return "Tool execution failed: missing path parameter."
        if old_text == "":
            return "Tool execution failed: missing old_string/search parameter."

        path = Path(file_path).expanduser()
        replace_all = cls._tool_bool_arg(args, False, "replace_all", "all")
        try:
            original = path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Tool execution failed: {exc}"

        occurrences = original.count(old_text)
        if not occurrences:
            return "Tool execution failed: search text was not found."

        edited = (
            original.replace(old_text, new_text)
            if replace_all
            else original.replace(old_text, new_text, 1)
        )
        try:
            path.write_text(edited, encoding="utf-8")
        except OSError as exc:
            return f"Tool execution failed: {exc}"

        count = occurrences if replace_all else 1
        return f"$ edit {shlex.quote(str(path))}\nreplaced {count} occurrence(s)"

    @classmethod
    def _history_tool_content(cls, args: dict) -> str:
        query = cls._tool_arg(args, "query", "search", "pattern")
        lines = ["$ history"]
        if query:
            lines.append(f"query: {query}")
        lines.append(
            "Persistent chat history search is not configured in this API process."
        )
        return "\n".join(lines)

    @classmethod
    def _task_tool_content(cls, args: dict) -> str:
        operation = args.get("operation")
        if isinstance(operation, dict):
            action = str(operation.get("action") or "list").lower()
        else:
            action = cls._tool_arg(args, "action", "operation") or "list"
            action = action.lower()

        lines = [f"$ task {action}"]
        if action == "list":
            lines.append("[]")
        else:
            lines.append(
                "Task tool is recognized, but no local task backend is configured."
            )
        return "\n".join(lines)

    @classmethod
    def _question_tool_content(cls, args: dict) -> str:
        questions = args.get("questions")
        if not isinstance(questions, list):
            question = cls._tool_arg(args, "question", "prompt", "input")
            questions = [{"question": question}] if question else []

        lines = ["$ question"]
        if not questions:
            lines.append("No questions were provided.")
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                lines.append(f"{index}. {item}")
                continue
            prompt = item.get("question") or item.get("prompt") or ""
            lines.append(f"{index}. {prompt}")
            options = item.get("options")
            if isinstance(options, list) and options:
                first = options[0]
                if isinstance(first, dict):
                    label = first.get("label") or first.get("value") or ""
                else:
                    label = str(first)
                lines.append(f"default_option: {label}")
        lines.append(
            "Interactive user input is unavailable through this API; continue with a sensible default."
        )
        return cls._truncate_tool_output("\n".join(lines))

    @classmethod
    def _capability_stub_tool_content(cls, name: str, args: dict) -> str:
        action = cls._tool_arg(args, "action", "operation", "name")
        suffix = f" {action}" if action else ""
        return (
            f"$ {name}{suffix}\n"
            f"{name} is recognized, but no local {name} backend is configured."
        )

    async def _execute_tool_call(self, tool_call: dict) -> dict:
        name = str(tool_call.get("function", {}).get("name", "")).strip()
        normalized_name = name.lower()
        args = self._tool_call_arguments(tool_call)
        description = str(args.get("description", "")).strip()

        lines = []
        if description:
            lines.append(f"# {description}")

        if normalized_name in READ_TOOL_NAMES:
            lines.append(await asyncio.to_thread(self._read_tool_content, args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in GLOB_TOOL_NAMES:
            lines.append(await asyncio.to_thread(self._glob_tool_content, args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in GREP_TOOL_NAMES:
            lines.append(await asyncio.to_thread(self._grep_tool_content, args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in MEMORY_TOOL_NAMES:
            lines.append(await asyncio.to_thread(self._memory_tool_content, args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in WEBFETCH_TOOL_NAMES:
            lines.append(await self._webfetch_tool_content(args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in WRITE_TOOL_NAMES:
            lines.append(await asyncio.to_thread(self._write_tool_content, args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in EDIT_TOOL_NAMES:
            lines.append(await asyncio.to_thread(self._edit_tool_content, args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in HISTORY_TOOL_NAMES:
            lines.append(self._history_tool_content(args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in QUESTION_TOOL_NAMES:
            lines.append(self._question_tool_content(args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in TASK_TOOL_NAMES:
            lines.append(self._task_tool_content(args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        if normalized_name in CAPABILITY_STUB_TOOL_NAMES:
            lines.append(self._capability_stub_tool_content(normalized_name, args))
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": self._truncate_tool_output("\n".join(lines)),
            }

        command = self._build_shell_command(name, args)
        if command:
            lines.append(f"$ {command}")

        supported_shell = (
            normalized_name in SHELL_TOOL_NAMES
            or normalized_name in DIRECT_SHELL_TOOLS
            or normalized_name in TIME_TOOL_NAMES
        )
        if not supported_shell:
            lines.append(f"Tool execution failed: unsupported tool '{name}'.")
        elif not command:
            lines.append("Tool execution failed: missing command parameter.")
        else:
            logger.info("Executing bash tool call: %s", command)
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=TOOL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                lines.append(
                    f"Tool execution timed out after {TOOL_TIMEOUT_SECONDS} seconds."
                )
            else:
                stdout_text = stdout.decode(errors="replace").rstrip()
                stderr_text = stderr.decode(errors="replace").rstrip()
                if stdout_text:
                    lines.append(stdout_text)
                if stderr_text:
                    lines.append(stderr_text)
                if process.returncode:
                    lines.append(f"[exit code {process.returncode}]")

        content = self._truncate_tool_output("\n".join(lines))
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": content,
        }

    async def _execute_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        return await asyncio.gather(
            *(self._execute_tool_call(tool_call) for tool_call in tool_calls)
        )

    @staticmethod
    def _assistant_tool_message(content: str, tool_calls: list[dict]) -> dict:
        visible_content = strip_xml_tool_calls(content)
        return {
            "role": "assistant",
            "content": visible_content or None,
            "tool_calls": tool_calls,
        }

    async def _non_stream_chat(
        self,
        payload: dict,
        messages: list[dict],
        params: dict,
        tool_round: int = 0,
    ) -> dict:
        """Consume MiMo's stream and aggregate it into one OpenAI response."""
        try:
            content_parts = []
            reasoning_parts = []
            usage = None
            parser = _ReasoningStreamParser()

            async with self._client.stream(
                "POST",
                self.WEB_CHAT_URL,
                params=self.chat_params,
                json=payload,
            ) as response:
                if response.is_error:
                    await response.aread()
                response.raise_for_status()
                async for event, data in self._iter_mimo_events(response):
                    if event == "message":
                        for part, is_reasoning in parser.feed(
                            self._event_content(data)
                        ):
                            target = reasoning_parts if is_reasoning else content_parts
                            target.append(part)
                    elif event == "usage":
                        usage = self._normalize_usage(data)
                    elif event in ("error", "sensitive_query"):
                        raise RuntimeError(self._event_content(data) or event)
                    elif event == "finish":
                        break

            for part, is_reasoning in parser.feed("", final=True):
                target = reasoning_parts if is_reasoning else content_parts
                target.append(part)

            await asyncio.to_thread(
                usage_tracker.record,
                self.name,
                payload["modelConfig"]["model"],
                usage,
                False,
            )

            content = "".join(content_parts)
            tool_calls = parse_xml_tool_call(content)
            if tool_calls and tool_round < MAX_AUTO_TOOL_ROUNDS:
                tool_results = await self._execute_tool_calls(tool_calls)
                followup_messages = [
                    *messages,
                    self._assistant_tool_message(content, tool_calls),
                    *tool_results,
                ]
                followup_payload = self._build_payload(
                    followup_messages,
                    payload["modelConfig"]["model"],
                    params,
                )
                return await self._non_stream_chat(
                    followup_payload,
                    followup_messages,
                    params,
                    tool_round + 1,
                )

            return build_non_stream_response(
                content,
                payload["modelConfig"]["model"],
                "".join(reasoning_parts) or None,
                usage=usage,
            )

        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP error: %s - %s",
                exc.response.status_code,
                exc.response.text,
            )
            raise
        except Exception as exc:
            logger.error("Request failed: %s", exc)
            raise

    async def _stream_chat(
        self,
        payload: dict,
        messages: list[dict],
        params: dict,
        tool_round: int = 0,
    ) -> AsyncGenerator[dict, None]:
        """Convert MiMo's named SSE stream to OpenAI chat chunks."""
        try:
            parser = _ReasoningStreamParser()
            model = payload["modelConfig"]["model"]
            usage = None
            content_parts = []
            reasoning_parts = []
            visible_filter = _ToolCallStreamFilter()

            async with self._client.stream(
                "POST",
                self.WEB_CHAT_URL,
                params=self.chat_params,
                json=payload,
            ) as response:
                if response.is_error:
                    await response.aread()
                response.raise_for_status()
                async for event, data in self._iter_mimo_events(response):
                    if event == "message":
                        for part, is_reasoning in parser.feed(
                            self._event_content(data)
                        ):
                            target = reasoning_parts if is_reasoning else content_parts
                            target.append(part)
                            if is_reasoning:
                                yield parse_stream_chunk({"reasoning": part}, model)
                            else:
                                for chunk in visible_filter.feed(part):
                                    yield parse_stream_chunk({"content": chunk}, model)
                    elif event == "usage":
                        usage = self._normalize_usage(data)
                    elif event in ("error", "sensitive_query"):
                        raise RuntimeError(self._event_content(data) or event)
                    elif event == "finish":
                        break

            for part, is_reasoning in parser.feed("", final=True):
                target = reasoning_parts if is_reasoning else content_parts
                target.append(part)
                if is_reasoning:
                    yield parse_stream_chunk({"reasoning": part}, model)
                else:
                    for chunk in visible_filter.feed(part):
                        yield parse_stream_chunk({"content": chunk}, model)

            for chunk in visible_filter.feed("", final=True):
                yield parse_stream_chunk({"content": chunk}, model)

            content = "".join(content_parts)
            tool_calls = parse_xml_tool_call(content)

            await asyncio.to_thread(usage_tracker.record, self.name, model, usage, True)

            if tool_calls and tool_round < MAX_AUTO_TOOL_ROUNDS:
                tool_results = await self._execute_tool_calls(tool_calls)
                followup_messages = [
                    *messages,
                    self._assistant_tool_message(content, tool_calls),
                    *tool_results,
                ]
                followup_payload = self._build_payload(
                    followup_messages,
                    model,
                    params,
                )
                async for chunk in self._stream_chat(
                    followup_payload,
                    followup_messages,
                    params,
                    tool_round + 1,
                ):
                    yield chunk
                return

            if tool_calls:
                stream_tool_calls = []
                for index, tool_call in enumerate(tool_calls):
                    stream_tool_calls.append({"index": index, **tool_call})
                yield {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": stream_tool_calls,
                        },
                        "finish_reason": None,
                    }],
                }
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"

            yield {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
            }

        except Exception as exc:
            logger.error("Stream error: %s", exc)
            raise

    async def check_health(self) -> tuple[bool, str]:
        try:
            await self.chat_completion(
                messages=[{"role": "user", "content": "Reply OK"}],
                model="mimo-v2-flash",
            )
            return True, ""
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                return False, "Cookie hết hạn hoặc rỗng (upstream trả 401)"
            return False, (
                f"MiMo upstream trả HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            )
        except Exception as exc:
            return False, str(exc)

    async def close(self):
        await self._client.aclose()


class MimoClientManager:
    """Manage configured accounts with simple round-robin selection."""

    def __init__(self):
        self.clients: list[MimoWebClient] = []
        self._current_index = 0
        self._lock = threading.RLock()
        self._load_credentials()

    def _load_credentials(self):
        creds_list = load_credentials()
        clients = []
        for index, cred in enumerate(creds_list):
            if cred.get("is_active", True) and cred.get("cookies"):
                try:
                    client = MimoWebClient(cred)
                    client.credential_index = index
                    clients.append(client)
                except Exception as exc:
                    logger.warning(
                        "Failed to load credential '%s': %s",
                        cred.get("name"),
                        exc,
                    )
        with self._lock:
            self.clients = clients
            if self.clients:
                self._current_index %= len(self.clients)
            else:
                self._current_index = 0
        logger.info("Loaded %s MiMo web clients", len(self.clients))

    def reload(self):
        for client in self.clients:
            try:
                asyncio.create_task(client.close())
            except RuntimeError:
                pass
        self._load_credentials()

    def get_client(self) -> Optional[MimoWebClient]:
        with self._lock:
            if not self.clients:
                return None
            client = self.clients[self._current_index % len(self.clients)]
            self._current_index += 1
            return client

    def get_client_attempts(self) -> list[MimoWebClient]:
        first = self.get_client()
        if first is None:
            return []

        with self._lock:
            clients = list(self.clients)

        first_index = next(
            (index for index, client in enumerate(clients) if client is first),
            None,
        )
        if first_index is None:
            return [first]

        return [
            clients[(first_index + offset) % len(clients)]
            for offset in range(len(clients))
        ]

    @staticmethod
    def is_account_error(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403, 419, 440):
                return True
            text = exc.response.text
        else:
            status = None
            text = str(exc)

        lowered = (text or "").lower()
        if "sensitive_query" in lowered:
            return False

        keywords = (
            "session",
            "cookie",
            "access token",
            "service token",
            "servicetoken",
            "unauthorized",
            "forbidden",
            "login",
            "expired",
            "expire",
            "hết hạn",
            "het han",
            "đăng nhập",
            "dang nhap",
            "未登录",
            "登录",
            "登陆",
            "过期",
        )
        return any(keyword in lowered for keyword in keywords) and (
            status is None or status < 500
        )

    @staticmethod
    def describe_error(exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text.strip()
            if body:
                return f"HTTP {exc.response.status_code}: {body[:300]}"
            return f"HTTP {exc.response.status_code}"
        return str(exc)[:300]

    async def mark_client_failed(self, client: MimoWebClient, reason: str) -> None:
        await asyncio.to_thread(self._mark_client_failed_sync, client, reason)
        try:
            await client.close()
        except Exception:
            pass

    def _mark_client_failed_sync(self, client: MimoWebClient, reason: str) -> None:
        with self._lock:
            creds_list = load_credentials()
            target_index = getattr(client, "credential_index", None)

            if not (
                isinstance(target_index, int)
                and 0 <= target_index < len(creds_list)
            ):
                target_index = next(
                    (
                        index
                        for index, credential in enumerate(creds_list)
                        if credential.get("name", "default") == getattr(client, "name", None)
                        and credential.get("cookies", "") == getattr(client, "cookie_string", None)
                    ),
                    None,
                )

            if target_index is not None:
                creds_list[target_index]["is_active"] = False
                creds_list[target_index]["last_error"] = reason
                creds_list[target_index]["last_error_at"] = int(time.time())
                save_credentials(creds_list)

            self.clients = [item for item in self.clients if item is not client]
            if self.clients:
                self._current_index %= len(self.clients)
            else:
                self._current_index = 0

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.clients)


client_manager = MimoClientManager()
