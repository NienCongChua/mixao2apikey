import asyncio
import json
import logging
import threading
import time
import uuid
from typing import AsyncGenerator, AsyncIterator, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from app.config import load_credentials, save_credentials
from app.usage import usage_tracker
from app.utils import (
    build_non_stream_response,
    parse_cookies_from_string,
    parse_stream_chunk,
    parse_xml_tool_call,
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

    @classmethod
    def _messages_to_query(cls, messages: list[dict]) -> str:
        """
        MiMo accepts one query and stores history server-side. OpenAI callers
        send their full history, so serialize it into a fresh, stateless MiMo
        conversation for each request.
        """
        rendered = []
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

        if len(rendered) == 1 and messages[-1].get("role") == "user":
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
            "query": self._messages_to_query(messages),
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

    async def _execute_tool_call(self, tool_call: dict) -> dict:
        name = tool_call.get("function", {}).get("name", "")
        args = self._tool_call_arguments(tool_call)
        command = str(args.get("command", "")).strip()
        description = str(args.get("description", "")).strip()

        lines = []
        if description:
            lines.append(f"# {description}")
        if command:
            lines.append(f"$ {command}")

        if name != "bash":
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

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": "\n".join(lines),
        }

    async def _execute_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        return [await self._execute_tool_call(tool_call) for tool_call in tool_calls]

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
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
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
            content_buffer = ""
            content_emitted = False
            tool_prefixes = ("<tool_call", "<toolcall")

            def visible_content_chunks(part: str) -> list[str]:
                nonlocal content_buffer, content_emitted
                if content_emitted:
                    return [part] if part else []

                content_buffer += part
                stripped = content_buffer.lstrip()
                if not stripped:
                    return []

                looks_like_tool_prefix = any(
                    prefix.startswith(stripped) or stripped.startswith(prefix)
                    for prefix in tool_prefixes
                )
                if looks_like_tool_prefix:
                    return []

                content_emitted = True
                output = content_buffer
                content_buffer = ""
                return [output] if output else []

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
                                for chunk in visible_content_chunks(part):
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
                    for chunk in visible_content_chunks(part):
                        yield parse_stream_chunk({"content": chunk}, model)

            content = "".join(content_parts)
            tool_calls = parse_xml_tool_call(content)

            await asyncio.to_thread(usage_tracker.record, self.name, model, usage, True)

            if tool_calls and not content_emitted and tool_round < MAX_AUTO_TOOL_ROUNDS:
                tool_results = await self._execute_tool_calls(tool_calls)
                followup_messages = [
                    *messages,
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
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

            if tool_calls and not content_emitted:
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
            elif not content_emitted and content_buffer:
                yield parse_stream_chunk({"content": content_buffer}, model)
                finish_reason = "stop"
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
