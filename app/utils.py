"""
utils.py — Các hàm helper cho MiMo Web2API
"""

import json
import shlex
import time
import uuid
import logging
import re
from html import unescape
from typing import Optional

logger = logging.getLogger(__name__)

_TOOL_CALL_FUNCTION_RE = re.compile(
    r"<tool_?call>\s*<function\s*=\s*([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</function>\s*</tool_?call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_NAMED_RE = re.compile(
    r"<tool_?call>\s*<name>\s*([^<]+?)\s*</name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</tool_?call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_JSON_RE = re.compile(
    r"<tool_?call>\s*(\{.*?\})\s*</tool_?call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z0-9_.:-]+)>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_ARGUMENT_TAG_RE = re.compile(
    r"<([A-Za-z0-9_.:-]+)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_TAG_RE = re.compile(
    r"</?tool_?call\s*>",
    re.IGNORECASE,
)


def parse_xml_tool_call(content: str) -> Optional[list[dict]]:
    """
    Convert MiMo's XML-like tool-call transcript into OpenAI tool_calls.

    MiMo may emit:
    <tool_call>
    <function=bash>
    <parameter=command>whoami</parameter>
    </function>
    </tool_call>
    """
    if not content:
        return None

    parsed: list[tuple[int, tuple[int, int], dict]] = []

    for match in _TOOL_CALL_FUNCTION_RE.finditer(content):
        function_name, body = match.groups()
        params = _parse_tool_parameters(body, _TOOL_PARAMETER_RE)
        if params is None and not body.strip():
            params = {}
        if params is not None:
            parsed.append((
                match.start(),
                match.span(),
                _build_tool_call(function_name, params),
            ))

    for match in _TOOL_CALL_NAMED_RE.finditer(content):
        if _overlaps(match.span(), [span for _, span, _ in parsed]):
            continue
        function_name, body = match.groups()
        params = _parse_tool_parameters(body, _TOOL_ARGUMENT_TAG_RE)
        if params is None and not body.strip():
            params = {}
        if params is not None:
            parsed.append((
                match.start(),
                match.span(),
                _build_tool_call(function_name.strip(), params),
            ))

    for match in _TOOL_CALL_JSON_RE.finditer(content):
        if _overlaps(match.span(), [span for _, span, _ in parsed]):
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        function_name = payload.get("name")
        arguments = payload.get("arguments") or {}
        function_payload = payload.get("function")
        if isinstance(function_payload, dict):
            function_name = function_name or function_payload.get("name")
            arguments = function_payload.get("arguments", arguments)
        elif function_payload:
            function_name = function_name or function_payload
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not function_name or not isinstance(arguments, dict):
            continue
        parsed.append((
            match.start(),
            match.span(),
            _build_tool_call(str(function_name), {
                str(key): str(value)
                for key, value in arguments.items()
            }),
        ))

    if not parsed:
        return None

    parsed.sort(key=lambda item: item[0])
    return [tool_call for _, _, tool_call in parsed]


def strip_xml_tool_calls(content: str) -> str:
    """Remove complete tool-call blocks from assistant text."""
    if not content:
        return ""

    spans = _outer_tool_call_spans(content)
    if not spans:
        return content

    parts = []
    previous = 0
    for start, end in spans:
        parts.append(content[previous:start])
        previous = end
    parts.append(content[previous:])

    stripped = "".join(parts)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _outer_tool_call_spans(content: str) -> list[tuple[int, int]]:
    spans = []
    stack: list[re.Match] = []
    for match in _TOOL_CALL_TAG_RE.finditer(content):
        is_closing = match.group(0).startswith("</")
        if not is_closing:
            stack.append(match)
        elif stack:
            opening = stack.pop()
            if not stack:
                spans.append((opening.start(), match.end()))
    return spans


def _overlaps(span: tuple[int, int], existing: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in existing)


def _parse_tool_parameters(body: str, pattern: re.Pattern) -> Optional[dict[str, str]]:
    params: dict[str, str] = {}
    consumed_spans = []
    for param_match in pattern.finditer(body):
        key, value = param_match.groups()
        params[key] = unescape(value.strip())
        consumed_spans.append(param_match.span())

    if not params:
        return None

    remainder_parts = []
    previous = 0
    for start, end in consumed_spans:
        remainder_parts.append(body[previous:start])
        previous = end
    remainder_parts.append(body[previous:])
    if "".join(remainder_parts).strip():
        return None

    return params


def _build_tool_call(function_name: str, params: dict[str, str]) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(params, ensure_ascii=False),
        },
    }


def parse_stream_chunk(chunk: dict, model: str) -> dict:
    """
    Parse một chunk từ web API stream về format OpenAI SSE chunk.
    
    Web API MiMo thường trả về:
    {
        "content": "...",
        "reasoning": "...",   (optional)
        "finish_reason": null | "stop"
    }
    """
    delta = {"role": "assistant"}
    content = chunk.get("content") or chunk.get("text") or chunk.get("delta", "")
    reasoning = chunk.get("reasoning") or chunk.get("thinking") or None
    finish_reason = chunk.get("finish_reason")

    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    }


def build_non_stream_response(
    content: str,
    model: str,
    reasoning: Optional[str] = None,
    finish_reason: str = "stop",
    usage: Optional[dict] = None,
) -> dict:
    """
    Build response cho non-streaming request theo format OpenAI.
    """
    tool_calls = parse_xml_tool_call(content)
    if tool_calls:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        finish_reason = "tool_calls"
    else:
        message = {
            "role": "assistant",
            "content": content,
        }
        if reasoning:
            message["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    }


def build_error_response(message: str, code: str = "server_error") -> str:
    """
    Build error SSE message cho stream response.
    """
    error_chunk = {
        "error": {
            "message": message,
            "type": code,
        }
    }
    return f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"


def serialize_sse(data: dict) -> str:
    """
    Serialize dict thành SSE data format.
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def serialize_done() -> str:
    """SSE done signal."""
    return "data: [DONE]\n\n"


def parse_cookies_from_string(cookie_str: str) -> dict:
    """
    Parse cookie string từ browser thành dict.
    
    Ví dụ:
    "session=abc123; token=xyz; domain=.example.com"
    → {"session": "abc123", "token": "xyz"}
    """
    cookies = {}
    if not cookie_str:
        return cookies
    
    for item in cookie_str.split(";"):
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            cookies[key.strip()] = value.strip()
    
    return cookies


def parse_curl_command(curl_cmd: str) -> dict:
    """
    Parse a browser "Copy as cURL" command.

    Chromium normally exports cookies with ``-b``/``--cookie``. Supporting
    only a literal ``Cookie:`` header silently drops the login token.
    """
    result = {
        "url": "",
        "cookies": "",
        "user_agent": "",
        "headers": {},
        "body": None,
    }
    if not curl_cmd.strip():
        return result

    normalized = curl_cmd.replace("\\\r\n", " ").replace("\\\n", " ")
    try:
        tokens = shlex.split(normalized, posix=True)
    except ValueError:
        logger.warning("Could not parse cURL command")
        return result

    cookie_arg = ""
    body_arg = None
    i = 1 if tokens and tokens[0].lower() == "curl" else 0

    while i < len(tokens):
        token = tokens[i]

        if token in ("-H", "--header") and i + 1 < len(tokens):
            header = tokens[i + 1]
            i += 2
            if ":" in header:
                key, value = header.split(":", 1)
                result["headers"][key.strip()] = value.strip()
            continue

        if token.startswith("--header="):
            header = token.split("=", 1)[1]
            if ":" in header:
                key, value = header.split(":", 1)
                result["headers"][key.strip()] = value.strip()
            i += 1
            continue

        if token in ("-b", "--cookie") and i + 1 < len(tokens):
            cookie_arg = tokens[i + 1]
            i += 2
            continue

        if token.startswith("--cookie="):
            cookie_arg = token.split("=", 1)[1]
            i += 1
            continue

        if token in ("-A", "--user-agent") and i + 1 < len(tokens):
            result["user_agent"] = tokens[i + 1]
            i += 2
            continue

        if token.startswith("--user-agent="):
            result["user_agent"] = token.split("=", 1)[1]
            i += 1
            continue

        if token in ("--data", "--data-raw", "--data-binary", "-d") and i + 1 < len(tokens):
            body_arg = tokens[i + 1]
            i += 2
            continue

        if token.startswith(("--data=", "--data-raw=", "--data-binary=")):
            body_arg = token.split("=", 1)[1]
            i += 1
            continue

        if token == "--url" and i + 1 < len(tokens):
            result["url"] = tokens[i + 1]
            i += 2
            continue

        if token.startswith(("http://", "https://")) and not result["url"]:
            result["url"] = token

        i += 1

    lower_headers = {key.lower(): value for key, value in result["headers"].items()}
    result["cookies"] = lower_headers.get("cookie", "") or cookie_arg
    result["user_agent"] = (
        result["user_agent"] or lower_headers.get("user-agent", "")
    )

    if body_arg is not None:
        try:
            result["body"] = json.loads(body_arg)
        except json.JSONDecodeError:
            result["body"] = body_arg

    return result
