import json
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from httpx import AsyncClient, ASGITransport

from main import app
from app.access_keys import access_key_store
from app.config import settings
from app.mimo_client import client_manager, MIMO_MODELS
from app.usage import usage_tracker


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
def reset_api_key():
    original = settings.api_key
    original_keys = list(access_key_store._keys)
    settings.api_key = None
    access_key_store._keys = []
    yield
    settings.api_key = original
    access_key_store._keys = original_keys


# ─── GET /v1/models ───


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    for model_id in MIMO_MODELS:
        assert model_id in ids


@pytest.mark.asyncio
async def test_list_models_structure(client):
    resp = await client.get("/v1/models")
    model = resp.json()["data"][0]
    assert "id" in model
    assert model["object"] == "model"
    assert "created" in model
    assert model["owned_by"] == "mimo-web2api"


# ─── Authentication ───


@pytest.mark.asyncio
async def test_auth_required_when_api_key_set(client):
    settings.api_key = "test-secret"
    resp = await client.get("/v1/models")
    assert resp.status_code == 401
    assert "Missing API key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_rejects_wrong_key(client):
    settings.api_key = "test-secret"
    resp = await client.get(
        "/v1/models",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401
    assert "Invalid API key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_accepts_correct_key(client):
    settings.api_key = "test-secret"
    resp = await client.get(
        "/v1/models",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_skipped_when_no_api_key(client):
    settings.api_key = None
    resp = await client.get("/v1/models")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_required_when_local_api_key_exists(client):
    access_key_store._keys = [
        {
            "id": "local",
            "name": "Local",
            "key": "sk-local",
            "is_active": True,
            "created_at": 0,
            "last_used_at": 0,
        }
    ]
    resp = await client.get("/v1/models")
    assert resp.status_code == 401

    with patch.object(access_key_store, "mark_used"):
        resp = await client.get(
            "/v1/models",
            headers={"Authorization": "Bearer sk-local"},
        )
    assert resp.status_code == 200


# ─── POST /v1/chat/completions ───


@pytest.mark.asyncio
async def test_chat_completions_no_accounts(client):
    with patch.object(client_manager, "get_client", return_value=None):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert resp.status_code == 503
    assert "No active" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_chat_completions_success(client):
    mock_result = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "mimo-v2.5-pro",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(return_value=mock_result)

    with patch.object(client_manager, "get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["usage"]["total_tokens"] == 2


@pytest.mark.asyncio
async def test_chat_completions_model_fallback(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {},
        }
    )

    with patch.object(client_manager, "get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert resp.status_code == 200
    call_kwargs = mock_client.chat_completion.call_args
    assert call_kwargs.kwargs["model"] == settings.default_model


@pytest.mark.asyncio
async def test_chat_completions_forward_params(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {},
        }
    )

    with patch.object(client_manager, "get_client", return_value=mock_client):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 100,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
            },
        )
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_tokens"] == 100
    assert kwargs["frequency_penalty"] == 0.1
    assert kwargs["presence_penalty"] == 0.2


@pytest.mark.asyncio
async def test_chat_completions_forward_tool_params(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {},
        }
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run shell commands",
                "parameters": {"type": "object"},
            },
        }
    ]

    with patch.object(client_manager, "get_client", return_value=mock_client):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
        )

    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True


@pytest.mark.asyncio
async def test_chat_completions_upstream_error(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(side_effect=Exception("upstream timeout"))

    with patch.object(client_manager, "get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert resp.status_code == 502
    assert "upstream timeout" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_chat_completions_disables_bad_account_and_retries(client):
    bad_client = MagicMock()
    bad_client.name = "expired"
    bad_client.chat_completion = AsyncMock(side_effect=RuntimeError("session expired"))

    good_result = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "mimo-v2.5-pro",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Recovered"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    good_client = MagicMock()
    good_client.name = "healthy"
    good_client.chat_completion = AsyncMock(return_value=good_result)

    with patch.object(client_manager, "get_client_attempts", return_value=[bad_client, good_client]), \
         patch.object(client_manager, "mark_client_failed", new=AsyncMock()) as mark_failed:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Recovered"
    mark_failed.assert_awaited_once_with(bad_client, "session expired")
    good_client.chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_completions_stream(client):
    async def fake_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"role": "assistant", "content": "Hi"}}]}
        yield {"choices": [{"delta": {"content": " there"}, "finish_reason": "stop"}]}

    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(return_value=fake_stream())

    with patch.object(client_manager, "get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    lines = resp.text.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) >= 3
    assert data_lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_chat_completions_stream_retries_before_first_chunk(client):
    async def fake_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"role": "assistant", "content": "OK"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    bad_client = MagicMock()
    bad_client.name = "expired"
    bad_client.chat_completion = AsyncMock(side_effect=RuntimeError("cookie expired"))

    good_client = MagicMock()
    good_client.name = "healthy"
    good_client.chat_completion = AsyncMock(return_value=fake_stream())

    with patch.object(client_manager, "get_client_attempts", return_value=[bad_client, good_client]), \
         patch.object(client_manager, "mark_client_failed", new=AsyncMock()) as mark_failed:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert '"content": "OK"' in resp.text
    assert "data: [DONE]" in resp.text
    mark_failed.assert_awaited_once_with(bad_client, "cookie expired")


# ─── POST /api/config ───


@pytest.mark.asyncio
async def test_get_config(client):
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    assert "credentials" in resp.json()


@pytest.mark.asyncio
async def test_update_config(client, tmp_path):
    creds_file = tmp_path / "creds.json"
    with patch("app.routes.save_credentials") as mock_save, \
         patch("app.routes.client_manager") as mock_mgr:
        resp = await client.post(
            "/api/config",
            json={
                "credentials": [
                    {
                        "name": "test",
                        "cookies": "token=abc",
                        "user_agent": "ua",
                        "chat_url": "",
                        "api_key": "",
                        "is_active": True,
                    }
                ]
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["message"] == "Loaded 1 accounts"
    mock_save.assert_called_once()
    mock_mgr.reload.assert_called_once()


# ─── POST /api/parse-curl ───


@pytest.mark.asyncio
async def test_parse_curl(client):
    resp = await client.post(
        "/api/parse-curl",
        json={"curl": "curl 'https://example.com' -b 'token=abc' -A 'Mozilla'"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://example.com"
    assert body["cookies"] == "token=abc"
    assert body["user_agent"] == "Mozilla"


@pytest.mark.asyncio
async def test_parse_curl_empty(client):
    resp = await client.post("/api/parse-curl", json={"curl": ""})
    assert resp.status_code == 200
    assert resp.json()["url"] == ""


# ─── POST /api/reload ───


@pytest.mark.asyncio
async def test_reload_accounts(client):
    with patch("app.routes.client_manager") as mock_mgr:
        mock_mgr.count = 3
        resp = await client.post("/api/reload")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "count": 3}
    mock_mgr.reload.assert_called_once()


# ─── Usage / API keys ───


@pytest.mark.asyncio
async def test_usage_endpoint(client):
    with patch.object(usage_tracker, "snapshot", return_value={"totals": {"total_tokens": 3}}):
        resp = await client.get("/api/usage")
    assert resp.status_code == 200
    assert resp.json()["totals"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_reset_usage_endpoint(client):
    with patch.object(usage_tracker, "reset") as mock_reset:
        resp = await client.post("/api/usage/reset")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    mock_reset.assert_called_once()


@pytest.mark.asyncio
async def test_access_key_routes(client):
    with patch("app.routes.access_key_store") as store:
        store.list_public.return_value = []
        store.has_active_keys.return_value = False
        resp = await client.get("/api/access-keys")
    assert resp.status_code == 200
    assert resp.json()["api_keys"] == []

    with patch("app.routes.access_key_store") as store:
        store.add.return_value = {
            "id": "abc",
            "name": "test",
            "key": "sk-test",
            "key_preview": "sk-t...test",
            "is_active": True,
            "created_at": 0,
        }
        resp = await client.post(
            "/api/access-keys",
            json={"name": "test", "key": "sk-test"},
        )
    assert resp.status_code == 200
    assert resp.json()["api_key"]["key"] == "sk-test"


# ─── POST /api/test-account ───


@pytest.mark.asyncio
async def test_test_account_healthy(client):
    mock_health = AsyncMock(return_value=(True, ""))
    with patch("app.mimo_client.MimoWebClient") as MockClient:
        instance = MockClient.return_value
        instance.check_health = mock_health
        instance.close = AsyncMock()

        resp = await client.post(
            "/api/test-account",
            json={"cookies": "token=abc"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_test_account_unhealthy(client):
    with patch("app.mimo_client.MimoWebClient") as MockClient:
        instance = MockClient.return_value
        instance.check_health = AsyncMock(return_value=(False, "cookie expired"))
        instance.close = AsyncMock()

        resp = await client.post(
            "/api/test-account",
            json={"cookies": "token=expired"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "unhealthy"
    assert "cookie expired" in resp.json()["message"]


@pytest.mark.asyncio
async def test_test_account_error(client):
    with patch("app.mimo_client.MimoWebClient") as MockClient:
        MockClient.side_effect = Exception("bad credentials")

        resp = await client.post(
            "/api/test-account",
            json={"cookies": "bad"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"
    assert "bad credentials" in resp.json()["message"]


# ─── GET / ───


@pytest.mark.asyncio
async def test_web_ui(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "MiMo" in resp.text


@pytest.mark.asyncio
async def test_chat_ui(client):
    resp = await client.get("/chat")
    assert resp.status_code == 200
    assert "MiMo Chat" in resp.text


@pytest.mark.asyncio
async def test_overview_ui(client):
    resp = await client.get("/overview")
    assert resp.status_code == 200
    assert "Usage Overview" in resp.text


# ─── Thinking / Reasoning ───


@pytest.mark.asyncio
async def test_chat_thinking_true(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
    )
    with patch.object(client_manager, "get_client", return_value=mock_client):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "think"}],
                "thinking": True,
            },
        )
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["thinking"] is True


@pytest.mark.asyncio
async def test_chat_thinking_not_set(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
    )
    with patch.object(client_manager, "get_client", return_value=mock_client):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    _, kwargs = mock_client.chat_completion.call_args
    assert "thinking" not in kwargs


# ─── Web Search ───


@pytest.mark.asyncio
async def test_chat_web_search_enabled(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
    )
    with patch.object(client_manager, "get_client", return_value=mock_client):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "search this"}],
                "web_search": True,
            },
        )
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["web_search"] is True


@pytest.mark.asyncio
async def test_chat_web_search_not_set(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
    )
    with patch.object(client_manager, "get_client", return_value=mock_client):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    _, kwargs = mock_client.chat_completion.call_args
    assert "web_search" not in kwargs


# ─── Stop Sequences ───


@pytest.mark.asyncio
async def test_chat_stop_string(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
    )
    with patch.object(client_manager, "get_client", return_value=mock_client):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "list items"}],
                "stop": "STOP",
            },
        )
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["stop"] == "STOP"


@pytest.mark.asyncio
async def test_chat_stop_list(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
    )
    with patch.object(client_manager, "get_client", return_value=mock_client):
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "list items"}],
                "stop": ["STOP", "END"],
            },
        )
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["stop"] == ["STOP", "END"]


# ─── Content Parts ───


@pytest.mark.asyncio
async def test_chat_with_text_content_parts(client):
    mock_client = AsyncMock()
    mock_client.chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
            "model": "mimo-v2.5-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
    )
    with patch.object(client_manager, "get_client", return_value=mock_client):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "input_text", "text": "world"},
                        ],
                    }
                ],
            },
        )
    assert resp.status_code == 200
    _, kwargs = mock_client.chat_completion.call_args
    messages = kwargs["messages"]
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][1]["type"] == "input_text"
