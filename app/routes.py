from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, AsyncGenerator
import json
import logging

from app.access_keys import access_key_store
from app.models import (
    ChatRequest,
    ModelList,
    ModelInfo,
    AccountConfig,
)
from app.mimo_client import client_manager, MIMO_MODELS  
from app.config import settings, load_credentials, save_credentials  
from app.usage import usage_tracker
from app.utils import parse_curl_command

logger = logging.getLogger(__name__)
router = APIRouter()


async def verify_auth(authorization: Optional[str] = Header(None)):
    """Verify API key nếu được cấu hình"""
    auth_required = bool(settings.api_key) or access_key_store.has_active_keys()
    if auth_required:
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Use: Authorization: Bearer <your-api-key>"
            )
        scheme, _, key = authorization.partition(" ")
        valid_env_key = bool(settings.api_key) and key == settings.api_key
        valid_local_key = access_key_store.is_valid(key)
        if scheme.lower() != "bearer" or not (valid_env_key or valid_local_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        if valid_local_key:
            await run_in_threadpool(access_key_store.mark_used, key)
    return True


# ─── OpenAI Compatible Endpoints ───

@router.get("/v1/models", response_model=ModelList)
async def list_models(auth=Depends(verify_auth)):
    """GET /v1/models - Liệt kê models có sẵn"""
    models = []
    for model_id in MIMO_MODELS:
        models.append(ModelInfo(id=model_id))
    return ModelList(data=models)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    raw_request: Request,
    auth=Depends(verify_auth),
):
    """
    POST /v1/chat/completions - OpenAI-compatible chat endpoint
    """
    client = client_manager.get_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="No active MiMo accounts. Please add credentials via web UI."
        )

    # Map model name
    model = MIMO_MODELS.get(
        request.model,
        MIMO_MODELS.get(settings.default_model, "mimo-v2.5-pro"),
    )

    # Convert messages to dict
    messages = [m.model_dump(exclude_none=True) for m in request.messages]

    # Prepare kwargs
    kwargs = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens or 4096,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
    }

    if request.reasoning_effort:
        kwargs["reasoning_effort"] = request.reasoning_effort
    if request.thinking is not None:
        kwargs["thinking"] = request.thinking
    if request.stop:
        kwargs["stop"] = request.stop
    if request.web_search is not None:
        kwargs["web_search"] = request.web_search

    if request.stream:
        return StreamingResponse(
            _stream_response(client, messages, model, kwargs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    try:
        result = await client.chat_completion(
            messages=messages,
            model=model,
            stream=False,
            **kwargs,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


async def _stream_response(client, messages, model, kwargs) -> AsyncGenerator[str, None]:
    """Stream response ở định dạng SSE"""
    try:
        stream = await client.chat_completion(
            messages=messages,
            model=model,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_chunk = {
            "error": {"message": str(e), "type": "server_error"}
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


# ─── Web UI ───

@router.get("/api/config")
async def get_config():
    """GET /api/config - Lấy credentials hiện tại"""
    creds = load_credentials()
    return {"credentials": creds}


@router.post("/api/config")
async def update_config(config: AccountConfig):
    """POST /api/config - Cập nhật credentials"""
    creds = [c.model_dump() for c in config.credentials]
    save_credentials(creds)
    client_manager.reload()
    return {"status": "ok", "message": f"Loaded {len(creds)} accounts"}


@router.post("/api/parse-curl")
async def parse_curl(data: dict):
    return parse_curl_command(data.get("curl", ""))


@router.post("/api/test-account")
async def test_account(data: dict):
    """POST /api/test-account - Test account có hoạt động không"""
    from app.mimo_client import MimoWebClient
    try:
        client = MimoWebClient(data)
        healthy, message = await client.check_health()
        await client.close()
        return {
            "status": "healthy" if healthy else "unhealthy",
            "message": message,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/api/reload")
async def reload_accounts():
    """Reload credentials from disk without modifying the file."""
    client_manager.reload()
    return {"status": "ok", "count": client_manager.count}


@router.get("/api/usage")
async def get_usage():
    """Return token usage totals grouped by account/model."""
    data = usage_tracker.snapshot()
    existing = {account.get("name") for account in data.get("accounts", [])}
    empty_totals = {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_tokens": 0,
        "total_tokens": 0,
    }
    credentials = await run_in_threadpool(load_credentials)
    for credential in credentials:
        name = credential.get("name") or "default"
        if name in existing:
            continue
        data.setdefault("accounts", []).append({
            "name": name,
            "totals": dict(empty_totals),
            "models": [],
            "recent": [],
            "updated_at": 0,
        })
    return data


@router.post("/api/usage/reset")
async def reset_usage():
    usage_tracker.reset()
    return {"status": "ok"}


@router.get("/api/access-keys")
async def list_access_keys():
    return {
        "api_keys": access_key_store.list_public(),
        "env_api_key_enabled": bool(settings.api_key),
        "auth_required": bool(settings.api_key) or access_key_store.has_active_keys(),
    }


@router.post("/api/access-keys")
async def add_access_key(data: dict):
    raw_key = data.get("key")
    entry = access_key_store.add(
        name=str(data.get("name", "API key")),
        key=(str(raw_key).strip() if raw_key is not None else "") or None,
    )
    return {"status": "ok", "api_key": entry}


@router.delete("/api/access-keys/{key_id}")
async def delete_access_key(key_id: str):
    if not access_key_store.delete(key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "ok"}


@router.patch("/api/access-keys/{key_id}")
async def update_access_key(key_id: str, data: dict):
    if "is_active" not in data:
        raise HTTPException(status_code=400, detail="Missing is_active")
    if not access_key_store.set_active(key_id, bool(data["is_active"])):
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "ok"}


# ─── Serve web UI ───

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

@router.get("/")
async def web_ui():
    """Web management interface"""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>MiMo Web2API</h1><p>Web UI not found. Run from project root.</p>")


@router.get("/chat")
async def chat_ui():
    """Chat interface"""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "chat.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>MiMo Chat</h1><p>Chat UI not found.</p>")


@router.get("/overview")
async def overview_ui():
    """Usage overview interface"""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "overview.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>MiMo Overview</h1><p>Overview UI not found.</p>")


# Mount static files
def setup_static(app):
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    if os.path.exists(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")
