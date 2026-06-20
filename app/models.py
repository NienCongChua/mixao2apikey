from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal
import time
import secrets


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[dict]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    model: str = "mimo-v2.5-pro"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    thinking: Optional[Union[bool, dict]] = None
    web_search: Optional[bool] = None
    user: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning: Optional[str] = None


class ChoiceChunk(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[dict] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{secrets.token_hex(12)}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str = "fp_mimo_web2api"
    choices: List[ChoiceChunk]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    reasoning_content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"
    logprobs: Optional[dict] = None


class ChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{secrets.token_hex(12)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str = "fp_mimo_web2api"
    choices: List[Choice]
    usage: Usage = Usage()


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mimo-web2api"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Web UI models
class CredentialEntry(BaseModel):
    name: str
    cookies: str
    user_agent: str = ""
    chat_url: str = ""
    api_key: str = ""
    is_active: bool = True


class AccountConfig(BaseModel):
    credentials: List[CredentialEntry] = []
