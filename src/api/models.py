from pydantic import BaseModel, Field


# --- OpenAI 兼容 Chat Completions ---

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "deepseek-v4-pro"
    temperature: float = None
    stream: bool = True


class DeltaMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class ChoiceDelta(BaseModel):
    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: str = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChoiceDelta]


# --- File Management ---

class FileInfo(BaseModel):
    id: str
    filename: str
    bytes: int = 0
    created_at: str = ""


class FileListResponse(BaseModel):
    data: list[FileInfo]


class FileDeleteResponse(BaseModel):
    id: str
    deleted: bool = True


# --- Config ---

class RuntimeConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.65
    top_k: int = 8
    rerank_top_n: int = 3
    temperature: float = 0.7
    relevance_threshold: float = 0.3


class ConfigResponse(BaseModel):
    config: RuntimeConfig


# --- Chat History ---

class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]


# --- Error ---

class ErrorResponse(BaseModel):
    error: dict = Field(
        default_factory=lambda: {"type": "", "message": ""}
    )
