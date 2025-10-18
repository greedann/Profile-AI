from pydantic import BaseModel
from typing import List, Optional


class PromptGenerationRequest(BaseModel):
    prompt: str


class ChatMessage(BaseModel):
    role: str  # expected: "system" | "user" | "assistant"
    content: str


class ChatGenerationRequest(BaseModel):
    messages: List[ChatMessage]
    # optional tuning parameters (not required by all backends)
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ResponseModel(BaseModel):
    response: str
