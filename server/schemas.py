from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class PromptGenerationRequest(BaseModel):
    prompt: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    # limit content length to avoid excessively large payloads
    content: str = Field(..., min_length=0, max_length=10000)


class ChatGenerationRequest(BaseModel):
    messages: List[ChatMessage]


class ResponseModel(BaseModel):
    response: str


# --- Model management schemas ---
class ModelListResponse(BaseModel):
    models: List[str]


class ModelLoadRequest(BaseModel):
    model_name: str


class ModelLoadResponse(BaseModel):
    success: bool
    message: Optional[str] = None
