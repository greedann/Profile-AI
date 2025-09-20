from pydantic import BaseModel

class PromptGenerationRequest(BaseModel):
    prompt: str

class ResponseModel(BaseModel):
    response: str
