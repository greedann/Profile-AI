import asyncio
from fastapi import FastAPI, Depends, HTTPException
from schemas import (
    PromptGenerationRequest,
    ResponseModel,
    ChatGenerationRequest,
    ModelListResponse,
    ModelLoadRequest,
    ModelLoadResponse,
)
from services import BaseModelBackend, model_manager
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_backend() -> BaseModelBackend:
    # return current backend from model_manager (falls back to dummy)
    return model_manager.get_backend()


@app.post("/generate", response_model=ResponseModel)
async def generate(request: PromptGenerationRequest, backend: BaseModelBackend = Depends(get_backend)):
    # run blocking generation in threadpool
    try:
        response = await asyncio.to_thread(backend.generate, request.prompt)
        return ResponseModel(response=response)
    except Exception as e:
        logger.exception("Error during generation, detail: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error during generation.")


@app.post("/generate_chat", response_model=ResponseModel)
async def generate_chat(request: ChatGenerationRequest, backend: BaseModelBackend = Depends(get_backend)):
    try:
        # convert pydantic messages to plain dicts for backend
        messages = [m.dict() for m in request.messages]
        response = await asyncio.to_thread(backend.generate_chat, messages)
        return ResponseModel(response=response)
    except Exception as e:
        logger.exception("Error during generation, detail: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error during chat generation.")


@app.get("/models", response_model=ModelListResponse)
async def list_models():
    models = await asyncio.to_thread(model_manager.list_models)
    return ModelListResponse(models=models)


@app.post("/models/load", response_model=ModelLoadResponse)
async def load_model(req: ModelLoadRequest):
    result = await asyncio.to_thread(model_manager.load_model, req.model_name)
    return ModelLoadResponse(success=result.get("success", False), message=result.get("message"))


@app.get("/model/current")
def get_current_model():
    """Return the name of currently loaded model"""
    current = model_manager.get_current_model()
    return {
        "models": [
            current if current else ""
        ]
    }
