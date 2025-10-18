import os
from fastapi import FastAPI, Depends, HTTPException
from schemas import PromptGenerationRequest, ResponseModel, ChatGenerationRequest
from services import DummyModelBackend, BaseModelBackend

app = FastAPI()

BACKEND_TYPE = os.getenv("MODEL_BACKEND", "dummy")  # "real" or "dummy"

def get_backend() -> BaseModelBackend:
    if BACKEND_TYPE == "dummy":
        # return dummy backend for testing
        return DummyModelBackend()
    else:
        # return real backend for production
        from services import RealModelBackend
        # print current working directory
        model_path = os.getcwd() + "/models/yoda"
        return RealModelBackend(model_path=model_path)

@app.post("/generate", response_model=ResponseModel)
def generate(request: PromptGenerationRequest, backend: BaseModelBackend = Depends(get_backend)):
    response = backend.generate(request.prompt)
    return ResponseModel(response=response)


@app.post("/generate_chat", response_model=ResponseModel)
def generate_chat(request: ChatGenerationRequest, backend: BaseModelBackend = Depends(get_backend)):
    try:
        # convert pydantic messages to plain dicts for backend
        messages = [m.dict() for m in request.messages]
        response = backend.generate_chat(messages)
        return ResponseModel(response=response)
    except Exception as e:
        # return 500 with brief message
        raise HTTPException(status_code=500, detail=str(e))
