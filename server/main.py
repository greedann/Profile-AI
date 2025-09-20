import os
from fastapi import FastAPI, Depends
from schemas import PromptGenerationRequest, ResponseModel
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
