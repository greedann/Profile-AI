from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
        """
        pass


class RealModelBackend(BaseModelBackend):
    # Class variables for singleton pattern
    _model = None
    _tokenizer = None
    _initialized = False
    _init_lock = threading.Lock()
    _model_lock = threading.Lock()

    def __init__(self, model_path):
        cls = self.__class__
        if not cls._initialized:
            with cls._init_lock:
                if not cls._initialized:
                    # Lazy load model and tokenizer
                    try:
                        import torch
                        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
                        from peft import PeftModel

                        # Force GPU memory cleanup before loading
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                        )
                        base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
                        base_model = AutoModelForCausalLM.from_pretrained(
                            base_model_name,
                            quantization_config=bnb_config,
                            device_map="auto"
                        )
                        cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
                        cls._model = PeftModel.from_pretrained(base_model, model_path)
                        cls._model.eval()
                        cls._initialized = True
                        logger.info("Model loaded from %s", model_path)
                    except Exception:
                        logger.exception("Failed to initialize model from %s", model_path)
                        raise

    def _generate_from_formatted(self, formatted_prompt: str, max_new_tokens: int = 100) -> str:
        """
        formatted_prompt: already includes model-specific wrappers e.g. starts with '<s>[INST]' etc.
        """
        import torch
        cls = self.__class__
        model = cls._model
        tokenizer = cls._tokenizer
        # protect model usage with a lock to avoid concurrent calls that may not be thread-safe
        with cls._model_lock:
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            with torch.no_grad():
                # generate returns full sequences (prompt + generated). To avoid echoing
                # the input prompt back in the decoded text, slice out the newly generated
                # token ids using the input length.
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                )

        # outputs is a tensor of shape (num_return_sequences, seq_len)
        seq = outputs[0]
        # generated tokens are those after the input length
        gen_tokens = seq[input_ids.shape[-1]:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]", 1)[1].strip()
        return generated_text

    def generate(self, prompt: str) -> str:
        # keep backward compatible single-prompt interface
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        return self._generate_from_formatted(formatted_prompt)

    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role":"system"|"user"|"assistant", "content":"..."}, ...]
        Format the chat messages into the model's expected prompt structure.
        The format is:
        <s>
        <<SYS>>
        ...system messages...
        <</SYS>>
        [INST] user message 1 [/INST]
        assistant message 1
        [INST] user message 2 [/INST]
        assistant message 2
        ...
        """
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        parts = []
        if system_texts:
            parts.append("<<SYS>>\n" + "\n".join(system_texts) + "\n<</SYS>>\n")
        # For chat history, include assistant content after respective user blocks.
        # We'll go through messages in order and append formatted segments.
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                parts.append(f"[INST] {content} [/INST]\n")
            elif role == "assistant":
                # include assistant reply in history (plain)
                parts.append(content + "\n")
            # system already handled
        full_inner = "".join(parts).strip()
        # final formatted prompt for model
        formatted_prompt = f"<s>{full_inner}"
        return self._generate_from_formatted(formatted_prompt)

    @classmethod
    def unload_model(cls):
        """Unload model to free resources."""
        try:
            import torch
            import gc
            
            with cls._init_lock:
                if cls._model is not None:
                    del cls._model
                    cls._model = None
                if cls._tokenizer is not None:
                    del cls._tokenizer
                    cls._tokenizer = None
                
                cls._initialized = False
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache multiple times
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                logger.info("Model unloaded and CUDA cache cleared")
        except Exception:
            logger.exception("Error during model unload")


class DummyModelBackend(BaseModelBackend):
    def generate(self, prompt: str) -> str:
        # Return a dummy response for testing purposes
        return f"[DUMMY] This is a dummy response for the prompt: {prompt}"

    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        # simple deterministic dummy: echo last user message with tags + context count
        last_user = None
        system_count = 0
        for m in messages:
            if m.get("role") == "user":
                last_user = m.get("content")
            if m.get("role") == "system":
                system_count += 1
        return f"[DUMMY-CHAT] system_count={system_count} last_user={last_user}"


class ModelManager:
    """Simple manager to list available pretrained models and load by name.

    It scans the `models/` directory (one level) and treats each child as a model name.
    Loading a model will initialize RealModelBackend with the selected model path.
    """

    def __init__(self, models_root: Optional[str] = None):
        self.models_root = Path(models_root) if models_root else Path.cwd() / "models"
        self._lock = threading.Lock()
        self._current_backend: Optional[BaseModelBackend] = None
        self._current_model_name: Optional[str] = None

    def list_models(self) -> List[str]:
        if not self.models_root.exists():
            return []
        models = [p.name for p in self.models_root.iterdir() if p.is_dir()]
        return models

    def load_model(self, model_name: str) -> Dict[str, str]:
        model_path = self.models_root / model_name
        if not model_path.exists() or not model_path.is_dir():
            return {"success": False, "message": f"Model '{model_name}' not found"}

        with self._lock:
            try:
                # Completely unload previous model BEFORE loading new one
                if self._current_backend is not None:
                    if isinstance(self._current_backend, RealModelBackend):
                        RealModelBackend.unload_model()
                
                self._current_backend = None
                self._current_model_name = None
                
                # Longer delay to ensure complete cleanup
                import time
                time.sleep(1.0)

                # Now load the new model
                backend = RealModelBackend(str(model_path))
                self._current_backend = backend
                self._current_model_name = model_name
                return {"success": True, "message": f"Model '{model_name}' loaded"}
            except Exception as e:
                logger.exception("Failed to load model %s", model_name)
                self._current_backend = None
                self._current_model_name = None
                return {"success": False, "message": str(e)}

    def get_current_model(self) -> Optional[str]:
        """Return the name of currently loaded model"""
        with self._lock:
            return self._current_model_name

    def get_backend(self) -> BaseModelBackend:
        # fallback to dummy if none
        if self._current_backend is None:
            return DummyModelBackend()
        return self._current_backend


# singleton manager (module-level)
model_manager = ModelManager()
