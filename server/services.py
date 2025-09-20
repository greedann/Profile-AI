from abc import ABC, abstractmethod


class BaseModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class RealModelBackend(BaseModelBackend):
    # Class variables for singleton pattern
    _model = None
    _tokenizer = None
    _initialized = False

    def __init__(self, model_path):
        cls = self.__class__
        if not cls._initialized:
            # Lazy load model and tokenizer
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel
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

    def generate(self, prompt: str) -> str:
        import torch
        cls = self.__class__
        model = cls._model
        tokenizer = cls._tokenizer
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]", 1)[1].strip()
        return generated_text


class DummyModelBackend(BaseModelBackend):
    def generate(self, prompt: str) -> str:
        # Return a dummy response for testing purposes
        return f"[DUMMY] This is a dummy response for the prompt: {prompt}"
