from abc import ABC, abstractmethod
from typing import List, Dict


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

    def _generate_from_formatted(self, formatted_prompt: str, max_new_tokens: int = 100) -> str:
        """
        formatted_prompt: already includes model-specific wrappers e.g. starts with '<s>[INST]' etc.
        """
        import torch
        cls = self.__class__
        model = cls._model
        tokenizer = cls._tokenizer
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
        print(formatted_prompt)
        return self._generate_from_formatted(formatted_prompt)


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
