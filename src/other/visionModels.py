from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

from other.utils import load_model


class BaseLLM(ABC):
    def __init__(self, model_name: str, model_path: str = "/netscratch/thomas/models/", **kwargs):
        self.model_name = model_name
        self.model_path = model_path
        self.kwargs = kwargs  # Store additional arguments for subclasses
        self.llm = None
        self.stop_token_ids = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_prompt(self, question: str) -> str:
        pass

    def query(self, question: str) -> tuple:
        if self.llm is None:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load_model() first.")
        prompt = self.generate_prompt(question)
        return self.llm, prompt, self.stop_token_ids

    def get_model_name(self) -> str:
        return self.model_name

class LLAVA(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        super().__init__(model_name)
        self.load_model()

    def load_model(self):
        self.llm = LLM(model=load_model(model_name=self.model_name, model_path=self.model_path))

    def generate_prompt(self, question: str) -> str:
        return f"USER: <image>\n{question}\nASSISTANT:"

class LLAVANext(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", **kwargs):
        super().__init__(model_name, **kwargs)
        self.stop_token_ids = None  # Specify if there are specific IDs
        self.max_model_len = kwargs.get("max_model_len", 8192)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(model=load_model(model_name=self.model_name, model_path=self.model_path), max_model_len=self.max_model_len)

    def generate_prompt(self, question: str) -> str:
        return f"[INST] <image>\n{question} [/INST]"

class Qwen2VL(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_num_seqs = kwargs.get("max_num_seqs", 5)
        self.stop_token_ids = kwargs.get("stop_token_ids", None)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(model=load_model(model_name=self.model_name, model_path=self.model_path), max_num_seqs=self.max_num_seqs, **self.kwargs)

    def generate_prompt(self, question: str) -> str:
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n")

