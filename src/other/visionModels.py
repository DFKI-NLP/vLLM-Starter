from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams

class BaseLLM(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs  # Store additional arguments for subclasses
        self.llm = None
        self.stop_token_ids = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def construct_prompt(self, question: str) -> str:
        pass

    def query(self, question: str) -> tuple:
        """
            Query the loaded language model with a specific question.

            This method constructs a usable prompt from the provided question and returns the
            necessary components for model inference. It ensures that the model is
            properly loaded before proceeding.

            Parameters:
                question (str): The input query or question to be processed by the model.

            Returns:
                tuple: A tuple containing:
                    - llm: The loaded language model instance.
                    - prompt (str): The constructed prompt specific to the model.
                    - stop_token_ids (list or None): A list of stop token IDs to define
                      end-of-sequence conditions, or None if not specified.
            """
        if self.llm is None:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load_model() first.")
        prompt = self.construct_prompt(question)
        return self.llm, prompt, self.stop_token_ids

    def get_model_name(self) -> str:
        return self.model_name

class LLAVA(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        super().__init__(model_name)
        self.load_model()

    def load_model(self):
        self.llm = LLM(model=self.model_name)

    def construct_prompt(self, question: str) -> str:
        return f"USER: <image>\n{question}\nASSISTANT:"

class LLAVANext(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", **kwargs):
        super().__init__(model_name, **kwargs)
        self.stop_token_ids = None  # Set stop_token_ids to None for LLAVA Next models.
        self.max_model_len = kwargs.get("max_model_len", 8192)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(model=self.model_name, max_model_len=self.max_model_len)

    def construct_prompt(self, question: str) -> str:
        return f"[INST] <image>\n{question} [/INST]"

class Qwen2VL(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_num_seqs = kwargs.get("max_num_seqs", 5)
        self.stop_token_ids = kwargs.get("stop_token_ids", None)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(model=self.model_name, max_num_seqs=self.max_num_seqs, **self.kwargs)

    def construct_prompt(self, question: str) -> str:
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n")

