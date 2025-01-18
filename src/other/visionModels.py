from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams

class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
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

class AriaLLM(BaseLLM):
    def __init__(self, model_name: str = "rhymes-ai/Aria", **kwargs):
        super().__init__(model_name, **kwargs)
        self.tokenizer_mode = kwargs.get("tokenizer_mode", "slow")
        self.dtype = kwargs.get("dtype", "bfloat16")
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.trust_remote_code = kwargs.get("trust_remote_code", True)
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.stop_token_ids = kwargs.get(
            "stop_token_ids", [93532, 93653, 944, 93421, 1019, 93653, 93519]
        )
        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            tokenizer_mode=self.tokenizer_mode,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            trust_remote_code=self.trust_remote_code,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
        )

    def construct_prompt(self, question: str) -> str:
        return (f"<|im_start|>user\n<fim_prefix><|img|><fim_suffix>\n{question}"
                "<|im_end|>\n<|im_start|>assistant\n")

class ChameleonLLM(BaseLLM):
    def __init__(self, model_name: str = "facebook/chameleon-7b", **kwargs):
        super().__init__(model_name)
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        # Initialize the LLM with the specified parameters.
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache
        )

    def construct_prompt(self, question: str) -> str:
        return f"{question}<image>"

#Currently, does not work!
class DeepseekVL2(BaseLLM):
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl2-tiny", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.stop_token_ids = None  # Set stop_token_ids to None for Deepseek-VL2 models.
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.hf_overrides = {"architectures": ["DeepseekVLV2ForCausalLM"]}
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            hf_overrides=self.hf_overrides
        )

    def construct_prompt(self, question: str) -> str:
        return f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"

class GLM4V(BaseLLM):
    def __init__(self, model_name: str = "THUDM/glm-4v-9b", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 2048)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.stop_token_ids = kwargs.get("stop_token_ids", [151329, 151336, 151338])
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            trust_remote_code=True,
            enforce_eager=True,
            disable_mm_preprocessor_cache=self.kwargs.get("disable_mm_preprocessor_cache", False)
        )

    def construct_prompt(self, question: str) -> str:
        return f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"

class LLAVA(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        super().__init__(model_name)
        self.load_model()

    def load_model(self):
        self.llm = LLM(model=self.model_name)

    def construct_prompt(self, question: str) -> str:
        return f"USER: <image>\n{question}\nASSISTANT:"

class LLAVANext(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", max_model_len: int = 8192):
        super().__init__(model_name)
        self.stop_token_ids = None  # Set stop_token_ids to None for LLAVA Next models.
        self.max_model_len = max_model_len
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(model=self.model_name, max_model_len=self.max_model_len)

    def construct_prompt(self, question: str) -> str:
        return f"[INST] <image>\n{question} [/INST]"

class Qwen2VL(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", max_num_seqs: int = 5):
        super().__init__(model_name)
        self.max_num_seqs = max_num_seqs
        self.stop_token_ids = None
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(model=self.model_name, max_num_seqs=self.max_num_seqs)

    def construct_prompt(self, question: str) -> str:
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n")


