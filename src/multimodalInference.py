import PIL
from vllm import LLM, SamplingParams

from other.utils import load_model

data = PIL.Image.open("../data/example.jpg")
model_name = "Qwen/Qwen2-VL-7B-Instruct"

llm = LLM(
    model=load_model(model_name),
    max_num_seqs=5,
)
stop_token_ids = None

question = "Describe the image"
prompt_template = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                      "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                      f"{question}<|im_end|>\n"
                      "<|im_start|>assistant\n")

inputs = {
                "prompt": prompt_template,
                "multi_modal_data": {
                    "image": data
                },
            }

sampling_params = SamplingParams(temperature=0.2,
                                          max_tokens=128,
                                          stop_token_ids=stop_token_ids,
                                          )

outputs = llm.generate(inputs, sampling_params=sampling_params)
for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
