from vllm import SamplingParams
from vllm.assets.image import ImageAsset
import argparse

from other.visionModels import *

parser = argparse.ArgumentParser(description="Run a specific LLM model.")
parser.add_argument("--model", type=str, required=True, help="Name of the model to use (LLAVA, LLAVANext, Qwen2VL)")

if parser.model == "QWEN2VL":
    runner = Qwen2VL()

elif parser.model == "LLAVA":
    runner = LLAVA()

elif parser.model == "LLAVANEXT":
    runner = LLAVANext()

#llava_runner = Qwen2VL(model_name = "Qwen/Qwen2-VL-7B-Instruct", model_path = "/netscratch/thomas/models/") #Load the model
llm, prompt, stop_token_ids = runner.query("What is shown in the image?")

sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=stop_token_ids)
inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": ImageAsset("cherry_blossom").pil_image.convert("RGB")
            },
        }
outputs = llm.generate(inputs, sampling_params=sampling_params)

for o in outputs:
    generated_text = o.outputs[0].text
    print("Result=" +generated_text)