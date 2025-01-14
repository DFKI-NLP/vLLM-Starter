from vllm import SamplingParams
from vllm.assets.image import ImageAsset

from other.visionModels import Qwen2VL

llava_runner = Qwen2VL(model_name = "Qwen/Qwen2-VL-7B-Instruct", model_path = "/netscratch/thomas/models/") #Load the model
llm, prompt, stop_token_ids = llava_runner.query("What is shown in the image?")

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