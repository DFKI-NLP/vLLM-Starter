from vllm import LLM, SamplingParams
from pydantic import BaseModel
from typing import List, Optional, Union
from enum import Enum
from vllm.sampling_params import GuidedDecodingParams
from utils import load_model


cache_dir = "/netscratch/thomas/models/" #TODO Please change to your local directory on the cluster
model_name ="google/gemma-2-9b-it" # Please change to the model you want to use
prompt = "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's"

# Guided decoding by JSON using Pydantic schema
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType
json_schema = CarDescription.model_json_schema()



guided_params = GuidedDecodingParams(json=json_schema)
sampling_params = SamplingParams(temperature=0.2,
                                              max_tokens=3000,
                                              guided_decoding= guided_params
                                              )



llm = LLM(model=load_model(model_name=model_name, model_path=cache_dir)) #Load your model
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=3000)
outputs = llm.generate(prompt, sampling_params=sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated JSON: {generated_text!r}")
    print("---------------------------------------------------")




