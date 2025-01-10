from vllm import LLM, SamplingParams

from other.utils import load_model

cache_dir = "/netscratch/thomas/models/" #TODO Please change to your local directory on the cluster
model_name ="google/gemma-2-9b-it" # Please change to the model you want to use
prompt = "Write me a short story about a cat and a dog."

llm = LLM(model=load_model(model_name=model_name, model_path=cache_dir)) #Load your model
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=3000)
outputs = llm.generate(prompt, sampling_params=sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print("---------------------------------------------------")
