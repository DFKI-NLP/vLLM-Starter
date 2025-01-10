from vllm import LLM, SamplingParams


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


##Download?
"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

# Define the model name and save directory
model_name = "google/t5-large"
save_dir = "/netscratch/models/GogoleT5"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Load the model using vLLM
print(f"Loading model {model_name} with vLLM...")
llm = LLM(model=model_name)

# Save the model using vLLM's underlying Hugging Face components
print(f"Saving model and tokenizer to {save_dir}...")
llm.tokenizer.save_pretrained(save_dir)
llm.model.save_pretrained(save_dir)

print("Model and tokenizer saved successfully!")
"""