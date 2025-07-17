from vllm import LLM, SamplingParams
import argparse

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Argument parser setup
parser = argparse.ArgumentParser(description="Run vLLM model on sample prompts.")
parser.add_argument(
    "--model",
    type=str,
    default="facebook/opt-125m",
    help="The name or path of the model to use (default: 'facebook/opt-125m')"
)
args = parser.parse_args()
model = args.model
print(f"Loading model '{model}'")

llm = LLM(model=model)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")