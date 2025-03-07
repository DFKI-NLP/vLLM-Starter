from vllm import LLM, SamplingParams
import torch

availableGPUs = torch.cuda.device_count()
print(availableGPUs)

llm = LLM(model="/ds/models/hf-cache-slt/deepseek/DeepSeek-R1.gguf",
          tokenizer="/ds/models/hf-cache-slt/deepseek/deepseek-config",
          hf_config_path="/ds/models/hf-cache-slt/deepseek/deepseek-config",
          enforce_eager=True, tensor_parallel_size=availableGPUs, trust_remote_code=True,
          max_model_len=1024)
sampling_params = SamplingParams(temperature=0.5, max_tokens=2000)


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text\n: {generated_text}")
    print("-" * 80)


conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Why did the Roman Empire fall?",
    },
]
outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)
print_outputs(outputs)