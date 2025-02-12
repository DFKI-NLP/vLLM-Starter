from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os


model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_model = 'mistral-instruct-v0.2-awq'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# GEnerate path to save the model and tokenizer
hf_hub_cache = os.getenv("HF_HUB_CACHE")
quant_model = os.path.join(hf_hub_cache, "quantized-model")

# Save quantized model
model.save_quantized(quant_model)
tokenizer.save_pretrained(quant_model)

print(f'Model is quantized and saved at "{quant_model}"')